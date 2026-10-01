import json
import os
import shutil

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from vllm import LLM, SamplingParams
import vllm.envs as envs
from dataclasses import dataclass, field
from trl.scripts.utils import ScriptArguments, TrlParser, init_zero_verbose
from torch.utils.data import DataLoader
from typing import Optional
import argparse
import tempfile
from huggingface_hub import HfApi, upload_file

from datasets import Dataset, load_dataset, DatasetDict


def configure_tokenizer(model_name: str, tokenizer):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    tokenizer.chat_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{{bos_token}}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<|user|>\n{{message['content']}}\n"
        "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{message['content']}}{{eos_token}}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    return tokenizer



def parse_arguments():
    parser = argparse.ArgumentParser(description="Sample proofs from a model and verify them")
    parser.add_argument("--dataset", type=str, required=True, help="Parquet file containing theorem statements")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of proofs to sample per problem")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum iterations for verification")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for token handling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for generation")
    parser.add_argument("--repo_id", type=str, default="xxx98/lean-proofs-bsz2048", 
                        help="Hugging Face repository ID for uploading results")
    parser.add_argument("--chunck_idx", type=int, default=None, help="Chunk index for the dataset")
    parser.add_argument("--n_chunks", type=int, default=None, help="Number of chunks for the dataset")
    parser.add_argument("--max_examples", type=int, default=0, help="Optional cap on the number of examples to sample (0 = all)")
    return parser.parse_args()

class StopOnPhrases(StoppingCriteria):
    def __init__(self, stop_phrases, tokenizer):
        self.stop_phrases = stop_phrases
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Check if any of the stop phrases are in the generated text
        for phrase in self.stop_phrases:
            if phrase in generated_text:
                return True
        return False


def get_model_outputs(model_path, tokenizer, texts, max_tokens, temperature, n):
    
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    
    # Limit tensor_parallel_size to maximum 4, use remaining for data parallelism
    tensor_parallel_size = len(available_gpus) if "qwen" not in model_path.lower() else min(len(available_gpus), 4)
    tensor_parallel_size = tensor_parallel_size if "0.5b" not in model_path.lower() else min(tensor_parallel_size, 2)
    
    llm = LLM(model=model_path,
              enforce_eager=True,
              tensor_parallel_size=tensor_parallel_size,
              trust_remote_code=True, 
              # swap_space=60,
              gpu_memory_utilization=0.96)
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens, 
                                     n=n)
    completions = llm.generate(texts, sampling_params)
    generated_responses = []
    for completion in completions:
        # Each completion has n outputs due to n generations
        completion_responses = [output.text for output in completion.outputs]
        generated_responses.extend(completion_responses)
    generated_text_list = generated_responses
    
    return generated_text_list

def main():
    args = parse_arguments()
    
    # Add the checkpoint name to the output directory
    checkpoint_name = os.path.basename(args.model_name)
    args.output_dir = os.path.join(args.output_dir, f"{checkpoint_name}-temperature_{args.temperature}")

    # Create filename from output directory path
    # Clean up the path and convert to a safe filename
    clean_path = args.output_dir.strip("./").replace("/", "___")
    
    # Add chunk suffix if using chunking
    if args.chunck_idx is not None:
        assert args.n_chunks is not None, "n_chunks must be provided if chunck_idx is provided"
        filename = f"{clean_path}___chunk{args.chunck_idx}of{args.n_chunks}.json"
    else:
        assert args.n_chunks is None, "n_chunks must be None if chunck_idx is not provided"
        filename = f"{clean_path}.json"
    
    # If the filename in the repo already exists, skip
    api = HfApi()
    repo_id = args.repo_id
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"Created repository: {repo_id}")
    
    skip_existing = False  # TODO
    if skip_existing:
        if filename in api.list_repo_files(repo_id=repo_id, repo_type="dataset"):
            print(f"File {filename} already exists in repository {repo_id}, skipping...")
            return
        else:
            print(f"File {filename} does not exist in repository {repo_id}, generating...")
    
    test_file = os.path.join(args.dataset)
    
    # Load from parquet
    test_dataset = Dataset.from_parquet(test_file)
    if args.max_examples and args.max_examples > 0:
        test_dataset = test_dataset.select(range(min(args.max_examples, len(test_dataset))))

    if args.chunck_idx is not None:
        dataset_size = len(test_dataset)
        start_idx = args.chunck_idx * dataset_size // args.n_chunks
        end_idx = (args.chunck_idx + 1) * dataset_size // args.n_chunks if args.chunck_idx != args.n_chunks - 1 else dataset_size
        test_dataset = test_dataset.select(range(start_idx, end_idx))
    else:
        assert args.n_chunks is None, "n_chunks must be None if chunck_idx is not provided"

    checkpoint = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer = configure_tokenizer(checkpoint, tokenizer)

    texts = []
    for example in test_dataset:
        if "prompt" in example:
            prompt = example["prompt"]
        else:
            messages = example["messages"]
            assert len(messages) == 2, "messages must have length 2"
            prompt = [messages[0]]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    
    try:
        generated_text_list = get_model_outputs(os.path.join(checkpoint), tokenizer, texts, args.max_tokens, args.temperature, args.num_samples)
        # print("generated_text_list[0]:", generated_text_list[0])
        assert len(generated_text_list) == len(test_dataset) * args.num_samples, f"len(generated_text_list): {len(generated_text_list)}, len(test_dataset): {len(test_dataset)}, args.num_samples: {args.num_samples}"
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Save the data exactly as generated
    all_items = []
    for example_idx, example in enumerate(test_dataset):
        generated_texts = generated_text_list[example_idx * args.num_samples:(example_idx + 1) * args.num_samples]
        # Handle different dataset formats
        item = {
            "nums": example["nums"],
            "target": example["target"],
            "search_type": example["search_type"],
            "heuristic": example["heuristic"],
            "rating": example["rating"],
            "model_output_list": generated_texts,
            "type": "test"
        }
        
        all_items.append(item)
    
    # Upload to Hugging Face Hub
    
    # Save to temporary file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(all_items, tmp_file, indent=2)
        tmp_path = tmp_file.name
    
    try:
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
    
    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    print(f"File name: {filename}")


if __name__ == "__main__":
    main()
