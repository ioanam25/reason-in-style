import importlib
import re
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

class GenerationPipeline:
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        mock: bool = False,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
    ):
        self.mock = mock
        self.device = device
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        
        if not self.mock:
            if "cuda" not in device:
                raise ValueError("This script is vLLM-only and expects a CUDA device.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._init_vllm(model_name, device, tensor_parallel_size)

    def _init_vllm(self, model_name: str, device: str, tensor_parallel_size: int) -> None:
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is required for non-mock generation. Install it on a Linux CUDA machine."
            ) from exc

        print("Initializing vLLM generation engine...")
        dtype = "bfloat16" if "cuda" in device else "float16"
        self.SamplingParams = vllm.SamplingParams
        self.llm = vllm.LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )

    def generate_traces(self, questions: List[str], question_ids: List[str], k: int = 4) -> Dataset:
        data = {"question_id": [], "question": [], "trace": [], "answer": []}
        
        if self.mock:
            for q_id, q in tqdm(zip(question_ids, questions), total=len(questions), desc="Generating mock traces"):
                for i in range(k):
                    data["question_id"].append(q_id)
                    data["question"].append(q)
                    data["trace"].append(f"Reasoning trace {i} for question {q_id}... Therefore the result bounds are:\n<answer>{i}</answer>")
                    data["answer"].append(str(i))
            return Dataset.from_dict(data)

        # Batch Prompts Pre-computation
        prompts = []
        for q in questions:
            messages = [
                {"role": "user", "content": f"You are a strict logical reasoning assistant. Solve the following question. You MUST format your final numerical answer enclosed exactly within <answer> and </answer> XML tags.\n\nQuestion: {q}\n\nLet's think step-by-step."}
            ]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        sampling_params = self.SamplingParams(
            n=k,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )
        print("Submitting prompts to vLLM...")
        outputs = self.llm.generate(prompts, sampling_params)

        for q_id, q, req_out in tqdm(
            zip(question_ids, questions, outputs),
            total=len(questions),
            desc="Collecting traces",
        ):
            for out in req_out.outputs:
                gen_text = out.text.strip()
                answer_match = self._parse_answer(gen_text)
                if answer_match:
                    data["question_id"].append(q_id)
                    data["question"].append(q)
                    data["trace"].append(gen_text)
                    data["answer"].append(answer_match)
                        
        return Dataset.from_dict(data)

    def _parse_answer(self, text: str) -> Optional[str]:
        """Strict parsing heuristic requiring <answer> bounds"""
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        return "Unknown" # Discard trace downstream if it fails parsing completely


def load_gsm8k_questions() -> tuple[List[str], List[str]]:
    print("Loading GSM8K dataset from Hugging Face...")

    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        questions = gsm8k["question"]
    except NotImplementedError as exc:
        if "LocalFileSystem" not in str(exc):
            raise

        print(
            "Standard GSM8K loading hit a datasets/fsspec cache compatibility issue. "
            "Retrying with streaming mode..."
        )
        gsm8k_stream = load_dataset("gsm8k", "main", split="train", streaming=True)
        questions = [example["question"] for example in gsm8k_stream]

    ids = [f"gsm8k_{i}" for i in range(len(questions))]
    print(f"Loaded {len(questions)} base questions.")
    return questions, ids

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reasoning traces.")
    parser.add_argument("--mock", type=lambda x: str(x).lower() == 'true', default=False, help="Use mock generation logic")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch precision dtype")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Model HF ID")
    parser.add_argument("--output", type=str, default="generated_traces_dataset", help="Output path for HF Dataset")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    
    args = parser.parse_args()

    if args.mock:
        # Synthetic questions for execution validation
        questions = ["What is 2+2?", "Solve 3x=9"]
        ids = ["base_1", "base_2"]
    else:
        questions, ids = load_gsm8k_questions()
    
    print(
        f"Initializing GenerationPipeline on {args.device} "
        f"[mock={args.mock}, precision={args.precision}, tp={args.tensor_parallel_size}]"
    )
    pipeline = GenerationPipeline(
        model_name=args.model,
        mock=args.mock,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    dataset = pipeline.generate_traces(questions, ids, k=4)
    print(f"Generation complete. Dataset size: {len(dataset)}")
    
    dataset.save_to_disk(args.output)
    print(f"Dataset successfully saved to {args.output}")
