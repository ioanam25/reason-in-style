import fcntl
import json
import os
import pathlib
import shutil

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainerCallback,
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
from trl.scripts.utils import TrlParser, init_zero_verbose, ScriptArguments
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, snapshot_download

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional

import torch.nn.functional as F


@dataclass
class ModelConfigWithBase(ModelConfig):
    model_subfolder: Optional[str] = None
    skip_upload_optimizer_states: Optional[bool] = True
    skip_hub_upload: Optional[bool] = True


def ensure_model_cached(repo_id: str) -> str:
    """Download model once; other ranks/processes wait on a shared file lock.

    Local directories with config.json are returned as-is (no Hub lookup).
    """
    if os.path.isdir(repo_id) and os.path.isfile(os.path.join(repo_id, "config.json")):
        return os.path.abspath(repo_id)

    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    repo_cache = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    has_config = any(
        (pathlib.Path(root) / "config.json").is_file()
        for root, _, _ in os.walk(snapshots_dir)
    )
    if not has_config:
        no_exist = os.path.join(repo_cache, ".no_exist")
        if os.path.isdir(no_exist):
            shutil.rmtree(no_exist)

    lock_path = os.path.join(cache_dir, f".download-lock-{repo_id.replace('/', '--')}")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return snapshot_download(repo_id)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)



class SaveBestCheckpointCallback(TrainerCallback):
    """Copy the current val-best checkpoint to output_dir/best whenever eval_loss improves.

    HuggingFace still writes checkpoint-{step}; save_total_limit may rotate those.
    `best/` is a stable path for eval and is not matched by checkpoint-[0-9]+ rotation.
    """

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        src = state.best_model_checkpoint
        if not src or not os.path.isdir(src):
            return
        dest = os.path.join(args.output_dir, "best")
        marker = os.path.join(dest, "source_checkpoint.txt")
        abs_src = os.path.abspath(src)
        if os.path.isfile(marker):
            prev = pathlib.Path(marker).read_text().splitlines()[0].strip()
            if prev == abs_src:
                return
        tmp = dest + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        pathlib.Path(tmp, "source_checkpoint.txt").write_text(
            f"{abs_src}\nbest_metric={state.best_metric}\nglobal_step={state.global_step}\n"
        )
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.rename(tmp, dest)


class WeightedCompletionCollator(DataCollatorForCompletionOnlyLM):
    """Pass through per-example IS weights; drop them before the completion collator."""

    def __call__(self, features):
        weights = []
        has_w = False
        clean = []
        for ex in features:
            ex = dict(ex)
            if "is_weight" in ex:
                has_w = True
                weights.append(float(ex.pop("is_weight")))
            else:
                weights.append(1.0)
            clean.append(ex)
        batch = super().__call__(clean)
        if has_w:
            batch["is_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedSFTTrainer(SFTTrainer):
    """SFT NLL multiplied by p(c|x)/q(c|x) per example, then mean over the batch.

    Packing concatenates examples and destroys the IS estimator, so callers must
    disable packing when is_weight is present.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("is_weight", None)
        if weights is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        mask = shift_labels.ne(-100)
        tok_per_ex = mask.sum(dim=-1).clamp(min=1)
        seq_loss = (token_loss * mask).sum(dim=-1) / tok_per_ex
        w = weights.to(device=seq_loss.device, dtype=seq_loss.dtype)
        loss = (seq_loss * w).mean()

        if "labels" in inputs and not getattr(self.args, "use_liger", False):
            predictions = shift_logits.argmax(dim=-1)
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)
            accuracy = (
                (correct_tokens.sum() / total_tokens.sum()).item() if total_tokens.sum() > 0 else 0.0
            )
            if hasattr(self, "_metrics"):
                self._metrics.setdefault("mean_token_accuracy", []).append(accuracy)

        return (loss, outputs) if return_outputs else loss


def configure_tokenizer_and_response_template(model_name: str, tokenizer):
    """
    Use the model's native chat template when it exists. Otherwise fall back to a
    simple generic chat template so base causal LMs can still be used for SFT.
    """
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if getattr(tokenizer, "chat_template", None):
        tmpl = str(tokenizer.chat_template)
        # Local soft-special bases live under models/scas-soft-special-* (no "Qwen2.5" in path).
        if (
            "Qwen2.5" in model_name
            or "qwen" in model_name.lower()
            or "<|im_start|>" in tmpl
        ):
            return tokenizer, "<|im_start|>assistant\n"
        if "Llama-3.2" in model_name:
            return tokenizer, "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if "DeepSeek-R1-Distill-Qwen-1.5B" in model_name:
            return tokenizer, "<｜Assistant｜><think>\n"
        return tokenizer, "<|assistant|>\n"

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
    return tokenizer, "<|assistant|>\n"


def load_parquet_dataset(path: str) -> Dataset:
    return Dataset(pq.ParquetFile(path).read())


def _tokenized_cache_path(dataset_dir: str, split_file: str) -> str:
    return os.path.join(dataset_dir, ".hf_tokenized", pathlib.Path(split_file).stem)


def load_dataset_splits(args):
    from datasets import load_from_disk

    train_cache = _tokenized_cache_path(args.dataset_name, args.dataset_train_split)
    if not os.path.isfile(os.path.join(train_cache, "dataset_info.json")):
        raise FileNotFoundError(
            f"Missing tokenized cache {train_cache}. "
            "Run: python scripts/prepare_sft_hf_cache.py <parquet_dir> "
            "--tokenizer <student> --max-seq-length 4096 "
            "(or sbatch scripts/data/prepare-scas-sft-cache.slurm)."
        )
    train_dataset = load_from_disk(train_cache)

    if args.dataset_test_split != "test":
        test_cache = _tokenized_cache_path(args.dataset_name, args.dataset_test_split)
        if not os.path.isfile(os.path.join(test_cache, "dataset_info.json")):
            raise FileNotFoundError(f"Missing tokenized cache {test_cache}")
        test_dataset = load_from_disk(test_cache)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfigWithBase))
    args, training_args, model_config = parser.parse_args_and_config()

    train_dataset, test_dataset = load_dataset_splits(args)

    # Subsample for debugging
    # train_dataset = train_dataset.select(range(10000))  # TODO: remove this

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if model_config.model_subfolder is not None:
        model_kwargs["subfolder"] = model_config.model_subfolder

    ensure_model_cached(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    tokenizer, response_template = configure_tokenizer_and_response_template(
        model_config.model_name_or_path, tokenizer
    )

    use_weights = "is_weight" in train_dataset.column_names
    if use_weights:
        mean_w = float(sum(train_dataset["is_weight"]) / max(len(train_dataset), 1))
        print(
            f"IS weights: n={len(train_dataset)} mean_is_weight={mean_w:.4f}",
            flush=True,
        )
    if use_weights and getattr(training_args, "packing", False):
        print(
            "is_weight present: disabling packing so p/q is applied per example",
            flush=True,
        )
        training_args.packing = False
    if use_weights and not getattr(training_args, "group_by_length", False):
        # Same estimator; pad each microbatch to its own max instead of a random 4k pair.
        print(
            "is_weight present: enabling group_by_length so similar-length traces batch together",
            flush=True,
        )
        training_args.group_by_length = True

    collator = (
        WeightedCompletionCollator(response_template, tokenizer=tokenizer)
        if use_weights
        else DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    )
    trainer_cls = WeightedSFTTrainer if use_weights else SFTTrainer

    #peft_config = get_peft_config(model_config)
    #print("peft config:", peft_config)
    trainer = trainer_cls(
        model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        callbacks=[SaveBestCheckpointCallback()],
        #peft_config=peft_config,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Upload everything to hub (only from main process)
    if trainer.is_world_process_zero() and not model_config.skip_hub_upload:
        api = HfApi()
        api.create_repo(
            repo_id=f"xxx98/{os.path.basename(training_args.output_dir)}",
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=training_args.output_dir,
            repo_id=f"xxx98/{os.path.basename(training_args.output_dir)}",
            repo_type="model",
            ignore_patterns=["*.pt"] if model_config.skip_upload_optimizer_states else None,
        )


if __name__ == "__main__":
    main()
