#!/usr/bin/env python3
"""
Inspect decodings for one problem (one question_id) without running full eval.

For a chosen question_id with >=4 traces, this script prints:
- Ground truth traces (all 4)
- Greedy reconstructions from each trace's own (c,z)
- c and z summary stats for each trace
- Cross reconstructions:
    (c1+z1) vs (c2+z1)
    (c1+z2) vs (c2+z2)
These pairs should look very similar if c captures question-level info and z is trace-style.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.module import DisentangledLightningModule, get_decoder_sequence_start_token_id


SPLIT_PREFERENCE = ("validation", "val", "test", "train")
TRAIN_SPLIT_PREFERENCE = ("train",)
VAL_SPLIT_PREFERENCE = ("validation", "val")


def split_records_by_question(records: list[dict], validation_fraction: float) -> tuple[list[dict], list[dict]]:
    """
    Match training split logic in main.py:
    - group by question_id
    - sort question_ids
    - last fraction (rounded, at least 1, at most n-1) go to validation
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[str(r["question_id"])].append(r)

    question_ids = sorted(grouped)
    if len(question_ids) < 2 or validation_fraction <= 0:
        return records, records

    validation_groups = max(1, int(round(len(question_ids) * validation_fraction)))
    validation_groups = min(validation_groups, len(question_ids) - 1)
    validation_ids = set(question_ids[-validation_groups:])

    train_records: list[dict] = []
    validation_records: list[dict] = []
    for qid in question_ids:
        (validation_records if qid in validation_ids else train_records).extend(grouped[qid])
    return train_records, validation_records


def _pick_split(dd: DatasetDict, preferred: tuple[str, ...], dataset_path: Path) -> tuple[str, Dataset]:
    for name in preferred:
        if name in dd:
            return name, dd[name]
    available = ", ".join(sorted(dd.keys()))
    raise ValueError(f"No split in {preferred} for {dataset_path}. Available: {available}")


def load_records(dataset_path: Path, split: str = "auto") -> tuple[str, list[dict]]:
    """
    Load records and return (split_name, rows).

    split:
      - auto: pick first available from SPLIT_PREFERENCE
      - train: require a train split
      - validation: require a validation/val split
    """
    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if split == "auto":
            split_name, ds = _pick_split(loaded, SPLIT_PREFERENCE, dataset_path)
        elif split == "train":
            split_name, ds = _pick_split(loaded, TRAIN_SPLIT_PREFERENCE, dataset_path)
        elif split == "validation":
            split_name, ds = _pick_split(loaded, VAL_SPLIT_PREFERENCE, dataset_path)
        else:
            raise ValueError(f"Unknown split={split!r}")
        rows = [dict(row) for row in tqdm(ds, desc=f"Loading {split_name} records", unit="rec")]
        if not rows:
            raise ValueError(f"No records found at {dataset_path} split={split_name}")
        return split_name, rows

    if not isinstance(loaded, Dataset):
        raise TypeError(f"Unsupported dataset type: {type(loaded)!r}")

    if split != "auto":
        raise ValueError(
            f"Requested split={split!r}, but {dataset_path} is a single Dataset (no splits). "
            "Use --split auto, or save a DatasetDict with train/validation."
        )
    rows = [dict(row) for row in tqdm(loaded, desc="Loading records", unit="rec")]
    if not rows:
        raise ValueError(f"No records found at {dataset_path}")
    return "(no_split)", rows


def resolve_train_val_records(dataset_path: Path, validation_fraction: float) -> tuple[list[dict], list[dict], str]:
    """
    Reproduce training behavior from main.py.resolve_datasets when only --data-path is given.

    Returns (train_records, val_records, mode_str).
    """
    loaded = load_from_disk(str(dataset_path))

    # DatasetDict case
    if isinstance(loaded, DatasetDict):
        if "train" in loaded:
            train_ds = loaded["train"]
        else:
            # Training selects a preferred split as "train" if no explicit train.
            _, train_ds = _pick_split(loaded, ("train", "validation", "val", "test"), dataset_path)
        train_records = [dict(r) for r in train_ds]

        if "validation" in loaded:
            val_records = [dict(r) for r in loaded["validation"]]
        elif "val" in loaded:
            val_records = [dict(r) for r in loaded["val"]]
        else:
            train_records, val_records = split_records_by_question(train_records, validation_fraction)
        return train_records, val_records, "disk"

    # Flat Dataset case
    if isinstance(loaded, Dataset):
        records = [dict(r) for r in loaded]
        train_records, val_records = split_records_by_question(records, validation_fraction)
        return train_records, val_records, "disk"

    raise TypeError(f"Unsupported dataset object loaded from {dataset_path}: {type(loaded)!r}")


def filter_records_to_questions(records, max_questions, field="question_id", seed=0):
    if max_questions is None:
        return records
    qids = list({r.get(field) for r in records if r.get(field) is not None})
    if len(qids) <= max_questions:
        return records
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(qids, size=max_questions, replace=False))
    return [r for r in records if r.get(field) in chosen]


def _encoder_tokenizer_for_traces(model):
    enc_tok = getattr(model.encoder, "tokenizer", None)
    dec_tok = getattr(model.decoder, "tokenizer", None)
    if enc_tok is not None and dec_tok is not None:
        return enc_tok
    return dec_tok or enc_tok


def _decoder_tokenizer_for_decode(model):
    dec_tok = getattr(model.decoder, "tokenizer", None)
    if dec_tok is not None:
        return dec_tok
    return getattr(model.encoder, "tokenizer", None)


def _greedy_prefix_decode(module, c_z: torch.Tensor, device: torch.device, max_new_tokens: int = 256):
    """Prefix-conditional greedy decode (argmax each step)."""
    model = module.model.to(device)
    model.eval()
    tokenizer = _decoder_tokenizer_for_decode(model)
    if tokenizer is None:
        return None
    start_id = get_decoder_sequence_start_token_id(tokenizer)
    if start_id is None:
        return None

    generated = torch.tensor([[start_id]], device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            logits = model.decoder(generated, c_z)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def _filter_logits_top_k_top_p(
    logits: torch.Tensor, *, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.

    Expects logits shape [vocab]. Returns filtered logits where removed tokens are -inf.
    """
    x = logits
    if top_k is not None and top_k > 0 and top_k < x.numel():
        kth = torch.topk(x, k=top_k).values[-1]
        x = torch.where(x < kth, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype), x)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Remove tokens with cumulative prob above threshold; keep at least 1 token
        to_remove = cumprobs > top_p
        if to_remove.any():
            to_remove[0] = False
        filtered_sorted = torch.where(
            to_remove, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype), sorted_logits
        )
        # Scatter back
        x2 = torch.full_like(x, float("-inf"))
        x2.scatter_(0, sorted_idx, filtered_sorted)
        x = x2

    return x


def _sample_prefix_decode(
    module,
    c_z: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = None,
):
    """Prefix-conditional stochastic decode (temperature + top-p/top-k sampling)."""
    model = module.model.to(device)
    model.eval()
    tokenizer = _decoder_tokenizer_for_decode(model)
    if tokenizer is None:
        return None
    start_id = get_decoder_sequence_start_token_id(tokenizer)
    if start_id is None:
        return None

    temperature = float(temperature)
    if temperature <= 0:
        # Avoid divide-by-zero; fall back to greedy behavior.
        return _greedy_prefix_decode(module, c_z, device=device, max_new_tokens=max_new_tokens)

    generated = torch.tensor([[start_id]], device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            logits = model.decoder(generated, c_z)[:, -1, :].squeeze(0)  # [vocab]
            logits = logits / temperature
            logits = _filter_logits_top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                next_id = logits.argmax().view(1)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def encode_trace_to_c_z(module, trace: str, device: torch.device):
    model = module.model.to(device)
    model.eval()
    enc_tok = _encoder_tokenizer_for_traces(model)
    if enc_tok is None:
        return None, None
    enc = enc_tok([trace], padding=True, truncation=True, return_tensors="pt", max_length=model.max_len)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        c, z, _ = model(ids, mask, decode=False)
    return c, z


def greedy_decode_from_c_and_z(module, c: torch.Tensor, z: torch.Tensor, device: torch.device, max_new_tokens: int = 256):
    c_z = torch.cat([c, z], dim=-1)
    return _greedy_prefix_decode(module, c_z, device=device, max_new_tokens=max_new_tokens)


def sample_decode_from_c_and_z(
    module,
    c: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = None,
):
    c_z = torch.cat([c, z], dim=-1)
    return _sample_prefix_decode(
        module,
        c_z,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _summ(x: torch.Tensor | None):
    if x is None:
        return None
    x0 = x[0].detach().float().cpu().numpy()
    return {
        "shape": list(x0.shape),
        "norm": float(np.linalg.norm(x0)),
        "mean": float(x0.mean()),
        "std": float(x0.std()),
        "first8": [float(v) for v in x0[:8]],
    }


def _pick_qid_with_k_traces(groups: dict, k: int, seed: int):
    eligible = [qid for qid, g in groups.items() if len(g) >= k]
    if not eligible:
        return None
    rng = np.random.default_rng(seed)
    return str(rng.choice(eligible))


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type != "cuda":
        return device

    try:
        torch.cuda.init()
        return device
    except Exception as e:
        print(
            "\n[warn] --device=cuda requested but CUDA init failed.\n"
            f"       torch.cuda.is_available()={torch.cuda.is_available()}  "
            f"torch.cuda.device_count()={torch.cuda.device_count()}\n"
            f"       init_error={repr(e)}\n"
            "       Falling back to CPU.\n"
        )
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Inspect decodings for one question_id")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--data-path", type=str, required=True, help="Path to HF dataset on disk")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=("auto", "train", "validation", "both"),
        help=(
            "Which dataset split to inspect. If the dataset on disk does not provide a validation split, "
            "we reproduce training behavior from main.py and create validation by question_id using "
            "--validation-fraction."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="When the dataset has no val split, hold out this fraction of question_ids for validation (training logic).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optionally subsample distinct question_id for faster load (0/None = all).",
    )
    parser.add_argument(
        "--qid",
        type=str,
        default=None,
        help="question_id to inspect (must have >=4 traces). If omitted, pick random eligible.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of traces to print (default 4)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for decode")
    parser.add_argument(
        "--decode",
        type=str,
        default="greedy",
        choices=("greedy", "sample", "both"),
        help="Decoding strategy for per-trace reconstructions.",
    )
    parser.add_argument(
        "--cross-decode",
        type=str,
        default="sample",
        choices=("greedy", "sample", "both"),
        help="Decoding strategy for cross reconstructions (swap demos). Default is sampling.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (only for --*-decode sample/both)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p (only for --*-decode sample/both)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables; only for --*-decode sample/both)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of sampled decodes to print per (c,z) when sampling is enabled.",
    )
    parser.add_argument(
        "--gemini-styles",
        action="store_true",
        help="Load legacy Gemini-style checkpoints (nomic encoder + legacy absolute-pos decoder).",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    dataset_path = Path(args.data_path)

    def _load_splits() -> list[tuple[str, list[dict]]]:
        # If user explicitly requests train/validation/both, reproduce training resolve logic.
        if args.split in {"train", "validation", "both"}:
            train_recs, val_recs, _mode = resolve_train_val_records(
                dataset_path, validation_fraction=float(args.validation_fraction)
            )
            if args.split == "train":
                return [("train", train_recs)]
            if args.split == "validation":
                return [("validation", val_recs)]
            return [("train", train_recs), ("validation", val_recs)]

        # auto: keep previous behavior (pick first available split) for quick inspection
        name, recs = load_records(dataset_path, split="auto")
        return [(name, recs)]

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_hparams = ckpt_meta.get("hyper_parameters", {})
    del ckpt_meta
    load_overrides = {}
    if "lora_r" not in saved_hparams:
        load_overrides["lora_r"] = 0
    if args.gemini_styles:
        load_overrides.update(
            {
                "decoder_type": "legacy",
                "decoder_embed_dim": 256,
                "decoder_layers": 4,
                "decoder_heads": 4,
                "num_prefix_tokens": 4,
                "max_len": 2048,
                "model_name": "nomic-ai/nomic-embed-text-v2-moe",
                "pooling": "mean",
            }
        )
    module = DisentangledLightningModule.load_from_checkpoint(args.checkpoint, map_location=device, **load_overrides)
    module.eval()

    split_records = _load_splits()
    max_questions = None if (args.max_questions is None or args.max_questions <= 0) else args.max_questions
    need_k = max(4, args.k)

    for split_name, records in split_records:
        print("\n" + "=" * 72)
        print(f"SPLIT: {split_name}")
        print("=" * 72)

        records = filter_records_to_questions(records, max_questions, seed=args.seed)
        n_q = len({r.get("question_id") for r in records if r.get("question_id") is not None})
        print(f"  Loaded {len(records)} traces  ({n_q} distinct question_id)")

        groups = defaultdict(list)
        for r in records:
            if r.get("question_id") is not None:
                groups[str(r["question_id"])].append(r)

        qid = args.qid
        if qid is None:
            qid = _pick_qid_with_k_traces(groups, k=need_k, seed=args.seed)
        if qid is None or qid not in groups:
            print("  No eligible question_id found in this split (need >=k traces). Skipping.")
            continue
        if len(groups[qid]) < need_k:
            print(f"  qid='{qid}' has only {len(groups[qid])} traces in this split; need >= {need_k}. Skipping.")
            continue

        chosen = groups[qid][:need_k]
        print(f"\nInspecting qid='{qid}' with {len(chosen)} traces (printing first {need_k})")

        per = []
        for i, r in enumerate(chosen[:need_k], start=1):
            trace = r["trace"]
            answer = str(r.get("answer", ""))
            c, z = encode_trace_to_c_z(module, trace, device)
            decoded_greedy = None
            decoded_samples = None
            if c is not None and z is not None:
                if args.decode in {"greedy", "both"}:
                    decoded_greedy = greedy_decode_from_c_and_z(
                        module, c, z, device=device, max_new_tokens=args.max_new_tokens
                    )
                if args.decode in {"sample", "both"}:
                    top_k = None if int(args.top_k) <= 0 else int(args.top_k)
                    n_samples = max(1, int(args.n_samples))
                    decoded_samples = [
                        sample_decode_from_c_and_z(
                            module,
                            c,
                            z,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            top_k=top_k,
                        )
                        for _ in range(n_samples)
                    ]
            per.append(
                {
                    "i": i,
                    "answer": answer,
                    "trace": trace,
                    "decoded_greedy": decoded_greedy,
                    "decoded_samples": decoded_samples,
                    "c": c,
                    "z": z,
                }
            )

        for item in per:
            print(f"\nTrace {item['i']}/{len(per)}")
            if item["answer"]:
                print(f"  Answer (GT field): {item['answer']}")
            print("  Ground truth trace:")
            print(item["trace"])
            if args.decode in {"greedy", "both"}:
                print("\n  Reconstruction (greedy):")
                print(item["decoded_greedy"] if item["decoded_greedy"] is not None else "[generation failed]")
            if args.decode in {"sample", "both"}:
                print(f"\n  Reconstruction (sample; T={args.temperature} top_p={args.top_p} top_k={args.top_k}):")
                if not item["decoded_samples"]:
                    print("[generation failed]")
                else:
                    for j, dec in enumerate(item["decoded_samples"], start=1):
                        tag = f"[sample {j}/{len(item['decoded_samples'])}]"
                        print(f"\n  {tag}\n{dec if dec is not None else '[generation failed]'}")
            print(f"\n  c stats: {_summ(item['c'])}")
            print(f"  z stats: {_summ(item['z'])}")

        # Cross reconstructions from first two traces (exactly as requested)
        c1, z1 = per[0]["c"], per[0]["z"]
        c2, z2 = per[1]["c"], per[1]["z"]
        if c1 is None or z1 is None or c2 is None or z2 is None:
            print("\n[warn] Missing embeddings/tokenizer; skipping cross reconstructions.")
            continue

        top_k = None if int(args.top_k) <= 0 else int(args.top_k)
        n_samples = max(1, int(args.n_samples))

        def _cross_decode(label: str, fn):
            def _maybe_multi(c, z):
                if label.startswith("sample") and n_samples > 1:
                    return [fn(c, z) for _ in range(n_samples)]
                return fn(c, z)

            dec_c1z1 = _maybe_multi(c1, z1)
            dec_c2z1 = _maybe_multi(c2, z1)
            dec_c1z2 = _maybe_multi(c1, z2)
            dec_c2z2 = _maybe_multi(c2, z2)

            def _print_block(name: str, dec):
                if isinstance(dec, list):
                    for j, d in enumerate(dec, start=1):
                        print(f"\n{name} (sample {j}/{len(dec)})\n{d}")
                else:
                    print(f"\n{name}\n{dec}")

            print("\n" + "=" * 60)
            print(f"CROSS RECONSTRUCTIONS ({label})")
            print("=" * 60)
            print("\n(c1 + z1) vs (c2 + z1) — should look very similar")
            _print_block("[c1+z1]", dec_c1z1)
            _print_block("[c2+z1]", dec_c2z1)
            print("\n(c1 + z2) vs (c2 + z2) — should look very similar")
            _print_block("[c1+z2]", dec_c1z2)
            _print_block("[c2+z2]", dec_c2z2)

        if args.cross_decode in {"greedy", "both"}:
            _cross_decode(
                "greedy",
                lambda c, z: greedy_decode_from_c_and_z(module, c, z, device=device, max_new_tokens=args.max_new_tokens),
            )
        if args.cross_decode in {"sample", "both"}:
            _cross_decode(
                f"sample; T={args.temperature} top_p={args.top_p} top_k={args.top_k}",
                lambda c, z: sample_decode_from_c_and_z(
                    module,
                    c,
                    z,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=top_k,
                ),
            )


if __name__ == "__main__":
    main()

