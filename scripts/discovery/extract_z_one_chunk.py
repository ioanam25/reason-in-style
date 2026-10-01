#!/usr/bin/env python3
"""Extract one contiguous row range; designed to be invoked as a short-lived subprocess."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.eval.eval_gemini_styles import _encoder_tokenizer, load_gemini_with_styles  # noqa: E402
from src.module import DisentangledLightningModule  # noqa: E402


def force_eager(model):
    enc = getattr(model, "encoder", None)
    base = getattr(enc, "model", None) if enc is not None else None
    if base is None:
        return
    try:
        base.config.use_cache = False
    except Exception:
        pass
    # Prefer eager attention to avoid SDPA/flash CUDA hard-aborts.
    for obj in (base, getattr(base, "model", None)):
        if obj is None:
            continue
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg._attn_implementation = "eager"
                cfg.attn_implementation = "eager"
            except Exception:
                pass
        if hasattr(obj, "set_attn_implementation"):
            try:
                obj.set_attn_implementation("eager")
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hf-dataset", required=True)
    p.add_argument("--raw-json", default="")
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--row-start", type=int, required=True)
    p.add_argument("--row-end", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json, split="train"))
    shard = records[args.row_start:args.row_end]

    module = DisentangledLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=False
    )
    module.eval()
    max_len = int(module.hparams.get("max_len", 4096))
    model = module.model.to(device)
    model.eval()
    force_eager(model)
    tokenizer = _encoder_tokenizer(model)

    c_list, z_list = [], []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in tqdm(range(0, len(shard), args.batch_size), desc="chunk"):
            batch = shard[s:s + args.batch_size]
            traces = [r["trace"] for r in batch]
            enc = tokenizer(traces, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            c_s, z_s, _ = model(input_ids, attention_mask, decode=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            c_list.append(c_s.float().cpu().numpy())
            z_list.append(z_s.float().cpu().numpy())

    c_np = np.concatenate(c_list)
    z_np = np.concatenate(z_list)
    np.savez(
        args.out_npz,
        c_s=c_np,
        z_s=z_np,
        question_id=np.array([r["question_id"] for r in shard]),
        style_id=np.array([r.get("style_id", -1) for r in shard], dtype=np.int64),
        row_start=np.array(args.row_start, dtype=np.int64),
        row_end=np.array(args.row_end, dtype=np.int64),
    )
    print(f"Wrote {args.out_npz} c={c_np.shape}", flush=True)


if __name__ == "__main__":
    main()
