#!/usr/bin/env python3
"""Runs on the training cluster. Emits one JSON blob of MATH-500 Pass@k for every tracked arm.

Each arm is resolved under both protocols:
  legacy  - generated before the EOS/stop-token fix (truncated at max_tokens)
  eosfix  - generated with EOS_FIX=1 and explicit stop_token_ids

Protocol is decided by the `eos_fix` flag in pass_at_k.json when present, then by
a `finish_reason` field in the generations dump, then by the -eosfix dir suffix.
"""

from __future__ import annotations
import os

import json
from pathlib import Path

EVAL_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "model-evals"
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# arm, student, scale, prefix-glob (without the -eosfix suffix)
SPECS = [
    # --- 4B headline ---
    ("vanilla SFT", "4B-Thinking", "4B", "scas-standard-qwen3-4b-thinking_checkpoint-*-math500-passk"),
    ("vanilla SFT", "4B-Base", "4B", "scas-standard-qwen3-4b-base_checkpoint-*-math500-passk"),
    ("vanilla SFT", "4B-Instruct", "4B", "scas-standard-qwen3-4b-instruct_checkpoint-*-math500-passk"),
    ("no-SFT", "4B-Thinking", "4B", "scas-nosft-qwen3-4b-thinking_pretrained-math500-passk"),
    ("no-SFT", "4B-Instruct", "4B", "scas-nosft-qwen3-4b-instruct_pretrained-math500-passk"),
    ("no-SFT", "4B-Base", "4B", "scas-nosft-qwen3-4b-base_pretrained-math500-passk"),
    ("no-SFT", "1.7B", "1.7B", "scas-nosft-qwen3-1p7b_pretrained-math500-passk"),
    ("no-SFT", "0.6B", "0.6B", "scas-nosft-qwen3-0p6b_pretrained-math500-passk"),
    ("no-SFT", "1.7B-Base", "1.7B", "scas-nosft-qwen3-1p7b-base_pretrained-math500-passk"),
    ("no-SFT", "0.6B-Base", "0.6B", "scas-nosft-qwen3-0p6b-base_pretrained-math500-passk"),
    ("AE-GMM style_1", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced-style_1"),
    ("AE-GMM style_6", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced-style_6"),
    ("AE-GMM style_1", "4B-Instruct", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced-style_1"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "4B-Base", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "4B-Instruct", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    # --- 4B Phase A controls ---
    ("A1 random K=6", "4B-Thinking", "4B", "scas-aez-random-qwen3-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "4B-Base", "4B", "scas-aez-random-qwen3-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "4B-Instruct", "4B", "scas-aez-random-qwen3-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "4B-Thinking", "4B", "scas-aez-const1-qwen3-4b-thinking-qwen3-4b-thinking-k1_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "4B-Base", "4B", "scas-aez-const1-qwen3-4b-base-qwen3-4b-base-k1_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "4B-Thinking", "4B", "scas-aez-modc9-qwen3-4b-thinking-qwen3-4b-thinking-k9_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "4B-Base", "4B", "scas-aez-modc9-qwen3-4b-base-qwen3-4b-base-k9_checkpoint-*-math500-passk-balanced"),
    # --- 4B seed replicates ---
    ("AE-GMM seed 2", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6-opt-s2_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM seed 3", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6-opt-s3_checkpoint-*-math500-passk-balanced"),
    ("vanilla seed 2", "4B-Thinking", "4B", "scas-standard-qwen3-4b-thinking-opt-s2_checkpoint-*-math500-passk"),
    ("vanilla seed 3", "4B-Thinking", "4B", "scas-standard-qwen3-4b-thinking-opt-s3_checkpoint-*-math500-passk"),
    # --- 4B Phase B ---
    ("B1 class-balanced GMM", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "4B-Base", "4B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "4B-Base", "4B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "4B-Thinking", "4B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "4B-Base", "4B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "4B-Instruct", "4B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "1.7B", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "0.6B", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 1.7B remaining Phase A/B ---
    ("AE-GMM K=6", "1.7B", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "1.7B", "1.7B", "scas-aez-const1-qwen3-1p7b-qwen3-1p7b-k1_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "1.7B", "1.7B", "scas-aez-modc9-qwen3-1p7b-qwen3-1p7b-k9_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "1.7B", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 1.7B-Base remaining ---
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "1.7B-Base", "1.7B", "scas-aez-const1-qwen3-1p7b-base-qwen3-1p7b-base-k1_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "1.7B-Base", "1.7B", "scas-aez-modc9-qwen3-1p7b-base-qwen3-1p7b-base-k9_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 0.6B remaining ---
    ("AE-GMM K=6", "0.6B", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "0.6B", "0.6B", "scas-aez-const1-qwen3-0p6b-qwen3-0p6b-k1_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "0.6B", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "0.6B", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 0.6B-Base ---
    ("vanilla SFT", "0.6B-Base", "0.6B", "scas-standard-qwen3-0p6b-base_checkpoint-*-math500-passk"),
    ("AE-GMM K=6", "0.6B-Base", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "0.6B-Base", "0.6B", "scas-aez-random-qwen3-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A2 constant K=1", "0.6B-Base", "0.6B", "scas-aez-const1-qwen3-0p6b-base-qwen3-0p6b-base-k1_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "0.6B-Base", "0.6B", "scas-aez-modc9-qwen3-0p6b-base-qwen3-0p6b-base-k9_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "0.6B-Base", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "0.6B-Base", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("B3 descriptors", "0.6B-Base", "0.6B", "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    # --- 4B-Instruct remaining ---
    ("A2 constant K=1", "4B-Instruct", "4B", "scas-aez-const1-qwen3-4b-instruct-qwen3-4b-instruct-k1_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "4B-Instruct", "4B", "scas-aez-modc9-qwen3-4b-instruct-qwen3-4b-instruct-k9_checkpoint-*-math500-passk-balanced"),
    ("B1 class-balanced GMM", "4B-Instruct", "4B", "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    ("B2 special tokens", "4B-Instruct", "4B", "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 1.7B (post-trained Qwen3-1.7B) ---
    ("vanilla SFT", "1.7B", "1.7B", "scas-standard-qwen3-1p7b_checkpoint-*-math500-passk"),
    ("AE-GMM K=6", "1.7B", "1.7B", "scas-aez-covz-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "1.7B", "1.7B", "scas-aez-random-qwen3-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 1.7B-Base ---
    ("vanilla SFT", "1.7B-Base", "1.7B", "scas-standard-qwen3-1p7b-base_checkpoint-*-math500-passk"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "scas-aez-random-qwen3-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "scas-aez-random-qwen3-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    # --- scale: 0.6B ---
    ("vanilla SFT", "0.6B", "0.6B", "scas-standard-qwen3-0p6b_checkpoint-*-math500-passk"),
    ("vanilla SFT", "0.6B", "0.6B", "scas-standard-qwen3_checkpoint-*-math500-passk"),
    ("AE-GMM K=6", "0.6B", "0.6B", "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "scas-aez-covz-qwen3-qwen3-0p6b-k6-qwen3-opt_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "0.6B", "0.6B", "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-math500-passk-balanced"),
]


def curve(d: Path):
    f = d / "pass_at_k.json"
    if not f.exists():
        return None
    try:
        blob = json.loads(f.read_text())
    except Exception:
        return None
    pk = (
        blob.get("pass_at_k_cluster_prefix")
        or blob.get("pass_at_k")
        or blob.get("pass_at_k_standard")
        or {}
    )
    vals = {}
    for k in BUDGETS:
        v = pk.get(str(k), pk.get(k))
        if v is not None:
            vals[str(k)] = round(100 * float(v), 2)
    if not vals:
        return None
    return blob, vals


def detect_protocol(d: Path, blob: dict) -> str:
    if "eos_fix" in blob:
        return "eosfix" if blob["eos_fix"] else "legacy"
    if d.name.endswith("-eosfix"):
        return "eosfix"
    gens = d / "generations.jsonl"
    if gens.exists():
        try:
            with gens.open() as fh:
                head = fh.readline()
            if head and "finish_reason" in head:
                return "eosfix"
        except Exception:
            pass
    return "legacy"


def truncation_rate(d: Path):
    """Prefer pass_at_k.json; do not scan huge generations.jsonl over SSH."""
    f = d / "pass_at_k.json"
    if not f.exists():
        return None
    try:
        blob = json.loads(f.read_text())
    except Exception:
        return None
    for key in ("pct_finish_length", "pct_truncated", "truncation_rate"):
        if key in blob and blob[key] is not None:
            return round(float(blob[key]), 1)
    return None


def ckpt_step(name: str) -> int:
    """Step number embedded in the eval dir, used to prefer the latest checkpoint."""
    marker = "checkpoint-"
    if marker not in name:
        return -1
    tail = name.split(marker, 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else -1


def main() -> None:
    best: dict[tuple, dict] = {}
    seen_dirs = set()
    for arm, student, scale, pat in SPECS:
        for suffix in ("", "-eosfix"):
            for d in sorted(EVAL_ROOT.glob(pat + suffix)):
                if not d.is_dir() or d.name in seen_dirs:
                    continue
                got = curve(d)
                if not got:
                    continue
                blob, vals = got
                proto = detect_protocol(d, blob)
                seen_dirs.add(d.name)
                key = (arm, student, scale, proto)
                step = ckpt_step(d.name)
                prefer = 1 if "4b-final" in d.name else 0
                prev = best.get(key)
                if prev and (prev["_prefer"], prev["_step"]) >= (prefer, step):
                    continue
                best[key] = {
                    "arm": arm,
                    "student": student,
                    "scale": scale,
                    "protocol": proto,
                    "status": "complete",
                    "eval_dir": d.name,
                    "checkpoint": blob.get("checkpoint"),
                    "pct_finish_length": truncation_rate(d),
                    "pass_at_k_pct": vals,
                    "_step": step,
                    "_prefer": prefer,
                }
    out = []
    for row in best.values():
        row.pop("_step", None)
        row.pop("_prefer", None)
        out.append(row)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
