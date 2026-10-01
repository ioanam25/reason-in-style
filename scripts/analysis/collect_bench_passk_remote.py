#!/usr/bin/env python3
"""Runs on the training cluster. Emits Pass@k for AMC / OlympiadBench Phase A arms.

Same protocol tagging as collect_passk_remote.py (legacy vs eosfix).
Each row also carries a `benchmark` field: `amc` or `olympiad`.
"""

from __future__ import annotations
import os

import json
from pathlib import Path

EVAL_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "model-evals"
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# arm, student, scale, benchmark, prefix-glob (without the -eosfix suffix)
SPECS = [
    # ========== AMC ==========
    # --- Thinking ---
    ("vanilla SFT", "4B-Thinking", "4B", "amc",
     "scas-standard-qwen3-4b-thinking_checkpoint-*-amc-passk"),
    ("no-SFT", "4B-Thinking", "4B", "amc",
     "scas-nosft-qwen3-4b-thinking_pretrained-amc-passk"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "amc",
     "scas-aez-covz-qwen3-4b-thinking-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "4B-Thinking", "4B", "amc",
     "scas-aez-random-qwen3-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "4B-Thinking", "4B", "amc",
     "scas-aez-const1-qwen3-4b-thinking-qwen3-4b-thinking-k1_checkpoint-*-amc-passk-balanced"),
    # --- Base ---
    ("vanilla SFT", "4B-Base", "4B", "amc",
     "scas-standard-qwen3-4b-base_checkpoint-*-amc-passk"),
    ("no-SFT", "4B-Base", "4B", "amc",
     "scas-nosft-qwen3-4b-base_pretrained-amc-passk"),
    ("AE-GMM K=6", "4B-Base", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "4B-Base", "4B", "amc",
     "scas-aez-random-qwen3-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "4B-Base", "4B", "amc",
     "scas-aez-const1-qwen3-4b-base-qwen3-4b-base-k1_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "4B-Base", "4B", "amc",
     "scas-aez-modc9-qwen3-4b-base-qwen3-4b-base-k9_checkpoint-*-amc-passk-balanced"),
    # --- Instruct ---
    ("vanilla SFT", "4B-Instruct", "4B", "amc",
     "scas-standard-qwen3-4b-instruct_checkpoint-*-amc-passk"),
    ("no-SFT", "4B-Instruct", "4B", "amc",
     "scas-nosft-qwen3-4b-instruct_pretrained-amc-passk"),
    ("AE-GMM K=6", "4B-Instruct", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "4B-Instruct", "4B", "amc",
     "scas-aez-random-qwen3-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-amc-passk-balanced"),
    # --- 1.7B ---
    ("vanilla SFT", "1.7B", "1.7B", "amc",
     "scas-standard-qwen3-1p7b_checkpoint-*-amc-passk"),
    ("no-SFT", "1.7B", "1.7B", "amc",
     "scas-nosft-qwen3-1p7b_pretrained-amc-passk"),
    ("AE-GMM K=6", "1.7B", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "amc",
     "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "1.7B", "1.7B", "amc",
     "scas-aez-random-qwen3-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    # --- 1.7B-Base ---
    ("vanilla SFT", "1.7B-Base", "1.7B", "amc",
     "scas-standard-qwen3-1p7b-base_checkpoint-*-amc-passk"),
    ("no-SFT", "1.7B-Base", "1.7B", "amc",
     "scas-nosft-qwen3-1p7b-base_pretrained-amc-passk"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "amc",
     "scas-aez-random-qwen3-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "amc",
     "scas-aez-random-qwen3-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    # --- 0.6B ---
    ("vanilla SFT", "0.6B", "0.6B", "amc",
     "scas-standard-qwen3_checkpoint-*-amc-passk"),
    ("vanilla SFT", "0.6B", "0.6B", "amc",
     "scas-standard-qwen3-0p6b_checkpoint-*-amc-passk"),
    ("no-SFT", "0.6B", "0.6B", "amc",
     "scas-nosft-qwen3-0p6b_pretrained-amc-passk"),
    ("AE-GMM K=6", "0.6B", "0.6B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "amc",
     "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "0.6B", "0.6B", "amc",
     "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "amc",
     "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "1.7B", "1.7B", "amc",
     "scas-aez-const1-qwen3-1p7b-qwen3-1p7b-k1_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "1.7B", "1.7B", "amc",
     "scas-aez-modc9-qwen3-1p7b-qwen3-1p7b-k9_checkpoint-*-amc-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "1.7B", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "1.7B-Base", "1.7B", "amc",
     "scas-aez-const1-qwen3-1p7b-base-qwen3-1p7b-base-k1_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "1.7B-Base", "1.7B", "amc",
     "scas-aez-modc9-qwen3-1p7b-base-qwen3-1p7b-base-k9_checkpoint-*-amc-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("B2 special tokens", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "1.7B-Base", "1.7B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("vanilla SFT", "0.6B-Base", "0.6B", "amc",
     "scas-standard-qwen3-0p6b-base_checkpoint-*-amc-passk"),
    ("no-SFT", "0.6B-Base", "0.6B", "amc",
     "scas-nosft-qwen3-0p6b-base_pretrained-amc-passk"),
    ("AE-GMM K=6", "0.6B-Base", "0.6B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "0.6B-Base", "0.6B", "amc",
     "scas-aez-random-qwen3-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "0.6B-Base", "0.6B", "amc",
     "scas-aez-const1-qwen3-0p6b-base-qwen3-0p6b-base-k1_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "0.6B-Base", "0.6B", "amc",
     "scas-aez-modc9-qwen3-0p6b-base-qwen3-0p6b-base-k9_checkpoint-*-amc-passk-balanced"),
    ("B1 class-balanced GMM", "0.6B-Base", "0.6B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("B2 special tokens", "0.6B-Base", "0.6B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "0.6B-Base", "0.6B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("A2 constant K=1", "4B-Instruct", "4B", "amc",
     "scas-aez-const1-qwen3-4b-instruct-qwen3-4b-instruct-k1_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "4B-Instruct", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "4B-Thinking", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-amc-passk-balanced"),
    ("B3 descriptors", "4B-Base", "4B", "amc",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-amc-passk-balanced"),

    # ========== OlympiadBench-EN ==========
    # --- Thinking ---
    ("vanilla SFT", "4B-Thinking", "4B", "olympiad",
     "scas-standard-qwen3-4b-thinking_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "4B-Thinking", "4B", "olympiad",
     "scas-nosft-qwen3-4b-thinking_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-thinking-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "4B-Thinking", "4B", "olympiad",
     "scas-aez-random-qwen3-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "4B-Thinking", "4B", "olympiad",
     "scas-aez-const1-qwen3-4b-thinking-qwen3-4b-thinking-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    # --- Base ---
    ("vanilla SFT", "4B-Base", "4B", "olympiad",
     "scas-standard-qwen3-4b-base_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "4B-Base", "4B", "olympiad",
     "scas-nosft-qwen3-4b-base_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "4B-Base", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "4B-Base", "4B", "olympiad",
     "scas-aez-random-qwen3-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "4B-Base", "4B", "olympiad",
     "scas-aez-const1-qwen3-4b-base-qwen3-4b-base-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "4B-Base", "4B", "olympiad",
     "scas-aez-modc9-qwen3-4b-base-qwen3-4b-base-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    # --- Instruct ---
    ("vanilla SFT", "4B-Instruct", "4B", "olympiad",
     "scas-standard-qwen3-4b-instruct_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "4B-Instruct", "4B", "olympiad",
     "scas-nosft-qwen3-4b-instruct_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "4B-Instruct", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "4B-Instruct", "4B", "olympiad",
     "scas-aez-random-qwen3-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    # --- 1.7B ---
    ("vanilla SFT", "1.7B", "1.7B", "olympiad",
     "scas-standard-qwen3-1p7b_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "1.7B", "1.7B", "olympiad",
     "scas-nosft-qwen3-1p7b_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "1.7B", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "1.7B", "1.7B", "olympiad",
     "scas-aez-random-qwen3-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    # --- 1.7B-Base ---
    ("vanilla SFT", "1.7B-Base", "1.7B", "olympiad",
     "scas-standard-qwen3-1p7b-base_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "1.7B-Base", "1.7B", "olympiad",
     "scas-nosft-qwen3-1p7b-base_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-random-qwen3-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-random-qwen3-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    # --- 0.6B ---
    ("vanilla SFT", "0.6B", "0.6B", "olympiad",
     "scas-standard-qwen3_checkpoint-*-olympiadbench-en-passk"),
    ("vanilla SFT", "0.6B", "0.6B", "olympiad",
     "scas-standard-qwen3-0p6b_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "0.6B", "0.6B", "olympiad",
     "scas-nosft-qwen3-0p6b_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "0.6B", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-qwen3-0p6b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "0.6B", "0.6B", "olympiad",
     "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "olympiad",
     "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "1.7B", "1.7B", "olympiad",
     "scas-aez-const1-qwen3-1p7b-qwen3-1p7b-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "1.7B", "1.7B", "olympiad",
     "scas-aez-modc9-qwen3-1p7b-qwen3-1p7b-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "1.7B", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-const1-qwen3-1p7b-base-qwen3-1p7b-base-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-modc9-qwen3-1p7b-base-qwen3-1p7b-base-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B1 class-balanced GMM", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B2 special tokens", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "1.7B-Base", "1.7B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("vanilla SFT", "0.6B-Base", "0.6B", "olympiad",
     "scas-standard-qwen3-0p6b-base_checkpoint-*-olympiadbench-en-passk"),
    ("no-SFT", "0.6B-Base", "0.6B", "olympiad",
     "scas-nosft-qwen3-0p6b-base_pretrained-olympiadbench-en-passk"),
    ("AE-GMM K=6", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-random-qwen3-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-const1-qwen3-0p6b-base-qwen3-0p6b-base-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-modc9-qwen3-0p6b-base-qwen3-0p6b-base-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B1 class-balanced GMM", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-bal-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B2 special tokens", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-sp-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "0.6B-Base", "0.6B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A2 constant K=1", "4B-Instruct", "4B", "olympiad",
     "scas-aez-const1-qwen3-4b-instruct-qwen3-4b-instruct-k1_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "4B-Instruct", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-instruct-qwen3-4b-instruct-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "4B-Thinking", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-thinking-qwen3-4b-thinking-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("B3 descriptors", "4B-Base", "4B", "olympiad",
     "scas-aez-covz-qwen3-4b-final-gmm-desc-qwen3-4b-base-qwen3-4b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
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
    """Truncation % from pass_at_k.json when present.

    Do not scan generations.jsonl here: AMC/Olympiad dumps are huge and the
    SSH collector was dying while walking them. MATH collector still computes
    this from dumps; bench plots do not use Trunc%.
    """
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
    for arm, student, scale, bench, pat in SPECS:
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
                key = (arm, student, scale, bench, proto)
                step = ckpt_step(d.name)
                prefer = 1 if "final-gmm" in d.name else 0
                prev = best.get(key)
                if prev and (prev["_prefer"], prev["_step"]) >= (prefer, step):
                    continue
                best[key] = {
                    "arm": arm,
                    "student": student,
                    "scale": scale,
                    "benchmark": bench,
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
