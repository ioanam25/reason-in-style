"""Resolve a model-evals directory name to (arm, student, scale, benchmark, protocol).

Uses the same arm/student labels as `collect_passk_remote.py` and
`collect_bench_passk_remote.py` so style metrics join cleanly onto the Pass@k
registries.
"""

from __future__ import annotations
import os

import fnmatch
import json
from pathlib import Path

EVAL_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "model-evals"

# student dir token -> (label, scale)
STUDENTS = {
    "qwen3-4b-thinking": ("4B-Thinking", "4B"),
    "qwen3-4b-base": ("4B-Base", "4B"),
    "qwen3-4b-instruct": ("4B-Instruct", "4B"),
    "qwen3-1p7b": ("1.7B", "1.7B"),
    "qwen3-1p7b-base": ("1.7B-Base", "1.7B"),
    "qwen3-0p6b": ("0.6B", "0.6B"),
    "qwen3-0p6b-base": ("0.6B-Base", "0.6B"),
}

# benchmark -> (prefix-arm eval suffix, vanilla eval suffix)
BENCHES = {
    "math500": ("math500-passk-balanced", "math500-passk"),
    "amc": ("amc-passk-balanced", "amc-passk"),
    "olympiad": ("olympiadbench-en-passk-balanced", "olympiadbench-en-passk"),
}

# arm label -> (dataset stem template, K). `{s}` is the student dir token.
PREFIX_ARMS = {
    "AE-GMM K=6": ("covz-qwen3-4b-final-gmm-{s}-{s}", 6),
    "A1 random K=6": ("random-qwen3-{s}-{s}", 6),
    "A2 constant K=1": ("const1-{s}-{s}", 1),
    "A3 ModC K=9": ("modc9-{s}-{s}", 9),
    "B1 class-balanced GMM": ("covz-qwen3-4b-final-gmm-bal-{s}-{s}", 6),
    "B2 special tokens": ("covz-qwen3-4b-final-gmm-sp-{s}-{s}", 6),
    "B3 descriptors": ("covz-qwen3-4b-final-gmm-desc-{s}-{s}", 6),
}

# Older runs whose dir names predate the `-{student}-{student}` convention, plus
# the seed replicates. (arm, student, scale, benchmark, glob-without-eosfix)
LEGACY_SPECS = [
    ("AE-GMM K=6", "4B-Thinking", "4B", "math500", "scas-aez-covz-qwen3-4b-thinking-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "amc", "scas-aez-covz-qwen3-4b-thinking-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "4B-Thinking", "4B", "olympiad", "scas-aez-covz-qwen3-4b-thinking-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "4B-Instruct", "4B", "math500", "scas-aez-covz-qwen3-4b-instruct-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "4B-Base", "4B", "math500", "scas-aez-covz-qwen3-4b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "math500", "scas-aez-covz-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "math500", "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "amc", "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "olympiad", "scas-aez-covz-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    # The 1.7B students use the covz-qwen3-1p7b style basis, not covz-qwen3-4b-final-gmm.
    ("AE-GMM K=6", "1.7B", "1.7B", "math500", "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "amc", "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B", "1.7B", "olympiad", "scas-aez-covz-qwen3-1p7b-qwen3-1p7b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "math500", "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "amc", "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "1.7B-Base", "1.7B", "olympiad", "scas-aez-covz-qwen3-1p7b-base-qwen3-1p7b-base-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A1 random K=6", "1.7B", "1.7B", "math500", "scas-aez-random-qwen3-qwen3-1p7b-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "1.7B-Base", "1.7B", "math500", "scas-aez-random-qwen3-qwen3-1p7b-base-k6_checkpoint-*-math500-passk-balanced"),
    # 0.6B random keeps the single-token stem.
    ("A1 random K=6", "0.6B", "0.6B", "math500", "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("A1 random K=6", "0.6B", "0.6B", "amc", "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-amc-passk-balanced"),
    ("A1 random K=6", "0.6B", "0.6B", "olympiad", "scas-aez-random-qwen3-qwen3-0p6b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "math500", "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-math500-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "amc", "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-amc-passk-balanced"),
    ("A3 ModC K=9", "0.6B", "0.6B", "olympiad", "scas-aez-modc-style-qwen3-qwen3-0p6b-k9_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "math500", "scas-aez-covz-qwen3-qwen3-0p6b-k6-qwen3-opt_checkpoint-*-math500-passk-balanced"),
    # 0.6B AE-GMM keeps the bare `covz-qwen3` stem (no repeated student token).
    ("AE-GMM K=6", "0.6B", "0.6B", "math500", "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "amc", "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-amc-passk-balanced"),
    ("AE-GMM K=6", "0.6B", "0.6B", "olympiad", "scas-aez-covz-qwen3-qwen3-0p6b-k6_checkpoint-*-olympiadbench-en-passk-balanced"),
    ("AE-GMM K=6", "0.6B-Base", "0.6B", "math500", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-0p6b-base-qwen3-0p6b-base-k6_checkpoint-*-math500-passk-balanced"),
    # seed replicates (4B-Thinking only)
    ("AE-GMM seed 2", "4B-Thinking", "4B", "math500", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6-opt-s2_checkpoint-*-math500-passk-balanced"),
    ("AE-GMM seed 3", "4B-Thinking", "4B", "math500", "scas-aez-covz-qwen3-4b-final-gmm-qwen3-4b-thinking-qwen3-4b-thinking-k6-opt-s3_checkpoint-*-math500-passk-balanced"),
    ("vanilla seed 2", "4B-Thinking", "4B", "math500", "scas-standard-qwen3-4b-thinking-opt-s2_checkpoint-*-math500-passk"),
    ("vanilla seed 3", "4B-Thinking", "4B", "math500", "scas-standard-qwen3-4b-thinking-opt-s3_checkpoint-*-math500-passk"),
]

# arms whose prefix carries no style information; used as nulls in the analyses
NULL_ARMS = {"vanilla SFT", "A1 random K=6", "A2 constant K=1", "no-SFT"}

ARM_ORDER = {
    "no-SFT": -1,
    "vanilla SFT": 0,
    "AE-GMM K=6": 1,
    "A1 random K=6": 2,
    "A2 constant K=1": 3,
    "A3 ModC K=9": 4,
    "B1 class-balanced GMM": 5,
    "B2 special tokens": 6,
    "B3 descriptors": 7,
    "AE-GMM seed 2": 8,
    "AE-GMM seed 3": 9,
    "vanilla seed 2": 10,
    "vanilla seed 3": 11,
}

STUDENT_ORDER = {
    "4B-Thinking": 0,
    "4B-Base": 1,
    "4B-Instruct": 2,
    "1.7B": 3,
    "1.7B-Base": 4,
    "0.6B": 5,
    "0.6B-Base": 6,
}

STUDENT_MODEL = {
    "4B-Thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "4B-Base": "Qwen/Qwen3-4B-Base",
    "4B-Instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "1.7B": "Qwen/Qwen3-1.7B",
    "1.7B-Base": "Qwen/Qwen3-1.7B-Base",
    "0.6B": "Qwen/Qwen3-0.6B",
    "0.6B-Base": "Qwen/Qwen3-0.6B-Base",
}


def _build_specs() -> list[tuple[str, str, str, str, str]]:
    specs: list[tuple[str, str, str, str, str]] = []
    for stok, (label, scale) in STUDENTS.items():
        for bench, (pfx_suffix, van_suffix) in BENCHES.items():
            # vanilla SFT. The 0.6B post-trained student kept the bare `qwen3` stem.
            van_stems = [f"scas-standard-{stok}"]
            if stok == "qwen3-0p6b":
                van_stems.append("scas-standard-qwen3")
            for stem in van_stems:
                specs.append(("vanilla SFT", label, scale, bench, f"{stem}_checkpoint-*-{van_suffix}"))
            specs.append(
                ("no-SFT", label, scale, bench, f"scas-nosft-{stok}_pretrained-{van_suffix}")
            )
            for arm, (tmpl, _k) in PREFIX_ARMS.items():
                stem = tmpl.format(s=stok)
                specs.append((arm, label, scale, bench, f"scas-aez-{stem}-k*_checkpoint-*-{pfx_suffix}"))
    specs.extend(LEGACY_SPECS)
    return specs


SPECS = _build_specs()


ZSCORE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "zscore-variants"

# AE z-spaces that carry GMM K=6 assignments over the shared 111,834 teacher traces
AE_TAGS = ["covz", "covz-qwen3-1p7b", "covz-qwen3-4b", "covz-qwen3-4b-final"]
AE_DECODER_SIZE = {
    "covz": "0.6B",
    "covz-qwen3-1p7b": "1.7B",
    "covz-qwen3-4b": "4B (early)",
    "covz-qwen3-4b-final": "4B (final)",
}


def teacher_tag(dir_name: str) -> str | None:
    """Which AE z-space the arm's style labels came from, or None for non-AE arms.

    The style basis is baked into the dataset stem: `covz-qwen3-4b-final-gmm-*`
    arms use the 4B final AE, while the 1.7B and 0.6B arms predate it and use the
    AE whose decoder matches their own size.
    """
    stem, _ = _strip_proto(dir_name)
    if "covz-qwen3-4b-final" in stem:
        return "covz-qwen3-4b-final"
    if "covz-qwen3-1p7b" in stem:
        return "covz-qwen3-1p7b"
    if "scas-aez-covz-qwen3-" in stem:
        return "covz"
    return None


def _strip_proto(name: str) -> tuple[str, str]:
    if name.endswith("-eosfix"):
        return name[: -len("-eosfix")], "eosfix"
    return name, "legacy"


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


def resolve(dir_name: str) -> dict | None:
    """Map an eval dir name onto its arm/student/benchmark, or None if untracked.

    Longer glob patterns are preferred so that `...-base-...` variants win over
    the shorter non-base pattern they would otherwise also match.
    """
    stem, proto = _strip_proto(dir_name)
    # `-single-style_i` dumps hold one prefix only, so they carry no style contrast.
    if "-single-style" in stem:
        return None
    hits = [
        (arm, student, scale, bench, pat)
        for arm, student, scale, bench, pat in SPECS
        if fnmatch.fnmatchcase(stem, pat)
    ]
    if not hits:
        return None
    arm, student, scale, bench, pat = max(hits, key=lambda h: len(h[4]))
    return {
        "arm": arm,
        "student": student,
        "scale": scale,
        "benchmark": bench,
        "protocol": proto,
        "eval_dir": dir_name,
        "ckpt_step": ckpt_step(dir_name),
        "is_null_arm": arm in NULL_ARMS,
        "teacher_tag": teacher_tag(dir_name),
    }


def detect_protocol(d: Path) -> str:
    """Protocol from pass_at_k.json, then generations, then the dir suffix.

    4B-final MATH dumps often have `finish_reason` but no `eos_fix` field and no
    `-eosfix` suffix; treat those as EOS_FIX to match the Pass@k collectors.
    """
    f = d / "pass_at_k.json"
    if f.exists():
        try:
            blob = json.loads(f.read_text())
        except Exception:
            blob = {}
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


def dump_rank(dir_name: str) -> tuple[int, int]:
    """Prefer 4B-final style source, then latest checkpoint — same as Pass@k collectors."""
    prefer = 1 if "4b-final" in dir_name else 0
    return (prefer, ckpt_step(dir_name))


def iter_eval_dirs(eval_root: Path = EVAL_ROOT, protocol: str = "eosfix"):
    """Yield (path, meta) for tracked eval dirs under the requested protocol.

    When several dumps exist for one (arm, student, benchmark), 4B-final wins
    over older AE sources, then the latest checkpoint step.
    """
    best: dict[tuple, tuple[tuple[int, int], Path, dict]] = {}
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        meta = resolve(d.name)
        if meta is None:
            continue
        if detect_protocol(d) != protocol:
            continue
        if not (d / "generations.jsonl").exists():
            continue
        key = (meta["arm"], meta["student"], meta["benchmark"])
        rank = dump_rank(d.name)
        prev = best.get(key)
        if prev is not None and prev[0] >= rank:
            continue
        best[key] = (rank, d, meta)
    for _rank, d, meta in best.values():
        yield d, meta
