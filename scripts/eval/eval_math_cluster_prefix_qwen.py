#!/usr/bin/env python3
"""
Pass@k on MATH-500 for cluster-prefix (or oracle ModC-prefix) math decoders.

Paper (ModC §4.2): evaluate on MATH500 with balanced mode allocation across
[style_1]/[style_2] (or teacher prefixes), k in {1,2,4,...,256}.

Usage:
  python scripts/eval/eval_math_cluster_prefix_qwen.py \\
      --checkpoint checkpoints/scas-prefix-qwen3-4b-base \\
      --cluster-meta data/scas/cluster_gmm_assignments-qwen3-4b/cluster_meta.json \\
      --output-dir model-evals/scas-prefix-qwen3-4b-base-math500/balanced \\
      --samples-total 256 --sampling-mode balanced
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.prefix_passk import (  # noqa: E402
    allocate_integer_budget,
    balanced_weights,
    load_cluster_meta,
    multi_prefix_pass_at_k,
    style_sort_key,
)
from scripts.eval.eval_runtime import (  # noqa: E402
    choose_tensor_parallel_size,
    configure_tokenizer,
    load_num_attention_heads,
    load_vocab_size,
    standard_pass_at_k,
)
from scripts.eval.vllm_replicas import (  # noqa: E402
    count_visible_gpus,
    generate_with_data_parallel_replicas,
    resolve_data_parallel_replicas,
)
from scripts.math_answer_utils import answers_match  # noqa: E402
from scripts.eval.eval_scas_passk_utils import load_scas_val_problems  # noqa: E402
from scripts.scas_style_prefix import format_style_prefix  # noqa: E402


def _normalize_answer_str(a) -> str:
    """Stringify gold answers; collapse int-like floats (e.g. 142.0 -> '142')."""
    if isinstance(a, float) and a.is_integer():
        return str(int(a))
    if isinstance(a, (list, tuple)):
        parts = [_normalize_answer_str(x) for x in a]
        return parts[0] if len(parts) == 1 else ", ".join(parts)
    s = str(a).strip()
    # AoPS clock notation: 4{:}30 -> 4:30
    s = s.replace("{:", ":").replace(":}", ":")
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def _gsm8k_final_answer(text: str) -> str:
    """GSM8K gold is CoT with a final `#### <answer>` line."""
    s = str(text)
    if "####" in s:
        return s.rsplit("####", 1)[-1].strip().replace(",", "")
    return s.strip()


def benchmark_label(hf_dataset: str) -> str:
    """Human-readable benchmark name for logs / pass_at_k.json."""
    alias = hf_dataset.strip().lower()
    if alias in {"aime", "aime-2024-2025", "aime2024+2025", "aime_2024_2025"}:
        return "AIME-2024+2025"
    if alias in {
        "hmmt",
        "hmmt25",
        "hmmt-feb-2025",
        "hmmt_feb_2025",
        "matharena/hmmt_feb_2025",
    }:
        return "HMMT-Feb-2025"
    if alias in {"gsm8k", "openai/gsm8k"}:
        return "GSM8K"
    if alias in {"amc25", "amc-2025", "amc12-2025"}:
        return "AMC12-2025"
    if alias in {"amc", "amc23+aimo", "amc-mid"}:
        return "AMC"
    if alias in {"olympiadbench", "olympiad", "olympiadbench-en"}:
        return "OlympiadBench-EN"
    if alias in {"omni-math", "omnimath", "omni_math"}:
        return "Omni-MATH"
    if "aime" in alias:
        return "AIME"
    if "MATH-500" in hf_dataset or "math-500" in alias:
        return "MATH-500"
    return hf_dataset


def load_math500_problems(hf_dataset: str, max_problems: int) -> list[dict]:
    """Load MATH-500 / AIME / GSM8K / AMC / Olympiad-style HF datasets."""
    from datasets import Dataset, concatenate_datasets, load_dataset

    alias = hf_dataset.strip().lower()

    def _rows_from_ds(ds, *, subject_default: str = "", answer_fn=None, q_keys=None):
        q_keys = q_keys or ("problem", "Problem", "question", "Question")
        rows = []
        for i, row in enumerate(ds):
            q = None
            for k in q_keys:
                if row.get(k) is not None:
                    q = row[k]
                    break
            if answer_fn is not None:
                a = answer_fn(row)
            else:
                a = row.get("answer") if row.get("answer") is not None else row.get("Answer")
            if q is None or a is None:
                raise KeyError(f"Dataset row missing problem/answer keys: {list(row.keys())}")
            qid = row.get("unique_id") or row.get("id") or row.get("ID") or i
            rows.append(
                {
                    "question_id": str(qid),
                    "question": str(q),
                    "answer": _normalize_answer_str(a),
                    "subject": str(row.get("subject", subject_default) or subject_default),
                    "level": row.get("level") or row.get("difficulty"),
                }
            )
        return rows

    if alias in {"aime", "aime-2024-2025", "aime2024+2025", "aime_2024_2025"}:
        d24 = load_dataset("HuggingFaceH4/aime_2024", split="train")
        d25 = load_dataset("math-ai/aime25", split="test")

        def _norm(ds, year_tag: str):
            rows = []
            for i, row in enumerate(ds):
                q = row.get("problem") or row.get("Problem")
                a = row.get("answer") if "answer" in row else row.get("Answer")
                qid = row.get("id") or row.get("unique_id") or row.get("ID") or f"{year_tag}-{i}"
                rows.append(
                    {
                        "unique_id": str(qid),
                        "problem": str(q),
                        "answer": _normalize_answer_str(a),
                        "subject": year_tag,
                        "level": None,
                    }
                )
            return Dataset.from_list(rows)

        ds = concatenate_datasets([_norm(d24, "aime2024"), _norm(d25, "aime2025")])
        problems = _rows_from_ds(ds)
    elif alias in {
        "hmmt",
        "hmmt25",
        "hmmt-feb-2025",
        "hmmt_feb_2025",
        "matharena/hmmt_feb_2025",
    }:
        ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
        problems = []
        for i, row in enumerate(ds):
            qid = row.get("problem_idx", i)
            ptype = row.get("problem_type")
            if isinstance(ptype, list):
                subj = ",".join(str(x) for x in ptype) if ptype else "hmmt-feb-2025"
            else:
                subj = str(ptype or "hmmt-feb-2025")
            problems.append(
                {
                    "question_id": f"hmmt25-{qid}",
                    "question": str(row["problem"]),
                    "answer": _normalize_answer_str(row["answer"]),
                    "subject": subj,
                    "level": None,
                }
            )
    elif alias in {"gsm8k", "openai/gsm8k"}:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = _rows_from_ds(
            ds,
            subject_default="gsm8k",
            answer_fn=lambda row: _gsm8k_final_answer(row["answer"]),
            q_keys=("question",),
        )
    elif alias in {"amc25", "amc-2025", "amc12-2025"}:
        # AMC 12 2025 A/B, text-only (no figures). Gold is the numeric/symbolic
        # choice value, not A-E. Post-Qwen3-0.6B-Base (Apr 2025) contest.
        ds = load_dataset("sonthenguyen/amc12-2025-non-figure", split="train")
        problems = _rows_from_ds(ds, subject_default="amc12-2025", q_keys=("question",))
    elif alias in {"amc", "amc23+aimo", "amc-mid"}:
        # Mid-hard gap between MATH-500 and AIME: AMC 2023 + AIMO validation AMC.
        amc23 = load_dataset("math-ai/amc23", split="test")
        aimo = load_dataset("AI-MO/aimo-validation-amc", split="train")

        def _amc_rows(ds, tag: str, q_key: str):
            rows = []
            for i, row in enumerate(ds):
                q = row.get(q_key) or row.get("problem") or row.get("question")
                a = row.get("answer")
                qid = row.get("id") or f"{tag}-{i}"
                rows.append(
                    {
                        "unique_id": f"{tag}-{qid}",
                        "problem": str(q),
                        "answer": _normalize_answer_str(a),
                        "subject": tag,
                        "level": None,
                    }
                )
            return Dataset.from_list(rows)

        ds = concatenate_datasets(
            [_amc_rows(amc23, "amc23", "question"), _amc_rows(aimo, "aimo-amc", "problem")]
        )
        problems = _rows_from_ds(ds)
    elif alias in {"olympiadbench", "olympiad", "olympiadbench-en"}:
        # Text-only English open-ended competition math (Hothan/OlympiadBench).
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        problems = []
        for i, row in enumerate(ds):
            # Prefer single-answer items for reliable Pass@k grading.
            if row.get("is_multiple_answer"):
                continue
            fa = row.get("final_answer")
            if not fa:
                continue
            a = fa[0] if isinstance(fa, (list, tuple)) else fa
            problems.append(
                {
                    "question_id": str(row.get("id", i)),
                    "question": str(row["question"]),
                    "answer": _normalize_answer_str(a),
                    "subject": str(row.get("subfield") or "olympiad"),
                    "level": row.get("difficulty"),
                }
            )
    elif alias in {"omni-math", "omnimath", "omni_math"}:
        # Full Omni-MATH is large + often symbolic; keep difficulty >= 7 as a hard OOD slice.
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
        problems = []
        for i, row in enumerate(ds):
            diff = row.get("difficulty")
            try:
                if diff is not None and float(diff) < 7.0:
                    continue
            except (TypeError, ValueError):
                pass
            a = row.get("answer")
            if a is None or not str(row.get("problem", "")).strip():
                continue
            problems.append(
                {
                    "question_id": str(row.get("id", i)),
                    "question": str(row["problem"]),
                    "answer": _normalize_answer_str(a),
                    "subject": str((row.get("domain") or ["omni"])[0] if isinstance(row.get("domain"), list) else row.get("domain") or "omni"),
                    "level": diff,
                }
            )
    else:
        try:
            ds = load_dataset(hf_dataset, split="test")
        except ValueError:
            ds = load_dataset(hf_dataset, split="train")
        problems = _rows_from_ds(ds)

    if max_problems and max_problems > 0:
        problems = problems[:max_problems]
    return problems


def load_style_names(cluster_meta: Path | None, oracle_styles: list[str] | None) -> tuple[list[str], dict[str, float]]:
    if oracle_styles:
        names = sorted(oracle_styles, key=style_sort_key)
        weights = balanced_weights(names)
        return names, weights
    meta_info = load_cluster_meta(cluster_meta)
    return meta_info["style_names"], meta_info["train_weights"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", buffering=1) as fout:
        for rec in rows:
            fout.write(json.dumps(rec) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _grade_generation_rows(rows: list[dict], timeout_sec: float, flush_every: int) -> list[dict]:
    from tqdm.auto import tqdm

    graded = []
    pbar = tqdm(total=len(rows), desc="Grading", unit="comp", file=sys.stderr)
    try:
        for i, rec in enumerate(rows):
            rec = dict(rec)
            rec["correct"] = bool(
                answers_match(rec.get("output") or "", rec.get("gold_answer"), timeout_sec=timeout_sec)
            )
            graded.append(rec)
            pbar.update(1)
            if flush_every > 0 and (i + 1) % flush_every == 0:
                pbar.set_postfix(n=i + 1)
    finally:
        pbar.close()
    return graded


def _write_prefix_passk(
    *,
    out_dir: Path,
    problems: list[dict],
    style_names: list[str],
    successes: list[dict],
    gen_lens_by_prefix: dict,
    args,
    samples_by_style: dict,
    eval_weights: dict,
    train_weights: dict,
    cluster_meta_path: Path | None,
    benchmark: str,
) -> dict:
    per_problem = []
    for pi, prob in enumerate(problems):
        p_by_style = {}
        n_by_style = {}
        for style in style_names:
            vals = successes[pi][style]
            p_by_style[style] = float(np.mean(vals)) if vals else 0.0
            n_by_style[style] = len(vals)
        p_pool = sum(eval_weights[s] * p_by_style[s] for s in style_names)
        per_problem.append(
            {
                "question_id": prob["question_id"],
                "subject": prob["subject"],
                **{f"p_{s}": p_by_style[s] for s in style_names},
                **{f"n_{s}": n_by_style[s] for s in style_names},
                "p_pool": float(p_pool),
            }
        )

    cluster_curve, std_curve, balanced_curve = {}, {}, {}
    by_style_curve = {s: {} for s in style_names}
    bw = balanced_weights(style_names)
    mixed_n256_alloc = allocate_integer_budget(256, bw)
    for k in args.budgets:
        cluster_vals, balanced_vals, std_vals = [], [], []
        style_vals = {s: [] for s in style_names}
        for pp in per_problem:
            p_map = {s: pp[f"p_{s}"] for s in style_names}
            cluster_vals.append(multi_prefix_pass_at_k(p_map, k, eval_weights))
            balanced_vals.append(multi_prefix_pass_at_k(p_map, k, bw))
            std_vals.append(standard_pass_at_k(pp["p_pool"], k))
            for s in style_names:
                style_vals[s].append(standard_pass_at_k(p_map[s], k))
        cluster_curve[str(k)] = float(np.mean(cluster_vals))
        balanced_curve[str(k)] = float(np.mean(balanced_vals))
        std_curve[str(k)] = float(np.mean(std_vals))
        for s in style_names:
            by_style_curve[s][str(k)] = float(np.mean(style_vals[s]))

    subjects = sorted({str(pp.get("subject") or "") for pp in per_problem if pp.get("subject")})
    pass_at_k_by_subject = {subj: {"mixed": {}, "by_style": {s: {} for s in style_names}} for subj in subjects}
    n_by_subject = {}
    for subj in subjects:
        subset = [pp for pp in per_problem if str(pp.get("subject") or "") == subj]
        n_by_subject[subj] = len(subset)
        bw_s = balanced_weights(style_names)
        for k in args.budgets:
            mixed_vals = []
            style_vals = {s: [] for s in style_names}
            for pp in subset:
                p_map = {s: pp[f"p_{s}"] for s in style_names}
                mixed_vals.append(multi_prefix_pass_at_k(p_map, k, bw_s))
                for s in style_names:
                    style_vals[s].append(standard_pass_at_k(p_map[s], k))
            pass_at_k_by_subject[subj]["mixed"][str(k)] = float(np.mean(mixed_vals))
            for s in style_names:
                pass_at_k_by_subject[subj]["by_style"][s][str(k)] = float(np.mean(style_vals[s]))

    results = {
        "benchmark": benchmark,
        "checkpoint": args.checkpoint,
        "cluster_meta": str(cluster_meta_path) if cluster_meta_path else None,
        "hf_dataset": args.hf_dataset if not args.test_parquet else None,
        "test_parquet": args.test_parquet,
        "prefix_modes": style_names,
        "style_token_format": args.style_token_format,
        "num_problems": len(problems),
        "samples_total": args.samples_total,
        "samples_per_style": samples_by_style,
        "mixed_n256_alloc": mixed_n256_alloc,
        "sampling_mode": args.sampling_mode,
        "eval_weights": eval_weights,
        "train_style_proportions": train_weights,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "budgets": args.budgets,
        "pass_at_k_cluster_prefix": cluster_curve,
        "pass_at_k_cluster_prefix_balanced_formula": balanced_curve,
        "pass_at_k_standard": std_curve,
        "pass_at_k_by_style": by_style_curve,
        "pass_at_k_by_subject": pass_at_k_by_subject,
        "n_by_subject": n_by_subject,
        "mean_output_len_by_prefix": {
            s: float(np.mean(gen_lens_by_prefix[s])) if gen_lens_by_prefix[s] else 0.0
            for s in style_names
        },
        "per_problem": per_problem,
    }
    (out_dir / "pass_at_k.json").write_text(json.dumps(results, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = args.budgets
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(ks, [balanced_curve[str(k)] for k in ks], "-o", color="black", label="mixed (k split across styles)")
        cmap = plt.cm.tab10
        for i, s in enumerate(style_names):
            ax.plot(ks, [by_style_curve[s][str(k)] for k in ks], "-", color=cmap(i % 10), label=s)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples)")
        ax.set_ylabel("Pass@k")
        ax.set_title(f"{benchmark} — {Path(args.checkpoint).parent.name}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pass_at_k.png", dpi=150)
    except Exception as e:
        print(f"Plot skipped: {e}")

    style_hdr = "".join(f"{s:>10}" for s in style_names)
    if pass_at_k_by_subject:
        for subj in sorted(pass_at_k_by_subject):
            label = {"aime2024": "AIME-2024", "aime2025": "AIME-2025"}.get(subj, subj)
            n_s = n_by_subject[subj]
            mixed = pass_at_k_by_subject[subj]["mixed"]
            bys = pass_at_k_by_subject[subj]["by_style"]
            print(f"\n=== {label} (n={n_s}) Pass@k ===")
            print(f"{'k':>6} {'mixed':>10}{style_hdr}")
            for k in args.budgets:
                bits = "".join(f"{bys[s][str(k)]:>10.4f}" for s in style_names)
                print(f"{k:>6} {mixed[str(k)]:>10.4f}{bits}")
    else:
        print(f"\n=== {benchmark} Pass@k ===")
        print(f"{'k':>6} {'mixed':>10}{style_hdr}")
        for k in args.budgets:
            bits = "".join(f"{by_style_curve[s][str(k)]:>10.4f}" for s in style_names)
            print(f"{k:>6} {balanced_curve[str(k)]:>10.4f}{bits}")
    print(f"mixed k=256 alloc (256/K per style): {mixed_n256_alloc}")
    print(f"\nWrote {out_dir / 'pass_at_k.json'}")
    return results


def grade_prefix_dump(out_dir: Path, args) -> None:
    gen_path = out_dir / "generations.jsonl"
    meta_path = out_dir / "eval_meta.json"
    if not gen_path.is_file():
        raise FileNotFoundError(f"missing {gen_path}")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    rows = _grade_generation_rows(_read_jsonl(gen_path), args.grade_timeout, args.grade_flush_every)
    _write_jsonl(gen_path, rows)

    style_names = list(meta.get("prefix_modes") or args.oracle_styles or [])
    if not style_names:
        raise ValueError("grade: eval_meta.json missing prefix_modes")
    problems_meta = meta.get("problems") or []
    q_order = [p["question_id"] for p in problems_meta] if problems_meta else []
    if not q_order:
        seen = []
        for rec in rows:
            qid = rec["question_id"]
            if qid not in seen:
                seen.append(qid)
        q_order = seen
    q_index = {qid: i for i, qid in enumerate(q_order)}
    problems = problems_meta or [{"question_id": qid, "subject": ""} for qid in q_order]
    successes = [{s: [] for s in style_names} for _ in problems]
    gen_lens_by_prefix = {s: [] for s in style_names}
    for rec in rows:
        pi = q_index[rec["question_id"]]
        style = rec["prefix_style"]
        successes[pi][style].append(int(bool(rec["correct"])))
        gen_lens_by_prefix.setdefault(style, []).append(int(rec.get("output_len") or 0))

    eval_weights = meta.get("eval_weights") or balanced_weights(style_names)
    train_weights = meta.get("train_style_proportions") or eval_weights
    samples_by_style = meta.get("samples_per_style") or {s: 0 for s in style_names}
    if not args.samples_total:
        args.samples_total = int(meta.get("samples_total") or 0)
    cluster_meta_path = Path(meta["cluster_meta"]) if meta.get("cluster_meta") else None
    benchmark = meta.get("benchmark") or benchmark_label(args.hf_dataset)
    _write_prefix_passk(
        out_dir=out_dir,
        problems=problems,
        style_names=style_names,
        successes=successes,
        gen_lens_by_prefix=gen_lens_by_prefix,
        args=args,
        samples_by_style=samples_by_style,
        eval_weights=eval_weights,
        train_weights=train_weights,
        cluster_meta_path=cluster_meta_path,
        benchmark=benchmark,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MATH-500 cluster-prefix Pass@k eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cluster-meta", type=str, default=None)
    parser.add_argument(
        "--oracle-styles",
        type=str,
        nargs="*",
        default=None,
        help="Use oracle teacher prefixes instead of cluster styles, e.g. DeepSeek-R1 gpt-oss-120b",
    )
    parser.add_argument(
        "--style-token-format",
        type=str,
        default="hard",
        choices=("hard", "soft_special"),
        help="hard: [style] text; soft_special: <style_i> dedicated special token string",
    )
    parser.add_argument("--hf-dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument(
        "--test-parquet",
        type=str,
        default=None,
        help="SCAS (or other) validation parquet with messages + answer columns",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--samples-total", type=int, default=256)
    parser.add_argument(
        "--samples-per-style",
        type=int,
        default=0,
        help="If >0, generate this many samples for every style (overrides --samples-total).",
    )
    parser.add_argument("--sampling-mode", choices=["train", "balanced"], default="balanced")
    parser.add_argument("--max-problems", type=int, default=0, help="0 = all 500")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--data-parallel-replicas", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--grade-timeout",
        type=float,
        default=2.0,
        help="Max seconds per completion for sympy grading (prevents hangs)",
    )
    parser.add_argument(
        "--grade-flush-every",
        type=int,
        default=100,
        help="Flush generations.jsonl every N graded completions",
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "grade", "all"],
        default="all",
        help="generate: vLLM dump only; grade: CPU sympy from generations.jsonl; all: both",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "grade":
        grade_prefix_dump(out_dir, args)
        return

    from transformers import AutoTokenizer

    cluster_meta_path = Path(args.cluster_meta) if args.cluster_meta else None
    if cluster_meta_path is None and not args.oracle_styles:
        raise ValueError("Provide --cluster-meta or --oracle-styles")

    style_names, train_weights = load_style_names(cluster_meta_path, args.oracle_styles)
    eval_weights = train_weights if args.sampling_mode == "train" else balanced_weights(style_names)
    if args.samples_per_style and args.samples_per_style > 0:
        samples_by_style = {s: int(args.samples_per_style) for s in style_names}
        args.samples_total = int(args.samples_per_style) * len(style_names)
    else:
        samples_by_style = allocate_integer_budget(args.samples_total, eval_weights)

    problems = (
        load_scas_val_problems(args.test_parquet, args.max_problems)
        if args.test_parquet
        else load_math500_problems(args.hf_dataset, args.max_problems)
    )
    if args.test_parquet:
        benchmark = "SCAS-val"
    else:
        benchmark = benchmark_label(args.hf_dataset)
    print(f"Loaded {len(problems)} problems from {args.test_parquet or args.hf_dataset}")
    print(f"Styles ({len(style_names)}): {style_names}")
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"Generation counts/problem: {samples_by_style} (total={args.samples_total})")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, fix_mistral_regex=True)
    tokenizer = configure_tokenizer(tokenizer)

    prompts, meta, samples_per_prompt = [], [], []
    for pi, prob in enumerate(problems):
        for style in style_names:
            # For soft_special, oracle styles are typically "style_1"; format as <style_1>.
            # If already "<style_1>", format_style_prefix would mis-wrap — normalize first.
            style_key = style[1:-1] if style.startswith("<") and style.endswith(">") else style
            prefix = format_style_prefix(style_key, args.style_token_format)
            messages = [{"role": "user", "content": f"{prefix}\n{prob['question']}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            meta.append((pi, style))
            samples_per_prompt.append(samples_by_style[style])

    token_prompts = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]

    n_gpus = count_visible_gpus()
    num_heads = load_num_attention_heads(args.checkpoint)
    vocab_size = load_vocab_size(args.checkpoint)
    tp = choose_tensor_parallel_size(n_gpus, num_heads, args.tensor_parallel_size, vocab_size)
    num_replicas = resolve_data_parallel_replicas(n_gpus, tp, args.data_parallel_replicas)
    max_model_len = max(args.max_tokens + 1024, 4096)
    print(f"vLLM: tp={tp}, replicas={num_replicas}, gpus={n_gpus}")

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if os.environ.get("EOS_FIX", "0") == "1":
        stop_ids = set()
        for tok_str in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid >= 0:
                stop_ids.add(int(tid))
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        sampling_kwargs["stop_token_ids"] = sorted(stop_ids)
        print(f"EOS_FIX: stop_token_ids={sampling_kwargs['stop_token_ids']}")

    outputs = generate_with_data_parallel_replicas(
        checkpoint=args.checkpoint,
        prompt_token_ids=token_prompts,
        tensor_parallel_size=tp,
        num_replicas=num_replicas,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        sampling_kwargs=sampling_kwargs,
        samples_per_prompt=samples_per_prompt,
    )

    from tqdm.auto import tqdm

    do_grade = args.stage != "generate"
    successes = [{s: [] for s in style_names} for _ in problems]
    gen_lens_by_prefix = {s: [] for s in style_names}
    gen_path = out_dir / "generations.jsonl"
    total_completions = sum(len(out.outputs) for out in outputs)
    dump_desc = "Grading" if do_grade else "Dumping"
    print(
        f"{dump_desc} {total_completions} completions"
        + (f" (timeout={args.grade_timeout}s/completion)" if do_grade else " (ungraded)")
        + "...",
        flush=True,
    )

    dumped = 0
    with gen_path.open("w", buffering=1) as fout:
        pbar = tqdm(total=total_completions, desc=dump_desc, unit="comp", file=sys.stderr)
        try:
            for (pi, style), out in zip(meta, outputs):
                prob = problems[pi]
                for comp in out.outputs:
                    text = tokenizer.decode(comp.token_ids, skip_special_tokens=True)
                    correct = None
                    if do_grade:
                        correct = bool(
                            answers_match(text, prob["answer"], timeout_sec=args.grade_timeout)
                        )
                        successes[pi][style].append(int(correct))
                    gen_lens_by_prefix[style].append(len(text))
                    fout.write(
                        json.dumps(
                            {
                                "question_id": prob["question_id"],
                                "prefix_style": style,
                                "correct": correct,
                                "gold_answer": prob["answer"],
                                "output_len": len(text),
                                "n_tokens": len(comp.token_ids),
                                "finish_reason": getattr(comp, "finish_reason", None),
                                "output": text[:20000],
                            }
                        )
                        + "\n"
                    )
                    dumped += 1
                    pbar.update(1)
                    if args.grade_flush_every > 0 and dumped % args.grade_flush_every == 0:
                        fout.flush()
        finally:
            pbar.close()
    print(f"{dump_desc} {dumped} completions -> {gen_path}", flush=True)

    eval_meta = {
        "eval_type": "cluster_prefix",
        "benchmark": benchmark,
        "checkpoint": args.checkpoint,
        "cluster_meta": str(cluster_meta_path) if cluster_meta_path else None,
        "hf_dataset": args.hf_dataset if not args.test_parquet else None,
        "test_parquet": args.test_parquet,
        "prefix_modes": style_names,
        "style_token_format": args.style_token_format,
        "num_problems": len(problems),
        "samples_total": args.samples_total,
        "samples_per_style": samples_by_style,
        "sampling_mode": args.sampling_mode,
        "eval_weights": eval_weights,
        "train_style_proportions": train_weights,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "budgets": args.budgets,
        "problems": [
            {"question_id": p["question_id"], "subject": p.get("subject", ""), "answer": p["answer"]}
            for p in problems
        ],
    }
    (out_dir / "eval_meta.json").write_text(json.dumps(eval_meta, indent=2))
    (out_dir / "generations.done").write_text(f"{dumped}\n")

    if args.stage == "generate":
        print(f"Wrote ungraded dump {gen_path} ({dumped} rows)")
        return

    _write_prefix_passk(
        out_dir=out_dir,
        problems=problems,
        style_names=style_names,
        successes=successes,
        gen_lens_by_prefix=gen_lens_by_prefix,
        args=args,
        samples_by_style=samples_by_style,
        eval_weights=eval_weights,
        train_weights=train_weights,
        cluster_meta_path=cluster_meta_path,
        benchmark=benchmark,
    )


if __name__ == "__main__":
    main()
