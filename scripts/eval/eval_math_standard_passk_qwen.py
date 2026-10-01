#!/usr/bin/env python3
"""
Pass@k on MATH-500 for standard (question-only) math decoders.

Paper (ModC §4.2 gray baseline): sample n completions per problem from a
single question-only prompt, grade \\boxed{} answers, report Pass@k.

Usage:
  python scripts/eval/eval_math_standard_passk_qwen.py \\
      --checkpoint checkpoints/scas-standard-qwen3-4b-base \\
      --output-dir model-evals/scas-standard-qwen3-4b-base-math500-passk \\
      --samples-per-problem 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from scripts.eval.eval_math_cluster_prefix_qwen import (  # noqa: E402
    benchmark_label,
    load_math500_problems,
)
from scripts.eval.eval_scas_passk_utils import load_scas_val_problems  # noqa: E402
from scripts.math_answer_utils import answers_match  # noqa: E402


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


def _write_standard_passk(out_dir: Path, args, benchmark: str, per_problem: list[dict], problems: list[dict]) -> dict:
    std_curve = {}
    for k in args.budgets:
        vals = [standard_pass_at_k(pp["p_success"], k) for pp in per_problem]
        std_curve[str(k)] = float(np.mean(vals))

    by_subject = {}
    for pp in per_problem:
        subj = str(pp.get("subject") or "")
        if not subj:
            continue
        by_subject.setdefault(subj, []).append(pp["p_success"])
    pass_at_k_by_subject = {}
    for subj, ps in by_subject.items():
        pass_at_k_by_subject[subj] = {
            str(k): float(np.mean([standard_pass_at_k(p, k) for p in ps])) for k in args.budgets
        }

    results = {
        "benchmark": benchmark,
        "eval_type": "standard_sft_question_only",
        "checkpoint": args.checkpoint,
        "hf_dataset": args.hf_dataset if not args.test_parquet else None,
        "test_parquet": args.test_parquet,
        "num_problems": len(problems),
        "samples_per_problem": args.samples_per_problem,
        "prompt_format": "question only (no style prefix)",
        "eos_fix": os.environ.get("EOS_FIX", "0") == "1",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "enable_thinking": None if args.enable_thinking is None else bool(args.enable_thinking),
        "max_tokens": args.max_tokens,
        "budgets": args.budgets,
        "pass_at_k_standard": std_curve,
        "pass_at_k_by_subject": pass_at_k_by_subject,
        "n_by_subject": {
            s: sum(1 for pp in per_problem if str(pp.get("subject") or "") == s)
            for s in pass_at_k_by_subject
        },
        "mean_p_success": float(np.mean([pp["p_success"] for pp in per_problem])),
        "per_problem": per_problem,
    }
    (out_dir / "pass_at_k.json").write_text(json.dumps(results, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = args.budgets
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(ks, [std_curve[str(k)] for k in ks], "-o", color="gray", label="Standard Pass@k")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples)")
        ax.set_ylabel("Pass@k")
        ax.set_title(f"{benchmark} standard — {Path(args.checkpoint).parent.name}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pass_at_k.png", dpi=150)
        print(f"Wrote {out_dir / 'pass_at_k.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # AIME 2024/2025 get their own tables; do not headline the pooled n=60 average.
    if pass_at_k_by_subject:
        for subj in sorted(pass_at_k_by_subject):
            n_s = results["n_by_subject"][subj]
            label = {"aime2024": "AIME-2024", "aime2025": "AIME-2025"}.get(subj, subj)
            curve = pass_at_k_by_subject[subj]
            print(f"\n=== {label} (n={n_s}) Standard Pass@k (question-only) ===")
            print(f"{'k':>6} {'Pass@k':>10}")
            for k in args.budgets:
                print(f"{k:>6} {curve[str(k)]:>10.4f}")
    else:
        print(f"\n=== {benchmark} Standard Pass@k (question-only) ===")
        print(f"{'k':>6} {'Pass@k':>10}")
        for k in args.budgets:
            print(f"{k:>6} {std_curve[str(k)]:>10.4f}")
    print(f"\nWrote {out_dir / 'pass_at_k.json'}")
    return results


def grade_standard_dump(out_dir: Path, args) -> None:
    from tqdm.auto import tqdm

    gen_path = out_dir / "generations.jsonl"
    meta_path = out_dir / "eval_meta.json"
    if not gen_path.is_file():
        raise FileNotFoundError(f"missing {gen_path}")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    raw = _read_jsonl(gen_path)
    graded_rows = []
    pbar = tqdm(total=len(raw), desc="Grading", unit="comp", file=sys.stderr)
    try:
        for rec in raw:
            rec = dict(rec)
            rec["correct"] = bool(
                answers_match(rec.get("output") or "", rec.get("gold_answer"), timeout_sec=args.grade_timeout)
            )
            graded_rows.append(rec)
            pbar.update(1)
    finally:
        pbar.close()
    _write_jsonl(gen_path, graded_rows)

    q_meta = {p["question_id"]: p for p in meta.get("problems") or []}
    by_q: dict[str, list[int]] = {}
    q_order: list[str] = []
    for rec in graded_rows:
        qid = rec["question_id"]
        if qid not in by_q:
            by_q[qid] = []
            q_order.append(qid)
        by_q[qid].append(int(bool(rec["correct"])))
    problems = meta.get("problems") or [{"question_id": qid, "subject": ""} for qid in q_order]
    per_problem = []
    for qid in q_order:
        flags = by_q[qid]
        subj = (q_meta.get(qid) or {}).get("subject", "")
        per_problem.append(
            {
                "question_id": qid,
                "subject": subj,
                "p_success": float(np.mean(flags)) if flags else 0.0,
                "n_samples": len(flags),
            }
        )
    if meta.get("samples_per_problem"):
        args.samples_per_problem = int(meta["samples_per_problem"])
    benchmark = meta.get("benchmark") or benchmark_label(args.hf_dataset)
    _write_standard_passk(out_dir, args, benchmark, per_problem, problems)


def main() -> None:
    parser = argparse.ArgumentParser(description="MATH-500 standard Pass@k (question-only)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf-dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument(
        "--test-parquet",
        type=str,
        default=None,
        help="SCAS (or other) validation parquet with messages + answer columns",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--samples-per-problem", type=int, default=256)
    parser.add_argument("--max-problems", type=int, default=0, help="0 = all 500")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1, help="vLLM top_k; -1 disables")
    parser.add_argument("--min-p", type=float, default=0.0, help="vLLM min_p; 0 disables")
    parser.add_argument(
        "--enable-thinking",
        type=int,
        choices=[0, 1],
        default=None,
        help="Qwen3 hybrid: 1=thinking, 0=no-thinking. Omit to leave chat-template default.",
    )
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

    from tqdm.auto import tqdm

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "grade":
        grade_standard_dump(out_dir, args)
        return

    from transformers import AutoTokenizer

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
    print(f"Samples per problem: {args.samples_per_problem}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, fix_mistral_regex=True)
    tokenizer = configure_tokenizer(tokenizer)

    prompts, meta = [], []
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = bool(args.enable_thinking)
    for pi, prob in enumerate(problems):
        messages = [{"role": "user", "content": prob["question"]}]
        text = tokenizer.apply_chat_template(messages, **chat_kwargs)
        prompts.append(text)
        meta.append(pi)
    if args.enable_thinking is not None:
        print(
            f"enable_thinking={bool(args.enable_thinking)} "
            f"prompt_suffix={prompts[0][-80:]!r}"
        )

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
        "n": args.samples_per_problem,
    }
    if args.top_k is not None and int(args.top_k) > 0:
        sampling_kwargs["top_k"] = int(args.top_k)
    if args.min_p is not None and float(args.min_p) > 0:
        sampling_kwargs["min_p"] = float(args.min_p)
    print(
        "sampling",
        {k: sampling_kwargs[k] for k in sampling_kwargs if k != "n"},
    )
    # vLLM is constructed with skip_tokenizer_init=True, so stop tokens must be
    # passed explicitly; Base checkpoints list only <|endoftext|> in their
    # generation_config while the chat template ends turns with <|im_end|>.
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
    )

    do_grade = args.stage != "generate"
    per_problem: list[dict] = []
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
            for pi, out in zip(meta, outputs):
                prob = problems[pi]
                correct_flags = []
                for comp in out.outputs:
                    text = tokenizer.decode(comp.token_ids, skip_special_tokens=True)
                    correct = None
                    if do_grade:
                        correct = bool(
                            answers_match(text, prob["answer"], timeout_sec=args.grade_timeout)
                        )
                        correct_flags.append(int(correct))
                    fout.write(
                        json.dumps(
                            {
                                "question_id": prob["question_id"],
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

                if do_grade:
                    p_success = float(np.mean(correct_flags)) if correct_flags else 0.0
                    per_problem.append(
                        {
                            "question_id": prob["question_id"],
                            "subject": prob["subject"],
                            "p_success": p_success,
                            "n_samples": len(correct_flags),
                        }
                    )
        finally:
            pbar.close()
    print(f"{dump_desc} {dumped} completions -> {gen_path}", flush=True)

    eval_meta = {
        "eval_type": "standard_sft_question_only",
        "benchmark": benchmark,
        "checkpoint": args.checkpoint,
        "hf_dataset": args.hf_dataset if not args.test_parquet else None,
        "test_parquet": args.test_parquet,
        "num_problems": len(problems),
        "samples_per_problem": args.samples_per_problem,
        "temperature": args.temperature,
        "top_p": args.top_p,
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

    _write_standard_passk(out_dir, args, benchmark, per_problem, problems)


if __name__ == "__main__":
    main()
