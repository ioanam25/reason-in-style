#!/usr/bin/env python3
"""Style-diversity pack for init-ladder Pass@k generations.

For covz (prefix) runs:
  - per-style Pass@k
  - length stats by style
  - embedding centroids + between/within style cosine geometry

For vanilla runs:
  - overall Pass@k + length stats
  - null "fake-style" geometry (random 6-way split) for comparison

Usage:
  python scripts/analysis/analyze_init_ladder_style_diversity.py \
    --eval-dirs-glob '<SCRATCH_ROOT>/.../model-evals/*qwen3-4b-*math500*' \
    --out-dir .../init_ladder_style_diversity \
    --embed-model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --max-per-style 400 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def pass_at_k(corrects: list[bool], k: int) -> float:
    """Unbiased Pass@k for n samples with c correct (Chen et al.)."""
    n = len(corrects)
    c = int(sum(corrects))
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def mean_pass_at_k_by_problem(rows: list[dict], k: int) -> float:
    by_q: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_q[str(r["question_id"])].append(bool(r["correct"]))
    if not by_q:
        return 0.0
    return float(np.mean([pass_at_k(v, k) for v in by_q.values()]))


def length_stats(lens: np.ndarray) -> dict:
    if lens.size == 0:
        return {"n": 0}
    qs = np.quantile(lens, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    return {
        "n": int(lens.size),
        "mean": float(lens.mean()),
        "std": float(lens.std()),
        "cv": float(lens.std() / max(lens.mean(), 1e-6)),
        "q10": qs[0],
        "q25": qs[1],
        "q50": qs[2],
        "q75": qs[3],
        "q90": qs[4],
    }


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def pairwise_centroid_cos(centroids: np.ndarray) -> dict:
    """centroids: [K, D] L2-normalized -> mean/min pairwise cosine distance (1-cos)."""
    K = centroids.shape[0]
    if K < 2:
        return {"mean_cos_sim": None, "mean_cos_dist": None, "min_cos_dist": None}
    sims = centroids @ centroids.T
    dists = []
    cos_sims = []
    for i in range(K):
        for j in range(i + 1, K):
            cos_sims.append(float(sims[i, j]))
            dists.append(float(1.0 - sims[i, j]))
    return {
        "mean_cos_sim": float(np.mean(cos_sims)),
        "mean_cos_dist": float(np.mean(dists)),
        "min_cos_dist": float(np.min(dists)),
        "max_cos_dist": float(np.max(dists)),
    }


def within_between(z: np.ndarray, labels: np.ndarray) -> dict:
    z = l2_normalize(z)
    labs = np.unique(labels)
    cents = []
    within = []
    for lab in labs:
        idx = np.where(labels == lab)[0]
        c = l2_normalize(z[idx].mean(axis=0, keepdims=True))[0]
        cents.append(c)
        # cosine distance to centroid
        sims = z[idx] @ c
        within.extend((1.0 - sims).tolist())
    cents = np.stack(cents, axis=0)
    between = pairwise_centroid_cos(cents)
    mean_within = float(np.mean(within)) if within else None
    sep = None
    if mean_within is not None and between["mean_cos_dist"] is not None:
        sep = float(between["mean_cos_dist"] / max(mean_within, 1e-6))
    return {
        "mean_within_cos_dist": mean_within,
        "between": between,
        "separation_ratio": sep,  # between / within; higher => more distinct styles
        "n_styles": int(len(labs)),
        "n_embed": int(len(z)),
    }


def parse_dir_name(name: str) -> dict:
    arm = "covz" if "aez-covz" in name else ("vanilla" if "standard" in name else "unknown")
    if "1p7b-base" in name:
        size = "qwen3-1p7b-base"
    elif "4b-instruct" in name:
        size = "qwen3-4b-instruct"
    elif "4b-thinking" in name:
        size = "qwen3-4b-thinking"
    elif "4b-base" in name:
        size = "qwen3-4b-base"
    else:
        size = "unknown"
    if "math500" in name:
        bench = "math500"
    elif "olympiad" in name:
        bench = "olympiadbench"
    elif "amc" in name:
        bench = "amc"
    elif "scas-val" in name:
        bench = "scasval"
    else:
        bench = "unknown"
    return {"size": size, "arm": arm, "bench": bench, "dir_name": name}


def load_generations(path: Path, max_rows: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            text = o.get("output") or o.get("text") or o.get("completion") or ""
            rows.append(
                {
                    "question_id": o["question_id"],
                    "prefix_style": o.get("prefix_style"),
                    "correct": bool(o["correct"]),
                    "output_len": int(o.get("output_len") or len(text)),
                    "text": text,
                }
            )
            if max_rows is not None and i + 1 >= max_rows:
                break
    return rows


def subsample_for_embed(rows: list[dict], max_per_style: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        key = r["prefix_style"] or "none"
        by[key].append(r)
    out = []
    for key, items in by.items():
        if len(items) <= max_per_style:
            out.extend(items)
        else:
            idx = rng.choice(len(items), size=max_per_style, replace=False)
            out.extend(items[i] for i in idx)
    return out


def embed_texts(texts: list[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if device.startswith("cuda") else None)
    model = model.to(device)
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # truncate long gens; stored texts already ~4k chars
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # last-token / mean pool — gte uses last token often; use attention-masked mean
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            embs.append(pooled.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def analyze_dir(
    eval_dir: Path,
    budgets: list[int],
    max_per_style: int,
    embed_model: str | None,
    device: str,
    batch_size: int,
    seed: int,
) -> dict:
    meta = parse_dir_name(eval_dir.name)
    gen_path = eval_dir / "generations.jsonl"
    pass_path = eval_dir / "pass_at_k.json"
    result = {**meta, "eval_dir": str(eval_dir)}
    if pass_path.exists():
        pk = json.loads(pass_path.read_text())
        result["reported_pass_at_256"] = (
            pk.get("pass_at_k_cluster_prefix", {}) or pk.get("pass_at_k_standard", {})
        ).get("256")
        result["mean_output_len_by_prefix"] = pk.get("mean_output_len_by_prefix")
    if not gen_path.exists():
        result["error"] = "missing generations.jsonl"
        return result

    rows = load_generations(gen_path)
    result["n_generations"] = len(rows)

    # length overall
    all_lens = np.array([r["output_len"] for r in rows], dtype=np.float64)
    result["length_overall"] = length_stats(all_lens)

    is_prefix = meta["arm"] == "covz" and any(r["prefix_style"] for r in rows)
    budgets = list(budgets)

    if is_prefix:
        styles = sorted({r["prefix_style"] for r in rows if r["prefix_style"]})
        per_style = {}
        for s in styles:
            sub = [r for r in rows if r["prefix_style"] == s]
            lens = np.array([r["output_len"] for r in sub], dtype=np.float64)
            per_style[s] = {
                "length": length_stats(lens),
                "pass_at_k": {str(k): mean_pass_at_k_by_problem(sub, k) for k in budgets},
                "n": len(sub),
            }
        result["per_style"] = per_style
        # spread of style means
        means = [per_style[s]["length"]["mean"] for s in styles if per_style[s]["length"].get("n", 0)]
        p256 = [per_style[s]["pass_at_k"].get("256", 0.0) for s in styles]
        result["length_mean_range"] = float(max(means) - min(means)) if means else None
        result["length_mean_cv_across_styles"] = float(np.std(means) / max(np.mean(means), 1e-6)) if means else None
        result["pass256_across_styles"] = {
            "mean": float(np.mean(p256)),
            "std": float(np.std(p256)),
            "min": float(np.min(p256)),
            "max": float(np.max(p256)),
            "range": float(np.max(p256) - np.min(p256)),
        }
    else:
        result["per_style"] = None
        result["pass_at_k_overall"] = {str(k): mean_pass_at_k_by_problem(rows, k) for k in budgets}

    # embeddings
    if embed_model:
        if is_prefix:
            sample = subsample_for_embed(rows, max_per_style, seed)
            labels = np.array([r["prefix_style"] for r in sample])
            # map to ints
            style_to_i = {s: i for i, s in enumerate(sorted(set(labels)))}
            lab_i = np.array([style_to_i[s] for s in labels])
        else:
            # null: random 6-way split of subsample
            rng = np.random.default_rng(seed)
            # take up to 6*max_per_style
            n_take = min(len(rows), 6 * max_per_style)
            idx = rng.choice(len(rows), size=n_take, replace=False)
            sample = [rows[i] for i in idx]
            lab_i = rng.integers(0, 6, size=len(sample))
            result["embed_note"] = "vanilla_null_random_6way_split"

        texts = [r["text"] if r["text"] else " " for r in sample]
        z = embed_texts(texts, embed_model, device, batch_size)
        geom = within_between(z, lab_i)
        result["embedding_geometry"] = geom

        if is_prefix:
            # also store centroid pairwise matrix summary only
            result["embedding_geometry"]["kind"] = "true_prefix_styles"
        else:
            result["embedding_geometry"]["kind"] = "null_random_split"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=str, required=True)
    ap.add_argument("--name-regex", type=str, default="qwen3-(1p7b-base|4b-base|4b-instruct|4b-thinking)")
    ap.add_argument("--bench-regex", type=str, default="math500|amc|olympiadbench-en|scas-val")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    ap.add_argument("--embed-model", type=str, default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--max-per-style", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.eval_root)
    name_re = re.compile(args.name_regex)
    bench_re = re.compile(args.bench_regex)
    dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and name_re.search(d.name) and bench_re.search(d.name)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_model = None if args.no_embed else args.embed_model
    all_results = []
    for d in dirs:
        print(f"=== {d.name} ===", flush=True)
        try:
            r = analyze_dir(
                d,
                budgets=args.budgets,
                max_per_style=args.max_per_style,
                embed_model=embed_model,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        except Exception as e:
            r = {**parse_dir_name(d.name), "eval_dir": str(d), "error": repr(e)}
            print("ERROR", e, flush=True)
        all_results.append(r)
        (out_dir / f"{d.name}.json").write_text(json.dumps(r, indent=2))

    # summary table
    summary = []
    for r in all_results:
        if r.get("error"):
            summary.append({**{k: r.get(k) for k in ("size", "arm", "bench")}, "error": r["error"]})
            continue
        eg = r.get("embedding_geometry") or {}
        summary.append(
            {
                "size": r["size"],
                "arm": r["arm"],
                "bench": r["bench"],
                "pass256": r.get("reported_pass_at_256"),
                "len_mean": (r.get("length_overall") or {}).get("mean"),
                "len_cv": (r.get("length_overall") or {}).get("cv"),
                "len_mean_range_styles": r.get("length_mean_range"),
                "pass256_style_range": (r.get("pass256_across_styles") or {}).get("range"),
                "embed_between_cos_dist": (eg.get("between") or {}).get("mean_cos_dist"),
                "embed_within_cos_dist": eg.get("mean_within_cos_dist"),
                "embed_separation_ratio": eg.get("separation_ratio"),
                "embed_kind": eg.get("kind"),
            }
        )

    (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown
    lines = [
        "# Init-ladder style diversity",
        "",
        "| model | arm | bench | pass@256 | len_cv | style_len_range | style_pass_range | embed_between | sep_ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        if s.get("error"):
            lines.append(f"| {s['size']} | {s['arm']} | {s['bench']} | ERR | | | | | |")
            continue

        def fmt(x, pct=False):
            if x is None:
                return "—"
            return f"{100*x:.1f}%" if pct else f"{x:.3f}"

        lines.append(
            "| {size} | {arm} | {bench} | {p} | {lcv} | {lr} | {pr} | {eb} | {sr} |".format(
                size=s["size"].replace("qwen3-", ""),
                arm=s["arm"],
                bench=s["bench"],
                p=fmt(s["pass256"], pct=True),
                lcv=fmt(s["len_cv"]),
                lr=fmt(s["len_mean_range_styles"]),
                pr=fmt(s["pass256_style_range"], pct=True),
                eb=fmt(s["embed_between_cos_dist"]),
                sr=fmt(s["embed_separation_ratio"]),
            )
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("Wrote", out_dir / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
