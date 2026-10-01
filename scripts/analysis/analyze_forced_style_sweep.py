#!/usr/bin/env python3
"""Forced-style sweep on existing balanced Pass@k generations.

For each question, all 6 style prefixes were already sampled. Measures whether
init stage preserves controllable diversity vs collapse:

  - per-style Pass@k / length
  - matched-question length separation (mean pairwise |Δlen| across styles)
  - matched-question TF-IDF centroid geometry (between-style cos dist)
  - collapse rate: fraction of questions with very low length CV across styles
  - vanilla null: random 6-way split on same questions

No new generation required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def pass_at_k(corrects: list[bool], k: int) -> float:
    n = len(corrects)
    c = int(sum(corrects))
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def mean_pass_at_k(rows: list[dict], k: int) -> float:
    by: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by[str(r["question_id"])].append(bool(r["correct"]))
    if not by:
        return 0.0
    return float(np.mean([pass_at_k(v, k) for v in by.values()]))


def parse_meta(name: str) -> dict:
    arm = "covz" if "aez-covz" in name else ("vanilla" if "standard" in name else "unknown")
    if "4b-final-gmm" in name:
        source = "4b-final-gmm"
    elif "1p7b-gmm" in name:
        source = "1p7b-gmm"
    elif "4b-gmm" in name:
        source = "4b-gmm-early"
    elif arm == "covz":
        source = "covz-legacy"
    else:
        source = arm

    # Student size: ...-gmm-<student>-<student>-k...
    m = re.search(r"gmm-(qwen3-[a-z0-9-]+)-(qwen3-[a-z0-9-]+)-k", name)
    if m:
        size = m.group(2).removeprefix("qwen3-")
    elif "0p6b-base" in name:
        size = "0p6b-base"
    elif "1p7b-base" in name:
        size = "1p7b-base"
    elif "4b-instruct" in name:
        size = "4b-instruct"
    elif "4b-thinking" in name:
        size = "4b-thinking"
    elif "4b-base" in name:
        size = "4b-base"
    elif "0p6b" in name:
        size = "0p6b"
    elif re.search(r"(?:^|[-_])qwen3-1p7b(?:-k|_|$)", name) or "-1p7b-k" in name:
        size = "1p7b"
    elif re.search(r"(?:^|[-_])qwen3-4b(?:-k|_|$)", name) or "-4b-k" in name:
        size = "4b"
    else:
        size = "unknown"

    if "math500" in name:
        bench = "math500"
    elif "olympiad" in name:
        bench = "olympiad"
    elif "amc" in name:
        bench = "amc"
    elif "scas-val" in name:
        bench = "scasval"
    else:
        bench = "unknown"
    return {"size": size, "arm": arm, "source": source, "bench": bench}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            rows.append(
                {
                    "question_id": str(o["question_id"]),
                    "prefix_style": o.get("prefix_style"),
                    "correct": bool(o["correct"]),
                    "output_len": int(o.get("output_len") or 0),
                    "text": (o.get("output") or o.get("text") or "")[:2000],
                }
            )
    return rows


def style_centroids_tfidf(texts: list[str], labels: list[str]) -> tuple[np.ndarray, list[str]]:
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    styles = sorted(set(labels))
    cents = []
    for s in styles:
        idx = [i for i, lab in enumerate(labels) if lab == s]
        cents.append(np.asarray(X[idx].mean(axis=0)).ravel())
    C = np.stack(cents, axis=0)
    # l2
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
    return C, styles


def pairwise_cos_dist(C: np.ndarray) -> dict:
    K = C.shape[0]
    sims = C @ C.T
    dists = [float(1 - sims[i, j]) for i in range(K) for j in range(i + 1, K)]
    return {
        "mean_cos_dist": float(np.mean(dists)),
        "min_cos_dist": float(np.min(dists)),
        "max_cos_dist": float(np.max(dists)),
    }


def matched_length_separation(by_q_style_lens: dict[str, dict[str, list[int]]], styles: list[str]) -> dict:
    """For each question with all styles, mean pairwise |mean_len_i - mean_len_j| / mean_len."""
    abs_gaps = []
    rel_gaps = []
    cvs = []
    collapse_len = 0
    n_q = 0
    for q, smap in by_q_style_lens.items():
        if any(s not in smap or not smap[s] for s in styles):
            continue
        means = np.array([float(np.mean(smap[s])) for s in styles], dtype=np.float64)
        n_q += 1
        cv = float(means.std() / max(means.mean(), 1e-6))
        cvs.append(cv)
        if cv < 0.05:
            collapse_len += 1
        for i in range(len(styles)):
            for j in range(i + 1, len(styles)):
                g = abs(means[i] - means[j])
                abs_gaps.append(g)
                rel_gaps.append(g / max(means.mean(), 1e-6))
    return {
        "n_questions_matched": n_q,
        "mean_pairwise_abs_len_gap": float(np.mean(abs_gaps)) if abs_gaps else None,
        "mean_pairwise_rel_len_gap": float(np.mean(rel_gaps)) if rel_gaps else None,
        "mean_len_cv_across_styles": float(np.mean(cvs)) if cvs else None,
        "collapse_rate_len_cv_lt_0.05": float(collapse_len / n_q) if n_q else None,
    }


def matched_tfidf_separation(rows: list[dict], styles: list[str], max_per_style_per_q: int, seed: int) -> dict:
    """Build one centroid text per (q, style) from up to M samples; measure style geometry."""
    rng = np.random.default_rng(seed)
    by: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = r["prefix_style"]
        if s not in styles:
            continue
        by[r["question_id"]][s].append(r["text"] or " ")

    # per-style pooled texts across questions (for global centroids)
    style_texts: dict[str, list[str]] = {s: [] for s in styles}
    # also per-question style mean embeddings via concatenated sample
    q_style_docs = []  # (q, style, doc)
    for q, smap in by.items():
        if any(s not in smap or not smap[s] for s in styles):
            continue
        for s in styles:
            items = smap[s]
            if len(items) > max_per_style_per_q:
                pick = [items[i] for i in rng.choice(len(items), size=max_per_style_per_q, replace=False)]
            else:
                pick = items
            doc = "\n".join(pick)
            style_texts[s].extend(pick)
            q_style_docs.append((q, s, doc))

    # Global style centroids
    all_texts = []
    all_labs = []
    for s in styles:
        # subsample for TF-IDF fit speed
        texts = style_texts[s]
        if len(texts) > 800:
            idx = rng.choice(len(texts), size=800, replace=False)
            texts = [texts[i] for i in idx]
        all_texts.extend(texts)
        all_labs.extend([s] * len(texts))
    C, st = style_centroids_tfidf(all_texts, all_labs)
    global_geom = pairwise_cos_dist(C)

    # Matched: for each question, style-doc vectors in same TF-IDF space, mean pairwise dist
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    docs = [d for _, _, d in q_style_docs]
    X = vec.fit_transform(docs)
    # map back
    q_to_rows = defaultdict(dict)
    for i, (q, s, _) in enumerate(q_style_docs):
        q_to_rows[q][s] = i

    matched_dists = []
    collapse_tfidf = 0
    n_q = 0
    for q, smap in q_to_rows.items():
        if any(s not in smap for s in styles):
            continue
        idxs = [smap[s] for s in styles]
        Xi = X[idxs]
        # normalize
        sims = cosine_similarity(Xi)
        dists = [float(1 - sims[i, j]) for i in range(len(styles)) for j in range(i + 1, len(styles))]
        md = float(np.mean(dists))
        matched_dists.append(md)
        n_q += 1
        if md < 0.05:
            collapse_tfidf += 1

    return {
        "global_style_centroid_geometry": global_geom,
        "matched_mean_pairwise_cos_dist": float(np.mean(matched_dists)) if matched_dists else None,
        "matched_median_pairwise_cos_dist": float(np.median(matched_dists)) if matched_dists else None,
        "collapse_rate_tfidf_dist_lt_0.05": float(collapse_tfidf / n_q) if n_q else None,
        "n_questions_matched_tfidf": n_q,
    }


def analyze_covz(rows: list[dict], budgets: list[int], seed: int) -> dict:
    styles = sorted({r["prefix_style"] for r in rows if r["prefix_style"]})
    per_style = {}
    for s in styles:
        sub = [r for r in rows if r["prefix_style"] == s]
        lens = np.array([r["output_len"] for r in sub], dtype=np.float64)
        per_style[s] = {
            "n": len(sub),
            "len_mean": float(lens.mean()),
            "len_std": float(lens.std()),
            "pass_at_k": {str(k): mean_pass_at_k(sub, k) for k in budgets},
        }
    by_q_style_lens: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["prefix_style"]:
            by_q_style_lens[r["question_id"]][r["prefix_style"]].append(r["output_len"])

    len_sep = matched_length_separation(by_q_style_lens, styles)
    tfidf_sep = matched_tfidf_separation(rows, styles, max_per_style_per_q=4, seed=seed)

    p256 = [per_style[s]["pass_at_k"]["256"] for s in styles]
    means = [per_style[s]["len_mean"] for s in styles]
    return {
        "styles": styles,
        "per_style": per_style,
        "pass256_style_range": float(max(p256) - min(p256)),
        "pass256_style_std": float(np.std(p256)),
        "len_mean_range": float(max(means) - min(means)),
        "matched_length": len_sep,
        "matched_tfidf": tfidf_sep,
        # primary diversity scores (higher => more style control)
        "diversity_score_len": len_sep.get("mean_pairwise_rel_len_gap"),
        "diversity_score_tfidf": tfidf_sep.get("matched_mean_pairwise_cos_dist"),
        "collapse_score": float(
            np.mean(
                [
                    len_sep.get("collapse_rate_len_cv_lt_0.05") or 0.0,
                    tfidf_sep.get("collapse_rate_tfidf_dist_lt_0.05") or 0.0,
                ]
            )
        ),
    }


def analyze_vanilla_null(rows: list[dict], budgets: list[int], seed: int) -> dict:
    """Null diversity: randomly assign 6 fake styles, same matched metrics."""
    rng = np.random.default_rng(seed)
    styles = [f"style_{i}" for i in range(1, 7)]
    # group by question then split samples into 6 bins
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    fake_rows = []
    for q, items in by_q.items():
        labs = rng.integers(0, 6, size=len(items))
        for r, lab in zip(items, labs):
            rr = dict(r)
            rr["prefix_style"] = styles[int(lab)]
            fake_rows.append(rr)
    out = analyze_covz(fake_rows, budgets, seed)
    out["note"] = "vanilla_null_random_6way"
    out["pass_at_k_overall"] = {str(k): mean_pass_at_k(rows, k) for k in budgets}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--name-regex", type=str, default="(1p7b-gmm|4b-final-gmm|aez-covz-qwen3-4b-|aez-covz-qwen3-1p7b-|scas-standard-qwen3)")
    ap.add_argument("--bench-regex", type=str, default="math500")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.eval_root)
    name_re = re.compile(args.name_regex)
    bench_re = re.compile(args.bench_regex)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and name_re.search(d.name) and bench_re.search(d.name))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for d in dirs:
        print(f"=== {d.name} ===", flush=True)
        meta = parse_meta(d.name)
        gen = d / "generations.jsonl"
        pk = d / "pass_at_k.json"
        rec = {**meta, "dir": d.name}
        if pk.exists():
            obj = json.loads(pk.read_text())
            rec["reported_pass256"] = (
                obj.get("pass_at_k_cluster_prefix") or obj.get("pass_at_k_standard") or {}
            ).get("256")
        if not gen.exists():
            rec["error"] = "missing generations"
            summary.append(rec)
            continue
        rows = load_rows(gen)
        rec["n_generations"] = len(rows)
        try:
            if meta["arm"] == "covz":
                rec["sweep"] = analyze_covz(rows, args.budgets, args.seed)
            else:
                rec["sweep"] = analyze_vanilla_null(rows, args.budgets, args.seed)
        except Exception as e:
            rec["error"] = repr(e)
            print("ERROR", e, flush=True)
        (out / f"{d.name}.json").write_text(json.dumps(rec, indent=2))
        summary.append(rec)

    (out / "all_results.json").write_text(json.dumps(summary, indent=2))

    # markdown figure table
    lines = [
        "# Forced-style sweep (matched questions, all 6 prefixes)",
        "",
        "Existing balanced Pass@k gens — no regen. Higher diversity / lower collapse => styles still controllable.",
        "",
        "| model | source | arm | bench | pass@256 | len_rel_gap | tfidf_dist | collapse | pass256_style_range |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r.get("error"):
            lines.append(f"| {r['size']} | {r.get('source','')} | {r['arm']} | {r['bench']} | ERR | | | | |")
            continue
        sw = r.get("sweep") or {}
        p = r.get("reported_pass256")
        lines.append(
            "| {size} | {source} | {arm} | {bench} | {p} | {lg} | {td} | {c} | {pr} |".format(
                size=r["size"],
                source=r.get("source", ""),
                arm=r["arm"] + ("/null" if r["arm"] == "vanilla" else ""),
                bench=r["bench"],
                p=f"{100*p:.1f}%" if p is not None else "—",
                lg=f"{sw.get('diversity_score_len'):.3f}" if sw.get("diversity_score_len") is not None else "—",
                td=f"{sw.get('diversity_score_tfidf'):.3f}" if sw.get("diversity_score_tfidf") is not None else "—",
                c=f"{sw.get('collapse_score'):.3f}" if sw.get("collapse_score") is not None else "—",
                pr=f"{100*sw.get('pass256_style_range'):.1f}%" if sw.get("pass256_style_range") is not None else "—",
            )
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("Wrote", out / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
