#!/usr/bin/env python3
"""Stage 2: style claims over the compact records written by extract_style_records.py.

Per (arm, student, benchmark) dump this computes:

  forced   rebuilt `tab:forced` metrics on EOS_FIX generations - per-style Pass@k,
           matched-question length separation, matched-question TF-IDF geometry,
           and the collapse rates
  probe    style identifiability - can the applied prefix be recovered from the
           generated text? TF-IDF+logreg, handcrafted+RF, a length-invariant
           density-only RF, and a length-only control, all on a question-level split
  realized intended prefix vs realized style - student generations scored in the
           teacher handcrafted space and assigned to the nearest teacher-side GMM
           centroid, giving a KxK confusion matrix
  comple   why prefixes move Pass@k - per-style solve sets, pairwise Jaccard,
           leave-one-style-out Pass@k, and the same split by problem difficulty

Every metric is reported against the arm's own chance level, and the vanilla /
A1-random arms provide the nulls.
"""

from __future__ import annotations
import os

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.handcrafted_style_features import (  # noqa: E402
    DENSITY_FEATURES,
    FEATURE_NAMES,
    PROFILE_FEATURES,
)
from scripts.style_registry import ZSCORE_ROOT, iter_eval_dirs  # noqa: E402

RECORDS_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_records"
DEFAULT_OUT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_claims"
TEACHER_FEATURES_NPZ = REPO_ROOT / "data/scas/cluster_handcrafted_sweep-qwen3-0p6b/handcrafted_features.npz"

BUDGETS = [1, 8, 32, 128, 256]
# Style space for nearest-centroid assignment: length-invariant densities plus one
# log-length axis, so a student that is uniformly more verbose than the teacher
# does not collapse onto the longest cluster.
STYLE_SPACE = DENSITY_FEATURES + ["log_n_words"]


# --------------------------------------------------------------------------- #
# Pass@k
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k for c correct out of n samples (Chen et al. 2021)."""
    if n <= 0 or k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)))


def mean_pass_at_k(df: pd.DataFrame, k: int) -> float | None:
    """Question-averaged Pass@k; None when no question has k samples."""
    g = df.groupby("question_id", observed=True)["correct"].agg(["size", "sum"])
    g = g[g["size"] >= k]
    if g.empty:
        return None
    vals = [pass_at_k(int(n), int(c), k) for n, c in zip(g["size"], g["sum"])]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Teacher-side style basis
# --------------------------------------------------------------------------- #
def load_teacher_basis(tag: str, k: int, features_npz: Path) -> dict:
    """Teacher GMM centroids for one AE z-space, in the student-comparable space."""
    gmm_path = ZSCORE_ROOT / tag / "gmm_from_kmeans" / f"assignments_gmm_k{k:02d}.parquet"
    assign = pd.read_parquet(gmm_path)

    feat = np.load(features_npz, allow_pickle=True)
    X = np.asarray(feat["X"], dtype=np.float64)
    names = [str(x) for x in feat["feature_names"].tolist()]
    fqid = feat["question_id"].astype(str)
    fstyle = feat["style_name"].astype(str)

    key_to_cluster = {
        (str(q), str(s)): int(c)
        for q, s, c in zip(assign["question_id"].astype(str), assign["style_name"].astype(str), assign["cluster_id"].astype(int))
    }
    labels = np.array([key_to_cluster.get((q, s), -1) for q, s in zip(fqid, fstyle)], dtype=np.int32)
    keep = labels >= 0
    if keep.sum() < 0.99 * len(labels):
        raise RuntimeError(f"{tag}: only {keep.sum()}/{len(labels)} teacher rows aligned to GMM labels")
    X, labels = X[keep], labels[keep]

    # style_1 = shortest cluster, matching the prefix SFT dataset builder
    mean_len = assign.groupby("cluster_id")["trace_len"].mean().sort_values()
    rank_map = {int(cid): rank for rank, cid in enumerate(mean_len.index)}
    style_of_row = np.array([f"style_{rank_map[int(c)] + 1}" for c in labels])
    styles = [f"style_{i + 1}" for i in range(k)]

    df = pd.DataFrame(X, columns=names)
    df["log_n_words"] = np.log1p(df["n_words"])
    S = df[STYLE_SPACE].to_numpy(dtype=np.float64)
    mu = S.mean(axis=0)
    sd = np.clip(S.std(axis=0), 1e-9, None)
    Sz = (S - mu) / sd

    centroids = np.stack([Sz[style_of_row == s].mean(axis=0) for s in styles], axis=0)
    profile = {
        s: {f: float(df.loc[style_of_row == s, f].mean()) for f in PROFILE_FEATURES}
        for s in styles
    }
    props = {s: float((style_of_row == s).mean()) for s in styles}
    return {
        "tag": tag,
        "k": k,
        "styles": styles,
        "centroids": centroids,
        "mu": mu,
        "sd": sd,
        "profile": profile,
        "props": props,
        "cluster_rank_map": {str(cid): rank for cid, rank in rank_map.items()},
        "n_traces": int(len(labels)),
    }


def assign_nearest_style(df: pd.DataFrame, basis: dict) -> np.ndarray:
    """Nearest teacher centroid for each student generation, in standardized space."""
    d = df.copy()
    d["log_n_words"] = np.log1p(d["n_words"].to_numpy(dtype=np.float64))
    S = d[STYLE_SPACE].to_numpy(dtype=np.float64)
    Sz = (S - basis["mu"]) / basis["sd"]
    # squared euclidean to every centroid
    d2 = ((Sz[:, None, :] - basis["centroids"][None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    return np.asarray(basis["styles"], dtype=object)[idx]


# --------------------------------------------------------------------------- #
# Workstream 1: forced-style geometry on EOS_FIX
# --------------------------------------------------------------------------- #
def matched_length_separation(df: pd.DataFrame, styles: list[str], col: str) -> dict:
    """Mean pairwise |style mean - style mean| per question, over questions with all styles."""
    piv = df.pivot_table(index="question_id", columns="prefix_style", values=col, aggfunc="mean", observed=True)
    piv = piv.reindex(columns=styles)
    piv = piv.dropna(axis=0, how="any")
    if piv.empty or len(styles) < 2:
        return {"n_questions_matched": 0}
    M = piv.to_numpy(dtype=np.float64)
    row_mean = np.clip(M.mean(axis=1), 1e-6, None)
    cv = M.std(axis=1) / row_mean
    iu = np.triu_indices(len(styles), k=1)
    gaps = np.abs(M[:, iu[0]] - M[:, iu[1]])
    rel = gaps / row_mean[:, None]
    return {
        "n_questions_matched": int(M.shape[0]),
        "mean_pairwise_abs_gap": float(gaps.mean()),
        "mean_pairwise_rel_gap": float(rel.mean()),
        "mean_cv_across_styles": float(cv.mean()),
        "collapse_rate_cv_lt_0.05": float((cv < 0.05).mean()),
        "per_style_mean": {s: float(v) for s, v in zip(styles, M.mean(axis=0))},
    }


def matched_tfidf_separation(texts: pd.DataFrame, styles: list[str]) -> dict:
    """Per-question style docs in a shared TF-IDF space; mean pairwise cosine distance."""
    if texts.empty or len(styles) < 2:
        return {"n_questions_matched": 0}
    docs = (
        texts.groupby(["question_id", "prefix_style"], observed=True)["text"]
        .apply(lambda s: "\n".join(s))
        .reset_index()
    )
    counts = docs.groupby("question_id", observed=True)["prefix_style"].nunique()
    full_qs = set(counts[counts == len(styles)].index)
    docs = docs[docs["question_id"].isin(full_qs)]
    if docs.empty:
        return {"n_questions_matched": 0}

    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(docs["text"].tolist())
    pos = {(q, s): i for i, (q, s) in enumerate(zip(docs["question_id"], docs["prefix_style"]))}

    dists, collapsed = [], 0
    for q in sorted(full_qs):
        idxs = [pos[(q, s)] for s in styles]
        sims = cosine_similarity(X[idxs])
        iu = np.triu_indices(len(styles), k=1)
        md = float(np.mean(1.0 - sims[iu]))
        dists.append(md)
        collapsed += int(md < 0.05)

    # global per-style centroids over the pooled docs
    cents = []
    for s in styles:
        rows = [i for i, st in enumerate(docs["prefix_style"]) if st == s]
        cents.append(np.asarray(X[rows].mean(axis=0)).ravel())
    C = np.stack(cents)
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
    gsims = C @ C.T
    iu = np.triu_indices(len(styles), k=1)
    return {
        "n_questions_matched": len(dists),
        "matched_mean_pairwise_cos_dist": float(np.mean(dists)),
        "matched_median_pairwise_cos_dist": float(np.median(dists)),
        "collapse_rate_tfidf_dist_lt_0.05": float(collapsed / len(dists)),
        "global_mean_cos_dist": float(np.mean(1.0 - gsims[iu])),
        "global_min_cos_dist": float(np.min(1.0 - gsims[iu])),
        "global_max_cos_dist": float(np.max(1.0 - gsims[iu])),
    }


def forced_style_block(df: pd.DataFrame, texts: pd.DataFrame, styles: list[str]) -> dict:
    per_style = {}
    for s in styles:
        sub = df[df["prefix_style"] == s]
        per_style[s] = {
            "n": int(len(sub)),
            "n_tokens_mean": float(sub["n_tokens"].mean()),
            "n_words_mean": float(sub["n_words"].mean()),
            "pct_finish_length": float(100.0 * (sub["finish_reason"] == "length").mean()),
            "acc_mean": float(sub["correct"].mean()),
            "pass_at_k": {str(k): mean_pass_at_k(sub, k) for k in [1, 8, 32]},
        }
    len_sep = matched_length_separation(df, styles, "n_tokens")
    tfidf_sep = matched_tfidf_separation(texts, styles)
    p1 = [per_style[s]["pass_at_k"]["1"] for s in styles if per_style[s]["pass_at_k"]["1"] is not None]
    return {
        "per_style": per_style,
        "matched_length_tokens": len_sep,
        "matched_tfidf": tfidf_sep,
        "pass1_style_range": float(max(p1) - min(p1)) if len(p1) > 1 else None,
        "len_sep_pct": (
            None if len_sep.get("mean_pairwise_rel_gap") is None
            else round(100.0 * len_sep["mean_pairwise_rel_gap"], 2)
        ),
        "tfidf_dist": tfidf_sep.get("matched_mean_pairwise_cos_dist"),
        "collapse_score": float(
            np.mean(
                [
                    len_sep.get("collapse_rate_cv_lt_0.05") or 0.0,
                    tfidf_sep.get("collapse_rate_tfidf_dist_lt_0.05") or 0.0,
                ]
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Workstream 2a: style identifiability
# --------------------------------------------------------------------------- #
def question_split(qids: np.ndarray, seed: int, test_frac: float = 0.3) -> np.ndarray:
    """Boolean test mask, split on whole questions so no question spans both folds."""
    uniq = np.unique(qids)
    rng = np.random.default_rng(seed)
    test_q = set(rng.choice(uniq, size=max(1, int(round(test_frac * len(uniq)))), replace=False).tolist())
    return np.array([q in test_q for q in qids])


def _probe_scores(y_true, y_pred, styles: list[str]) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=styles, zero_division=0)),
    }


def identifiability_block(
    df: pd.DataFrame, texts: pd.DataFrame, styles: list[str], seed: int, max_rows: int
) -> dict:
    """Can the applied prefix be recovered from the generation? Chance is 1/K."""
    k = len(styles)
    out: dict = {"n_styles": k, "chance": 1.0 / k if k else None}
    if k < 2:
        out["note"] = "single-style arm: identifiability undefined"
        return out

    # --- handcrafted-feature probes on the full record table ---
    d = df[df["prefix_style"].isin(styles)]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    y = d["prefix_style"].astype(str).to_numpy()
    q = d["question_id"].astype(str).to_numpy()
    test = question_split(q, seed)
    if test.all() or (~test).all():
        out["note"] = "degenerate question split"
        return out

    feature_sets = {
        "handcrafted_full": FEATURE_NAMES,
        "handcrafted_density": DENSITY_FEATURES,
        "length_only": ["n_tokens"],
    }
    for name, cols in feature_sets.items():
        X = d[cols].to_numpy(dtype=np.float32)
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"
        )
        clf.fit(X[~test], y[~test])
        out[name] = _probe_scores(y[test], clf.predict(X[test]), styles)
        out[name]["n_train"] = int((~test).sum())
        out[name]["n_test"] = int(test.sum())
        if name == "handcrafted_full":
            imp = sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1])[:8]
            out[name]["top_features"] = [{"feature": f, "importance": float(v)} for f, v in imp]

    # --- lexical probe on the sampled texts ---
    t = texts[texts["prefix_style"].isin(styles)]
    if len(t) >= 200:
        ty = t["prefix_style"].astype(str).to_numpy()
        tq = t["question_id"].astype(str).to_numpy()
        ttest = question_split(tq, seed)
        if not (ttest.all() or (~ttest).all()):
            vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
            Xt = vec.fit_transform(t["text"].tolist())
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(Xt[~ttest], ty[~ttest])
            out["tfidf_logreg"] = _probe_scores(ty[ttest], clf.predict(Xt[ttest]), styles)
            out["tfidf_logreg"]["n_train"] = int((~ttest).sum())
            out["tfidf_logreg"]["n_test"] = int(ttest.sum())

    best = max(
        (out[nm]["acc"] for nm in ("handcrafted_full", "tfidf_logreg") if nm in out),
        default=None,
    )
    out["best_acc"] = best
    out["best_acc_over_chance"] = None if best is None else float(best - 1.0 / k)
    return out


# --------------------------------------------------------------------------- #
# Workstream 2b: intended vs realized style
# --------------------------------------------------------------------------- #
def realized_block(df: pd.DataFrame, styles: list[str], basis: dict, seed: int, max_rows: int) -> dict:
    """Confusion between the prefix the student was given and the style it produced."""
    tstyles = basis["styles"]
    d = df[df["prefix_style"].isin(styles)]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    realized = assign_nearest_style(d, basis)

    mat = np.zeros((len(styles), len(tstyles)), dtype=np.float64)
    for i, s in enumerate(styles):
        m = d["prefix_style"].to_numpy() == s
        if not m.any():
            continue
        for j, t in enumerate(tstyles):
            mat[i, j] = float((realized[m] == t).mean())

    shared = [s for s in styles if s in tstyles]
    diag = float(np.mean([mat[styles.index(s), tstyles.index(s)] for s in shared])) if shared else None
    marginal = {t: float((realized == t).mean()) for t in tstyles}

    # Does the student reproduce the teacher's ordering of styles on each feature?
    prof_rho = {}
    for f in PROFILE_FEATURES:
        if f not in d.columns:
            continue
        student = np.array([float(d.loc[d["prefix_style"] == s, f].mean()) for s in shared])
        teacher = np.array([basis["profile"][s][f] for s in shared])
        if len(shared) >= 3 and np.std(student) > 0 and np.std(teacher) > 0:
            rho = float(np.corrcoef(pd.Series(student).rank(), pd.Series(teacher).rank())[0, 1])
            prof_rho[f] = rho

    return {
        "teacher_tag": basis["tag"],
        "intended_styles": styles,
        "realized_styles": tstyles,
        "confusion_row_normalized": mat.tolist(),
        "diagonal_mass": diag,
        "chance_diagonal": 1.0 / len(tstyles),
        "diagonal_over_chance": None if diag is None else float(diag - 1.0 / len(tstyles)),
        "realized_marginal": marginal,
        "teacher_marginal": basis["props"],
        "realized_entropy_bits": float(
            -sum(p * math.log2(p) for p in marginal.values() if p > 0)
        ),
        "max_entropy_bits": float(math.log2(len(tstyles))),
        "profile_rank_corr": prof_rho,
        "profile_rank_corr_mean": float(np.mean(list(prof_rho.values()))) if prof_rho else None,
        "n_scored": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Workstream 2c: style complementarity
# --------------------------------------------------------------------------- #
def complementarity_block(df: pd.DataFrame, styles: list[str], loso_k: int) -> dict:
    """Do styles solve different problems, or is one style carrying the arm?"""
    if len(styles) < 2:
        return {"note": "single-style arm: complementarity undefined"}

    solved: dict[str, set] = {}
    for s in styles:
        sub = df[df["prefix_style"] == s]
        g = sub.groupby("question_id", observed=True)["correct"].sum()
        solved[s] = set(g[g > 0].index)
    all_q = set(df["question_id"].unique())
    union = set().union(*solved.values())

    jac = {}
    for i, a in enumerate(styles):
        for b in styles[i + 1 :]:
            inter = len(solved[a] & solved[b])
            uni = len(solved[a] | solved[b])
            jac[f"{a}|{b}"] = float(inter / uni) if uni else None
    jvals = [v for v in jac.values() if v is not None]

    counts = defaultdict(int)
    for s in styles:
        for q in solved[s]:
            counts[q] += 1
    unique_solves = {s: int(sum(1 for q in solved[s] if counts[q] == 1)) for s in styles}

    all_k = mean_pass_at_k(df, loso_k)
    loso = {}
    for s in styles:
        sub = df[df["prefix_style"] != s]
        v = mean_pass_at_k(sub, loso_k)
        loso[s] = {
            "pass_at_k": v,
            "delta_vs_all": None if (v is None or all_k is None) else float(all_k - v),
        }

    # difficulty from this arm's own per-question success rate
    p_success = df.groupby("question_id", observed=True)["correct"].mean()
    qcut = pd.qcut(p_success.rank(method="first"), 4, labels=["Q1 hardest", "Q2", "Q3", "Q4 easiest"])
    bucket = dict(zip(p_success.index, qcut))
    d = df.copy()
    d["difficulty"] = d["question_id"].map(bucket)
    by_diff = {}
    for lab in ["Q1 hardest", "Q2", "Q3", "Q4 easiest"]:
        sub = d[d["difficulty"] == lab]
        if sub.empty:
            continue
        per = {s: float(sub.loc[sub["prefix_style"] == s, "correct"].mean()) for s in styles}
        vals = [v for v in per.values() if v == v]
        by_diff[lab] = {
            "n_questions": int(sub["question_id"].nunique()),
            "acc_by_style": per,
            "acc_range": float(max(vals) - min(vals)) if len(vals) > 1 else None,
            "union_solve_rate": float(
                len({q for s in styles for q in solved[s]} & set(sub["question_id"].unique()))
                / max(1, sub["question_id"].nunique())
            ),
            "mean_style_solve_rate": float(
                np.mean(
                    [
                        len(solved[s] & set(sub["question_id"].unique())) / max(1, sub["question_id"].nunique())
                        for s in styles
                    ]
                )
            ),
        }

    best_style_cov = max(len(solved[s]) for s in styles) / max(1, len(all_q))
    return {
        "loso_k": loso_k,
        "pass_at_k_all_styles": all_k,
        "leave_one_style_out": loso,
        "max_loso_delta": (
            max((v["delta_vs_all"] for v in loso.values() if v["delta_vs_all"] is not None), default=None)
        ),
        "pairwise_jaccard": jac,
        "mean_pairwise_jaccard": float(np.mean(jvals)) if jvals else None,
        "min_pairwise_jaccard": float(np.min(jvals)) if jvals else None,
        "per_style_solved": {s: len(solved[s]) for s in styles},
        "unique_solves": unique_solves,
        "union_coverage": float(len(union) / max(1, len(all_q))),
        "best_single_style_coverage": float(best_style_cov),
        "union_gain_over_best_style": float(len(union) / max(1, len(all_q)) - best_style_cov),
        "n_questions": len(all_q),
        "by_difficulty": by_diff,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def null_styles(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Assign fake balanced prefixes so vanilla dumps yield a matched null."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    labels = [f"style_{i + 1}" for i in range(k)]
    d["prefix_style"] = [labels[i] for i in rng.integers(0, k, size=len(d))]
    return d


def analyze_dump(rec_dir: Path, bases: dict, args) -> dict:
    meta = json.loads((rec_dir / "meta.json").read_text())
    df = pd.read_parquet(rec_dir / "records.parquet")
    df["prefix_style"] = df["prefix_style"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["finish_reason"] = df["finish_reason"].astype(str)

    tpath = rec_dir / "texts.jsonl"
    texts = pd.read_json(tpath, lines=True) if tpath.exists() and tpath.stat().st_size else pd.DataFrame(
        columns=["question_id", "prefix_style", "text"]
    )
    if not texts.empty:
        texts["question_id"] = texts["question_id"].astype(str)
        texts["prefix_style"] = texts["prefix_style"].astype(str)

    styles = sorted([s for s in df["prefix_style"].unique() if s and s != "None"])
    is_vanilla = len(styles) == 0
    if is_vanilla:
        # vanilla SFT carries no prefix; give it random balanced pseudo-styles so the
        # identifiability and complementarity numbers have a same-shape null.
        df = null_styles(df, args.null_k, args.seed)
        texts = null_styles(texts, args.null_k, args.seed) if not texts.empty else texts
        styles = [f"style_{i + 1}" for i in range(args.null_k)]

    res = {
        **{k: meta[k] for k in ("arm", "student", "scale", "benchmark", "protocol", "eval_dir") if k in meta},
        "teacher_tag": meta.get("teacher_tag"),
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "n_styles": len(styles),
        "styles": styles,
        "pseudo_styles": bool(is_vanilla),
        "pct_finish_length": float(100.0 * (df["finish_reason"] == "length").mean()),
        "n_tokens_mean": float(df["n_tokens"].mean()),
        "pass_at_k": {str(k): mean_pass_at_k(df, k) for k in BUDGETS},
    }
    res["forced"] = forced_style_block(df, texts, styles)
    res["identifiability"] = identifiability_block(df, texts, styles, args.seed, args.max_probe_rows)
    res["complementarity"] = complementarity_block(df, styles, args.loso_k)

    tag = meta.get("teacher_tag") or args.default_teacher_tag
    basis = bases.get(tag)
    if basis is not None and len(styles) > 1:
        res["realized"] = realized_block(df, styles, basis, args.seed, args.max_probe_rows)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--features-npz", type=Path, default=TEACHER_FEATURES_NPZ)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--null-k", type=int, default=6)
    ap.add_argument("--loso-k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-probe-rows", type=int, default=60000)
    ap.add_argument("--default-teacher-tag", default="covz-qwen3-4b-final")
    ap.add_argument("--only", default=None, help="substring filter on the eval dir name")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="reuse per_dump/*.json when present (default: on, for Slurm requeues)",
    )
    ap.add_argument("--force", action="store_true", help="recompute even if per_dump exists")
    args = ap.parse_args()
    if args.force:
        args.skip_existing = False

    dirs = sorted(d for d in args.records_root.iterdir() if d.is_dir() and (d / "records.parquet").exists())
    preferred = {p.name for p, _ in iter_eval_dirs(protocol="eosfix")}
    dirs = [d for d in dirs if d.name in preferred]
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]
    print(f"{len(dirs)} preferred EOS_FIX record dirs (4B-final over older AE)", flush=True)

    tags = sorted({json.loads((d / "meta.json").read_text()).get("teacher_tag") for d in dirs} - {None})
    tags = sorted(set(tags) | {args.default_teacher_tag})
    bases = {}
    for tag in tags:
        print(f"loading teacher basis {tag}", flush=True)
        bases[tag] = load_teacher_basis(tag, args.k, args.features_npz)

    out_per = args.out_dir / "per_dump"
    out_per.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dirs:
        out_path = out_per / f"{d.name}.json"
        if args.skip_existing and out_path.exists():
            try:
                res = json.loads(out_path.read_text())
                rows.append(res)
                print(f"skip (exists) {d.name}", flush=True)
                continue
            except Exception:
                pass
        print(f"analyze {d.name}", flush=True)
        try:
            res = analyze_dump(d, bases, args)
        except Exception as exc:  # keep going; one bad dump should not sink the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        out_path.write_text(json.dumps(res, indent=2))
        rows.append(res)
        # checkpoint registry after each dump so requeues keep partial progress
        (args.out_dir / "style_claims_registry.json").write_text(json.dumps(rows, indent=2))
        ident = res.get("identifiability", {})
        real = res.get("realized", {})
        print(
            f"  {res['arm']:24} {res['student']:11} {res['benchmark']:9} "
            f"ident={ident.get('best_acc')} diag={real.get('diagonal_mass')} "
            f"lensep={res['forced'].get('len_sep_pct')}%",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "style_claims_registry.json").write_text(json.dumps(rows, indent=2))
    teacher_dump = {
        tag: {
            "styles": b["styles"],
            "props": b["props"],
            "profile": b["profile"],
            "cluster_rank_map": b["cluster_rank_map"],
            "n_traces": b["n_traces"],
        }
        for tag, b in bases.items()
    }
    (args.out_dir / "teacher_style_profiles.json").write_text(json.dumps(teacher_dump, indent=2))
    print(f"wrote {len(rows)} dumps -> {args.out_dir}")


if __name__ == "__main__":
    main()
