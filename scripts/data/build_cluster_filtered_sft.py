#!/usr/bin/env python3
"""Build vanilla (no prefix) SFT parquets filtered by teacher-side GMM style.

Source is the AE-GMM K=6 prefix dataset. Prefixes are stripped so training is
ordinary question→trace SFT; the GMM labels are used only to *select* rows.

Variants (written under cluster_prefix/<name>/k6):

  gmm-filter-style1..6    train on that GMM style only (shortest→longest)
  gmm-filter-rebal        upsample/downsample so each style has median count
  gmm-filter-rand-rebal   random subset with the same n as rebal (no GMM)
  gmm-filter-rand-style1  random subset with the same n as style_1 (no GMM)
  gmm-filter-is-rebal     keep every trace; per-question p=Unif(6)
  gmm-filter-is-obs       keep every trace; per-question p=Unif(observed styles)
  gmm-filter-is-global    keep every trace; global w ∝ 1/q(c) (rebal analogue)

Importance sampling (is-rebal). One row = one teacher trace on a question.
q is the per-question empirical style distribution, smoothed as

  q(c|x) = [n_c + K ε] / [K (1+ε)]   then renormalized over the C=6 styles

where K = #teacher traces on that question (9, or 18 for 633 questions) and
n_c = # of those traces whose GMM label is c. The k-sum in the writeup is over
those traces, not over styles: a 6-term sum would not be the empirical q(c|x).

p(c|x) defaults to Unif(1..6), the rebal target. Unobserved styles still have
q>0 after ε, but no traces, so the Monte Carlo only hits the observed support
and raw E_q[p/q] = (styles present)/6. Weights are mean-normalized to 1 so the
loss scale matches vanilla SFT. `--p-mode observed` uses Unif(styles that
actually appear on x) instead, which gives every question total weight 1
without that missing-mass factor.

Validation is prefix-stripped but not resampled, matching the balanced-prefix
builder (eval set stays fixed).

Usage:
  python scripts/data/build_cluster_filtered_sft.py --all
  python scripts/data/build_cluster_filtered_sft.py --variant style1
"""

from __future__ import annotations
import os

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scas_style_prefix import strip_style_prefix  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
DEFAULT_SRC = SROOT / "cluster_prefix" / "covz-qwen3-4b-final-gmm" / "k6"
DEFAULT_OUT_ROOT = SROOT / "cluster_prefix"

STYLE_VARIANTS = tuple(f"style{i}" for i in range(1, 7))
VARIANTS = STYLE_VARIANTS + (
    "rebal",
    "rand-rebal",
    "rand-style1",
    "is-rebal",
    "is-obs",
    "is-global",
)
ARM_NAME = {f"style{i}": f"gmm-filter-style{i}" for i in range(1, 7)}
ARM_NAME.update(
    {
        "rebal": "gmm-filter-rebal",
        "rand-rebal": "gmm-filter-rand-rebal",
        "rand-style1": "gmm-filter-rand-style1",
        "is-rebal": "gmm-filter-is-rebal",
        "is-obs": "gmm-filter-is-obs",
        "is-global": "gmm-filter-is-global",
    }
)
N_STYLES = 6
DEFAULT_IS_EPS = 0.01


def decode_msgs(raw):
    if isinstance(raw, str):
        return json.loads(raw), True
    if hasattr(raw, "tolist"):
        return [dict(m) for m in raw.tolist()], False
    return [dict(m) for m in raw], False


def encode_msgs(msgs, was_str):
    return json.dumps(msgs) if was_str else np.array(msgs, dtype=object)


def user_idx(msgs) -> int:
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            return i
    raise ValueError("no user turn found")


def strip_row(raw):
    msgs, was_str = decode_msgs(raw)
    i = user_idx(msgs)
    msgs[i] = dict(msgs[i])
    msgs[i]["content"] = strip_style_prefix(msgs[i]["content"])
    return encode_msgs(msgs, was_str)


def style_col(df: pd.DataFrame) -> pd.Series:
    if "cluster_style_name" in df.columns:
        return df["cluster_style_name"].astype(str)
    return df["style_name"].astype(str)


def strip_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["messages"] = [strip_row(m) for m in out["messages"]]
    return out


def resample_equal(df: pd.DataFrame, styles: pd.Series, target: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    tmp = df.copy()
    tmp["_style"] = styles.to_numpy()
    for _, grp in tmp.groupby("_style", sort=True):
        idx = rng.choice(len(grp), size=target, replace=len(grp) < target)
        parts.append(grp.iloc[idx])
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
        .drop(columns=["_style"])
    )


def random_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(int(n), len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def add_importance_weights(
    df: pd.DataFrame,
    eps: float,
    p_mode: str,
    mean_norm: bool,
    n_styles: int = N_STYLES,
) -> tuple[pd.DataFrame, dict]:
    """Keep every row. Weight trace (x, c) by p(c|x)/q(c|x).

    q_raw(c|x) = (n_c + K ε) / (K (1+ε)), then divide by sum_{c'=1..C} q_raw
    so q is a distribution on the C GMM styles. p is Unif(C) or Unif(observed).
    """
    if "question_id" not in df.columns:
        raise ValueError("IS variants need question_id")
    if p_mode not in ("uniform", "observed", "global"):
        raise ValueError(f"unknown p_mode {p_mode}")

    out = df.copy()
    styles = style_col(out).astype(str)
    out["_style"] = styles.to_numpy()
    k_per_q = out.groupby("question_id")["_style"].transform("size").to_numpy(dtype=np.float64)
    n_c = out.groupby(["question_id", "_style"], sort=False)["_style"].transform("size").to_numpy(dtype=np.float64)
    s_per_q = out.groupby("question_id")["_style"].transform("nunique").to_numpy(dtype=np.float64)

    if p_mode == "global":
        n_total = float(len(out))
        n_style = styles.value_counts().to_dict()
        q = np.array(
            [(n_style[s] + eps) / (n_total + n_styles * eps) for s in styles],
            dtype=np.float64,
        )
        p = np.full(len(out), 1.0 / n_styles, dtype=np.float64)
    else:
        q_raw = (n_c + k_per_q * eps) / (k_per_q * (1.0 + eps))
        z = (1.0 + n_styles * eps) / (1.0 + eps)
        q = q_raw / z
        if p_mode == "uniform":
            p = np.full(len(out), 1.0 / n_styles, dtype=np.float64)
        else:
            p = 1.0 / s_per_q
    w_raw = p / q
    w = w_raw / w_raw.mean() if mean_norm else w_raw.copy()

    out["is_q"] = q
    out["is_p"] = p
    out["is_n_c"] = n_c
    out["is_K"] = k_per_q
    out["is_n_styles_on_q"] = s_per_q
    out["is_weight_raw"] = w_raw
    out["is_weight"] = w
    out = out.drop(columns=["_style"])

    by_style = {}
    for name, grp in out.groupby(styles, sort=True):
        by_style[str(name)] = {
            "n": int(len(grp)),
            "mean_weight": float(grp["is_weight"].mean()),
            "mean_weight_raw": float(grp["is_weight_raw"].mean()),
            "sum_weight": float(grp["is_weight"].sum()),
        }
    stats = {
        "eps": float(eps),
        "p_mode": p_mode,
        "mean_norm": bool(mean_norm),
        "n_styles": int(n_styles),
        "n_rows": int(len(out)),
        "n_questions": int(out["question_id"].nunique()),
        "mean_weight": float(w.mean()),
        "mean_weight_raw": float(w_raw.mean()),
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "p50_weight": float(np.median(w)),
        "p90_weight": float(np.percentile(w, 90)),
        "p99_weight": float(np.percentile(w, 99)),
        "mean_styles_per_q": float(s_per_q.mean()),
        "by_style": by_style,
    }
    return out, stats


def build_variant(
    train: pd.DataFrame,
    variant: str,
    seed: int,
    eps: float = DEFAULT_IS_EPS,
    p_mode: str = "uniform",
    mean_norm: bool = True,
) -> tuple[pd.DataFrame, dict]:
    styles = style_col(train)
    counts = Counter(styles.tolist())
    info: dict = {
        "variant": variant,
        "rows_in": int(len(train)),
        "counts_in": {k: int(v) for k, v in sorted(counts.items())},
    }
    median = int(np.median(list(counts.values())))
    n_style1 = int(counts.get("style_1", 0))

    if variant in STYLE_VARIANTS:
        style_name = f"style_{variant.removeprefix('style')}"
        keep = train[styles.to_numpy() == style_name].reset_index(drop=True)
        info["filter"] = style_name
    elif variant == "rebal":
        keep = resample_equal(train, styles, median, seed)
        info["filter"] = "rebalance"
        info["target_per_style"] = median
    elif variant == "rand-rebal":
        n = median * len(counts)
        keep = random_subset(train, n, seed)
        info["filter"] = "random"
        info["n_match"] = "rebal_total"
        info["n"] = int(n)
    elif variant == "rand-style1":
        keep = random_subset(train, n_style1, seed)
        info["filter"] = "random"
        info["n_match"] = "style_1"
        info["n"] = n_style1
    elif variant == "is-rebal":
        keep, is_stats = add_importance_weights(train, eps=eps, p_mode=p_mode, mean_norm=mean_norm)
        info["filter"] = "importance_sampling"
        info["is"] = is_stats
    elif variant == "is-obs":
        keep, is_stats = add_importance_weights(
            train, eps=eps, p_mode="observed", mean_norm=mean_norm
        )
        info["filter"] = "importance_sampling_observed"
        info["is"] = is_stats
    elif variant == "is-global":
        keep, is_stats = add_importance_weights(
            train, eps=eps, p_mode="global", mean_norm=mean_norm
        )
        info["filter"] = "importance_sampling_global"
        info["is"] = is_stats
    else:
        raise ValueError(f"unknown variant {variant}")

    info["rows_out"] = int(len(keep))
    info["counts_out"] = dict(Counter(style_col(keep).tolist()))
    return keep, info


def write_variant(
    src: Path,
    out_root: Path,
    variant: str,
    seed: int,
    k: int,
    eps: float = DEFAULT_IS_EPS,
    p_mode: str = "uniform",
    mean_norm: bool = True,
) -> Path:
    if variant == "is-obs":
        p_mode = "observed"
    elif variant == "is-global":
        p_mode = "global"
    arm = ARM_NAME[variant]
    out = out_root / arm / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "src": str(src),
        "variant": variant,
        "arm": arm,
        "seed": seed,
        "k": k,
        "eps": float(eps),
        "p_mode": p_mode,
        "mean_norm": bool(mean_norm),
        "splits": {},
    }

    val_path = src / "validation.parquet"
    val = strip_split(pd.read_parquet(val_path))
    val.to_parquet(out / "validation.parquet", index=False)
    meta["splits"]["validation"] = {
        "variant": "passthrough_stripped",
        "rows": int(len(val)),
        "counts": dict(Counter(style_col(val).tolist())),
    }

    train = pd.read_parquet(src / "train.parquet")
    built, info = build_variant(train, variant, seed, eps=eps, p_mode=p_mode, mean_norm=mean_norm)
    built = strip_split(built)
    built.to_parquet(out / "train.parquet", index=False)
    meta["splits"]["train"] = info
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    extra = ""
    if variant in ("is-rebal", "is-obs", "is-global"):
        extra = (
            f"  mean_w_raw={info['is']['mean_weight_raw']:.4f}"
            f"  mean_w={info['is']['mean_weight']:.4f}"
            f"  max_w={info['is']['max_weight']:.4f}"
        )
    print(
        f"{arm}: train {info['rows_in']} -> {info['rows_out']}  {info.get('counts_out')}{extra}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--eps", type=float, default=DEFAULT_IS_EPS)
    ap.add_argument("--p-mode", choices=("uniform", "observed"), default="uniform")
    ap.add_argument("--no-mean-norm", action="store_true")
    args = ap.parse_args()

    if not args.all and args.variant is None:
        raise SystemExit("pass --all or --variant")
    variants = VARIANTS if args.all else (args.variant,)
    for v in variants:
        write_variant(
            args.src,
            args.out_root,
            v,
            args.seed,
            args.k,
            eps=args.eps,
            p_mode=args.p_mode,
            mean_norm=not args.no_mean_norm,
        )
    print("DONE cluster-filtered SFT parquets")


if __name__ == "__main__":
    main()
