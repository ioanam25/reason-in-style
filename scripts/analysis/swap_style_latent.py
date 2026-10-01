#!/usr/bin/env python3
"""Workstream 3: hold the content latent fixed and swap the style latent.

Two experiments on one AE checkpoint (default `covz-qwen3-4b-final`):

3a  Teacher-forced NLL swap matrix. For a trace `s` whose GMM cluster is `k`, we
    rebuild the decoder prefix from `[c(s); z']` for a range of substituted `z'`
    and score the reconstruction NLL of `s` itself. If `z` is a style code the
    matrix minimises on its diagonal, and the `c`/`z` 2x2 ablation shows that
    swapping content hurts far more than swapping style.

    conditions: [c(s); z(s)]      identity
                [c(s); mu_j]      j = 1..K, the GMM cluster means
                [c(s); z(s')]     another trace's style
                [c(s'); z(s)]     another trace's content
                [c(s'); z(s')]    both swapped

3b  Qualitative / restyle rendering. Greedy-decode `[c(s); mu_j]` for every j,
    store the teacher gold boxed answer, then score with analyze_ae_restyle.py.

Usage:
  python scripts/analysis/swap_style_latent.py --tag covz-qwen3-4b-final --n-traces 900
  python scripts/analysis/swap_style_latent.py --skip-nll --n-qualitative 120 --max-new-tokens 1024
"""

from __future__ import annotations
import os

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.handcrafted_style_features import PROFILE_FEATURES, extract_features  # noqa: E402
from scripts.style_registry import ZSCORE_ROOT  # noqa: E402

DEFAULT_OUT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "latent_swap"
# matches TRACE_DATA_PATH / SCAS_TRACE_JSON from scripts/scas_model_size.sh
DEFAULT_HF_DATASET = "scas_traces_dataset"
DEFAULT_RAW_JSON = str(REPO_ROOT / "configs/scas/scas_modc_dataset.json")


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_module(checkpoint: str, device: torch.device):
    from src.module import DisentangledLightningModule

    print(f"loading {checkpoint}", flush=True)
    module = DisentangledLightningModule.load_from_checkpoint(
        checkpoint, map_location="cpu", strict=False
    )
    module.eval()
    module.model.to(device)
    for p in module.model.parameters():
        p.requires_grad_(False)
    return module


def decoder_start_id(module) -> int:
    from scripts.eval.eval_gemini_styles import get_decoder_sequence_start_token_id

    hp = getattr(module, "hparams", {})
    sid = hp.get("decoder_start_token_id") if hasattr(hp, "get") else None
    if sid is not None:
        return int(sid)
    tok = module.model.decoder.tokenizer
    sid = get_decoder_sequence_start_token_id(tok)
    if sid is None:
        raise RuntimeError("no decoder start token available")
    return int(sid)


def load_cluster_labels(tag: str, k: int) -> pd.DataFrame:
    path = ZSCORE_ROOT / tag / "gmm_from_kmeans" / f"assignments_gmm_k{k:02d}.parquet"
    df = pd.read_parquet(path)
    mean_len = df.groupby("cluster_id")["trace_len"].mean().sort_values()
    rank = {int(c): r for r, c in enumerate(mean_len.index)}
    df["style"] = [f"style_{rank[int(c)] + 1}" for c in df["cluster_id"]]
    return df[["question_id", "style_name", "cluster_id", "style", "trace_len"]]


# --------------------------------------------------------------------------- #
# encoding / scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode(module, traces: list[str], device, max_len: int, batch_size: int):
    from scripts.eval.eval_gemini_styles import _encoder_tokenizer

    tok = _encoder_tokenizer(module.model)
    cs, zs = [], []
    for i in range(0, len(traces), batch_size):
        enc = tok(
            traces[i : i + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            c, z, _ = module.model(
                enc["input_ids"].to(device), enc["attention_mask"].to(device), decode=False
            )
        cs.append(c.float().cpu())
        zs.append(z.float().cpu())
    return torch.cat(cs), torch.cat(zs)


def decoder_batch(module, traces: list[str], device, max_len: int, start_id: int):
    """Tokenize for the decoder exactly as TraceCollator does at training time."""
    tok = module.model.decoder.tokenizer
    dec = tok(
        traces,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max(1, max_len - 1),
    )
    ids, mask = dec["input_ids"], dec["attention_mask"]
    b = ids.shape[0]
    ids = torch.cat([torch.full((b, 1), start_id, dtype=ids.dtype), ids], dim=1)
    mask = torch.cat([torch.ones((b, 1), dtype=mask.dtype), mask], dim=1)
    return ids[:, :max_len].to(device), mask[:, :max_len].to(device)


@torch.no_grad()
def per_sample_nll(decoder, ids, mask, c_z, chunk: int) -> torch.Tensor:
    """Teacher-forced NLL per sequence, same shift/mask convention as training."""
    hidden = decoder.get_hidden(ids, c_z)
    shift_h = hidden[:, :-1, :]
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100

    B, L, _ = shift_h.shape
    tot = torch.zeros(B, device=ids.device, dtype=torch.float32)
    cnt = torch.zeros(B, device=ids.device, dtype=torch.float32)
    for i in range(0, L, chunk):
        h = shift_h[:, i : i + chunk, :]
        lab = labels[:, i : i + chunk]
        logits = decoder.project_hidden(h).float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100, reduction="none"
        ).view(lab.shape)
        valid = (lab != -100).float()
        tot += (ce * valid).sum(dim=1)
        cnt += valid.sum(dim=1)
    return tot / cnt.clamp(min=1.0)


@torch.no_grad()
def greedy_decode(module, c_z, device, start_id: int, max_new_tokens: int) -> list[str]:
    """Batched greedy decode from a latent prefix (no KV cache; matches the eval path)."""
    decoder = module.model.decoder
    tok = decoder.tokenizer
    eos = tok.eos_token_id
    B = c_z.shape[0]
    gen = torch.full((B, 1), start_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(max_new_tokens - 1):
            hidden = decoder.get_hidden(gen, c_z)
            logits = decoder.project_hidden(hidden[:, -1:, :]).float()
            nxt = logits[:, -1, :].argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            gen = torch.cat([gen, nxt[:, None]], dim=1)
            done = done | (nxt == eos)
            if bool(done.all()):
                break
    return [tok.decode(row[1:], skip_special_tokens=True) for row in gen]


# --------------------------------------------------------------------------- #
# 3a: NLL swap matrix
# --------------------------------------------------------------------------- #
def swap_matrix(module, recs, mus, styles, device, args) -> dict:
    start_id = decoder_start_id(module)
    rng = np.random.default_rng(args.seed)
    decoder = module.model.decoder

    traces = [r["trace"] for r in recs]
    true_style = np.array([r["_style"] for r in recs])
    print(f"encoding {len(traces)} traces", flush=True)
    C, Z = encode(module, traces, device, args.max_len, args.encode_batch_size)

    # partner trace for the c/z ablation: a different question, so content really differs
    qids = np.array([r["question_id"] for r in recs])
    partner = np.empty(len(recs), dtype=np.int64)
    for i in range(len(recs)):
        for _ in range(20):
            j = int(rng.integers(0, len(recs)))
            if qids[j] != qids[i]:
                partner[i] = j
                break
        else:
            partner[i] = (i + 1) % len(recs)

    mus_t = torch.as_tensor(mus, dtype=torch.float32)
    conditions = ["identity", "z_other", "c_other", "both_other"] + [f"mu:{s}" for s in styles]
    rows = []
    t0 = time.time()
    for start in range(0, len(recs), args.nll_batch_size):
        end = min(start + args.nll_batch_size, len(recs))
        idx = np.arange(start, end)
        ids, mask = decoder_batch(module, traces[start:end], device, args.max_len, start_id)
        c = C[idx].to(device)
        z = Z[idx].to(device)
        c_p = C[partner[idx]].to(device)
        z_p = Z[partner[idx]].to(device)

        vals: dict[str, np.ndarray] = {}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for cond in conditions:
                if cond == "identity":
                    cz = torch.cat([c, z], dim=-1)
                elif cond == "z_other":
                    cz = torch.cat([c, z_p], dim=-1)
                elif cond == "c_other":
                    cz = torch.cat([c_p, z], dim=-1)
                elif cond == "both_other":
                    cz = torch.cat([c_p, z_p], dim=-1)
                else:
                    j = styles.index(cond.split(":", 1)[1])
                    cz = torch.cat([c, mus_t[j].to(device).expand(len(idx), -1)], dim=-1)
                vals[cond] = per_sample_nll(decoder, ids, mask, cz, args.logit_chunk).cpu().numpy()

        for n, i in enumerate(idx):
            row = {"i": int(i), "question_id": qids[i], "true_style": true_style[i]}
            row.update({cond: float(vals[cond][n]) for cond in conditions})
            rows.append(row)
        if start % (args.nll_batch_size * 10) == 0:
            done = end / len(recs)
            el = time.time() - t0
            print(f"  nll {end}/{len(recs)} ({100 * done:.0f}%) elapsed={el / 60:.1f}m", flush=True)

    df = pd.DataFrame(rows)

    # K x K matrix: rows = true cluster, cols = substituted mu
    mat = np.full((len(styles), len(styles)), np.nan)
    for i, s in enumerate(styles):
        sub = df[df["true_style"] == s]
        if sub.empty:
            continue
        for j, t in enumerate(styles):
            mat[i, j] = float(sub[f"mu:{t}"].mean())

    diag = np.array([mat[i, i] for i in range(len(styles))])
    off = np.array(
        [mat[i, j] for i in range(len(styles)) for j in range(len(styles)) if i != j]
    )
    argmin_is_diag = float(np.mean([int(np.nanargmin(mat[i]) == i) for i in range(len(styles))]))

    # per-trace paired swap cost, so the null can be a sign-flip permutation
    own = np.array([df.loc[n, f"mu:{df.loc[n, 'true_style']}"] for n in df.index])
    other = np.array(
        [
            np.mean([df.loc[n, f"mu:{t}"] for t in styles if t != df.loc[n, "true_style"]])
            for n in df.index
        ]
    )
    per_trace_cost = other - own
    perm = permutation_null(per_trace_cost, args.n_perm, args.seed)

    def m(col: str) -> float:
        return float(df[col].mean())

    return {
        "styles": styles,
        "n_traces": int(len(df)),
        "nll_matrix_true_by_substituted": mat.tolist(),
        "diagonal_mean_nll": float(np.nanmean(diag)),
        "offdiagonal_mean_nll": float(np.nanmean(off)),
        "swap_cost_nats": float(np.nanmean(off) - np.nanmean(diag)),
        "row_argmin_on_diagonal_rate": argmin_is_diag,
        "row_argmin_chance": 1.0 / len(styles),
        "per_trace_swap_cost_mean": float(per_trace_cost.mean()),
        "per_trace_swap_cost_sem": float(per_trace_cost.std(ddof=1) / np.sqrt(len(per_trace_cost))),
        "permutation_null": perm,
        "ablation_2x2": {
            "c_own_z_own": m("identity"),
            "c_own_z_other": m("z_other"),
            "c_other_z_own": m("c_other"),
            "c_other_z_other": m("both_other"),
            "cost_of_swapping_z": m("z_other") - m("identity"),
            "cost_of_swapping_c": m("c_other") - m("identity"),
            "c_over_z_cost_ratio": (
                (m("c_other") - m("identity")) / (m("z_other") - m("identity"))
                if abs(m("z_other") - m("identity")) > 1e-9
                else None
            ),
        },
        "per_trace_table": df.to_dict(orient="list"),
    }


def permutation_null(cost: np.ndarray, n_perm: int, seed: int) -> dict:
    """Sign-flip null for the paired swap cost being greater than zero."""
    rng = np.random.default_rng(seed)
    obs = float(cost.mean())
    draws = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(cost))
        draws[i] = float((cost * signs).mean())
    p = float((np.sum(np.abs(draws) >= abs(obs)) + 1) / (n_perm + 1))
    return {
        "n_perm": n_perm,
        "observed": obs,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "p_value": p,
    }


# --------------------------------------------------------------------------- #
# 3b: qualitative rendering
# --------------------------------------------------------------------------- #
def _gold_answer(trace: str) -> str:
    from scripts.math_answer_utils import parse_boxed_answer

    return parse_boxed_answer(trace or "")


def qualitative(module, recs, mus, styles, device, args) -> dict:
    start_id = decoder_start_id(module)
    mus_t = torch.as_tensor(mus, dtype=torch.float32)
    traces = [r["trace"] for r in recs]
    C, Z = encode(module, traces, device, args.max_len, args.encode_batch_size)

    samples = []
    for i, r in enumerate(recs):
        c = C[i : i + 1].to(device).expand(len(styles), -1)
        cz = torch.cat([c, mus_t.to(device)], dim=-1)
        texts = greedy_decode(module, cz, device, start_id, args.max_new_tokens)
        # identity decode for reference
        ident = greedy_decode(
            module,
            torch.cat([C[i : i + 1].to(device), Z[i : i + 1].to(device)], dim=-1),
            device,
            start_id,
            args.max_new_tokens,
        )[0]
        samples.append(
            {
                "question_id": r["question_id"],
                "true_style": r["_style"],
                "teacher_style_name": r.get("style_name"),
                "gold_answer": _gold_answer(r.get("trace") or ""),
                "teacher_trace": r.get("trace") or "",
                "identity_decode": ident,
                "decodes": {s: t for s, t in zip(styles, texts)},
            }
        )
        if (i + 1) % 5 == 0:
            print(f"  decoded {i + 1}/{len(recs)}", flush=True)

    # feature profile of the decodes, per substituted style
    prof: dict[str, dict[str, float]] = {}
    for s in styles:
        feats = [extract_features(smp["decodes"][s]) for smp in samples]
        prof[s] = {f: float(np.mean([x[f] for x in feats])) for f in PROFILE_FEATURES}
    return {
        "n_questions": len(samples),
        "max_new_tokens": int(args.max_new_tokens),
        "decode_profile": prof,
        "samples": samples,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="covz-qwen3-4b-final")
    ap.add_argument("--checkpoint", default=None, help="defaults to the tag's extract_meta.json")
    ap.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    ap.add_argument("--raw-json", default=DEFAULT_RAW_JSON)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-traces", type=int, default=900)
    ap.add_argument("--n-qualitative", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=0, help="0 = module hparams max_len")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--encode-batch-size", type=int, default=8)
    ap.add_argument("--nll-batch-size", type=int, default=4)
    ap.add_argument("--logit-chunk", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip-qualitative", action="store_true")
    ap.add_argument(
        "--skip-nll",
        action="store_true",
        help="generation-only restyle: skip the teacher-forced NLL matrix",
    )
    args = ap.parse_args()

    from scripts.discovery.clustering_gmm import enrich_records
    from scripts.eval.eval_gemini_styles import load_gemini_with_styles

    tag_dir = ZSCORE_ROOT / args.tag
    meta = json.loads((tag_dir / "extract_meta.json").read_text())
    checkpoint = args.checkpoint or meta["checkpoint"]
    out_dir = args.out_dir or (DEFAULT_OUT / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = load_cluster_labels(args.tag, args.k)
    styles = [f"style_{i + 1}" for i in range(args.k)]

    print("loading teacher traces", flush=True)
    records = enrich_records(
        load_gemini_with_styles(args.hf_dataset, args.raw_json, split="train")
    )
    key_to_style = {
        (str(q), str(s)): st
        for q, s, st in zip(labels_df["question_id"], labels_df["style_name"], labels_df["style"])
    }
    for r in records:
        r["_style"] = key_to_style.get((str(r["question_id"]), str(r.get("style_name"))))
    n_labelled = sum(1 for r in records if r["_style"] is not None)
    print(f"{n_labelled}/{len(records)} traces carry a GMM label", flush=True)

    # Cluster means come from the already-extracted z for this tag. The npz keeps the
    # `enrich_records` row order, so verify on question_id before trusting positions.
    zz = np.load(tag_dir / "embeddings_train_z.npz", allow_pickle=True)
    z_all = np.asarray(zz["z_s"], dtype=np.float32)
    z_q = zz["question_id"].astype(str)
    if len(z_all) != len(records):
        raise RuntimeError(
            f"z embeddings ({len(z_all)}) do not line up with records ({len(records)})"
        )
    rec_q = np.array([str(r["question_id"]) for r in records])
    if not np.array_equal(z_q, rec_q):
        raise RuntimeError("z embedding row order does not match the record order")
    row_style = np.array([r["_style"] for r in records], dtype=object)
    mus = np.stack([z_all[row_style == s].mean(axis=0) for s in styles], axis=0)
    print(f"z dim={mus.shape[1]} cluster sizes={[int((row_style == s).sum()) for s in styles]}", flush=True)

    module = load_module(checkpoint, device)
    max_len = args.max_len or int(module.hparams.get("max_len", 4096))
    args.max_len = max_len
    print(f"max_len={max_len}", flush=True)

    rng = np.random.default_rng(args.seed)
    if not args.skip_nll:
        per_cluster = max(1, args.n_traces // len(styles))
        pick = []
        for s in styles:
            cand = np.flatnonzero(row_style == s)
            pick.extend(rng.choice(cand, size=min(per_cluster, len(cand)), replace=False).tolist())
        pick = sorted(int(i) for i in pick)
        recs = [records[i] for i in pick]
        print(f"swap matrix on {len(recs)} traces", flush=True)

        result = {
            "tag": args.tag,
            "checkpoint": checkpoint,
            "k": args.k,
            "max_len": max_len,
            "swap": swap_matrix(module, recs, mus, styles, device, args),
        }
        (out_dir / "swap_matrix.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result["swap"].items() if k != "per_trace_table"}, indent=2)[:2000])
    else:
        print("skipping NLL swap matrix (--skip-nll)", flush=True)

    if not args.skip_qualitative:
        qpick = []
        seen_q = set()
        for i in rng.permutation(len(records)):
            r = records[int(i)]
            if r["_style"] is None or r["question_id"] in seen_q:
                continue
            seen_q.add(r["question_id"])
            qpick.append(int(i))
            if len(qpick) >= args.n_qualitative:
                break
        qrecs = [records[i] for i in qpick]
        print(f"qualitative decodes on {len(qrecs)} questions x {len(styles)} styles", flush=True)
        qual = qualitative(module, qrecs, mus, styles, device, args)
        qual.update(
            {
                "tag": args.tag,
                "checkpoint": checkpoint,
                "k": args.k,
                "max_len": max_len,
                "max_new_tokens": int(args.max_new_tokens),
                "seed": int(args.seed),
            }
        )
        (out_dir / "qualitative_decodes.json").write_text(json.dumps(qual, indent=2))
        print(
            f"wrote {out_dir / 'qualitative_decodes.json'} "
            f"({qual['n_questions']} questions, max_new_tokens={args.max_new_tokens})",
            flush=True,
        )

    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
