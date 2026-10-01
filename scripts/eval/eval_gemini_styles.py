#!/usr/bin/env python3
"""
Evaluate a checkpoint on the Gemini dataset with ground truth style labels.

Computes:
  1. Reconstruction quality (loss + sample decodes from train set)
  2. Cosine similarity gaps (within-question, between-question, within-style, between-style)
  3. K-Means (k=4) on z(s) and c(s) -> ARI / NMI against ground truth style_id
  4. Linear probes for style_id on c(s) and z(s)
  5. UMAP colored by style_id and by question_id

Usage:
    python scripts/eval/eval_gemini_styles.py \
        --checkpoint checkpoints/gemini-recon-smollm2-lora/last.ckpt \
        --output-dir eval_results/gemini-recon-smollm2-lora \
        --device cuda
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# tqdm's monitor thread can abort the interpreter mid-run on this stack.
tqdm.monitor_interval = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.module import DisentangledLightningModule, get_decoder_sequence_start_token_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gemini_with_styles(
    hf_dataset_path: str = "gemini_style_traces_dataset",
    raw_json_path: str = "gemini_style_dataset.json",
    split: str = "all",
):
    """Load HF dataset and join with raw JSON to recover style_id / style_name."""
    hf_ds = load_from_disk(hf_dataset_path)
    if isinstance(hf_ds, DatasetDict):
        if split == "all":
            records = []
            for split_name in ("train", "validation", "val", "test"):
                if split_name in hf_ds:
                    records.extend(dict(row) for row in hf_ds[split_name])
        elif split in hf_ds:
            records = [dict(row) for row in hf_ds[split]]
        else:
            raise ValueError(f"Split {split!r} not in dataset. Available: {list(hf_ds.keys())}")
    else:
        records = [dict(row) for row in hf_ds]

    labeled = sum(int(r.get("style_id", -1)) >= 0 for r in records)
    if labeled == len(records) and len(records) > 0:
        print(f"  Using style labels from HF dataset ({len(records)} records)")
        return records

    raw = json.loads(Path(raw_json_path).read_text())
    lookup = {}
    for item in raw:
        qid = item.get("question_id")
        if qid is None:
            qid = f"gemini_{item['problem_id']}"
        key = (qid, item["raw_response"][:200])
        lookup[key] = {
            "style_id": item["style_id"],
            "style_name": item["style_name"],
        }

    matched, unmatched = 0, 0
    for r in records:
        key = (r["question_id"], r["trace"][:200])
        if key in lookup:
            r.update(lookup[key])
            matched += 1
        else:
            r["style_id"] = -1
            r["style_name"] = "unknown"
            unmatched += 1

    print(f"  Style label join: {matched} matched, {unmatched} unmatched out of {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _encoder_tokenizer(model):
    enc_tok = getattr(model.encoder, "tokenizer", None)
    dec_tok = getattr(model.decoder, "tokenizer", None)
    if enc_tok is not None and dec_tok is not None:
        return enc_tok
    return dec_tok or enc_tok


def extract_embeddings(module, records, batch_size, device, max_len=None):
    # No tqdm here: its monitor thread races the main thread into a CPython
    # refcount assertion ("object refcount : 2") that aborts the interpreter
    # a few seconds into extraction.
    model = module.model.to(device)
    tokenizer = _encoder_tokenizer(model)
    if max_len is None:
        max_len = model.max_len
    model.eval()

    starts = range(0, len(records), batch_size)
    total = len(starts)
    c_batches, z_batches = [], []
    for bi, start in enumerate(starts):
        batch = records[start : start + batch_size]
        traces = [r["trace"] for r in batch]
        enc = tokenizer(
            traces, padding=True, truncation=True, return_tensors="pt", max_length=max_len
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        if device.type == "cuda":
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                c_s, z_s, _ = model(input_ids, attention_mask, decode=False)
            torch.cuda.synchronize()
        else:
            with torch.no_grad():
                c_s, z_s, _ = model(input_ids, attention_mask, decode=False)
        c_batches.append(c_s.detach().float().cpu().numpy())
        z_batches.append(z_s.detach().float().cpu().numpy())
        del c_s, z_s, input_ids, attention_mask, enc
        if bi % 200 == 0 or bi == total - 1:
            print(f"  extract batch {bi + 1}/{total}", flush=True)

    return np.concatenate(c_batches), np.concatenate(z_batches)



def embeddings_cache_path(output_dir: Path, checkpoint: str, n_records: int) -> Path:
    ckpt_stem = Path(checkpoint).stem
    return output_dir / f"embeddings_{ckpt_stem}_{n_records}rec.npz"


def load_or_extract_embeddings(module, records, batch_size, device, max_len, cache_path: Path):
    if cache_path.exists():
        print(f"  Loading cached embeddings: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        if len(cached["c_s"]) == len(records):
            return cached["c_s"], cached["z_s"]
        print(f"  Cache size mismatch ({len(cached['c_s'])} vs {len(records)}), re-extracting")

    print(f"  Extracting embeddings (batch_size={batch_size})...")
    c_s, z_s = extract_embeddings(module, records, batch_size, device, max_len=max_len)
    np.savez(
        cache_path,
        c_s=c_s,
        z_s=z_s,
        question_id=np.array([r["question_id"] for r in records]),
        style_id=np.array([r.get("style_id", -1) for r in records]),
    )
    print(f"  Cached to {cache_path}")
    return c_s, z_s


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------

def _greedy_prefix_decode(module, c_z, device, max_new_tokens=256):
    model = module.model.to(device)
    model.eval()
    dec_tok = getattr(model.decoder, "tokenizer", None)
    if dec_tok is None:
        return None
    start_id = get_decoder_sequence_start_token_id(dec_tok)
    if start_id is None:
        return None
    generated = torch.tensor([[start_id]], device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens - 1):
            logits = model.decoder(generated, c_z)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == dec_tok.eos_token_id:
                break
            generated = torch.cat([generated, next_id], dim=1)
    return dec_tok.decode(generated[0, 1:], skip_special_tokens=True)


def evaluate_reconstruction(module, records, device, max_len, num_samples=8, max_decode_tokens=512):
    model = module.model.to(device)
    model.eval()
    enc_tok = _encoder_tokenizer(model)
    dec_tok = getattr(model.decoder, "tokenizer", None) or enc_tok

    results = []
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(records), size=min(num_samples, len(records)), replace=False)

    for idx in idxs:
        r = records[int(idx)]
        trace = r["trace"]

        enc = enc_tok(trace, padding=False, truncation=True, return_tensors="pt", max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
            c_s, z_s, _ = model(input_ids, attention_mask, decode=False)
            c_z = torch.cat([c_s, z_s], dim=-1)

        decoded = _greedy_prefix_decode(module, c_z, device, max_new_tokens=max_decode_tokens)

        results.append({
            "question_id": r.get("question_id", ""),
            "style_id": r.get("style_id", -1),
            "style_name": r.get("style_name", ""),
            "trace_preview": trace[:500],
            "decoded_preview": (decoded or "")[:500],
            "trace_len": len(trace),
            "decoded_len": len(decoded) if decoded else 0,
        })

        print(f"\n  Sample {len(results)}: qid={r.get('question_id','')} style={r.get('style_name','')}")
        print(f"    GT:      {trace[:200]}...")
        print(f"    Decoded: {(decoded or '[failed]')[:200]}...")

    return results


# ---------------------------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------------------------

def cluster_eval(embeddings, true_labels, space_name, k=4, n_init=20, seed=42):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    pred = km.fit_predict(embeddings)
    ari = adjusted_rand_score(true_labels, pred)
    nmi = normalized_mutual_info_score(true_labels, pred)
    print(f"  {space_name}: ARI={ari:.4f}  NMI={nmi:.4f}")
    return {"space": space_name, "k": k, "ari": float(ari), "nmi": float(nmi), "pred": pred.tolist()}


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_gap(embeddings, labels, label_name, max_pairs=100_000, seed=0):
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    emb = embeddings / norms

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    eligible = [idxs for idxs in groups.values() if len(idxs) >= 2]
    if not eligible:
        return {"label": label_name, "within": float("nan"), "between": float("nan"), "gap": float("nan")}

    within_sims = []
    for _ in range(max_pairs):
        idxs = eligible[int(rng.integers(0, len(eligible)))]
        a, b = rng.choice(idxs, size=2, replace=False)
        within_sims.append(float((emb[a] * emb[b]).sum()))

    labs = list(groups.keys())
    between_sims = []
    for _ in range(max_pairs):
        la, lb = rng.choice(labs, size=2, replace=False)
        a = rng.choice(groups[la])
        b = rng.choice(groups[lb])
        between_sims.append(float((emb[a] * emb[b]).sum()))

    w, b = np.mean(within_sims), np.mean(between_sims)
    return {"label": label_name, "within": float(w), "between": float(b), "gap": float(w - b)}


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def probe_style(embeddings, labels, space_name, sgd_epochs=20):
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"space": space_name, "accuracy": 0.0, "chance": 0.0}

    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    clf = SGDClassifier(loss="log_loss", max_iter=200, random_state=42, n_jobs=-1)
    classes = np.unique(y_tr)
    for _ in range(sgd_epochs):
        clf.partial_fit(X_tr, y_tr, classes=classes)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    chance = 1.0 / len(unique)
    print(f"  {space_name}: accuracy={acc:.4f}  chance={chance:.4f}")
    return {"space": space_name, "accuracy": float(acc), "chance": float(chance), "n_classes": int(len(unique))}


# ---------------------------------------------------------------------------
# UMAP visualizations
# ---------------------------------------------------------------------------

def plot_umaps(c_s, z_s, style_ids, style_names, question_ids, output_dir, seed=42):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plots.")
        return

    n_neighbors = max(2, min(15, len(c_s) - 1))

    print("  Fitting UMAP for c(s)...")
    emb_c = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(c_s)
    print("  Fitting UMAP for z(s)...")
    emb_z = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(z_s)

    unique_styles = sorted(set(style_names))
    style_to_int = {s: i for i, s in enumerate(unique_styles)}
    style_colors = np.array([style_to_int[s] for s in style_names])

    # --- Figure 1: colored by style ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc0 = axes[0].scatter(emb_c[:, 0], emb_c[:, 1], c=style_colors, cmap="tab10", s=14, alpha=0.7)
    axes[0].set_title("UMAP of c(s) colored by STYLE")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    sc1 = axes[1].scatter(emb_z[:, 0], emb_z[:, 1], c=style_colors, cmap="tab10", s=14, alpha=0.7)
    axes[1].set_title("UMAP of z(s) colored by STYLE")
    axes[1].set_xlabel("UMAP-1")

    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plt.cm.tab10(style_to_int[s] / max(len(unique_styles) - 1, 1)),
               markersize=8, label=s) for s in unique_styles]
    fig.legend(handles=handles, loc="lower center", ncol=len(unique_styles), fontsize=10)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = output_dir / "umap_by_style.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 2: colored by question_id ---
    from sklearn.preprocessing import LabelEncoder
    qid_enc = LabelEncoder().fit_transform(question_ids)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].scatter(emb_c[:, 0], emb_c[:, 1], c=qid_enc, cmap="tab20", s=14, alpha=0.5)
    axes[0].set_title("UMAP of c(s) colored by question_id")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    axes[1].scatter(emb_z[:, 0], emb_z[:, 1], c=qid_enc, cmap="tab20", s=14, alpha=0.5)
    axes[1].set_title("UMAP of z(s) colored by question_id")
    axes[1].set_xlabel("UMAP-1")

    plt.tight_layout()
    path = output_dir / "umap_by_question.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemini style evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf-dataset", type=str, default="gemini_style_traces_dataset")
    parser.add_argument("--raw-json", type=str, default="gemini_style_dataset.json")
    parser.add_argument(
        "--split",
        choices=["all", "train", "validation", "val", "test"],
        default="all",
        help="HF dataset split to evaluate (DatasetDict only)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on number of traces (random subsample)",
    )
    parser.add_argument("--output-dir", type=str, default="eval_results/gemini-style-eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-decode-samples", type=int, default=8)
    parser.add_argument("--max-decode-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help="Single K-Means k (0 = auto from number of unique style_id labels)",
    )
    parser.add_argument(
        "--cluster-ks",
        type=int,
        nargs="+",
        default=None,
        help="Run K-Means for each k (overrides --num-clusters), e.g. --cluster-ks 2 3 4 5",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=str,
        default=None,
        help="Path to .npz cache (default: output-dir/embeddings_{ckpt}_{n}rec.npz)",
    )
    parser.add_argument(
        "--reextract-embeddings",
        action="store_true",
        help="Ignore existing embedding cache and recompute",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data with style labels
    print("Loading Gemini dataset with style labels...")
    records = load_gemini_with_styles(args.hf_dataset, args.raw_json, split=args.split)
    if args.max_records > 0 and len(records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[int(i)] for i in pick]
        print(f"  Subsampled to {len(records)} records")
    style_ids = np.array([r["style_id"] for r in records])
    style_names = [r["style_name"] for r in records]
    question_ids = [r["question_id"] for r in records]

    n_with_style = (style_ids >= 0).sum()
    n_styles = len({s for s in style_ids if s >= 0})
    if args.cluster_ks:
        k_values = sorted(set(args.cluster_ks))
    elif args.num_clusters > 0:
        k_values = [args.num_clusters]
    else:
        k_values = [n_styles]
    print(f"  {len(records)} records, {n_with_style} with style labels, k={k_values}")
    for sid in sorted(set(style_ids)):
        if sid >= 0:
            name = next(r["style_name"] for r in records if r["style_id"] == sid)
            count = (style_ids == sid).sum()
            print(f"    style_id={sid} ({name}): {count} traces")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    module = DisentangledLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=False,
    )
    module.eval()

    saved_hparams = module.hparams
    max_len = saved_hparams.get("max_len", 2048)
    print(f"  max_len={max_len}, decoder_type={saved_hparams.get('decoder_type','?')}")

    results = {"checkpoint": args.checkpoint, "split": args.split, "max_records": args.max_records, "hparams": dict(saved_hparams)}

    # 1. Extract embeddings (cached)
    print("\n" + "=" * 70)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 70)
    cache_path = (
        Path(args.embeddings_cache)
        if args.embeddings_cache
        else embeddings_cache_path(output_dir, args.checkpoint, len(records))
    )
    if args.reextract_embeddings and cache_path.exists():
        cache_path.unlink()
        print(f"  Removed cache: {cache_path}")
    c_s, z_s = load_or_extract_embeddings(
        module, records, args.batch_size, device, max_len, cache_path,
    )
    print(f"  c(s) shape: {c_s.shape}, z(s) shape: {z_s.shape}")

    # 2. Reconstruction samples
    print("\n" + "=" * 70)
    print("RECONSTRUCTION QUALITY")
    print("=" * 70)
    decode_results = evaluate_reconstruction(
        module, records, device, max_len,
        num_samples=args.num_decode_samples,
        max_decode_tokens=args.max_decode_tokens,
    )
    results["reconstruction_samples"] = decode_results

    # 3. Cosine similarity gaps
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 70)

    valid_mask = style_ids >= 0
    c_valid = c_s[valid_mask]
    z_valid = z_s[valid_mask]
    sids_valid = style_ids[valid_mask]
    qids_valid = np.array(question_ids)[valid_mask]

    from sklearn.preprocessing import LabelEncoder
    qid_labels = LabelEncoder().fit_transform(qids_valid)

    print("\n  c(s) gaps:")
    c_qid_gap = cosine_gap(c_valid, qid_labels, "c_by_question")
    print(f"    by question_id: within={c_qid_gap['within']:.4f}  between={c_qid_gap['between']:.4f}  gap={c_qid_gap['gap']:.4f}")
    c_style_gap = cosine_gap(c_valid, sids_valid, "c_by_style")
    print(f"    by style_id:    within={c_style_gap['within']:.4f}  between={c_style_gap['between']:.4f}  gap={c_style_gap['gap']:.4f}")

    print("\n  z(s) gaps:")
    z_qid_gap = cosine_gap(z_valid, qid_labels, "z_by_question")
    print(f"    by question_id: within={z_qid_gap['within']:.4f}  between={z_qid_gap['between']:.4f}  gap={z_qid_gap['gap']:.4f}")
    z_style_gap = cosine_gap(z_valid, sids_valid, "z_by_style")
    print(f"    by style_id:    within={z_style_gap['within']:.4f}  between={z_style_gap['between']:.4f}  gap={z_style_gap['gap']:.4f}")

    results["cosine_gaps"] = {
        "c_by_question": c_qid_gap, "c_by_style": c_style_gap,
        "z_by_question": z_qid_gap, "z_by_style": z_style_gap,
    }

    # 4. K-Means clustering -> ARI / NMI (one or more k)
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING vs GROUND TRUTH STYLE")
    print("=" * 70)
    cluster_results = {"k_values": k_values, "z_style": {}, "c_style": {}}
    for k in k_values:
        print(f"\n  k={k}")
        cluster_results["z_style"][str(k)] = cluster_eval(
            z_valid, sids_valid, "z(s)", k=k, seed=args.seed,
        )
        cluster_results["c_style"][str(k)] = cluster_eval(
            c_valid, sids_valid, "c(s)", k=k, seed=args.seed,
        )
    results["clustering"] = cluster_results

    # 5. Linear probes for style_id
    print("\n" + "=" * 70)
    print("LINEAR PROBES: style_id classification")
    print("=" * 70)
    probe_results = {}
    probe_results["z_style"] = probe_style(z_valid, sids_valid, "z(s)")
    probe_results["c_style"] = probe_style(c_valid, sids_valid, "c(s)")
    results["probes"] = probe_results

    # 6. UMAP
    print("\n" + "=" * 70)
    print("UMAP VISUALIZATIONS")
    print("=" * 70)
    plot_umaps(
        c_valid, z_valid,
        sids_valid, [style_names[i] for i, m in enumerate(valid_mask) if m],
        qids_valid.tolist(), output_dir, seed=args.seed,
    )

    # Save results
    results_no_pred = {k: v for k, v in results.items()}
    for space in ("z_style", "c_style"):
        for k_str, entry in results_no_pred.get("clustering", {}).get(space, {}).items():
            if isinstance(entry, dict) and "pred" in entry:
                results_no_pred["clustering"][space][k_str] = {
                    kk: vv for kk, vv in entry.items() if kk != "pred"
                }

    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(results_no_pred, indent=2, default=str))
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'k':>3}  {'z ARI':>8}  {'z NMI':>8}  {'c ARI':>8}  {'c NMI':>8}")
    for k in k_values:
        zs = cluster_results["z_style"][str(k)]
        cs = cluster_results["c_style"][str(k)]
        print(
            f"  {k:3d}  {zs['ari']:8.4f}  {zs['nmi']:8.4f}  "
            f"{cs['ari']:8.4f}  {cs['nmi']:8.4f}"
        )
    print(f"  z(s) style probe:  {probe_results['z_style']['accuracy']:.4f}  "
          f"(chance={probe_results['z_style']['chance']:.4f})")
    print(f"  c(s) style probe:  {probe_results['c_style']['accuracy']:.4f}  "
          f"(chance={probe_results['c_style']['chance']:.4f})")
    print(f"  z(s) style gap:    {z_style_gap['gap']:.4f}")
    print(f"  c(s) question gap: {c_qid_gap['gap']:.4f}")


if __name__ == "__main__":
    main()
