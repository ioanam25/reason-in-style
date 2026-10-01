#!/usr/bin/env python3
"""
Assign GMM cluster labels to the HF validation split for cluster-prefix SFT.

Do NOT fit a separate GMM on validation — that would define different clusters
than train. Instead:
  1. Refit the final train GMM (K + seed from cluster_meta.json) on cached train z.
  2. Extract z for HF validation rows with the trace-only checkpoint.
  3. predict() and map cluster ids using the train cluster_to_style mapping.

Prereq: clustering_gmm.py outputs in --assignments-dir:
  train.parquet, cluster_meta.json, embeddings_train_z.npz

Writes:
  validation.parquet
  embeddings_val_z.npz (cache)
  updates cluster_meta.json with val_summary

Usage:
  python scripts/discovery/assign_gmm_validation_split.py \\
      --assignments-dir data/scas/cluster_gmm_assignments-qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.assign_z_cluster_styles import build_assignment_df  # noqa: E402
from scripts.discovery.clustering_gmm import enrich_records, fit_gmm, summarize_assignments  # noqa: E402
from scripts.clustering_utils import l2_normalize  # noqa: E402
from scripts.eval.eval_gemini_styles import load_gemini_with_styles, load_or_extract_embeddings  # noqa: E402


def load_style_maps(meta: dict) -> tuple[dict[int, str], dict[int, int]]:
    name_map = {int(k): v for k, v in meta["cluster_to_style"].items()}
    id_map = {int(k): v for k, v in meta["style_id_map"].items()}
    return name_map, id_map


def assign_hf_validation_split(
    *,
    assignments_dir: Path,
    checkpoint: str | None = None,
    hf_dataset: str | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int | None = None,
    reextract: bool = False,
) -> pd.DataFrame:
    assignments_dir = Path(assignments_dir)
    meta_path = assignments_dir / "cluster_meta.json"
    train_emb_path = assignments_dir / "embeddings_train_z.npz"
    train_assign_path = assignments_dir / "train.parquet"
    val_out = assignments_dir / "validation.parquet"
    val_emb_path = assignments_dir / "embeddings_val_z.npz"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not train_emb_path.is_file():
        raise FileNotFoundError(f"Missing {train_emb_path}")
    if not train_assign_path.is_file():
        raise FileNotFoundError(f"Missing {train_assign_path}")

    meta = json.loads(meta_path.read_text())
    checkpoint = checkpoint or meta.get("checkpoint")
    if not checkpoint:
        manifest = assignments_dir / "embedding_manifest.json"
        if manifest.is_file():
            checkpoint = json.loads(manifest.read_text()).get("checkpoint")
    if not checkpoint:
        raise ValueError("checkpoint required (pass --checkpoint or set in cluster_meta.json)")

    hf_dataset = hf_dataset or meta.get("hf_dataset", "scas_traces_dataset")
    seed = int(seed if seed is not None else meta.get("seed", 42))
    selected_k = int(meta["selected_k"])
    name_map, id_map = load_style_maps(meta)

    print(f"Loading train z from {train_emb_path}")
    train_cached = np.load(train_emb_path, allow_pickle=False)
    z_train = l2_normalize(train_cached["z_s"])
    if len(z_train) != int(meta.get("n_autoencoder_train", len(z_train))):
        print(f"  warning: z_train rows={len(z_train)} meta n_autoencoder_train={meta.get('n_autoencoder_train')}")

    print(f"Refitting GMM on train z (K={selected_k}, seed={seed})...")
    gmm = fit_gmm(z_train, selected_k, seed)

    print(f"Loading HF validation split from {hf_dataset}...")
    val_records = load_gemini_with_styles(hf_dataset, meta.get("raw_json", ""), split="validation")
    val_records = enrich_records(val_records)
    print(f"  n_val_records={len(val_records)}")

    import torch
    from src.module import DisentangledLightningModule

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\nLoading checkpoint: {checkpoint}")
    module = DisentangledLightningModule.load_from_checkpoint(
        checkpoint, map_location="cpu", strict=False,
    )
    module.eval()
    max_len = module.hparams.get("max_len", 4096)

    if reextract and val_emb_path.exists():
        val_emb_path.unlink()

    print("\nExtracting z(s) for HF validation split...")
    _c_val, z_val = load_or_extract_embeddings(
        module,
        val_records,
        batch_size,
        torch_device,
        max_len,
        val_emb_path,
    )
    z_val = l2_normalize(z_val)

    print("Predicting validation cluster labels with train-fitted GMM...")
    val_labels = gmm.predict(np.asarray(z_val, dtype=np.float64))
    val_assign = build_assignment_df(val_records, val_labels, name_map, id_map)
    val_assign.to_parquet(val_out, index=False)
    print(f"Wrote {val_out}")

    meta["checkpoint"] = checkpoint
    meta["hf_dataset"] = hf_dataset
    meta["seed"] = seed
    meta["validation_assignment"] = "gmm_train_fit_predict_hf_validation"
    meta["val_summary"] = summarize_assignments(val_assign, "hf_validation")
    val_counts = val_assign["style_name"].value_counts()
    style_names = meta.get("style_names", sorted(val_assign["style_name"].unique()))
    meta["val_style_proportions"] = {
        sn: float(val_counts.get(sn, 0) / len(val_assign))
        for sn in style_names
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Updated {meta_path}")
    print("Val style counts:", meta["val_summary"]["style_counts"])
    return val_assign


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign train-fitted GMM labels to HF validation split",
    )
    parser.add_argument(
        "--assignments-dir",
        type=Path,
        required=True,
        help="GMM output dir (train.parquet, cluster_meta.json, embeddings_train_z.npz)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reextract-embeddings", action="store_true")
    args = parser.parse_args()

    assign_hf_validation_split(
        assignments_dir=args.assignments_dir,
        checkpoint=args.checkpoint,
        hf_dataset=args.hf_dataset,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        reextract=args.reextract_embeddings,
    )


if __name__ == "__main__":
    main()
