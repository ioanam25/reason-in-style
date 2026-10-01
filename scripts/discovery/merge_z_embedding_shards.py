#!/usr/bin/env python3
"""Merge parallel z(s) embedding shards into a single NPZ for clustering_gmm.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge z embedding shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Default: <output-dir>/embeddings_train_z.npz",
    )
    args = parser.parse_args()

    shard_dir = args.output_dir / "embedding_shards"
    shards = sorted(shard_dir.glob("shard_*_of_*.npz"))
    if not shards:
        shards = sorted(shard_dir.glob("embeddings_train_z.shard*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards in {shard_dir}")

    num_shards = int(np.load(shards[0])["num_shards"])
    if len(shards) != num_shards:
        raise RuntimeError(f"Expected {num_shards} shards, found {len(shards)} in {shard_dir}")

    c_parts, z_parts, qid_parts, sid_parts = [], [], [], []
    expected_start = 0
    for path in shards:
        data = np.load(path, allow_pickle=False)
        start = int(data["row_start"])
        end = int(data["row_end"])
        if start != expected_start:
            raise RuntimeError(f"Gap in shard rows: expected start {expected_start}, got {start} ({path})")
        expected_start = end
        c_parts.append(data["c_s"])
        z_parts.append(data["z_s"])
        qid_parts.append(data["question_id"])
        sid_parts.append(data["style_id"])

    c_s = np.concatenate(c_parts, axis=0)
    z_s = np.concatenate(z_parts, axis=0)
    question_id = np.concatenate(qid_parts, axis=0)
    style_id = np.concatenate(sid_parts, axis=0)

    out = args.output_npz or (args.output_dir / "embeddings_train_z.npz")
    np.savez(out, c_s=c_s, z_s=z_s, question_id=question_id, style_id=style_id)
    print(f"Merged {len(shards)} shards -> {out}  (n={len(z_s)})")

    manifest_path = args.output_dir / "embedding_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("n_records") not in (None, len(z_s)):
            raise RuntimeError(
                f"Manifest n_records={manifest.get('n_records')} != merged n={len(z_s)}"
            )


if __name__ == "__main__":
    main()
