#!/usr/bin/env python3
"""Orchestrate chunked z-extract via subprocess-per-chunk (survives CUDA aborts)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.eval.eval_gemini_styles import load_gemini_with_styles  # noqa: E402


def shard_bounds(n: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return start, end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hf-dataset", required=True)
    p.add_argument("--raw-json", default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-rows", type=int, default=256, help="rows per subprocess chunk")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = args.output_dir
    shard_dir = out / "embedding_shards"
    partial_dir = shard_dir / f"partial_shard{args.shard_id:03d}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.npz"

    print("Loading records...", flush=True)
    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json, split="train"))
    start, end = shard_bounds(len(records), args.shard_id, args.num_shards)
    n = end - start
    print(f"Shard {args.shard_id}/{args.num_shards}: [{start}:{end}) n={n}", flush=True)

    py = sys.executable
    worker = str(REPO_ROOT / "scripts" / "extract_z_one_chunk.py")
    chunk_rows = args.chunk_rows
    n_chunks = (n + chunk_rows - 1) // chunk_rows

    for ci in range(n_chunks):
        out_npz = partial_dir / f"chunk_{ci:05d}.npz"
        if out_npz.exists():
            print(f"skip existing chunk {ci}", flush=True)
            continue
        r0 = start + ci * chunk_rows
        r1 = min(end, r0 + chunk_rows)
        print(f"=== chunk {ci}/{n_chunks}: rows [{r0}:{r1}) ===", flush=True)

        def run(bs: int) -> int:
            cmd = [
                py, "-u", worker,
                "--checkpoint", args.checkpoint,
                "--hf-dataset", args.hf_dataset,
                "--raw-json", args.raw_json or "",
                "--out-npz", str(out_npz),
                "--row-start", str(r0),
                "--row-end", str(r1),
                "--batch-size", str(bs),
                "--device", args.device,
            ]
            print("CMD", " ".join(cmd), flush=True)
            return subprocess.call(cmd)

        rc = run(args.batch_size)
        if rc != 0 or not out_npz.exists():
            print(f"chunk {ci} failed rc={rc}; retry batch_size=1", flush=True)
            out_npz.unlink(missing_ok=True)
            rc = run(1)
        if rc != 0 or not out_npz.exists():
            # last resort: split into single-row subprocesses
            print(f"chunk {ci} still failing; per-row fallback", flush=True)
            out_npz.unlink(missing_ok=True)
            row_parts = []
            for r in range(r0, r1):
                part = partial_dir / f"chunk_{ci:05d}_row_{r}.npz"
                if not part.exists():
                    cmd = [
                        py, "-u", worker,
                        "--checkpoint", args.checkpoint,
                        "--hf-dataset", args.hf_dataset,
                        "--raw-json", args.raw_json or "",
                        "--out-npz", str(part),
                        "--row-start", str(r),
                        "--row-end", str(r + 1),
                        "--batch-size", "1",
                        "--device", args.device,
                    ]
                    rc2 = subprocess.call(cmd)
                    if rc2 != 0 or not part.exists():
                        raise SystemExit(f"row {r} extract failed rc={rc2}")
                row_parts.append(part)
            cs, zs, qids, sids = [], [], [], []
            for part in row_parts:
                d = np.load(part, allow_pickle=True)
                cs.append(d["c_s"]); zs.append(d["z_s"]); qids.append(d["question_id"]); sids.append(d["style_id"])
            np.savez(
                out_npz,
                c_s=np.concatenate(cs),
                z_s=np.concatenate(zs),
                question_id=np.concatenate(qids),
                style_id=np.concatenate(sids),
                row_start=np.array(r0, dtype=np.int64),
                row_end=np.array(r1, dtype=np.int64),
            )
            for part in row_parts:
                part.unlink(missing_ok=True)
            print(f"Wrote {out_npz} via per-row fallback", flush=True)

    parts = sorted(partial_dir.glob("chunk_*.npz"))
    # exclude per-row leftovers
    parts = [p for p in parts if "_row_" not in p.name]
    cs, zs, qids, sids = [], [], [], []
    for part in parts:
        d = np.load(part, allow_pickle=True)
        cs.append(d["c_s"]); zs.append(d["z_s"]); qids.append(d["question_id"]); sids.append(d["style_id"])
    c_s = np.concatenate(cs); z_s = np.concatenate(zs)
    assert len(c_s) == n, (len(c_s), n)
    np.savez(
        shard_path,
        c_s=c_s, z_s=z_s,
        row_start=np.array(start, dtype=np.int64),
        row_end=np.array(end, dtype=np.int64),
        shard_id=np.array(args.shard_id, dtype=np.int64),
        num_shards=np.array(args.num_shards, dtype=np.int64),
        question_id=np.concatenate(qids),
        style_id=np.concatenate(sids),
    )
    print(f"Wrote {shard_path}", flush=True)
    if args.shard_id == 0:
        (out / "embedding_manifest.json").write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "hf_dataset": args.hf_dataset,
            "n_records": len(records),
            "num_shards": args.num_shards,
            "chunk_rows": args.chunk_rows,
            "extractor": "extract_z_embeddings_chunked.py/subprocess",
        }, indent=2))


if __name__ == "__main__":
    main()
