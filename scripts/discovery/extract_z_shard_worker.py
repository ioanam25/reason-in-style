#!/usr/bin/env python3
"""Orchestrate one shard via subprocess-per-chunk (survives CUDA hard aborts)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.eval.eval_gemini_styles import load_gemini_with_styles  # noqa: E402


def shard_range(n: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    return (n * shard_id) // num_shards, (n * (shard_id + 1)) // num_shards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hf-dataset", required=True)
    p.add_argument("--raw-json", default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--stagger-sec", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.stagger_sec > 0:
        print(f"stagger {args.stagger_sec:.0f}s shard {args.shard_id}", flush=True)
        time.sleep(args.stagger_sec)

    out = args.output_dir
    shard_dir = out / "embedding_shards"
    partial_dir = shard_dir / f"partial_shard{args.shard_id:03d}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.npz"
    if shard_path.exists():
        print(f"exists {shard_path}", flush=True)
        return

    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json, split="train"))
    start, end = shard_range(len(records), args.shard_id, args.num_shards)
    n = end - start
    print(f"Shard {args.shard_id}/{args.num_shards}: [{start}:{end}) n={n}", flush=True)

    py = sys.executable
    worker = str(REPO_ROOT / "scripts" / "extract_z_one_chunk.py")
    chunk = args.chunk_rows
    n_chunks = (n + chunk - 1) // chunk

    for ci in range(n_chunks):
        out_npz = partial_dir / f"chunk_{ci:05d}.npz"
        if out_npz.exists():
            print(f"skip chunk {ci}", flush=True)
            continue
        r0 = start + ci * chunk
        r1 = min(end, r0 + chunk)
        print(f"=== chunk {ci}/{n_chunks} rows [{r0}:{r1}) ===", flush=True)

        def run(bs: int) -> int:
            return subprocess.call([
                py, "-u", worker,
                "--checkpoint", args.checkpoint,
                "--hf-dataset", args.hf_dataset,
                "--raw-json", args.raw_json or "",
                "--out-npz", str(out_npz),
                "--row-start", str(r0),
                "--row-end", str(r1),
                "--batch-size", str(bs),
                "--device", args.device,
            ])

        rc = run(args.batch_size)
        if rc != 0 or not out_npz.exists():
            print(f"chunk {ci} failed rc={rc}; retry bs=1", flush=True)
            out_npz.unlink(missing_ok=True)
            rc = run(1)
        if rc != 0 or not out_npz.exists():
            # per-row last resort
            print(f"chunk {ci} per-row fallback", flush=True)
            out_npz.unlink(missing_ok=True)
            parts = []
            for r in range(r0, r1):
                part = partial_dir / f"chunk_{ci:05d}_row_{r}.npz"
                if not part.exists():
                    rc2 = subprocess.call([
                        py, "-u", worker,
                        "--checkpoint", args.checkpoint,
                        "--hf-dataset", args.hf_dataset,
                        "--raw-json", args.raw_json or "",
                        "--out-npz", str(part),
                        "--row-start", str(r),
                        "--row-end", str(r + 1),
                        "--batch-size", "1",
                        "--device", args.device,
                    ])
                    if rc2 != 0 or not part.exists():
                        raise SystemExit(f"row {r} failed rc={rc2}")
                parts.append(part)
            cs, zs, qids, sids = [], [], [], []
            for part in parts:
                d = np.load(part)
                cs.append(d["c_s"]); zs.append(d["z_s"]); qids.append(d["question_id"]); sids.append(d["style_id"])
            np.savez(out_npz, c_s=np.concatenate(cs), z_s=np.concatenate(zs),
                     question_id=np.concatenate(qids), style_id=np.concatenate(sids),
                     row_start=np.array(r0,dtype=np.int64), row_end=np.array(r1,dtype=np.int64))
            for part in parts:
                part.unlink(missing_ok=True)

    parts = sorted(p for p in partial_dir.glob("chunk_*.npz") if "_row_" not in p.name)
    cs = [np.load(p)["c_s"] for p in parts]
    zs = [np.load(p)["z_s"] for p in parts]
    qids = [np.load(p)["question_id"] for p in parts]
    sids = [np.load(p)["style_id"] for p in parts]
    assert sum(len(x) for x in cs) == n
    np.savez(
        shard_path,
        c_s=np.concatenate(cs), z_s=np.concatenate(zs),
        row_start=np.array(start, dtype=np.int64), row_end=np.array(end, dtype=np.int64),
        shard_id=np.array(args.shard_id, dtype=np.int64),
        num_shards=np.array(args.num_shards, dtype=np.int64),
        question_id=np.concatenate(qids), style_id=np.concatenate(sids),
    )
    print(f"Wrote {shard_path}", flush=True)
    if args.shard_id == 0:
        (out / "embedding_manifest.json").write_text(json.dumps({
            "checkpoint": args.checkpoint, "n_records": len(records),
            "num_shards": args.num_shards, "extractor": "subprocess-chunk+eager",
        }, indent=2))


if __name__ == "__main__":
    main()
