# Reproduce the main pipeline

End-to-end map from the public
[SCAS verified teacher pool](https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool)
to Style-SFT / IS-obs Pass@k numbers.

Assume `source scripts/env.sh` so `DATA_ROOT`, `ARCHIVE_ROOT` (`SROOT`), and
`CKPT_ROOT` are set.

## Artifacts

| Stage | Default location |
|-------|------------------|
| AE traces | `${DATA_ROOT}/scas_traces_dataset` |
| Vanilla SFT parquet | `${DATA_ROOT}/scas/standard_full_sft` |
| AE checkpoint | `${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz/` |
| z + GMM | `${ARCHIVE_ROOT}/zscore-variants/covz-qwen3-4b-final/` |
| Prefix data | `${ARCHIVE_ROOT}/cluster_prefix/covz-qwen3-4b-final-gmm/k6/` |
| Student ckpt | `${ARCHIVE_ROOT}/cluster_ckpt/<ARM>-<SIZE>/k6-opt/` |
| Evals | `${ARCHIVE_ROOT}/model-evals/` |

Tag note: train jobs write `…-vicreg-b8-covz`; analysis / Style-SFT often label the
same run `covz-qwen3-4b-final`. Symlink or set `TAG` consistently.

## Step 0 — data from Hugging Face

```bash
python scripts/data/build_scas_traces_from_hf.py \
  --output-dir "${DATA_ROOT}/scas_traces_dataset"

python scripts/data/build_scas_standard_sft.py \
  --output-dir "${DATA_ROOT}/scas/standard_full_sft"
# optional oracle teacher prefixes:
#   … --also-modc --modc-output-dir "${DATA_ROOT}/scas/modc_prefix_full_sft"
```

Nine teachers: gemma-4-31b-it, gpt-5-chat_2025-10-03, gpt-oss-120b,
llama-3.3-70b-instruct, olmo-3.1-32b-instruct, phi-4-reasoning-plus,
qwen2.5-72b-instruct, qwen3-32b, qwen3.5-27b.

## Step 1 — autoencoder (covz)

```bash
sbatch scripts/discovery/train-scas-trace-only-qwen3-4b-vicreg-b8-covz.slurm
# optional scale twins:
# sbatch scripts/discovery/train-scas-trace-only-qwen3-0p6b-vicreg-b8-covz.slurm
# sbatch scripts/discovery/train-scas-trace-only-qwen3-1p7b-vicreg-b8-covz.slurm
```

Recipe (all three sizes): `--training-mode trace`, `lambda_cov=0.15`,
`--vicreg-apply-to z`, `d_c=d_z=256`, 8 prefix tokens, 4 epochs, batch 1 /
accum 2 / 8 GPUs, max length 4096, frozen Qwen3 decoder + LoRA r=16.
See [`autoencoder-training.md`](autoencoder-training.md).

## Step 2 — extract z

```bash
sbatch --export=ALL,TAG=covz-qwen3-4b-final \
  scripts/discovery/extract-scas-z-8gpu.slurm
```

## Step 3 — GMM K=6

```bash
sbatch --export=ALL,TAG=covz-qwen3-4b-final \
  scripts/discovery/fit-scas-gmm-sweep-assignments.slurm
```

Expect assignments under
`${ARCHIVE_ROOT}/zscore-variants/${TAG}/` (often `gmm_from_kmeans/assignments_gmm_k06.parquet`).

## Step 4 — Style-SFT dataset + train

```bash
# build [style_i] parquet, tokenize per student size, train
bash scripts/submit/submit-scas-ae-gmm-ladder.sh
```

Or manually:

```bash
sbatch --export=ALL,K=6,STYLES=covz-qwen3-4b-final \
  scripts/data/build-scas-ae-decoder-gmm-prefix.slurm
sbatch --export=ALL,SIZE=qwen3-0p6b-base,K=6,STYLES=covz-qwen3-4b-final \
  scripts/data/prepare-scas-ae-gmm-ladder-data.slurm
sbatch --export=ALL,SIZE=qwen3-0p6b-base,STYLE=covz-qwen3-4b-final,K=6 \
  scripts/train/train-scas-ae-gmm-ladder-opt.slurm
```

Tokenize cache helper: `python scripts/prepare_sft_hf_cache.py <parquet_dir> --tokenizer <student>`.

## Step 5 — IS arms (obs / rebal / global)

Build filtered / reweighted prefix copies from the same GMM labels, then:

```bash
bash scripts/submit/submit-is-prefix-4b.sh submit
# also: submit-is-prefix-1p7b.sh , submit_cluster_filtered_0p6b.sh
```

## Step 6 — Pass@k

Decode settings for reported tables (set explicitly; do not rely on raw SLURM defaults):

```bash
export TEMPERATURE=0.6 TOP_P=0.95 EOS_FIX=1
export SAMPLES_PER_STYLE=256   # or SAMPLES_TOTAL=256 for question-only
```

Prefer the submit wrappers (`scripts/submit/submit-is-prefix-*-eval.sh`,
`submit-aime-hmmt-0p6b.sh`, `submit-scas-aez-passk-evals.sh`) which set these.

Students: `Qwen/Qwen3-{0.6B,1.7B,4B}-Base` (and init-ladder chat /
instruct / thinking ablations).

## Canonical vs skip

**Use for reported tables:** covz AE → GMM K=6 → Empirical / random / vanilla /
IS-obs / IS-rebal / IS-global / nosft; Pass@k with T=0.6, top_p=0.95, EOS fix.

**Skip or label as ablation:** k-means-only and gradient-cluster arms.

## Analysis scripts

`scripts/analysis/` builds figure probes (latent swap, occupancy, teacher
identifiability). They expect finished eval dumps under `${ARCHIVE_ROOT}/model-evals/`.
