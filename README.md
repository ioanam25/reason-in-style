# Reason in Style

Code for **Reason in Style**: unsupervised discovery of reasoning
styles in math reasoning traces, and style-prefix / importance-sampled SFT
of Qwen3-Base students.

## What this code does

1. Train a content–style autoencoder on SCAS teacher writeups → latents `c(s)`, `z(s)`.
2. Fit a GMM with `K=6` on `z`.
3. Fine-tune Qwen3-Base students with hard `[style_i]` prefixes and/or IS weights.
4. Evaluate Pass@k on MATH-500 / AIME / AMC / HMMT / OlympiadBench.

Decode settings used for the reported tables:

| Setting | Value |
|---------|-------|
| samples | n=256 (or 256×K balanced across styles) |
| temperature | 0.6 |
| top_p | 0.95 |
| max_tokens | 4096 |

## Setup

```bash
git clone https://github.com/ioanam25/reason-in-style.git
cd reason-in-style
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: SFT trainer + vLLM eval stack
pip install -r requirements-novllm.txt
```

Paths (override for your cluster):

```bash
export REPO_ROOT=$PWD
export DATA_ROOT=/path/to/datasets      # HF caches, scas_traces_dataset, SFT parquet
export SCRATCH_ROOT=/path/to/scratch    # checkpoints, archives, logs
export ARCHIVE_ROOT=$SCRATCH_ROOT/archive
source scripts/env.sh
```

Set `HF_TOKEN` / `WANDB_API_KEY` in the environment. Edit `#SBATCH --partition` / `--qos`
(or export `PARTITION` / `QOS`) for your site.

## Data (Hugging Face)

Teacher pool:

**[Student-Centric-Answer-Sampling/scas_verified_teacher_pool](https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool)**

Nine teachers (gemma-4-31b-it, gpt-5-chat, gpt-oss-120b, llama-3.3-70b-instruct,
olmo-3.1-32b-instruct, phi-4-reasoning-plus, qwen2.5-72b-instruct, qwen3-32b, qwen3.5-27b).

### 0. Build AE traces + vanilla SFT parquet

```bash
# On-disk DatasetDict for autoencoder / z-extract / GMM
python scripts/data/build_scas_traces_from_hf.py \
  --output-dir "${DATA_ROOT}/scas_traces_dataset"

# Question-only SFT parquet (vanilla baseline); add --also-modc for teacher prefixes
python scripts/data/build_scas_standard_sft.py \
  --output-dir "${DATA_ROOT}/scas/standard_full_sft"
```

`question_id` in the traces dataset is `{source_dataset}::{id}` so IDs do not collide across sources.

## Reproduce the main pipe (ordered)

Full detail and artifact layout: [`docs/reproduce-pipeline.md`](docs/reproduce-pipeline.md).
AE hyperparams: [`docs/autoencoder-training.md`](docs/autoencoder-training.md).

```bash
# 1) AE covz (4B primary; 0.6B / 1.7B twins optional)
sbatch scripts/discovery/train-scas-trace-only-qwen3-4b-vicreg-b8-covz.slurm

# 2) Extract z  (set TAG to match your run, e.g. covz-qwen3-4b-final)
sbatch --export=ALL,TAG=covz-qwen3-4b-final scripts/discovery/extract-scas-z-8gpu.slurm

# 3) GMM K=6 → assignments under ${ARCHIVE_ROOT}/zscore-variants/${TAG}/
sbatch --export=ALL,TAG=covz-qwen3-4b-final scripts/discovery/fit-scas-gmm-sweep-assignments.slurm

# 4) Prefix SFT data + tokenize + train (orchestrator)
bash scripts/submit/submit-scas-ae-gmm-ladder.sh

# 5) IS-obs / rebal / global (example 4B)
bash scripts/submit/submit-is-prefix-4b.sh dry    # inspect
bash scripts/submit/submit-is-prefix-4b.sh submit

# 6) Pass@k — always set decode env
export TEMPERATURE=0.6 TOP_P=0.95 EOS_FIX=1 SAMPLES_PER_STYLE=256
# then use scripts/eval/*-passk*.slurm or scripts/submit/submit-*-eval.sh
```

**Note:** some raw eval SLURM files default to `T=1.0` / `top_p=1.0`. Reported numbers require the env above (submit helpers already set it).

## Key entrypoints

| Stage | Script |
|-------|--------|
| HF → AE traces | `scripts/data/build_scas_traces_from_hf.py` |
| HF → vanilla SFT | `scripts/data/build_scas_standard_sft.py` |
| AE train | `scripts/discovery/train-scas-trace-only-qwen3-*-vicreg-b8-covz.slurm` |
| z extract | `scripts/discovery/extract-scas-z-8gpu.slurm` |
| GMM | `scripts/discovery/fit_scas_gmm_sweep_assignments.py` |
| Tokenize SFT | `scripts/prepare_sft_hf_cache.py` |
| Student SFT | `mod_c/train.py` / `scripts/train/train-scas-prefix-arm-opt.slurm` |
| Pass@k | `scripts/eval/eval_math_cluster_prefix_qwen.py` |

## Layout

```
main.py                 # AE training CLI
src/                    # Lightning module, losses, encoder/decoder
mod_c/                  # SFT trainer (TRL)
configs/scas/           # SFT metadata
scripts/data/           # HF → traces / prefix / IS builders
scripts/discovery/      # AE, z extract, GMM
scripts/train/          # student SFT SLURM
scripts/eval/           # Pass@k
scripts/analysis/       # figure probes
scripts/submit/         # multi-job submitters
docs/  tests/
```

## License

MIT — see [`LICENSE`](LICENSE).
