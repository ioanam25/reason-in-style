# Disentangling reasoning: data, architecture, training, losses

This document describes the SCAS trace autoencoder: how train/validation data are built, model shapes, what is frozen vs LoRA-tuned, batching, and how each loss term enters the objective (including \(\lambda\) weights and DDP scaling).

For CLI defaults, see `main.py`. The SCAS runs that produced the paper latents are `scripts/discovery/train-scas-trace-only-qwen3-*-vicreg-b8-covz.slurm`. The style labels used downstream are the **Qwen3-4B** covz checkpoint, GMM \(K{=}6\).

---

## 1. Dataset: train vs validation

### 1.1 Loading and splits (`main.py` → `resolve_datasets`)

Training reads `--data-path` (a Hugging Face `Dataset` on disk or a `DatasetDict`).

| Scenario | Train split | Validation split |
|----------|-------------|-------------------|
| **`DatasetDict` with `train` + `validation`/`val`** | `train` | `validation` if present, else `val` |
| **`DatasetDict` without val** | All records from preferred split | Held out via **question-level** split using `--validation-fraction` (default **0.1**) |
| **Flat `Dataset` (single table)** | Same as above: **grouped by `question_id`**, last fraction of **questions** (sorted) go to val | Complement |
| **`--val-data-path` set** | `train` from `--data-path` | From second path: prefers `validation` / `val` / `train` / `test` |

**Important:** Splits are by **`question_id`**, not by row. Every trace for a question stays in the same split, so validation never shares a question with training.

### 1.2 What is a training example? (`src/module.py` → `TraceDataset`)

- Records are expected to include at least **`question_id`** and **`trace`** (and often `answer`, etc.).
- The dataset builds **pairs of traces** \((t_a, t_b)\) that share the same **`question_id`** (positive pairs for contrastive losses).
- **Training** `TraceDataset`: if `--max-traces-per-question K` with `K > 0`, each epoch **re-samples** up to `K` traces per question, then forms all \(\binom{K}{2}\) pairs (`resample()` on `on_train_epoch_start`). This is **data augmentation** across epochs.
- **Validation** `TraceDataset`: **no** `max_traces_per_question`; it uses **all** traces per question (subject to having ≥2 traces to form pairs).

### 1.3 Batching in the DataLoader

- `--batch-size` is the number of **(trace_a, trace_b) pairs** per step per GPU.
- The collator stacks **both** traces in each pair → a batch of **`2 × batch_size`** tokenized sequences (shape `[2B, L]`).

---

## 2. Architecture: encoder, projections, decoder

### 2.1 High-level flow (`src/models.py` → `DisentangledModel.forward`)

1. **Encoder** maps tokenized trace → pooled vector **`h`** (one vector per sequence in the batch).
2. **`c_head`** maps **`h` → c(s)** (context-sensitive / “which problem”).
3. **`z_head`** maps **`h` → z(s)** (intended to carry reasoning style / variation not explained by `c`).
4. **Decoder** takes **concatenated** \([c(s); z(s)]\), projects to **`num_prefix_tokens`** prefix embeddings, prepends them to the token embeddings of a frozen pretrained LM, and trains LoRA on that LM. Reconstruction loss predicts the **same** trace tokens (shifted causal LM).

### 2.2 Encoder (`BaseModelEncoder`)

- **Default HF model:** `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (long-context text encoder).
- **Output dimension** `D_enc = model.config.hidden_size` (for this checkpoint, **1536** — see project plan / HF config; heads resize automatically).
- **Pooling** (`--pooling`):
  - `auto`: for GTE-Qwen2 models, uses **`last_token`** (last non-pad token); otherwise **`mean`**.
  - Can force `mean` or `last_token`.
- **LoRA** (optional, `lora_r > 0`): adapters on attention projections; default target modules **`q_proj`, `k_proj`, `v_proj`, `o_proj`** (`LORA_DEFAULTS` in `models.py`).
- **Gradient checkpointing** can be enabled on the encoder (`--gradient-checkpointing`) to save memory.

Compatibility: a small **monkey-patch** restores `DynamicCache.get_usable_length` for this model’s remote code vs newer `transformers`.

### 2.3 Projection heads (`ProjectionHead`)

- Separate heads **`c_head`** and **`z_head`**.
- **Input:** `D_enc`
- **Output:** **`d_c`** and **`d_z`** (CLI: `--d-c`, `--d-z`, default **256** each).
- **Depth:** `--projection-layers` **4** on the SCAS covz jobs.
- Structure: **LayerNorm** → alternating **Linear + GELU** → final **Linear** to `d_c` / `d_z`.

### 2.4 Decoder (SCAS: pretrained Qwen3 + LoRA)

SCAS does not train the scratch RoPE decoder. Each covz job freezes a Qwen3 LM and trains LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`.

| Quantity | SCAS covz |
|----------|-----------|
| `--decoder-type` | `pretrained` |
| Checkpoints | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` |
| LM weights | frozen (`--decoder-freeze-lm true`) |
| LoRA | \(r{=}16\), \(\alpha{=}32\), dropout 0.05 |
| Prefix | \(d_c + d_z = 512\), **8** prefix tokens |
| Downstream styles | **4B** covz run (`covz-qwen3-4b-final`), then GMM \(K{=}6\) |

---

## 3. Training: freeze schedule, optimizers, batch scale

### 3.1 What is frozen vs fine-tuned vs full training

| Component | SCAS covz |
|-----------|-----------|
| Encoder **base** weights | Frozen (LoRA only) |
| Encoder **LoRA** | Trainable from epoch 0 (`--freeze-encoder-epochs 0`); LR = `--encoder-lr` \(10^{-6}\) |
| `c_head`, `z_head` | Trainable; LR = `--lr` \(10^{-4}\) |
| Decoder LM | Frozen |
| Decoder LoRA | Trainable; LR = `--lr` \(10^{-4}\) |

### 3.2 Optimizer (`configure_optimizers`)

- **AdamW** with two groups when LoRA is active:
  - Non-encoder parameters: **`--lr`** (default `1e-4`).
  - Encoder trainable (LoRA) parameters: **`--encoder-lr`** (default `1e-6`).
- **`--lr-schedule`**: SCAS covz jobs leave the default **`constant`**.

### 3.3 Precision and scale

- Default **`--precision bf16-mixed`** in `main.py` / SLURM.
- **DDP:** SCAS uses 1 node × 8 GPUs, strategy `ddp_find_unused_parameters_true`. `world_size = 8`.

**Effective pairs per optimizer step (single job):**

\[
\text{pairs per step} = \text{batch\_size} \times \text{accumulate\_grad\_batches} \times \text{world\_size}
\]

SCAS covz: `batch_size=1`, `accumulate_grad_batches=2`, `micro_batch_size=1`, `world_size=8`, so \(1 \times 2 \times 8 = 16\) pairs per optimizer step.

**Token tensor per step:** each pair batch is **`2 × batch_size`** sequences per GPU before accumulation; `--max-len` is **4096**.

### 3.4 Duration

- **`--max-epochs`**: default **20** in `main.py`; SCAS covz jobs use **4**.
- Steps per epoch \(\approx\) `ceil(train_pairs / (batch_size × world_size))` with additional accumulation factor.

---

## 4. Losses and \(\lambda\) weights

All scalar losses below are combined into **`train/loss`** and **`val/loss`**. Logged names include `l_c`, `l_z`, `l_rec`, `l_var`, `l_cov`, `l_ent`, `l_rev`, `l_div`, `l_len`.

### 4.1 Definitions (conceptual)

| Symbol | Code | Role |
|--------|------|------|
| \( \mathcal{L}_c \) | `infonce_loss(c_all)` | InfoNCE on **c**: adjacent rows in `c_all` are positive pairs (same batch ordering as paired traces); encourages **same-question** traces to align in **c** space. |
| \( \mathcal{L}_z \) | `orthogonal_decorrelation_loss(c_s, z_s)` | Penalizes **cosine alignment** between **c** and **z** per trace (encourages disentangling). Computed on **local** batch only (not gathered). |
| \( \mathcal{L}_{rec} \) | `chunked_reconstruction_loss(...)` | Causal LM CE on trace tokens from decoder hidden states; padding masked with `ignore_index=-100`. |
| \( \mathcal{L}_{var} \) | `variance_regularization_loss` | VICReg variance floor. SCAS uses `--vicreg-apply-to z`, so this is on **z only**. |
| \( \mathcal{L}_{cov} \) | `covariance_regularization_loss` | Mean squared off-diagonal covariance, same heads as \( \mathcal{L}_{var} \). SCAS sets \(\lambda_{\mathrm{cov}}=0.15\). |
| \( \mathcal{L}_{ent} \) | `entropy_regularization_loss(z_all)` | Softmax over cosine similarities of z to other z; loss = **negative entropy** → **maximize entropy** (spread similarity mass). |
| \( \mathcal{L}_{rev} \) | `-infonce_loss(z_all)` | **Anti–InfoNCE** on z: pushes the standard InfoNCE objective **down** (discourages z from forming the same pairwise structure as c). |
| \( \mathcal{L}_{div} \) | `within_group_diversity_loss(z_all)` | Mean cosine similarity between **adjacent** z pairs (same-question pairs in batch layout); **minimized** → push same-question z apart. |

**Gathering:** For distributed training, `c_all` and `z_all` are `all_gather` stacks of shape `[world_size × N, D]` with gradients only through the local shard (standard CLIP/SimCLR trick). That increases the number of negatives for InfoNCE-style terms computed on the full stack.

### 4.2 \(\lambda\) hyperparameters (`main.py`)

| CLI flag | `main.py` default | SCAS covz |
|----------|----------------------|----------------|
| `--lambda-c` | **1.0** | **1.0** |
| `--lambda-z` | **1.0** | **1.0** |
| `--lambda-rec` | **1.0** | **1.0** |
| `--lambda-var` | **1.0** | **1.0** (on \(z\) only) |
| `--lambda-cov` | **0.0** | **0.15** |
| `--lambda-div` | **1.0** | **1.0** |
| `--lambda-len` | **0.0** | **0.0** |
| `--lambda-ent` | **0.0** | **0.0** |
| `--lambda-rev` | **0.0** | **0.0** |

### 4.3 Total loss formulas (training vs validation)

**Training** (`training_step`) — note the **`world_size` (`ws`)** multipliers so terms that use **gathered** tensors are **not** implicitly down-weighted by DDP gradient averaging:

\[
\mathcal{L}_{\text{train}}
= ws \cdot \lambda_c \mathcal{L}_c
+ \lambda_z \mathcal{L}_z
+ \lambda_{rec} \mathcal{L}_{rec}
+ ws \cdot \lambda_{var} \mathcal{L}_{var}
+ ws \cdot \lambda_{cov} \mathcal{L}_{cov}
+ ws \cdot \lambda_{ent} \mathcal{L}_{ent}
+ ws \cdot \lambda_{rev} \mathcal{L}_{rev}
+ ws \cdot \lambda_{div} \mathcal{L}_{div}
+ ws \cdot \lambda_{len} \mathcal{L}_{len}
\]

**Validation** (`validation_step`) — **no** extra `ws` factor (standard mean over GPUs for logging):

\[
\mathcal{L}_{\text{val}}
= \lambda_c \mathcal{L}_c
+ \lambda_z \mathcal{L}_z
+ \lambda_{rec} \mathcal{L}_{rec}
+ \lambda_{var} \mathcal{L}_{var}
+ \lambda_{cov} \mathcal{L}_{cov}
+ \lambda_{ent} \mathcal{L}_{ent}
+ \lambda_{rev} \mathcal{L}_{rev}
+ \lambda_{div} \mathcal{L}_{div}
+ \lambda_{len} \mathcal{L}_{len}
\]

---

## 5. Related scripts

- **`scripts/discovery/extract_z_embeddings.py`**: \(z\) from a covz checkpoint.
- **`scripts/discovery/clustering_gmm.py`**, **`fit_scas_gmm_sweep_assignments.py`**: GMM on \(z\) (paper: \(K{=}6\) on the 4B run).
- Loss and training details that match these jobs: [`autoencoder-training.md`](autoencoder-training.md).

---

## 6. Quick dimension cheat sheet

| Tensor / object | Shape / size |
|-----------------|---------------|
| `input_ids` (one batch) | `[2B, L]` with `L ≤ max_len` |
| Encoder pooled `h` | `[2B, D_enc]` (D_enc = **1536** for GTE-Qwen2-1.5B) |
| `c_s` | `[2B, d_c]` |
| `z_s` | `[2B, d_z]` |
| `c_z_concat` (decoder prefix input) | `[2B, d_c + d_z]` |
| Prefix embeddings | `[2B, num_prefix_tokens, decoder_embed_dim]` |
| Decoder hidden (for loss) | `[2B, L, decoder_embed_dim]` |

---

## 7. SCAS configuration (covz)

Jobs: `scripts/discovery/train-scas-trace-only-qwen3-{0p6b,1p7b,4b}-vicreg-b8-covz.slurm`. Data: `scas_traces_dataset` (question-level train/val split; all traces per question, `--max-traces-per-question 0`).

| Parameter | Value |
|-----------|-------|
| Encoder | `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, LoRA \(r{=}8\) on q/k/v/o, last-token pool |
| Decoder | frozen Qwen3-0.6B / 1.7B / 4B, LoRA \(r{=}16\), \(\alpha{=}32\) |
| Heads | 4 layers, \(d_c{=}d_z{=}256\), 8 prefix tokens |
| VICReg | `--vicreg-apply-to z`, \(\lambda_{\mathrm{cov}}{=}0.15\), \(\lambda_{\mathrm{var}}{=}1\) |
| Other \(\lambda\) | \(c,z,\mathrm{rec},\mathrm{div}{=}1\); \(\mathrm{len},\mathrm{ent},\mathrm{rev}{=}0\) |
| LR | \(10^{-4}\) heads and decoder LoRA, \(10^{-6}\) encoder LoRA, constant schedule |
| Batch | 1 pair/GPU, accum 2, micro-batch 1, 8 GPUs → 16 pairs/step |
| Length / epochs | max length 4096, 4 epochs, bf16-mixed, gradient checkpointing on |
| Downstream | GMM \(K{=}6\) on the **4B** \(z\); those labels are the Style-SFT style IDs |
