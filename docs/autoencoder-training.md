# SCAS autoencoder training

Details for the **trace-only** disentangling AE used to learn style latents \(z\) on SCAS multi-teacher traces. Code: [`main.py`](../main.py), [`src/losses.py`](../src/losses.py), [`src/module.py`](../src/module.py). Broader architecture notes: [`docs/training-and-architecture.md`](training-and-architecture.md).

---

## 1. Goal and mode

| Item | Value |
|------|--------|
| Mode | `--training-mode trace` |
| Input | Teacher solution traces (same question, different teachers) |
| Outputs | \(c(s)\): context / problem-sensitive; \(z(s)\): residual “style” latent |
| Downstream | GMM \(K{=}6\) on the 4B \(z\) → `[style_i]` prefix SFT |

**Trace-only:** both \(c\) and \(z\) are encoded from the **trace** text (not a separate question encoder path).

---

## 2. SCAS covz jobs

Same recipe at three decoder sizes. Submit:

```bash
sbatch scripts/discovery/train-scas-trace-only-qwen3-0p6b-vicreg-b8-covz.slurm
sbatch scripts/discovery/train-scas-trace-only-qwen3-1p7b-vicreg-b8-covz.slurm
sbatch scripts/discovery/train-scas-trace-only-qwen3-4b-vicreg-b8-covz.slurm
```

| Item | Value |
|------|--------|
| Data | `scas_traces_dataset` |
| Encoder | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` + LoRA \(r{=}8\) |
| Decoders | frozen `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`; LoRA \(r{=}16\), \(\alpha{=}32\) on `q/k/v/o` |
| Prefix / latents | 8 prefix tokens, \(d_c{=}d_z{=}256\), 4 projection layers |
| VICReg | `--vicreg-apply-to z`, \(\lambda_{\mathrm{cov}}{=}0.15\) |
| Max length / epochs | 4096, **4** epochs |
| Devices | 8 GPUs, `ddp_find_unused_parameters_true`, bf16-mixed |
| Batch | 1 pair/GPU, accum 2, micro-batch 1 → **16** pairs/step |
| Checkpoints | `${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-{0p6b,1p7b,4b}-vicreg-b8-covz` |
| Style labels | GMM \(K{=}6\) on the **4B** checkpoint |

---

## 3. Forward path (trace mode)

1. Tokenize each trace in the batch (pair → 2 sequences).
2. Encoder → pooled \(h\); heads → \(c(s)\), \(z(s)\).
3. Concat \([c; z]\) → project to **8 prefix embeddings** → prepend to decoder token embeddings.
4. Causal LM over the **same** trace; reconstruction = next-token CE (padding ignored).

Contrastive terms gather embeddings across GPUs (`_gather_with_backprop`) so InfoNCE sees the global batch.

---

## 4. Loss terms

Implemented in [`src/losses.py`](../src/losses.py); assembled in `LightningModule._compute_losses`.

| Log key | Symbol | Definition | Intended effect |
|---------|--------|------------|-----------------|
| `l_c` | \(L_c\) | InfoNCE on \(c\) with **adjacent same-question pairs** as positives | \(c\) clusters by problem |
| `l_z` | \(L_z\) | Mean squared cosine between \(c\) and \(z\) (same trace) | Decorrelate \(c\) and \(z\) |
| `l_rec` | \(L_{\mathrm{rec}}\) | Causal LM CE on trace tokens (chunked for memory) | Reconstruct the trace from prefixes |
| `l_var` | \(L_{\mathrm{var}}\) | VICReg variance hinge: \(\mathrm{mean}_d \mathrm{ReLU}(\gamma - \mathrm{std}_d)\), \(\gamma=1\). On \(c\) and \(z\), or on \(z\) only with `--vicreg-apply-to z` | Avoid dimensional collapse |
| `l_cov` | \(L_{\mathrm{cov}}\) | VICReg covariance: mean squared off-diagonal of the feature covariance. Same heads as \(L_{\mathrm{var}}\) | Stop a low-rank / 1-D \(z\) |
| `l_ent` | \(L_{\mathrm{ent}}\) | Negative mean entropy of softmaxed pairwise sims on \(z\) | Push similarity mass to be uniform (usually **off**) |
| `l_rev` | \(L_{\mathrm{rev}}\) | **Negative** InfoNCE on \(z\) (same pairing as \(L_c\)) | Push same-question \(z\) apart via reverse contrastive (usually **off**; still **logged**) |
| `l_div` | \(L_{\mathrm{div}}\) | Mean cosine similarity of same-question \(z\) pairs | Minimize → diversify \(z\) within a question |
| `l_len` | \(L_{\mathrm{len}}\) | Squared correlation of \(z\) with trace length | Decorrelate style from length (usually **off**) |

### 4.1 Total objective

With world size \(W{=}8\) and `sync_world_size=True` on train:

\[
\begin{aligned}
L =\;
& W\,\lambda_c\,L_c
+ \lambda_z\,L_z
+ \lambda_{\mathrm{rec}}\,L_{\mathrm{rec}} \\
&+ W\,\lambda_{\mathrm{var}}\,L_{\mathrm{var}}
+ W\,\lambda_{\mathrm{cov}}\,L_{\mathrm{cov}}
+ W\,\lambda_{\mathrm{ent}}\,L_{\mathrm{ent}}
+ W\,\lambda_{\mathrm{rev}}\,L_{\mathrm{rev}}
+ W\,\lambda_{\mathrm{div}}\,L_{\mathrm{div}}
+ W\,\lambda_{\mathrm{len}}\,L_{\mathrm{len}}
\end{aligned}
\]

Notes:

- \(L_z\) and \(L_{\mathrm{rec}}\) are **not** multiplied by \(W\).
- Validation uses the same formula with \(W=1\) for the scaled terms’ factor in code (`sync_world_size=False` → `ws=1`).
- Raw components (`l_*`) are always logged; a term with \(\lambda=0\) does **not** affect \(L\) but can still show large values (especially `l_rev`).

### 4.2 Lambdas (SCAS covz)

| Weight | CLI | SCAS covz | `main.py` default |
|--------|-----|---------|-------------------|
| \(\lambda_c\) | `--lambda-c` | **1.0** | 1.0 |
| \(\lambda_z\) | `--lambda-z` | **1.0** | 1.0 |
| \(\lambda_{\mathrm{rec}}\) | `--lambda-rec` | **1.0** | 1.0 |
| \(\lambda_{\mathrm{var}}\) | `--lambda-var` | **1.0** | 1.0 |
| \(\lambda_{\mathrm{cov}}\) | `--lambda-cov` | **0.15** on the covz runs (`--vicreg-apply-to z`) | 0.0 |
| \(\lambda_{\mathrm{div}}\) | `--lambda-div` | **1.0** | 1.0 |
| \(\lambda_{\mathrm{len}}\) | `--lambda-len` | **0.0** (unset) | 0.0 |
| \(\lambda_{\mathrm{rev}}\) | `--lambda-rev` | **0.0** (unset) | 0.0 |
| \(\lambda_{\mathrm{ent}}\) | `--lambda-ent` | **0.0** (unset) | 0.0 |

The covz autoencoders (`scripts/discovery/train-scas-trace-only-qwen3-*-vicreg-b8-covz.slurm`) train with \(\lambda_{\mathrm{cov}}=0.15\) and VICReg on \(z\) only. With those weights the objective is

\[
L \approx W L_c + L_z + L_{\mathrm{rec}} + W L_{\mathrm{var}} + 0.15\, W L_{\mathrm{cov}} + W L_{\mathrm{div}}
\]

### 4.3 How to read metrics (important)

- Prefer **`val/l_rec`**, **`val/l_var`**, **`val/l_c`**, **`val/l_div`**, **`val/l_cov`** over total `val/loss`.
- Total loss can be **negative** because \(L_{\mathrm{div}}\) is a mean cosine (can be \(<0\)). With \(W=8\), \(W\cdot L_{\mathrm{div}}\) often dominates the scalar.
- Large negative **`l_rev`** in W&B with \(\lambda_{\mathrm{rev}}=0\) is **expected logging**, not an active term.
- Falling **`l_var` → 0** means the latent is collapsing in variance. \(\lambda_{\mathrm{cov}}=0.15\) on \(z\) is there to keep that covariance from going rank-1.

---

## 5. Optimizer and schedule

From Lightning module / CLI (unless overridden in SLURM):

| Item | SCAS covz |
|------|-----------|
| Optimizer | AdamW |
| LR (heads / decoder LoRA) | \(10^{-4}\) (`main.py` default; the slurm does not override it) |
| Encoder LoRA LR | \(10^{-6}\) |
| Schedule | constant |
| Freeze encoder epochs | 0 (LoRA trainable from the start) |

---

## 6. Data batching

- `--batch-size` = number of **same-question trace pairs** per GPU.
- Collator expands each pair → **`2 × batch_size`** sequences.
- `--max-traces-per-question 0` on SCAS = use all available traces when forming pairs (no per-epoch subsample cap).
- Splits are by **`question_id`** (no question leakage into val).

---

## 7. Throughput / resource notes

The covz jobs are memory-conservative: batch 1, accum 2, micro-batch 1, max length 4096, gradient checkpointing on the encoder and the decoder. Strategy is `ddp_find_unused_parameters_true`.

---

## 8. Checkpoints and resume

- Lightning saves under `--checkpoint-dir` (`last.ckpt` plus epoch checkpoints). Resume only from that run's own `last.ckpt`.
- After training: extract \(z\) (`scripts/discovery/`), fit GMM \(K{=}6\) on the 4B latents, then build prefix / IS SFT (`scripts/data/`).

---

## 9. Related files

| Path | Role |
|------|------|
| `scripts/discovery/train-scas-trace-only-qwen3-*-vicreg-b8-covz.slurm` | SCAS covz launches |
| `scripts/scas_model_size.sh` | Size → decoder name and max length |
| `src/losses.py` | Loss implementations |
| `src/module.py` | `_compute_losses`, train/val steps |
| `src/models.py` | Encoder / heads / decoder |
| `docs/training-and-architecture.md` | Architecture and the same SCAS recipe |
