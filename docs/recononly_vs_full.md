# Recon-only vs Full-loss: Gemini Pretrained Decoder Experiment

## Setup

| Item | Value |
|------|-------|
| Dataset | Gemini style traces: 300 questions × 4 styles = 1,200 traces |
| Train / val split | 1,080 / 120 (by `question_id`, 10% val) |
| Encoder | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` + LoRA r=8 |
| Decoder | `HuggingFaceTB/SmolLM2-135M` (frozen) + LoRA r=16 + 8 prefix tokens |
| Trainable params | 7.6M / 1.7B total |
| Max length | 2048 |
| Epochs | 80 (resumed from earlier checkpoints) |
| Eval checkpoint | `last.ckpt` (epoch 79) for both runs |

**Recon-only losses:** `λ_rec=1`, all others = 0.

**Full-loss (optA):** `λ_c=λ_z=λ_rec=λ_var=λ_div=1`, `λ_rev=λ_ent=0`.

---

## Training outcomes

| Metric | Recon-only | Full-loss |
|--------|-----------|-----------|
| Final train loss (epoch 79) | 0.299 | 0.103 |
| Final val loss (epoch 79) | 2.816 | 4.548 |
| Best val checkpoint | epoch 4 | epoch 23 |

**Conclusion 1 — Perfect train reconstruction was not achieved.** Even after 80 epochs, train loss remained well above zero (0.30 recon-only, 0.10 full-loss). Val loss diverged strongly (2.82 and 4.55), and the best validation checkpoints occurred early (epochs 4 and 23), indicating overfitting or a train/val mismatch rather than convergence to a near-zero reconstruction objective.

---

## Disentanglement: cosine gaps

Higher gap = embeddings cluster more within the label group than between groups.

| Space | Label | Recon-only gap | Full-loss gap | Desired for c(s) | Desired for z(s) |
|-------|-------|----------------|---------------|------------------|------------------|
| c(s) | question_id | **0.3619** | **0.9340** | high | low |
| c(s) | style_id | 0.0929 | **-0.0024** | low | — |
| z(s) | style_id | 0.0955 | **1.1755** | — | high |
| z(s) | question_id | 0.3771 | **-0.3022** | — | low |

**Conclusion 2 — Full-loss cleanly separates content and style; recon-only does not.** Under full-loss, c(s) has a large question gap (0.93) and ~zero style gap (-0.002), while z(s) has a large style gap (1.18) and negative question gap (-0.30). Under recon-only, both c(s) and z(s) show moderate question gaps (~0.36–0.38) and small style gaps (~0.09–0.10), meaning neither latent is style-specific or question-specific.

---

## Disentanglement: K-Means (k=4) vs ground-truth style_id

| Space | Recon-only ARI | Recon-only NMI | Full-loss ARI | Full-loss NMI |
|-------|----------------|----------------|---------------|---------------|
| z(s) | 0.8466 | 0.8231 | **0.7667** | **0.7805** |
| c(s) | 0.9094 | 0.8735 | **-0.0024** | **0.0001** |

**Conclusion 3 — High z(s) style ARI under recon-only is misleading because c(s) also clusters styles.** Recon-only z(s) ARI is 0.85, but c(s) ARI is even higher (0.91). Full-loss z(s) ARI is 0.77 with c(s) ARI near chance (-0.002), which matches the intended design: style lives in z, content in c.

---

## Disentanglement: linear style probes (4 classes, chance = 25%)

| Space | Recon-only accuracy | Full-loss accuracy |
|-------|---------------------|-------------------|
| z(s) | 96.25% | **96.67%** |
| c(s) | **97.08%** | **23.33%** (chance) |

**Conclusion 4 — Only full-loss assigns style information primarily to z(s).** Both runs achieve ~97% style classification from z(s). Recon-only also reaches 97% from c(s), so style is duplicated across both spaces. Full-loss reduces c(s) style probe to chance (23%), while keeping z(s) at 97%.

---

## Reconstruction quality (greedy decode, 8 samples)

Qualitative summary from eval logs (not a numeric aggregate):

| Run | Pattern |
|-----|---------|
| Recon-only | Strong on some Algorithmic List traces (near-verbatim opens); failures on math-heavy and long Academic traces |
| Full-loss | Occasional strong openings (e.g. Pure Equation sample 1); frequent mid-trace drift |

**Conclusion 5 — Greedy decode does not show perfect reconstruction for either run.** Sample decodes are mixed: some step-list traces match well, others diverge immediately. This is consistent with non-zero train loss and should be validated with teacher-forced NLL on the full train set (see follow-up evals).

---

## Summary table

| Goal | Recon-only | Full-loss | Winner |
|------|-----------|-----------|--------|
| Train loss → 0 | No (0.30) | No (0.10, lower) | Full-loss (closer) |
| c(s) encodes question | Partial (gap 0.36) | Yes (gap 0.93) | Full-loss |
| z(s) encodes style | Partial (dup in c) | Yes (gap 1.18, probe 97%) | Full-loss |
| z(s) independent of question | No (gap 0.38) | Yes (gap -0.30) | Full-loss |
| Greedy reconstruction | Mixed | Mixed | Inconclusive |

---

## Takeaways

1. **Use full-loss (optA) for disentanglement on Gemini** — it is the only configuration that puts question identity in c(s) and style in z(s) by all metrics (gaps, ARI, probes).

2. **Recon-only is not a useful pretraining stage** for this task — it does not isolate style in z(s) and does not achieve better generation quality in the sampled eval.

3. **Reconstruction remains an open problem** — train loss did not reach 0; 8 prefix tokens + SmolLM2 + dual tokenizers may be insufficient. Best val checkpoints (epochs 4 and 23) may reconstruct better than `last.ckpt`; teacher-forced NLL on the train set is the right metric.

4. **Next evals:** train NLL, within-question swap decoding, cross-problem z transfer, gradient clustering vs style_id — see `scripts/eval_gemini_extended.py`. Submit both checkpoints in parallel (one GPU each):

```bash
bash scripts/submit-gemini-ext-parallel.sh
```

Embeddings are cached at `eval_results/*/embeddings_{ckpt}_1200rec.npz` and reused by both `eval_gemini_styles.py` and extended eval (no re-encoding on reruns).

---

## Artifacts

| Run | Checkpoint | Eval dir | W&B |
|-----|------------|----------|-----|
| Recon-only | `checkpoints/gemini-recon-smollm2-lora/last.ckpt` | `eval_results/gemini-recon-smollm2-lora/` | `recon-only-smollm2-lora16` |
| Full-loss | `checkpoints/gemini-full-smollm2-lora/last.ckpt` | `eval_results/gemini-full-smollm2-lora/` | `full-loss-smollm2-lora16` |

Eval script: `scripts/eval/eval_gemini_styles.py`  
Eval slurm log: `slurm_logs/eval-gemini-styles-23881629.out`
