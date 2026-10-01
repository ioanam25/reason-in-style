# Scripts map

## Main path (start here)

| Stage | Path |
|-------|------|
| HF → AE traces | `data/build_scas_traces_from_hf.py` |
| HF → vanilla / ModC SFT | `data/build_scas_standard_sft.py` |
| AE covz train | `discovery/train-scas-trace-only-qwen3-*-vicreg-b8-covz.slurm` |
| Extract z | `discovery/extract-scas-z-8gpu.slurm` |
| GMM | `discovery/fit_scas_gmm_sweep_assignments.py` |
| Prefix build | `data/build-scas-ae-decoder-gmm-prefix.slurm` |
| Tokenize | `prepare_sft_hf_cache.py` + `data/prepare-scas-*.slurm` |
| Style / IS SFT | `train/train-scas-prefix-arm-opt.slurm`, `train/train-scas-ae-gmm-ladder-opt.slurm` |
| Orchestrators | `submit/submit-scas-ae-gmm-ladder.sh`, `submit/submit-is-prefix-*.sh` |
| Pass@k | `eval/eval_math_cluster_prefix_qwen.py`, `eval/eval_math_standard_passk_qwen.py` |

## Supporting

| Area | Path |
|------|------|
| Env / sizes | `env.sh`, `scas_model_size.sh` |
| IS data | `data/build_cluster_filtered_sft.py`, `data/build_is_prefix_parquet_copy.py` |
| Random / control prefixes | `data/build_scas_random_*`, `data/build_control_prefix_variants.py` |
| Figure probes | `analysis/` |

