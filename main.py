import argparse
from datetime import timedelta
from functools import partial
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, load_from_disk
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.module import DisentangledLightningModule, DisentangledDataModule, get_decoder_sequence_start_token_id


def parse_bool(value):
    if value is None or isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def build_smoke_test_data():
    return [
        {"question_id": "1", "question": "q1", "trace": "trace 1a", "answer": "1"},
        {"question_id": "1", "question": "q1", "trace": "trace 1b", "answer": "1"},
        {"question_id": "2", "question": "q2", "trace": "trace 2a", "answer": "2"},
        {"question_id": "2", "question": "q2", "trace": "trace 2b", "answer": "2"},
    ]


def dataset_to_records(dataset):
    return [dict(row) for row in dataset]


def load_records(path):
    loaded = load_from_disk(path)
    if isinstance(loaded, Dataset):
        return dataset_to_records(loaded)
    if isinstance(loaded, DatasetDict):
        return {split_name: dataset_to_records(split_dataset) for split_name, split_dataset in loaded.items()}
    raise TypeError(f"Unsupported dataset object loaded from {path}: {type(loaded)!r}")


def select_split(loaded, preferred_splits):
    if isinstance(loaded, list):
        return loaded

    for split_name in preferred_splits:
        if split_name in loaded:
            return loaded[split_name]

    available = ", ".join(sorted(loaded))
    raise ValueError(f"Requested split not found. Available splits: {available}")


def split_records_by_question(records, validation_fraction):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["question_id"]].append(record)

    question_ids = sorted(grouped)
    if len(question_ids) < 2 or validation_fraction <= 0:
        return records, records

    validation_groups = max(1, int(round(len(question_ids) * validation_fraction)))
    validation_groups = min(validation_groups, len(question_ids) - 1)

    validation_ids = set(question_ids[-validation_groups:])
    train_records = []
    validation_records = []
    for question_id in question_ids:
        target = validation_records if question_id in validation_ids else train_records
        target.extend(grouped[question_id])
    return train_records, validation_records


def resolve_datasets(args):
    if args.data_path is None:
        smoke_data = build_smoke_test_data()
        return smoke_data, smoke_data, "smoke-test"

    loaded_train = load_records(args.data_path)

    if args.val_data_path is not None:
        loaded_val = load_records(args.val_data_path)
        train_records = select_split(loaded_train, ["train"])
        validation_records = select_split(loaded_val, ["validation", "val", "train", "test"])
        return train_records, validation_records, "disk"

    if isinstance(loaded_train, dict):
        train_records = select_split(loaded_train, ["train", "validation", "val", "test"])
        if "validation" in loaded_train:
            validation_records = loaded_train["validation"]
        elif "val" in loaded_train:
            validation_records = loaded_train["val"]
        else:
            train_records, validation_records = split_records_by_question(train_records, args.validation_fraction)
        return train_records, validation_records, "disk"

    train_records, validation_records = split_records_by_question(loaded_train, args.validation_fraction)
    return train_records, validation_records, "disk"


def build_training_strategy(args):
    dist_timeout = timedelta(seconds=args.dist_timeout_seconds)

    if args.strategy == "fsdp":
        try:
            from pytorch_lightning.strategies import FSDPStrategy
        except ImportError:
            from lightning.pytorch.strategies import FSDPStrategy

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_num_params,
        )

        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=args.fsdp_sharding_strategy,
            state_dict_type=args.fsdp_state_dict_type,
            cpu_offload=args.fsdp_cpu_offload,
            use_orig_params=True,
            timeout=dist_timeout,
        )

    # For DDP variants, build an explicit strategy object so we can raise the
    # process-group collective timeout. The NCCL watchdog default (30 min) can
    # kill an otherwise-healthy multi-hour run on a transient fabric stall,
    # which the slower large-decoder jobs are most exposed to.
    if args.strategy in ("ddp", "ddp_find_unused_parameters_true"):
        try:
            from pytorch_lightning.strategies import DDPStrategy
        except ImportError:
            from lightning.pytorch.strategies import DDPStrategy

        return DDPStrategy(
            find_unused_parameters=(args.strategy == "ddp_find_unused_parameters_true"),
            timeout=dist_timeout,
        )

    return args.strategy

def main():
    parser = argparse.ArgumentParser(description="Train Disentangling Reasoning Pipeline")

    # --- Trainer / hardware ---
    parser.add_argument("--accelerator", type=str, default="cuda", help="Accelerator (cuda, mps, cpu)")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs per node (1 = single-GPU)")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes (multi-node DDP)")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy: auto, ddp, ddp_find_unused_parameters_true, fsdp, etc.")
    parser.add_argument("--dist-timeout-seconds", type=int, default=7200, help="Collective (NCCL) process-group timeout in seconds for ddp/fsdp strategies")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for training")
    parser.add_argument("--fsdp-min-num-params", type=int, default=10_000_000, help="Minimum parameter count for FSDP size-based auto wrapping")
    parser.add_argument("--fsdp-sharding-strategy", type=str, default="FULL_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"], help="FSDP sharding strategy when --strategy fsdp")
    parser.add_argument("--fsdp-state-dict-type", type=str, default="sharded", choices=["full", "sharded"], help="Checkpoint/state-dict type for FSDP")
    parser.add_argument("--fsdp-cpu-offload", type=parse_bool, default=False, help="Enable FSDP CPU offload")

    # --- Data ---
    parser.add_argument("--mock", type=parse_bool, default=None, help="Use mock encoder")
    parser.add_argument("--data-path", type=str, default="openthoughts_math_5k_dataset", help="Path to a saved Hugging Face dataset or dataset dict for training")
    parser.add_argument("--val-data-path", type=str, default=None, help="Optional path to a separate saved validation dataset")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction when splitting a single dataset by question_id")
    parser.add_argument("--max-traces-per-question", type=int, default=0, help="If >0, subsample this many traces per question before forming pairs (re-sampled each epoch for augmentation). 0 = use all traces.")

    # --- Training schedule ---
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size in question-pair units")
    parser.add_argument("--max-epochs", type=int, default=20, help="Number of training epochs when not using fast_dev_run")
    parser.add_argument("--fast-dev-run", type=parse_bool, default=None, help="Override fast_dev_run; defaults to true only in smoke-test mode")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for heads and decoder (AdamW)")
    parser.add_argument("--encoder-lr", type=float, default=1e-6, help="Learning rate for encoder params (used after unfreeze)")
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule: constant or cosine annealing")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Number of batches to accumulate before each optimizer step (effective batch = batch_size * accumulate)")
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Micro-batch size for decoder forward (0 = process full batch at once; 1 halves peak memory when batch_size pairs expand to 2 traces)")
    parser.add_argument("--gradient-clip-val", type=float, default=10.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--gradient-clip-algorithm", type=str, default="norm", choices=["norm", "value"], help="Gradient clipping algorithm")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers for parallel data loading")

    # --- Loss weights ---
    parser.add_argument("--lambda-c", type=float, default=1.0, help="Weight for InfoNCE contrastive loss on c(s)")
    parser.add_argument("--lambda-z", type=float, default=1.0, help="Weight for orthogonal decorrelation loss between c(s) and z(s)")
    parser.add_argument("--lambda-rec", type=float, default=1.0, help="Weight for reconstruction (causal LM) loss")
    parser.add_argument("--lambda-var", type=float, default=1.0, help="Weight for VICReg-style variance regularizer on c(s) and z(s)")
    parser.add_argument("--lambda-ent", type=float, default=0.0, help="Weight for similarity-entropy regularizer on z(s) (0 to disable)")
    parser.add_argument("--lambda-rev", type=float, default=0.0, help="Weight for reversed InfoNCE on z(s) — pushes same-question z(s) apart (0 to disable; causes collapse at scale)")
    parser.add_argument("--lambda-div", type=float, default=1.0, help="Weight for within-question diversity loss on z(s) — prevents z(s) collapse (0 to disable)")
    parser.add_argument("--lambda-cov", type=float, default=0.0, help="Weight for VICReg-style covariance regularizer on c(s) and z(s) (prevents low-rank/1-D collapse)")
    parser.add_argument("--vicreg-apply-to", type=str, default="both", choices=["both", "z"], help="Apply VICReg variance/covariance regularizers to both heads or to z(s) only")
    parser.add_argument("--lambda-len", type=float, default=0.0, help="Weight for length nuisance-decorrelation on z(s) — penalizes corr(z, trace length); unsupervised (0 to disable)")
    parser.add_argument(
        "--training-mode",
        type=str,
        default="trace",
        choices=["trace", "modc"],
        help="trace: c,z from trace encoder. modc: c from question, z from trace, decoder sees question+trace.",
    )
    parser.add_argument("--question-max-len", type=int, default=256, help="Max encoder tokens for questions in modc training mode")

    # --- Encoder ---
    parser.add_argument("--model-name", type=str, default="Alibaba-NLP/gte-Qwen2-1.5B-instruct", help="Encoder model name")
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "mean", "last_token"], help="Encoder pooling strategy (auto picks based on model)")
    parser.add_argument("--max-len", type=int, default=4096, help="Maximum tokenized trace length")
    parser.add_argument("--val-max-len", type=int, default=0, help="Validation max tokenized trace length (0 = use --max-len)")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="Freeze encoder for the first N epochs (0 to never freeze)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (0 to disable LoRA and train full encoder)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None, help="LoRA target modules (default: q_proj k_proj v_proj o_proj)")
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True, help="Enable gradient checkpointing on the encoder")
    parser.add_argument(
        "--encoder-attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation for the encoder (flash_attention_2 needs flash-attn package)",
    )
    parser.add_argument(
        "--decoder-attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation for the pretrained decoder",
    )

    # --- Projection heads ---
    parser.add_argument("--d-c", type=int, default=256, help="Dimension of c(s) context-sensitive head")
    parser.add_argument("--d-z", type=int, default=256, help="Dimension of z(s) context-insensitive head")
    parser.add_argument("--projection-layers", type=int, default=4, help="Number of linear layers in the c(s) and z(s) projection heads")

    # --- Multi-head z ---
    parser.add_argument("--n-z-heads", type=int, default=1, help="Number of z sub-heads (1 = standard single z head; >1 = structured decomposition)")
    parser.add_argument("--z-head-names", type=str, nargs="+", default=None, help="Names for z sub-heads (e.g. strategy verbosity metacognition)")

    # --- Decoder ---
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="rope",
        choices=["rope", "legacy", "pretrained"],
        help=(
            "Decoder implementation: 'rope' (default), 'legacy' (absolute pos embeddings; Gemini-era checkpoints), "
            "or 'pretrained' (HF causal LM + latent prefix)."
        ),
    )
    parser.add_argument(
        "--decoder-model-name",
        type=str,
        default=None,
        help="HF model name/path for decoder when --decoder-type=pretrained (e.g. 'gpt2').",
    )
    parser.add_argument(
        "--decoder-freeze-lm",
        type=parse_bool,
        default=True,
        help="When --decoder-type=pretrained, freeze the LM weights and train only the latent prefix projection.",
    )
    parser.add_argument("--decoder-lora-r", type=int, default=0, help="LoRA rank for pretrained decoder LM (0 disables).")
    parser.add_argument("--decoder-lora-alpha", type=int, default=16, help="LoRA alpha for pretrained decoder LM.")
    parser.add_argument("--decoder-lora-dropout", type=float, default=0.05, help="LoRA dropout for pretrained decoder LM.")
    parser.add_argument(
        "--decoder-lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="LoRA target modules for pretrained decoder LM (module name substrings).",
    )
    parser.add_argument("--decoder-embed-dim", type=int, default=512, help="Decoder transformer hidden dimension")
    parser.add_argument("--decoder-layers", type=int, default=6, help="Number of decoder transformer layers")
    parser.add_argument("--decoder-heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--num-prefix-tokens", type=int, default=8, help="Number of prefix tokens projected from [c;z]")
    parser.add_argument("--decoder-gradient-checkpointing", type=parse_bool, default=False, help="Enable gradient checkpointing across decoder layers")

    # --- Checkpointing ---
    parser.add_argument("--enable-checkpointing", type=parse_bool, default=None, help="Override checkpoint saving; defaults to true only for real-data non-fast-dev-run training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory where Lightning checkpoints are stored")
    parser.add_argument("--save-top-k", type=int, default=1, help="How many best checkpoints to keep when validation loss is monitored")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a Lightning checkpoint to resume training from")

    # --- Logging ---
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Trainer logging cadence in optimizer steps")
    parser.add_argument("--num-sanity-val-steps", type=int, default=2, help="How many validation batches to run before training (ignored when --validate-before-train is enabled)")
    parser.add_argument("--validate-before-train", type=parse_bool, default=None, help="Run a full validation epoch before training starts; defaults to true for real-data non-fast-dev-run training")
    parser.add_argument("--limit-val-batches", type=float, default=1.0, help="Fraction or count of validation batches to run each validation epoch")
    parser.add_argument("--wandb", type=parse_bool, default=None, help="Enable Weights & Biases logging; defaults to true only for real-data non-fast-dev-run training")
    parser.add_argument("--wandb-project", type=str, default="disentangling-reasoning", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional Weights & Biases run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional Weights & Biases entity/team")
    parser.add_argument("--wandb-save-dir", type=str, default="wandb", help="Directory for local Weights & Biases files")
    parser.add_argument("--wandb-offline", type=parse_bool, default=False, help="Run Weights & Biases in offline mode")
    parser.add_argument("--wandb-log-model", type=parse_bool, default=False, help="Upload model checkpoints to Weights & Biases when checkpointing is enabled")
    parser.add_argument("--wandb-watch", type=parse_bool, default=False, help="Watch model parameters and gradients in Weights & Biases")
    parser.add_argument("--wandb-watch-log", type=str, default="all", help="Weights & Biases watch mode: gradients, parameters, or all")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=50, help="Step frequency for Weights & Biases gradient/parameter watching")

    args = parser.parse_args()

    train_data, val_data, data_mode = resolve_datasets(args)
    use_mock_encoder = args.mock if args.mock is not None else data_mode == "smoke-test"
    fast_dev_run = args.fast_dev_run if args.fast_dev_run is not None else data_mode == "smoke-test"
    enable_checkpointing = (
        args.enable_checkpointing
        if args.enable_checkpointing is not None
        else data_mode == "disk" and not fast_dev_run
    )
    enable_wandb = (
        args.wandb
        if args.wandb is not None
        else data_mode == "disk" and not fast_dev_run
    )
    validate_before_train = (
        args.validate_before_train
        if args.validate_before_train is not None
        else data_mode == "disk" and not fast_dev_run and args.resume_from is None
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    callbacks = []
    logger = False
    if enable_checkpointing:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="disentangling-{epoch:02d}",
                monitor="val/loss",
                mode="min",
                save_top_k=args.save_top_k,
                save_last=True,
            )
        )
    if enable_wandb:
        wandb_save_dir = Path(args.wandb_save_dir)
        wandb_save_dir.mkdir(parents=True, exist_ok=True)
        try:
            wandb_resume = "allow" if args.resume_from else None
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                save_dir=str(wandb_save_dir),
                offline=args.wandb_offline,
                log_model=args.wandb_log_model,
                resume=wandb_resume,
            )
            logger.log_hyperparams(
                {
                    "accelerator": args.accelerator,
                    "devices": args.devices,
                    "num_nodes": args.num_nodes,
                    "strategy": args.strategy,
                    "precision": args.precision,
                    "data_mode": data_mode,
                    "train_records": len(train_data),
                    "val_records": len(val_data),
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "model_name": args.model_name,
                    "max_len": args.max_len,
                    "d_c": args.d_c,
                    "d_z": args.d_z,
                    "decoder_embed_dim": args.decoder_embed_dim,
                    "decoder_layers": args.decoder_layers,
                    "decoder_heads": args.decoder_heads,
                    "num_prefix_tokens": args.num_prefix_tokens,
                    "freeze_encoder_epochs": args.freeze_encoder_epochs,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "encoder_lr": args.encoder_lr,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "checkpointing": enable_checkpointing,
                    "wandb_watch": args.wandb_watch,
                    "wandb_watch_log": args.wandb_watch_log,
                    "wandb_watch_log_freq": args.wandb_watch_log_freq,
                    "log_every_n_steps": args.log_every_n_steps,
                    "validate_before_train": validate_before_train,
                    "gradient_clip_val": args.gradient_clip_val,
                    "gradient_clip_algorithm": args.gradient_clip_algorithm,
                }
            )
        except Exception as e:
            print(
                "WARNING: Weights & Biases logger failed to initialize "
                f"({type(e).__name__}: {e}). Falling back to CSVLogger."
            )
            logger = CSVLogger(save_dir=str(checkpoint_dir / "logs"), name="csv")

    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    max_tpq = args.max_traces_per_question if args.max_traces_per_question > 0 else None

    print(
        "Initializing Lightning Trainer "
        f"with accelerator={args.accelerator}, devices={args.devices}, "
        f"num_nodes={args.num_nodes}, strategy={args.strategy}, "
        f"precision={args.precision}, "
        f"mock_encoder={use_mock_encoder}, data_mode={data_mode}, "
        f"train_records={len(train_data)}, val_records={len(val_data)}, "
        f"max_traces_per_question={max_tpq}, "
        f"fast_dev_run={fast_dev_run}, enable_checkpointing={enable_checkpointing}, "
        f"enable_wandb={enable_wandb}, validate_before_train={validate_before_train}"
    )

    module = DisentangledLightningModule(
        mock_encoder=use_mock_encoder,
        model_name=args.model_name,
        pooling=args.pooling,
        max_len=args.max_len,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        lr_schedule=args.lr_schedule,
        d_c=args.d_c,
        d_z=args.d_z,
        projection_layers=args.projection_layers,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        num_prefix_tokens=args.num_prefix_tokens,
        decoder_type=args.decoder_type,
        decoder_gradient_checkpointing=args.decoder_gradient_checkpointing,
        decoder_model_name=args.decoder_model_name,
        decoder_freeze_lm=args.decoder_freeze_lm,
        decoder_lora_r=args.decoder_lora_r,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
        decoder_lora_target_modules=args.decoder_lora_target_modules,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        gradient_checkpointing=args.gradient_checkpointing,
        lambda_c=args.lambda_c,
        lambda_z=args.lambda_z,
        lambda_rec=args.lambda_rec,
        lambda_var=args.lambda_var,
        lambda_ent=args.lambda_ent,
        lambda_rev=args.lambda_rev,
        lambda_div=args.lambda_div,
        lambda_cov=args.lambda_cov,
        lambda_len=args.lambda_len,
        vicreg_apply_to=args.vicreg_apply_to,
        training_mode=args.training_mode,
        question_max_len=args.question_max_len,
        train_record_count=len(train_data),
        val_record_count=len(val_data),
        n_z_heads=args.n_z_heads,
        z_head_names=args.z_head_names,
        micro_batch_size=args.micro_batch_size,
        encoder_attn_implementation=args.encoder_attn_implementation,
        decoder_attn_implementation=args.decoder_attn_implementation,
    )
    decoder_start_id = None
    if args.decoder_type == "pretrained":
        decoder_start_id = get_decoder_sequence_start_token_id(
            getattr(module.model.decoder, "tokenizer", None)
        )
        if decoder_start_id is None and not use_mock_encoder:
            print(
                "WARNING: pretrained decoder has no bos/eos/pad token id; "
                "decoder start token will not be prepended.",
                flush=True,
            )
    datamodule = DisentangledDataModule(
        train_data, val_data,
        batch_size=args.batch_size,
        encoder_tokenizer=module.model.encoder.tokenizer,
        decoder_tokenizer=getattr(module.model.decoder, "tokenizer", None) or module.model.encoder.tokenizer,
        max_len=args.max_len,
        val_max_len=args.val_max_len if args.val_max_len > 0 else None,
        num_workers=args.num_workers,
        max_traces_per_question=max_tpq,
        decoder_start_token_id=decoder_start_id,
        training_mode=args.training_mode,
        question_max_len=args.question_max_len,
    )
    if enable_wandb and args.wandb_watch:
        logger.watch(module.model, log=args.wandb_watch_log, log_freq=args.wandb_watch_log_freq)

    strategy = build_training_strategy(args)

    # Lightning limitation: FSDPPrecision currently doesn't support norm-based gradient clipping.
    # Auto-adjust to keep long runs from crashing at the first optimizer step.
    gradient_clip_algorithm = args.gradient_clip_algorithm
    if args.strategy == "fsdp" and gradient_clip_algorithm == "norm" and (args.gradient_clip_val or 0.0) > 0:
        print(
            "WARNING: --strategy fsdp does not support gradient_clip_algorithm='norm' "
            "with FSDPPrecision. Switching to gradient_clip_algorithm='value'.",
            flush=True,
        )
        gradient_clip_algorithm = "value"

    sanity_val_steps = 0 if validate_before_train else args.num_sanity_val_steps
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        fast_dev_run=fast_dev_run,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=sanity_val_steps,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val or None,
        gradient_clip_algorithm=gradient_clip_algorithm,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        logger=logger
    )

    if validate_before_train:
        print("Running full validation before training...", flush=True)
        trainer.validate(module, datamodule=datamodule)

    trainer.fit(module, datamodule, ckpt_path=args.resume_from)
    if enable_checkpointing:
        checkpoint_callback = next(
            callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)
        )
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    if enable_wandb:
        run_name = logger.experiment.name or args.wandb_run_name or logger.version
        print(f"W&B run: {run_name}")
        print(f"W&B save dir: {logger.save_dir}")
    print("Training initialization and trace pass complete.")

if __name__ == "__main__":
    main()
