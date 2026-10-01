import random

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from itertools import combinations
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models import DisentangledModel
from src.losses import (
    infonce_loss,
    orthogonal_decorrelation_loss,
    reconstruction_loss,
    chunked_reconstruction_loss,
    variance_regularization_loss,
    covariance_regularization_loss,
    length_decorrelation_loss,
    entropy_regularization_loss,
    within_group_diversity_loss,
)


def get_decoder_sequence_start_token_id(tokenizer) -> int | None:
    """First token for decoder-only LM sequences (teacher forcing or greedy decode).

    Prefer a real BOS when present; otherwise EOS (common for decoder-only LMs), then PAD.
    """
    if tokenizer is None:
        return None
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return int(tid)
    return None


def _gather_with_backprop(tensor: torch.Tensor) -> torch.Tensor:
    """all_gather embeddings across GPUs, preserving gradients on the local shard.

    The returned tensor has shape [world_size * N, D].  Gradients flow only
    through the local rank's slice (standard CLIP / SimCLR trick).
    Returns the input unchanged when not running distributed.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


class TraceDataset(Dataset):
    """
    Builds (trace_a, trace_b) positive pairs from same-question groups.

    If max_traces_per_question is set, each call to resample() picks a random
    subset of that many traces per question before forming C(k,2) pairs.
    Call resample() at the start of each epoch to get fresh pairs.
    """

    def __init__(self, data_list, max_traces_per_question: int | None = None, seed: int = 0):
        self._grouped: dict[str, list[dict]] = defaultdict(list)
        for item in data_list:
            self._grouped[item["question_id"]].append(item)
        self._max_tpq = max_traces_per_question
        self._rng = random.Random(seed)
        self.paired_data: list[tuple[dict, dict]] = []
        self.resample()

    def resample(self) -> None:
        self.paired_data = []
        for traces in self._grouped.values():
            if len(traces) < 2:
                continue
            if self._max_tpq and len(traces) > self._max_tpq:
                subset = self._rng.sample(traces, self._max_tpq)
            else:
                subset = traces
            self.paired_data.extend(combinations(subset, 2))

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        return self.paired_data[idx]
        
def collate_trace_pairs(batch, vocab_size=1000):
    # Legacy helper — returns raw trace strings (used by tests).
    traces = []
    for t1, t2 in batch:
        traces.extend([t1['trace'], t2['trace']])
    return traces

class TraceCollator:
    """Tokenizes traces inside DataLoader workers so CPU tokenization overlaps with GPU compute."""

    def __init__(
        self,
        encoder_tokenizer=None,
        decoder_tokenizer=None,
        max_len=2048,
        vocab_size=1000,
        decoder_start_token_id: int | None = None,
        training_mode: str = "trace",
        question_max_len: int = 256,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.decoder_start_token_id = decoder_start_token_id
        self.training_mode = training_mode
        self.question_max_len = question_max_len

    def _prepend_decoder_start(self, decoder_input_ids, decoder_attention_mask):
        if self.decoder_start_token_id is None:
            return decoder_input_ids, decoder_attention_mask
        sid = int(self.decoder_start_token_id)
        batch_size = decoder_input_ids.shape[0]
        start_col = torch.full((batch_size, 1), sid, dtype=decoder_input_ids.dtype)
        decoder_input_ids = torch.cat([start_col, decoder_input_ids], dim=1)
        decoder_attention_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=decoder_attention_mask.dtype), decoder_attention_mask],
            dim=1,
        )
        if decoder_input_ids.shape[1] > self.max_len:
            decoder_input_ids = decoder_input_ids[:, : self.max_len]
            decoder_attention_mask = decoder_attention_mask[:, : self.max_len]
        return decoder_input_ids, decoder_attention_mask

    def __call__(self, batch):
        if self.training_mode == "modc":
            return self._collate_modc(batch)

        traces = []
        for t1, t2 in batch:
            traces.extend([t1['trace'], t2['trace']])

        if self.encoder_tokenizer is not None and self.decoder_tokenizer is not None:
            # Dual-tokenization path: encoder ids/mask may differ from decoder ids/mask.
            enc = self.encoder_tokenizer(
                traces, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_len,
            )
            encoder_input_ids = enc["input_ids"]
            encoder_attention_mask = enc["attention_mask"]

            tok_max = self.max_len
            if self.decoder_start_token_id is not None:
                tok_max = max(1, self.max_len - 1)
            dec = self.decoder_tokenizer(
                traces, padding=True, truncation=True,
                return_tensors="pt", max_length=tok_max,
            )
            decoder_input_ids = dec["input_ids"]
            decoder_attention_mask = dec["attention_mask"]
            decoder_input_ids, decoder_attention_mask = self._prepend_decoder_start(
                decoder_input_ids, decoder_attention_mask
            )
            return {
                "encoder_input_ids": encoder_input_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            }

        # Back-compat single-tokenizer path (encoder + decoder share ids).
        tokenizer = self.decoder_tokenizer or self.encoder_tokenizer
        if tokenizer is not None:
            tok_max = self.max_len
            if self.decoder_start_token_id is not None:
                tok_max = max(1, self.max_len - 1)
            encoded = tokenizer(
                traces, padding=True, truncation=True,
                return_tensors="pt", max_length=tok_max,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if self.decoder_start_token_id is not None:
                sid = int(self.decoder_start_token_id)
                batch_size = input_ids.shape[0]
                start_col = torch.full((batch_size, 1), sid, dtype=input_ids.dtype)
                input_ids = torch.cat([start_col, input_ids], dim=1)
                attention_mask = torch.cat(
                    [torch.ones((batch_size, 1), dtype=attention_mask.dtype), attention_mask],
                    dim=1,
                )
                if input_ids.shape[1] > self.max_len:
                    input_ids = input_ids[:, : self.max_len]
                    attention_mask = attention_mask[:, : self.max_len]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        # Mock tokenization (no real tokenizer)
        token_rows = []
        cap = self.max_len - 1 if self.decoder_start_token_id is not None else self.max_len
        cap = max(1, cap)
        for trace in traces:
            token_values = [((ord(ch) % (self.vocab_size - 1)) + 1) for ch in trace][:cap]
            if not token_values:
                token_values = [1]
            if self.decoder_start_token_id is not None:
                token_values = [int(self.decoder_start_token_id)] + token_values
            if len(token_values) < self.max_len:
                token_values.extend([0] * (self.max_len - len(token_values)))
            else:
                token_values = token_values[: self.max_len]
            token_rows.append(token_values)
        input_ids = torch.tensor(token_rows, dtype=torch.long)
        attention_mask = input_ids.ne(0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _collate_modc(self, batch):
        traces = []
        questions = []
        decoder_texts = []
        for t1, t2 in batch:
            for item in (t1, t2):
                traces.append(item["trace"])
                questions.append(item["question"])
                decoder_texts.append(f"{item['question']}\n{item['trace']}")

        if self.encoder_tokenizer is None or self.decoder_tokenizer is None:
            raise RuntimeError("ModC training mode requires encoder and decoder tokenizers.")

        enc_trace = self.encoder_tokenizer(
            traces, padding=True, truncation=True, return_tensors="pt", max_length=self.max_len
        )
        enc_question = self.encoder_tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=min(self.question_max_len, self.max_len),
        )
        tok_max = self.max_len
        if self.decoder_start_token_id is not None:
            tok_max = max(1, self.max_len - 1)
        dec = self.decoder_tokenizer(
            decoder_texts, padding=True, truncation=True, return_tensors="pt", max_length=tok_max
        )
        decoder_input_ids = dec["input_ids"]
        decoder_attention_mask = dec["attention_mask"]
        decoder_input_ids, decoder_attention_mask = self._prepend_decoder_start(
            decoder_input_ids, decoder_attention_mask
        )
        out = {
            "encoder_trace_input_ids": enc_trace["input_ids"],
            "encoder_trace_attention_mask": enc_trace["attention_mask"],
            "encoder_question_input_ids": enc_question["input_ids"],
            "encoder_question_attention_mask": enc_question["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return out


class DisentangledDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        batch_size=2,
        encoder_tokenizer=None,
        decoder_tokenizer=None,
        max_len=2048,
        val_max_len=None,
        vocab_size=1000,
        num_workers=0,
        max_traces_per_question: int | None = None,
        decoder_start_token_id: int | None = None,
        training_mode: str = "trace",
        question_max_len: int = 256,
    ):
        super().__init__()
        self.max_traces_per_question = max_traces_per_question
        self.train_dataset = TraceDataset(train_data, max_traces_per_question=max_traces_per_question)
        self.val_dataset = TraceDataset(val_data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_collator = TraceCollator(
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            max_len=max_len,
            vocab_size=vocab_size,
            decoder_start_token_id=decoder_start_token_id,
            training_mode=training_mode,
            question_max_len=question_max_len,
        )
        self.val_collator = TraceCollator(
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            max_len=val_max_len if val_max_len is not None else max_len,
            vocab_size=vocab_size,
            decoder_start_token_id=decoder_start_token_id,
            training_mode=training_mode,
            question_max_len=question_max_len,
        )

    def on_train_epoch_start(self):
        if self.max_traces_per_question:
            self.train_dataset.resample()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

class DisentangledLightningModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size=None,
        lambda_c=1.0,
        lambda_z=1.0,
        lambda_rec=1.0,
        lambda_var=1.0,
        lambda_ent=0.0,
        lambda_rev=0.0,
        lambda_div=0.0,
        lambda_cov=0.0,
        lambda_len=0.0,
        vicreg_apply_to="both",
        training_mode="trace",
        question_max_len=256,
        max_len=4096,
        mock_encoder=False,
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        pooling="auto",
        lr=1e-4,
        encoder_lr=1e-6,
        lr_schedule="constant",
        projection_layers=4,
        d_c=256,
        d_z=256,
        decoder_embed_dim=512,
        decoder_layers=6,
        decoder_heads=8,
        num_prefix_tokens=8,
        decoder_type: str = "rope",
        decoder_gradient_checkpointing=False,
        decoder_model_name: str | None = None,
        decoder_freeze_lm: bool = True,
        decoder_lora_r: int = 0,
        decoder_lora_alpha: int = 16,
        decoder_lora_dropout: float = 0.05,
        decoder_lora_target_modules=None,
        freeze_encoder_epochs=2,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        gradient_checkpointing=True,
        train_record_count=None,
        val_record_count=None,
        n_z_heads=1,
        z_head_names=None,
        micro_batch_size=0,
        encoder_attn_implementation="sdpa",
        decoder_attn_implementation="sdpa",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DisentangledModel(
            vocab_size=vocab_size,
            max_len=max_len,
            mock_encoder=mock_encoder,
            model_name=model_name,
            pooling=pooling,
            d_c=d_c,
            d_z=d_z,
            projection_layers=projection_layers,
            decoder_embed_dim=decoder_embed_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            num_prefix_tokens=num_prefix_tokens,
            decoder_type=decoder_type,
            decoder_gradient_checkpointing=decoder_gradient_checkpointing,
            decoder_model_name=decoder_model_name,
            decoder_freeze_lm=decoder_freeze_lm,
            decoder_lora_r=decoder_lora_r,
            decoder_lora_alpha=decoder_lora_alpha,
            decoder_lora_dropout=decoder_lora_dropout,
            decoder_lora_target_modules=decoder_lora_target_modules,
            n_z_heads=n_z_heads,
            z_head_names=z_head_names,
            encoder_attn_implementation=encoder_attn_implementation,
            decoder_attn_implementation=decoder_attn_implementation,
        )
        self._encoder_frozen = False
        self._lora_applied = False

        if not mock_encoder and lora_r > 0:
            self.model.encoder.apply_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self._lora_applied = True

    def _freeze_encoder(self):
        """Freeze all encoder params (including LoRA adapters)."""
        if self._encoder_frozen:
            return
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        self._encoder_frozen = True
        if self.global_rank == 0:
            print("Encoder frozen (all params including LoRA)")

    def _unfreeze_encoder(self):
        """Unfreeze encoder. With LoRA, only adapter params become trainable;
        base weights stay frozen (handled by peft)."""
        if not self._encoder_frozen:
            return
        if self._lora_applied:
            from peft import PeftModel
            if isinstance(self.model.encoder.model, PeftModel):
                for name, p in self.model.encoder.model.named_parameters():
                    p.requires_grad = "lora_" in name or "lora_embedding" in name
            else:
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
        else:
            for p in self.model.encoder.parameters():
                p.requires_grad = True
        self._encoder_frozen = False
        trainable, total = self.model.encoder.lora_param_summary()
        if self.global_rank == 0:
            print(
                f"Encoder unfrozen: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M params trainable"
                f"{' (LoRA adapters only)' if self._lora_applied else ''}"
            )

    def _ensure_backbone_train_mode(self):
        """Force the HF encoder/decoder backbones into train mode.

        `from_pretrained` returns models in eval mode and Lightning does not
        always flip pretrained submodules back to train (see the "modules in
        eval mode" warning). The custom gte/Qwen forward only activates
        gradient checkpointing when `self.training` is True, so leaving the
        backbone in eval silently stores every layer's activations and OOMs at
        long sequence lengths. Restoring train mode here keeps checkpointing on.
        """
        enc = getattr(self.model, "encoder", None)
        enc_model = getattr(enc, "model", None) if enc is not None else None
        if enc_model is not None and not getattr(enc, "mock", False):
            enc_model.train()
        dec = getattr(self.model, "decoder", None)
        dec_lm = getattr(dec, "lm", None) if dec is not None else None
        if dec_lm is not None:
            dec_lm.train()

    def on_train_start(self):
        if self.hparams.gradient_checkpointing:
            self.model.encoder.enable_gradient_checkpointing()

        if self.hparams.freeze_encoder_epochs > 0:
            self._freeze_encoder()

        self._ensure_backbone_train_mode()

        train_batches = None
        val_batches = None
        if self.trainer is not None:
            train_batches = getattr(self.trainer, "num_training_batches", None)
            val_batches = getattr(self.trainer, "num_val_batches", None)
            if isinstance(val_batches, list):
                val_batches = sum(val_batches)

        summary_metrics = {
            "meta/train_record_count": float(self.hparams.train_record_count or 0),
            "meta/val_record_count": float(self.hparams.val_record_count or 0),
            "meta/train_pair_count": float(len(self.trainer.datamodule.train_dataset)) if self.trainer and self.trainer.datamodule else 0.0,
            "meta/val_pair_count": float(len(self.trainer.datamodule.val_dataset)) if self.trainer and self.trainer.datamodule else 0.0,
            "meta/train_batches": float(train_batches or 0),
            "meta/val_batches": float(val_batches or 0),
        }
        self.log_dict(summary_metrics, logger=True)

    def on_train_epoch_start(self):
        if self._encoder_frozen and self.current_epoch >= self.hparams.freeze_encoder_epochs:
            self._unfreeze_encoder()
        self._ensure_backbone_train_mode()

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids=None, decode=True):
        # Back-compat: older callers pass (input_ids, attention_mask).
        return self.model(
            encoder_input_ids,
            encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decode=decode,
        )

    def _encode_slice(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.model.encoder(encoder_input_ids, encoder_attention_mask)
        return self.model.c_head(h), self.model.z_head(h)

    def _encode_traces(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_input_ids.shape[0]
        micro_bs = int(self.hparams.micro_batch_size or 0)
        if micro_bs <= 0 or micro_bs >= batch_size:
            return self._encode_slice(encoder_input_ids, encoder_attention_mask)

        c_parts: list[torch.Tensor] = []
        z_parts: list[torch.Tensor] = []
        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            enc_ids = encoder_input_ids[start:end]
            enc_mask = encoder_attention_mask[start:end]
            if self.training:
                c_i, z_i = grad_checkpoint(
                    self._encode_slice,
                    enc_ids,
                    enc_mask,
                    use_reentrant=False,
                )
            else:
                c_i, z_i = self._encode_slice(enc_ids, enc_mask)
            c_parts.append(c_i)
            z_parts.append(z_i)
        return torch.cat(c_parts, dim=0), torch.cat(z_parts, dim=0)

    def _decoder_logits_fn(self):
        project_fn = getattr(self.model.decoder, "project_hidden", None)
        if project_fn is None:
            return None
        return project_fn

    def _decoder_slice_loss(
        self,
        c_s_slice: torch.Tensor,
        z_s_slice: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_z_concat = torch.cat([c_s_slice, z_s_slice], dim=-1)
        decoder_hidden = self.model.decoder.get_hidden(decoder_input_ids, c_z_concat)
        return chunked_reconstruction_loss(
            self.model.decoder.lm_head,
            decoder_hidden,
            decoder_input_ids,
            decoder_attention_mask,
            project_fn=self._decoder_logits_fn(),
        )

    def _decoder_reconstruction_loss(
        self,
        c_s: torch.Tensor,
        z_s: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder forward + chunked CE; optional micro-batching along batch dim."""
        batch_size = decoder_input_ids.shape[0]
        micro_bs = int(self.hparams.micro_batch_size or 0)
        if micro_bs <= 0 or micro_bs >= batch_size:
            c_z_concat = torch.cat([c_s, z_s], dim=-1)
            decoder_hidden = self.model.decoder.get_hidden(decoder_input_ids, c_z_concat)
            return chunked_reconstruction_loss(
                self.model.decoder.lm_head,
                decoder_hidden,
                decoder_input_ids,
                decoder_attention_mask,
                project_fn=self._decoder_logits_fn(),
            )

        total_loss = decoder_input_ids.new_zeros(())
        chunks = 0
        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            if self.training:
                chunk_loss = grad_checkpoint(
                    self._decoder_slice_loss,
                    c_s[start:end],
                    z_s[start:end],
                    decoder_input_ids[start:end],
                    decoder_attention_mask[start:end],
                    use_reentrant=False,
                )
            else:
                chunk_loss = self._decoder_slice_loss(
                    c_s[start:end],
                    z_s[start:end],
                    decoder_input_ids[start:end],
                    decoder_attention_mask[start:end],
                )
            total_loss = total_loss + chunk_loss
            chunks += 1
        return total_loss / chunks

    def _compute_losses(self, batch, *, sync_world_size: bool):
        if self.hparams.training_mode == "modc":
            eq_ids = batch["encoder_question_input_ids"].to(self.device)
            eq_mask = batch["encoder_question_attention_mask"].to(self.device)
            et_ids = batch["encoder_trace_input_ids"].to(self.device)
            et_mask = batch["encoder_trace_attention_mask"].to(self.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

            c_s, z_s, z_trace, decoder_hidden = self.model.forward_modc(
                eq_ids,
                eq_mask,
                et_ids,
                et_mask,
                decoder_input_ids=decoder_input_ids,
                decode=True,
            )
            z_for_div = z_trace
        else:
            if "encoder_input_ids" in batch:
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
            else:
                encoder_input_ids = batch["input_ids"].to(self.device)
                encoder_attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = encoder_input_ids
                decoder_attention_mask = encoder_attention_mask

            c_s, z_s = self._encode_traces(encoder_input_ids, encoder_attention_mask)
            z_trace = z_s
            z_for_div = z_s

        c_all = _gather_with_backprop(c_s)
        z_all = _gather_with_backprop(z_for_div)

        l_c = infonce_loss(c_all)
        l_z = orthogonal_decorrelation_loss(c_s, z_for_div)
        l_rec = self._decoder_reconstruction_loss(
            c_s, z_for_div, decoder_input_ids, decoder_attention_mask
        )
        # At d=256 the covariance penalty has 66x more off-diagonal terms than at d=32
        # and swamps the InfoNCE gradient, driving l_c above chance. "z" keeps the
        # VICReg pressure off the content head.
        if getattr(self.hparams, "vicreg_apply_to", "both") == "z":
            l_var = variance_regularization_loss(z_all)
            l_cov = covariance_regularization_loss(z_all)
        else:
            l_var = variance_regularization_loss(c_all) + variance_regularization_loss(z_all)
            l_cov = covariance_regularization_loss(c_all) + covariance_regularization_loss(z_all)
        l_ent = entropy_regularization_loss(z_all)
        l_rev = -infonce_loss(z_all)

        # Length nuisance-decorrelation (unsupervised): length = trace token count,
        # gathered in the same order as z_all so rows align across ranks.
        seq_len_local = decoder_attention_mask.sum(dim=1).float()
        len_all = _gather_with_backprop(seq_len_local)
        l_len = length_decorrelation_loss(z_all, len_all)

        n_z_heads = getattr(self.model, "n_z_heads", 1)
        if n_z_heads > 1 and hasattr(self.model.z_head, "d_per_head"):
            d_per = self.model.z_head.d_per_head
            l_div = z_all.new_zeros(())
            for hi in range(n_z_heads):
                z_sub = z_all[:, hi * d_per : (hi + 1) * d_per]
                l_div = l_div + within_group_diversity_loss(z_sub)
            l_div = l_div / n_z_heads
        else:
            l_div = within_group_diversity_loss(z_all)

        ws = float(self.trainer.world_size) if sync_world_size else 1.0
        l_total = (
            ws * self.hparams.lambda_c * l_c
            + self.hparams.lambda_z * l_z
            + self.hparams.lambda_rec * l_rec
            + ws * self.hparams.lambda_var * l_var
            + ws * self.hparams.lambda_cov * l_cov
            + ws * self.hparams.lambda_ent * l_ent
            + ws * self.hparams.lambda_rev * l_rev
            + ws * self.hparams.lambda_div * l_div
            + ws * self.hparams.lambda_len * l_len
        )
        batch_size = decoder_input_ids.shape[0]
        metrics = {
            "loss": l_total,
            "l_c": l_c,
            "l_z": l_z,
            "l_rec": l_rec,
            "l_var": l_var,
            "l_cov": l_cov,
            "l_ent": l_ent,
            "l_rev": l_rev,
            "l_div": l_div,
            "l_len": l_len,
            "batch_size": batch_size,
            "non_padding_tokens": decoder_input_ids.ne(0).sum().float(),
            "tokenized_seq_len_mean": decoder_attention_mask.sum(dim=1).float().mean(),
            "contrastive_n": float(c_all.shape[0]),
            "n_z_heads": n_z_heads,
            "z_all": z_all,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Mid-run validation can flip the pretrained backbones back to eval,
            # which disables the built-in gradient checkpointing. Restore it.
            self._ensure_backbone_train_mode()
        metrics = self._compute_losses(batch, sync_world_size=True)
        batch_size = metrics["batch_size"]
        for key in ("loss", "l_c", "l_z", "l_rec", "l_var", "l_cov", "l_ent", "l_rev", "l_div", "l_len"):
            self.log(
                f"train/{key}",
                metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=key == "loss",
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(
            "train/non_padding_tokens",
            metrics["non_padding_tokens"],
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/tokenized_seq_len_mean",
            metrics["tokenized_seq_len_mean"],
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/encoder_frozen",
            float(self._encoder_frozen),
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/contrastive_n",
            metrics["contrastive_n"],
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        n_z_heads = metrics["n_z_heads"]
        z_all = metrics["z_all"]
        if n_z_heads > 1 and hasattr(self.model.z_head, "d_per_head"):
            d_per = self.model.z_head.d_per_head
            for hi in range(n_z_heads):
                z_sub = z_all[:, hi * d_per : (hi + 1) * d_per]
                head_name = self.model.z_head.head_names[hi]
                self.log(
                    f"train/l_div_{head_name}",
                    within_group_diversity_loss(z_sub),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )
                self.log(
                    f"train/l_var_{head_name}",
                    variance_regularization_loss(z_sub),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        return metrics["loss"]

    def on_before_optimizer_step(self, optimizer):
        total_norm = torch.zeros((), device=self.device)
        parameter_count = 0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            total_norm = total_norm + parameter.grad.detach().data.norm(2).pow(2)
            parameter_count += parameter.numel()

        if parameter_count == 0:
            return

        grad_norm = total_norm.sqrt()
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, logger=True, batch_size=1, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        metrics = self._compute_losses(batch, sync_world_size=False)
        l_total = metrics["loss"]
        l_c = metrics["l_c"]
        l_z = metrics["l_z"]
        l_rec = metrics["l_rec"]
        l_var = metrics["l_var"]
        l_cov = metrics["l_cov"]
        l_ent = metrics["l_ent"]
        l_rev = metrics["l_rev"]
        l_div = metrics["l_div"]
        l_len = metrics["l_len"]
        batch_size = metrics["batch_size"]
        self.log("val/loss", l_total, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_c", l_c, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_z", l_z, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_rec", l_rec, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_var", l_var, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_cov", l_cov, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_ent", l_ent, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_rev", l_rev, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_div", l_div, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/l_len", l_len, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/non_padding_tokens", metrics["non_padding_tokens"], on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val/tokenized_seq_len_mean", metrics["tokenized_seq_len_mean"], on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self._collect_val_z(metrics.get("z_all"))

        return l_total

    # Cap on rows kept per validation epoch for the collapse monitor.
    VAL_Z_MAX_ROWS = 4096

    def on_validation_epoch_start(self):
        self._val_z_chunks = []

    def _collect_val_z(self, z_all):
        """Stash a bounded sample of z for the epoch-end collapse check."""
        if z_all is None:
            return
        chunks = getattr(self, "_val_z_chunks", None)
        if chunks is None:
            chunks = self._val_z_chunks = []
        if sum(c.shape[0] for c in chunks) >= self.VAL_Z_MAX_ROWS:
            return
        chunks.append(z_all.detach().float().cpu())

    def on_validation_epoch_end(self):
        """Log the effective rank of z.

        A healthy z spreads variance across many dimensions; a collapsed one
        puts nearly all of it in a handful, which teacher-agreement metrics do
        not penalise.
        """
        chunks = getattr(self, "_val_z_chunks", None)
        self._val_z_chunks = []
        if not chunks:
            return
        z = torch.cat(chunks, dim=0)[: self.VAL_Z_MAX_ROWS]
        if z.shape[0] < 2 or z.shape[1] < 1:
            return
        zc = (z - z.mean(dim=0, keepdim=True)).double()
        cov = (zc.T @ zc) / zc.shape[0]
        try:
            ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        except Exception:
            return
        total = float(ev.sum())
        if total <= 0.0:
            return
        eff_rank = (total ** 2) / float((ev ** 2).sum())
        self.log("val/z_eff_rank", eff_rank, on_step=False, on_epoch=True,
                 logger=True, batch_size=1, sync_dist=True)
        self.log("val/z_eff_rank_frac", eff_rank / z.shape[1], on_step=False,
                 on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        self.log("val/z_pc1_frac", float(ev.max()) / total, on_step=False,
                 on_epoch=True, logger=True, batch_size=1, sync_dist=True)

    def configure_optimizers(self):
        # IMPORTANT: the encoder can be frozen for the first N epochs and unfrozen later.
        # Lightning builds the optimizer once, so we must include encoder params up-front
        # (even if currently requires_grad=False) so they can be trained after unfreeze.
        #
        # When LoRA is applied, restrict the encoder param group to adapter params only.
        encoder_param_ids = {id(p) for p in self.model.encoder.parameters()}
        other_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]

        if self._lora_applied and hasattr(self.model.encoder, "model"):
            encoder_params = [
                p
                for name, p in self.model.encoder.model.named_parameters()
                if ("lora_" in name or "lora_embedding" in name)
            ]
        else:
            encoder_params = list(self.model.encoder.parameters())

        param_groups = [
            {"params": other_params, "lr": self.hparams.lr},
        ]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": self.hparams.encoder_lr})

        optimizer = torch.optim.AdamW(param_groups)

        if self.hparams.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs,
            )
            return [optimizer], [scheduler]
        return optimizer
