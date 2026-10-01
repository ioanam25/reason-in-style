import fcntl
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# Patch DynamicCache for gte-Qwen2 custom code compatibility with transformers>=4.45
# The model's trust_remote_code modeling_qwen.py calls DynamicCache.get_usable_length()
# which was removed in recent transformers versions.
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
            if layer_idx is not None and layer_idx < len(self):
                return self[layer_idx][0].shape[-2]
            return 0
        DynamicCache.get_usable_length = _get_usable_length
except ImportError:
    pass


def _hf_hub_cache_dir() -> str:
    return os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
    )


def warm_trust_remote_code_modules(model_name: str) -> None:
    """Import trust_remote_code modules once; DDP ranks must not race this."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cache_dir = _hf_hub_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    lock_path = os.path.join(cache_dir, f".warm-lock-{model_name.replace('/', '--')}")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            auto_map = getattr(config, "auto_map", None) or {}
            for auto_cls in ("AutoTokenizer", "AutoModel", "AutoModelForCausalLM"):
                spec = auto_map.get(auto_cls)
                if spec:
                    get_class_from_dynamic_module(spec, model_name)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the hidden state at the last non-padding position for each row."""
    seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths]


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1)


def _flash_attn_runtime_available() -> bool:
    try:
        import flash_attn_2_cuda  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation != "flash_attention_2":
        return attn_implementation
    if not _flash_attn_runtime_available():
        print(
            "WARNING: flash_attn CUDA extension unavailable; falling back to sdpa.",
            flush=True,
        )
        return "sdpa"
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available at model init; using sdpa until training starts on GPU.",
            flush=True,
        )
        return "sdpa"
    return "flash_attention_2"


class BaseModelEncoder(nn.Module):
    POOLING_AUTO = {
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct": "last_token",
        "Alibaba-NLP/gte-Qwen2-7B-instruct": "last_token",
    }

    LORA_DEFAULTS = {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }

    def __init__(
        self,
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        mock=False,
        base_dim=768,
        pooling="auto",
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.mock = mock
        self.base_dim = base_dim
        self.tokenizer = None
        self.model_name = model_name
        self._lora_applied = False
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        if pooling == "auto":
            self.pooling = self.POOLING_AUTO.get(model_name, "mean")
        else:
            self.pooling = pooling

        if not self.mock:
            warm_trust_remote_code_modules(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            attn_impl = _resolve_attn_implementation(attn_implementation)
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )
            self.base_dim = self.model.config.hidden_size

    def apply_lora(self, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
        """Wrap the encoder with LoRA adapters.  Base weights are frozen;
        only the low-rank A/B matrices are trainable."""
        if self.mock or self._lora_applied:
            return
        from peft import LoraConfig, get_peft_model
        if target_modules is None:
            target_modules = self.LORA_DEFAULTS["target_modules"]
        cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, cfg)
        self._lora_applied = True

    def lora_param_summary(self) -> tuple[int, int]:
        """Return (trainable_params, total_params) after LoRA is applied."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def enable_gradient_checkpointing(self):
        if self.mock or not hasattr(self.model, "gradient_checkpointing_enable"):
            return
        # use_reentrant=False + enable_input_require_grads are REQUIRED for a
        # frozen (LoRA) base: otherwise checkpointing keeps all activations.
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def disable_gradient_checkpointing(self):
        if not self.mock and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask):
        if self.mock:
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, self.base_dim, device=self._device_anchor.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "last_token":
            return _last_token_pool(hidden_states, attention_mask)
        return _mean_pool(hidden_states, attention_mask)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        if num_layers < 1:
            raise ValueError("ProjectionHead needs at least 1 layer.")

        layers = [nn.LayerNorm(input_dim)]
        in_features = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.GELU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RotaryPositionEmbedding(nn.Module):
    """Precomputes and applies RoPE frequencies."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a (B, H, L, D) tensor. cos/sin are (L, D/2)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x.shape[-2], :d2].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :d2].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPECausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking, supporting a non-rotated prefix."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        rope_len = L - num_prefix
        cos, sin = self.rope(rope_len, device=x.device)

        if num_prefix > 0:
            q_prefix, q_seq = q[:, :, :num_prefix], q[:, :, num_prefix:]
            k_prefix, k_seq = k[:, :, :num_prefix], k[:, :, num_prefix:]
            q_seq = _apply_rope(q_seq, cos, sin)
            k_seq = _apply_rope(k_seq, cos, sin)
            q = torch.cat([q_prefix, q_seq], dim=2)
            k = torch.cat([k_prefix, k_seq], dim=2)
        else:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class RoPETransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPECausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, num_prefix=num_prefix)
        x = x + self.ff(self.norm2(x))
        return x


class AutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        prefix_dim,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        num_prefix_tokens=8,
        max_len=8192,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.gradient_checkpointing = gradient_checkpointing

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.prefix_proj = nn.Linear(prefix_dim, num_prefix_tokens * embed_dim)
        self.register_buffer("_cached_attn_mask", torch.empty(0), persistent=False)

        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def _get_attn_mask(self, total_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._cached_attn_mask
        needs_rebuild = (
            cached.numel() == 0
            or cached.shape != (total_len, total_len)
            or cached.device != device
            or cached.dtype != dtype
        )
        if needs_rebuild:
            mask = torch.triu(
                torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            if self.num_prefix_tokens > 0:
                mask[: self.num_prefix_tokens, : self.num_prefix_tokens] = 0.0
            self._cached_attn_mask = mask
        return self._cached_attn_mask

    def get_hidden(self, token_ids, c_z_concat):
        """Return decoder hidden states [B, L, embed_dim] without the lm_head projection."""
        B, L = token_ids.shape

        prefix_emb = self.prefix_proj(c_z_concat).view(B, self.num_prefix_tokens, self.embed_dim)
        tok_emb = self.token_emb(token_ids)
        combined = torch.cat([prefix_emb, tok_emb], dim=1)
        attn_mask = self._get_attn_mask(combined.shape[1], combined.device, combined.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                combined = _checkpoint(
                    lambda states, mask, block=layer: block(states, attn_mask=mask, num_prefix=self.num_prefix_tokens),
                    combined,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                combined = layer(combined, attn_mask=attn_mask, num_prefix=self.num_prefix_tokens)

        return self.final_norm(combined[:, self.num_prefix_tokens:, :])

    def forward(self, token_ids, c_z_concat):
        """Full forward returning logits [B, L, vocab_size]. Use get_hidden + chunked_lm_loss for training."""
        return self.lm_head(self.get_hidden(token_ids, c_z_concat))


class LegacyAutoregressiveDecoder(nn.Module):
    """
    Legacy absolute-position decoder used by older Gemini-era checkpoints.

    Key differences vs RoPE decoder:
    - learned absolute positional embeddings `pos_emb`
    - `nn.TransformerEncoder` blocks (matches legacy state_dict key names)
    """

    def __init__(
        self,
        vocab_size: int,
        prefix_dim: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_prefix_tokens: int = 4,
        max_len: int = 2048,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.max_len = max_len
        self.gradient_checkpointing = gradient_checkpointing

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        self.prefix_proj = nn.Linear(prefix_dim, num_prefix_tokens * embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def get_hidden(self, token_ids: torch.Tensor, c_z_concat: torch.Tensor) -> torch.Tensor:
        """Return decoder hidden states [B, L, embed_dim] without the lm_head projection."""
        B, L = token_ids.shape
        if L > self.max_len:
            raise ValueError(f"Legacy decoder max_len={self.max_len} but got L={L}")

        prefix_emb = self.prefix_proj(c_z_concat).view(B, self.num_prefix_tokens, self.embed_dim)

        tok_e = self.token_emb(token_ids)
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        pos_e = self.pos_emb(positions)
        seq_emb = tok_e + pos_e

        combined = torch.cat([prefix_emb, seq_emb], dim=1)  # [B, m+L, D]

        total_len = self.num_prefix_tokens + L
        mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=token_ids.device),
            diagonal=1,
        )
        mask[: self.num_prefix_tokens, : self.num_prefix_tokens] = 0.0

        hidden = self.transformer(combined, mask=mask, is_causal=False)
        return hidden[:, self.num_prefix_tokens :, :]

    def forward(self, token_ids: torch.Tensor, c_z_concat: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.get_hidden(token_ids, c_z_concat))


class PretrainedLatentDecoder(nn.Module):
    """Latent-prefix adapter over a pretrained causal LM decoder."""

    def __init__(
        self,
        model_name: str,
        prefix_dim: int,
        num_prefix_tokens: int,
        freeze_lm: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.model_name = model_name
        self.num_prefix_tokens = int(num_prefix_tokens)
        self._lora_applied = False
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be >= 0")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        # Many decoder-only LMs ship without a pad token; use eos for padding to keep masks consistent.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_impl = _resolve_attn_implementation(attn_implementation)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )

        self.embed_dim = int(getattr(self.lm.config, "hidden_size", self.lm.get_input_embeddings().embedding_dim))
        self.vocab_size = int(getattr(self.lm.config, "vocab_size", self.lm.get_output_embeddings().weight.shape[0]))
        # Keep compatibility with training code that expects `.lm_head(hidden)` -> logits.
        self.lm_head = self.lm.get_output_embeddings()

        # Training only uses hidden states (lm_head is re-applied chunked in the
        # reconstruction loss). Newer HF supports `logits_to_keep` to skip the
        # full-vocab logits projection over the whole sequence, which otherwise
        # materializes a [B, L, vocab] tensor we immediately discard (~2.4 GB for
        # a 3B model at L=4096). Detect support once so get_hidden can avoid it.
        import inspect

        try:
            base_forward = self.lm.get_base_model().forward if hasattr(self.lm, "get_base_model") else self.lm.forward
            forward_params = inspect.signature(base_forward).parameters
        except (TypeError, ValueError):
            forward_params = {}
        if "logits_to_keep" in forward_params:
            self._logits_to_keep_kwarg = "logits_to_keep"
        elif "num_logits_to_keep" in forward_params:
            self._logits_to_keep_kwarg = "num_logits_to_keep"
        else:
            self._logits_to_keep_kwarg = None

        self.prefix_proj = nn.Linear(prefix_dim, self.num_prefix_tokens * self.embed_dim)

        if lora_r and int(lora_r) > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=lora_target_modules,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, cfg)
            self._lora_applied = True

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            # If LoRA is enabled, keep adapter params trainable.
            if self._lora_applied:
                for name, p in self.lm.named_parameters():
                    p.requires_grad = "lora_" in name or "lora_embedding" in name

        # Memory saver: checkpointing trades compute for activation memory.
        # MUST run after the PEFT wrap + freeze. With a frozen base, gradient
        # checkpointing only frees activations when (a) use_reentrant=False and
        # (b) the embedding output is forced to require grad via
        # enable_input_require_grads. Without both, the reentrant checkpoint
        # silently retains every layer's activations and the 3B decoder OOMs at
        # L=4096 during the forward pass.
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing and hasattr(self.lm, "gradient_checkpointing_enable"):
            try:
                self.lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.lm.gradient_checkpointing_enable()
            if hasattr(self.lm, "enable_input_require_grads"):
                self.lm.enable_input_require_grads()

    def _language_model_backbone(self) -> nn.Module:
        """Transformer stack only (no lm_head). Avoids tied-embedding / FSDP issues."""
        lm = self.lm
        if hasattr(lm, "get_base_model"):
            lm = lm.get_base_model()
        if hasattr(lm, "model"):
            return lm.model
        raise AttributeError(f"Cannot locate transformer backbone on {type(lm)}")

    def _make_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        if self.num_prefix_tokens == 0:
            # If tokenizer has a pad token, mask it out; otherwise assume no padding.
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                return torch.ones((B, L), device=token_ids.device, dtype=torch.long)
            return (token_ids != pad_id).to(dtype=torch.long)

        prefix_mask = torch.ones((B, self.num_prefix_tokens), device=token_ids.device, dtype=torch.long)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            seq_mask = torch.ones((B, L), device=token_ids.device, dtype=torch.long)
        else:
            seq_mask = (token_ids != pad_id).to(dtype=torch.long)
        return torch.cat([prefix_mask, seq_mask], dim=1)

    def get_hidden(self, token_ids: torch.Tensor, c_z_concat: torch.Tensor) -> torch.Tensor:
        """Return decoder hidden states [B, L, embed_dim] (excluding prefix positions)."""
        B, L = token_ids.shape
        if self.num_prefix_tokens > 0:
            prefix_emb = self.prefix_proj(c_z_concat).view(B, self.num_prefix_tokens, self.embed_dim)
            seq_emb = self.lm.get_input_embeddings()(token_ids)
            inputs_embeds = torch.cat([prefix_emb, seq_emb], dim=1)
        else:
            inputs_embeds = self.lm.get_input_embeddings()(token_ids)

        attn_mask = self._make_attention_mask(token_ids)

        # Run the transformer backbone only: skip lm_head and avoid
        # output_hidden_states=True (which materializes every layer's activations).
        # lm_head is applied in chunked_reconstruction_loss during training.
        outputs = self._language_model_backbone()(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
        )
        last_hidden = outputs.last_hidden_state  # (B, prefix+L, D)
        return last_hidden[:, self.num_prefix_tokens :, :]

    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map decoder hidden states to logits; FSDP-safe for tied lm_head weights."""
        import torch.nn.functional as F
        from contextlib import ExitStack

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError:
            FSDP = None

        def _logits(h: torch.Tensor) -> torch.Tensor:
            lm = self.lm
            if hasattr(lm, "get_base_model"):
                lm = lm.get_base_model()
            weight = lm.get_output_embeddings().weight
            return F.linear(h, weight)

        if FSDP is None:
            return _logits(hidden)

        fsdp_mods = [m for m in self.modules() if isinstance(m, FSDP)]
        if not fsdp_mods:
            return _logits(hidden)

        # Gather every local FSDP shard (embed + lm_head are tied but wrapped separately).
        with ExitStack() as stack:
            for mod in fsdp_mods:
                stack.enter_context(
                    FSDP.summon_full_params(mod, writeback=False, recurse=False)
                )
            return _logits(hidden)

    def forward(self, token_ids: torch.Tensor, c_z_concat: torch.Tensor) -> torch.Tensor:
        """Return logits [B, L, vocab_size] aligned to token_ids positions."""
        B, L = token_ids.shape
        if self.num_prefix_tokens > 0:
            prefix_emb = self.prefix_proj(c_z_concat).view(B, self.num_prefix_tokens, self.embed_dim)
            seq_emb = self.lm.get_input_embeddings()(token_ids)
            inputs_embeds = torch.cat([prefix_emb, seq_emb], dim=1)
        else:
            inputs_embeds = self.lm.get_input_embeddings()(token_ids)

        attn_mask = self._make_attention_mask(token_ids)
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
        )
        return outputs.logits[:, self.num_prefix_tokens :, :]


class MultiHeadZ(nn.Module):
    """Structured z decomposition: multiple sub-heads with independent projections.

    Each sub-head produces a separate z sub-vector. They are concatenated
    to form the full z(s) used by the decoder, but losses can target
    individual axes for interpretable disentanglement.

    When n_z_heads=1 (default), this is equivalent to a single ProjectionHead.
    """

    def __init__(
        self,
        input_dim: int,
        d_z: int,
        n_heads: int = 1,
        num_layers: int = 2,
        z_head_names: list[str] | None = None,
    ):
        super().__init__()
        if n_heads < 1:
            raise ValueError("n_z_heads must be >= 1")
        if d_z % n_heads != 0:
            raise ValueError(f"d_z ({d_z}) must be divisible by n_z_heads ({n_heads})")

        self.n_heads = n_heads
        self.d_per_head = d_z // n_heads
        self.d_z = d_z

        if z_head_names is not None and len(z_head_names) != n_heads:
            raise ValueError(f"z_head_names length ({len(z_head_names)}) != n_z_heads ({n_heads})")
        self.head_names = z_head_names or [f"z_{i}" for i in range(n_heads)]

        self.heads = nn.ModuleList([
            ProjectionHead(input_dim=input_dim, output_dim=self.d_per_head, num_layers=num_layers)
            for _ in range(n_heads)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns concatenated z(s) of shape [B, d_z]."""
        parts = [head(h) for head in self.heads]
        return torch.cat(parts, dim=-1)

    def forward_split(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of per-head z vectors, each [B, d_per_head]."""
        return [head(h) for head in self.heads]


class DisentangledModel(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        mock_encoder=False,
        base_dim=768,
        d_c=256,
        d_z=256,
        max_len=8192,
        projection_layers=2,
        decoder_embed_dim=512,
        decoder_layers=6,
        decoder_heads=8,
        num_prefix_tokens=8,
        pooling="auto",
        decoder_type: str = "rope",
        decoder_gradient_checkpointing: bool = False,
        decoder_model_name: str | None = None,
        decoder_freeze_lm: bool = True,
        decoder_lora_r: int = 0,
        decoder_lora_alpha: int = 16,
        decoder_lora_dropout: float = 0.05,
        decoder_lora_target_modules=None,
        n_z_heads: int = 1,
        z_head_names: list[str] | None = None,
        encoder_attn_implementation: str = "sdpa",
        decoder_attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.max_len = max_len
        self.n_z_heads = n_z_heads
        self.encoder = BaseModelEncoder(
            model_name=model_name,
            mock=mock_encoder,
            base_dim=base_dim,
            pooling=pooling,
            attn_implementation=encoder_attn_implementation,
        )

        encoder_dim = self.encoder.base_dim
        self.c_head = ProjectionHead(input_dim=encoder_dim, output_dim=d_c, num_layers=projection_layers)

        if n_z_heads > 1:
            self.z_head = MultiHeadZ(
                input_dim=encoder_dim, d_z=d_z, n_heads=n_z_heads,
                num_layers=projection_layers, z_head_names=z_head_names,
            )
        else:
            self.z_head = ProjectionHead(input_dim=encoder_dim, output_dim=d_z, num_layers=projection_layers)

        decoder_vocab_size = vocab_size
        if decoder_type == "pretrained":
            if decoder_model_name is None:
                raise ValueError("decoder_model_name is required when decoder_type='pretrained'")
            self.decoder = PretrainedLatentDecoder(
                model_name=decoder_model_name,
                prefix_dim=d_c + d_z,
                num_prefix_tokens=num_prefix_tokens,
                freeze_lm=decoder_freeze_lm,
                lora_r=decoder_lora_r,
                lora_alpha=decoder_lora_alpha,
                lora_dropout=decoder_lora_dropout,
                lora_target_modules=decoder_lora_target_modules,
                gradient_checkpointing=decoder_gradient_checkpointing,
                attn_implementation=decoder_attn_implementation,
            )
            decoder_vocab_size = self.decoder.vocab_size
        else:
            if decoder_vocab_size is None:
                if self.encoder.tokenizer is None:
                    decoder_vocab_size = 1000
                elif decoder_type == "legacy":
                    # Match legacy checkpoints that sized embeddings by tokenizer.vocab_size.
                    decoder_vocab_size = int(getattr(self.encoder.tokenizer, "vocab_size", len(self.encoder.tokenizer)))
                else:
                    # Safer for modern models that add special tokens beyond vocab_size.
                    decoder_vocab_size = len(self.encoder.tokenizer)

        if decoder_type == "legacy":
            self.decoder = LegacyAutoregressiveDecoder(
                vocab_size=decoder_vocab_size,
                prefix_dim=d_c + d_z,
                embed_dim=decoder_embed_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                num_prefix_tokens=num_prefix_tokens,
                max_len=max_len,
                gradient_checkpointing=decoder_gradient_checkpointing,
            )
        elif decoder_type == "rope":
            self.decoder = AutoregressiveDecoder(
                vocab_size=decoder_vocab_size,
                prefix_dim=d_c + d_z,
                embed_dim=decoder_embed_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                num_prefix_tokens=num_prefix_tokens,
                max_len=max_len,
            )
        elif decoder_type == "pretrained":
            # decoder already built above (needs decoder_vocab_size resolution first)
            pass
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type!r} (expected 'rope', 'legacy', or 'pretrained')")

    def tokenize_traces(self, traces: list[str], max_len=None):
        if max_len is None:
            max_len = self.max_len

        if hasattr(self.decoder, "tokenizer") and self.decoder.tokenizer is not None:
            encoded = self.decoder.tokenizer(
                traces,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_len,
            )
            return encoded["input_ids"].to(self.encoder._device_anchor.device)

        if self.encoder.tokenizer is not None:
            encoded = self.encoder.tokenizer(
                traces,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_len,
            )
            return encoded["input_ids"].to(self.encoder._device_anchor.device)

        token_rows = []
        for trace in traces:
            token_values = [((ord(ch) % (self.decoder.vocab_size - 1)) + 1) for ch in trace][:max_len]
            if not token_values:
                token_values = [1]
            if len(token_values) < max_len:
                token_values.extend([0] * (max_len - len(token_values)))
            token_rows.append(token_values)
        return torch.tensor(token_rows, device=self.encoder._device_anchor.device, dtype=torch.long)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids=None, decode=True):
        """
        Dual-tokenization support:
        - encoder_* are used for encoder + projection heads.
        - decoder_input_ids (if provided) are used for the autoregressive decoder path.
          If omitted, we decode using encoder_input_ids (back-compat).
        """
        h = self.encoder(encoder_input_ids, encoder_attention_mask)

        c_s = self.c_head(h)
        z_s = self.z_head(h)

        decoder_hidden = None
        if decode:
            c_z_concat = torch.cat([c_s, z_s], dim=-1)
            token_ids = decoder_input_ids if decoder_input_ids is not None else encoder_input_ids
            decoder_hidden = self.decoder.get_hidden(token_ids, c_z_concat)

        return c_s, z_s, decoder_hidden

    def forward_modc(
        self,
        encoder_question_ids,
        encoder_question_attention_mask,
        encoder_trace_ids,
        encoder_trace_attention_mask,
        decoder_input_ids=None,
        decode=True,
    ):
        """
        ModC-aligned forward: c from question, z from trace.
        Decoder reconstructs question + trace tokens.
        """
        h_q = self.encoder(encoder_question_ids, encoder_question_attention_mask)
        h_t = self.encoder(encoder_trace_ids, encoder_trace_attention_mask)
        c_s = self.c_head(h_q)
        z_s = self.z_head(h_t)
        z_trace = z_s

        decoder_hidden = None
        if decode:
            c_z_concat = torch.cat([c_s, z_s], dim=-1)
            token_ids = decoder_input_ids if decoder_input_ids is not None else encoder_trace_ids
            decoder_hidden = self.decoder.get_hidden(token_ids, c_z_concat)

        return c_s, z_s, z_trace, decoder_hidden
