import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

def infonce_loss(c_s: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Context-Sensitive InfoNCE Loss
    Assumes c_s is of shape [2N, D], structured as:
    [s_{1,1}, s_{1,2}, s_{2,1}, s_{2,2}, ..., s_{N,1}, s_{N,2}]
    
    Pairs are adjacent.
    """
    if c_s.shape[0] == 0:
        return c_s.new_zeros((), requires_grad=True)
    if c_s.shape[0] % 2 != 0:
        raise ValueError("InfoNCE expects an even number of embeddings arranged in adjacent positive pairs.")
    if c_s.shape[0] == 2:
        # With only one positive pair there are no negatives, so the contrastive term is undefined.
        return c_s.new_zeros((), requires_grad=True)
        
    c_s_norm = F.normalize(c_s, dim=-1)
    # Cosine similarity matrix [2N, 2N]
    sim_matrix = torch.matmul(c_s_norm, c_s_norm.T) / tau
    
    # Mask to ignore self-similarity
    diag_mask = torch.eye(c_s.shape[0], dtype=torch.bool, device=c_s.device)
    sim_matrix.masked_fill_(diag_mask, float('-inf'))
    
    # For element 2i, the positive is 2i+1. For 2i+1, the positive is 2i.
    target = torch.empty(c_s.shape[0], dtype=torch.long, device=c_s.device)
    target[0::2] = torch.arange(1, c_s.shape[0], 2, device=c_s.device)
    target[1::2] = torch.arange(0, c_s.shape[0], 2, device=c_s.device)
    
    loss = F.cross_entropy(sim_matrix, target)
    return loss

def orthogonal_decorrelation_loss(c_s: torch.Tensor, z_s: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonal Decorrelation Loss
    Penalizes linear correlation between c(s) and z(s) for the same trace.
    L_z = (1/2N) sum ( (c^T * z) / (|c| * |z| + eps) )^2
    """
    c_s_norm = torch.norm(c_s, p=2, dim=-1, keepdim=True)
    z_s_norm = torch.norm(z_s, p=2, dim=-1, keepdim=True)
    
    dot_product = (c_s * z_s).sum(dim=-1, keepdim=True)
    
    cosine_sim = dot_product / (c_s_norm * z_s_norm + eps)
    
    # Squared correlation
    loss = (cosine_sim ** 2).mean()
    return loss

def variance_regularization_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg-style variance regularizer.
    Penalizes each embedding dimension whose standard deviation falls below *gamma*.
    L_var = (1/D) sum_d max(0, gamma - std(x_d) + eps)
    Uses bare std (no eps inside sqrt) so the 1/sqrt singularity provides
    finite gradients even at near-collapse.
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros((), requires_grad=True)
    std = embeddings.std(dim=0)  # (D,)
    return F.relu(gamma - std + eps).mean()


def covariance_regularization_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    VICReg-style covariance regularizer.
    Penalizes the squared off-diagonal entries of the feature covariance matrix,
    spreading information across dimensions. A variance hinge alone does not stop
    a low-rank (e.g. 1-D antipodal) solution; this covariance term does.
    L_cov = (1/D) * sum_{i != j} Cov(x)_{ij}^2
    """
    n = embeddings.shape[0]
    d = embeddings.shape[1]
    if n < 2:
        return embeddings.new_zeros((), requires_grad=True)
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (n - 1)  # (D, D)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def length_decorrelation_loss(
    embeddings: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Length nuisance-decorrelation regularizer (fully unsupervised).

    Penalizes the squared Pearson correlation between each latent dimension and a
    scalar nuisance variable (here: trace length), averaged over dimensions:

        L_len = (1/D) * sum_d corr(z[:, d], length)^2

    Length is computable directly from the trace (token count) — it is NOT a
    teacher label — so this keeps the objective unsupervised while explicitly
    stopping z from degenerating into a length/verbosity axis. Without it, an
    unsupervised z trivially rediscovers length (which in this data is strongly
    confounded with teacher identity) and calls it "style".
    """
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.new_zeros((), requires_grad=True)
    lengths = lengths.to(embeddings.dtype).reshape(-1)
    z = embeddings - embeddings.mean(dim=0, keepdim=True)          # (N, D)
    l = lengths - lengths.mean()                                   # (N,)
    z_std = z.std(dim=0) + eps                                     # (D,)
    l_std = l.std() + eps                                          # scalar
    cov = (z * l.unsqueeze(-1)).mean(dim=0)                        # (D,)
    corr = cov / (z_std * l_std)                                   # (D,)
    # Sum (not mean) over dims: ~ total length-variance captured by z (R^2-like).
    # Mean-over-D would dilute a real single-axis length collapse to ~0.
    return corr.pow(2).sum()


def entropy_regularization_loss(embeddings: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Similarity-entropy regularizer.
    For each embedding, compute a softmax distribution over cosine similarities
    to all other embeddings, then return the negative mean entropy.
    Minimising this loss maximises entropy → uniform similarity → no clustering.
    """
    if embeddings.shape[0] < 3:
        return embeddings.new_zeros((), requires_grad=True)
    e = F.normalize(embeddings, dim=-1)
    sim = e @ e.T / tau                          # (N, N)
    # mask out self-similarity
    mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, float('-inf'))
    log_p = F.log_softmax(sim, dim=-1)           # (N, N)
    p = log_p.exp()
    # zero out the diagonal (was -inf → softmax 0, but 0*-inf = nan)
    p = p.masked_fill(mask, 0.0)
    log_p = log_p.masked_fill(mask, 0.0)
    entropy = -(p * log_p).sum(dim=-1)           # (N,)  per-row entropy
    return -entropy.mean()                        # minimize → maximize entropy


def reconstruction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction Loss (Causal language modeling)
    logits: [B, L, V]
    target_ids: [B, L]
    attention_mask: [B, L] — positions with 0 are treated as padding and ignored.
    """
    if logits.size(1) < 2 or target_ids.size(1) < 2:
        return logits.new_zeros((), requires_grad=True)

    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = target_ids[:, 1:].clone()

    if attention_mask is not None:
        pad_positions = attention_mask[:, 1:] == 0
        shift_labels[pad_positions] = -100

    shift_labels = shift_labels.contiguous().view(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss


def chunked_reconstruction_loss(
    lm_head: torch.nn.Linear,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 128,
    project_fn=None,
) -> torch.Tensor:
    """
    Memory-efficient reconstruction loss that never materializes [B, L, V].
    Projects hidden -> logits in chunks of `chunk_size` tokens along the
    sequence dimension, computes per-chunk CE, then averages.

    Each chunk is wrapped in torch.utils.checkpoint so the logits and CE
    intermediates are freed after forward and recomputed during backward,
    keeping peak memory proportional to one chunk rather than the full sequence.

    hidden: [B, L, embed_dim]  (decoder hidden states, NOT logits)
    lm_head: nn.Linear(embed_dim, vocab_size)
    project_fn: optional callable(hidden_chunk) -> logits_chunk; use for FSDP-safe projection
    target_ids: [B, L]
    attention_mask: [B, L]
    """
    if hidden.size(1) < 2 or target_ids.size(1) < 2:
        return hidden.new_zeros((), requires_grad=True)

    shift_hidden = hidden[:, :-1, :]
    shift_labels = target_ids[:, 1:].clone()

    if attention_mask is not None:
        pad_positions = attention_mask[:, 1:] == 0
        shift_labels[pad_positions] = -100

    B, L, D = shift_hidden.shape
    total_loss = shift_hidden.new_zeros(())
    total_tokens = 0

    def _chunk_ce(hidden_chunk, labels_chunk):
        if project_fn is not None:
            logits = project_fn(hidden_chunk)
        else:
            logits = lm_head(hidden_chunk)
        logits = logits.contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(
            logits, labels_chunk, ignore_index=-100, reduction="sum",
        ).unsqueeze(0)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk_labels = shift_labels[:, start:end].contiguous().view(-1)
        valid = (chunk_labels != -100).sum().item()
        if valid == 0:
            continue
        chunk_loss = _checkpoint(
            _chunk_ce,
            shift_hidden[:, start:end, :],
            chunk_labels,
            use_reentrant=False,
        ).squeeze(0)
        total_loss = total_loss + chunk_loss
        total_tokens += valid

    if total_tokens == 0:
        return hidden.new_zeros((), requires_grad=True)
    return total_loss / total_tokens


def within_group_diversity_loss(z_s: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Within-question diversity loss.
    Assumes z_s is of shape [2N, D] with adjacent pairs from the same question:
    [z_{1,1}, z_{1,2}, z_{2,1}, z_{2,2}, ...].
    Returns the mean cosine similarity of same-question pairs.
    Minimizing this pushes same-question z(s) apart without risking collapse.
    """
    if z_s.shape[0] < 2 or z_s.shape[0] % 2 != 0:
        return z_s.new_zeros((), requires_grad=True)
    z_norm = F.normalize(z_s, dim=-1)
    # Cosine similarity between each adjacent pair
    cos_sim = (z_norm[0::2] * z_norm[1::2]).sum(dim=-1)  # (N,)
    # Hinge: penalize only same-question similarity above `margin`; no reward
    # for going antipodal (this prevents the 1-D / low-rank z collapse).
    return F.relu(cos_sim - margin).mean()
