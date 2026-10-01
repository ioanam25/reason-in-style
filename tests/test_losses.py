import pytest
import torch
from src.losses import infonce_loss, orthogonal_decorrelation_loss, reconstruction_loss, variance_regularization_loss, entropy_regularization_loss
from src.models import DisentangledModel

def test_orthogonality_bounds():
    """Test 3.5a: Feed explicitly orthogonal synthetic vectors. Assert L_z is strictly 0.0."""
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    
    loss = orthogonal_decorrelation_loss(c, z)
    assert loss.item() < 1e-7

def test_infonce_symmetry():
    """Test 3.5b: Verify InfoNCE is invariant to the permutation of positive pairs."""
    # Create 2 pairs (4 embeddings)
    c_s1 = torch.randn(4, 128)
    # pair 1: index 0 and 1
    # pair 2: index 2 and 3
    loss1 = infonce_loss(c_s1)
    
    # swap pair 1 elements (index 0 and 1 swapped)
    c_s2 = c_s1.clone()
    c_s2[0] = c_s1[1]
    c_s2[1] = c_s1[0]
    
    loss2 = infonce_loss(c_s2)
    assert torch.isclose(loss1, loss2)

@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(), reason="Needs Accelerator")
def test_gradient_autograd_check():
    """Test 3.5c: backward() on a synthetic batch. Assert grads are not None and no NaNs."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create explicit leaf tensors to mock neural network outputs
    c_s = torch.randn(4, 128, device=device, requires_grad=True)
    z_s = torch.randn(4, 128, device=device, requires_grad=True)
    logits = torch.randn(4, 16, 1000, device=device, requires_grad=True)
    token_ids = torch.randint(0, 1000, (4, 16), device=device)
    
    l_c = infonce_loss(c_s)
    l_z = orthogonal_decorrelation_loss(c_s, z_s)
    l_rec = reconstruction_loss(logits, token_ids)
    
    loss = l_c + l_z + l_rec
    loss.backward()
    
    assert c_s.grad is not None
    assert not torch.isnan(c_s.grad).any()
    assert z_s.grad is not None
    assert not torch.isnan(z_s.grad).any()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_variance_regularization_collapsed():
    """Constant embeddings (collapsed) should produce a high variance loss."""
    embeddings = torch.ones(8, 64)  # all identical rows → std=0 per dim
    loss = variance_regularization_loss(embeddings)
    # gamma=1.0, std=0 → relu(1.0 - 0 + 1e-4) ≈ 1.0 per dim
    assert loss.item() > 0.9


def test_variance_regularization_spread():
    """Well-spread embeddings should produce near-zero variance loss."""
    torch.manual_seed(42)
    embeddings = torch.randn(64, 32) * 5.0  # std ≈ 5 >> gamma=1
    loss = variance_regularization_loss(embeddings)
    assert loss.item() < 1e-4


def test_variance_regularization_single_sample():
    """Single sample should return zero (can't compute std)."""
    embeddings = torch.randn(1, 32)
    loss = variance_regularization_loss(embeddings)
    assert loss.item() == 0.0


def test_entropy_regularization_uniform():
    """Uniformly spread embeddings should have near-max entropy → loss near -log(N-1)."""
    torch.manual_seed(0)
    # Orthogonal-ish embeddings: high entropy
    embeddings = torch.randn(16, 64)
    loss = entropy_regularization_loss(embeddings)
    # Max entropy for 15 neighbours = log(15) ≈ 2.71
    # Loss = -entropy, so should be around -2.7
    assert loss.item() < 0.0, "Entropy loss should be negative for spread embeddings"


def test_entropy_regularization_clustered():
    """Duplicated embeddings should have lower entropy than spread embeddings."""
    torch.manual_seed(0)
    # Create near-duplicate pairs: each anchor repeated → softmax peaks on duplicate
    base = torch.randn(8, 32)
    duplicated = torch.cat([base, base + 1e-5 * torch.randn_like(base)], dim=0)
    loss_duplicate = entropy_regularization_loss(duplicated)
    # Spread embeddings
    spread = torch.randn(16, 32) * 3.0
    loss_spread = entropy_regularization_loss(spread)
    # Duplicated should have lower entropy (higher / less negative loss)
    assert loss_duplicate.item() > loss_spread.item()


def test_entropy_regularization_too_few():
    """Fewer than 3 embeddings should return zero."""
    embeddings = torch.randn(2, 32)
    loss = entropy_regularization_loss(embeddings)
    assert loss.item() == 0.0
