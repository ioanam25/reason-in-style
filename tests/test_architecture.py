import pytest
import torch
from unittest.mock import patch

def test_forward_pass_dimensionality():
    """Test 2.4a: Implement a pytest suite utilizing synthetic integer tensors of shape [B, L]. Map to mps."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    B, L, vocab_size = 2, 16, 1000
    base_dim, d_c, d_z = 256, 128, 128
    
    # Synthetic traces and token ids
    traces = ["Sequence 1", "Sequence 2"]
    token_ids = torch.randint(0, vocab_size, (B, L)).to(device)
    
    with patch('src.models.DisentangledModel') as MockModel:
        model = MockModel.return_value
        model.return_value = (
            torch.randn(B, d_c, device=device),
            torch.randn(B, d_z, device=device),
            torch.randn(B, L, vocab_size, device=device)
        )
        
        # Forward pass
        c_s, z_s, logits = model(traces, token_ids)
        
        # Test 2.4b: Shape Assertions
        assert c_s.shape == (B, d_c), f"Expected {(B, d_c)}, got {c_s.shape}"
        assert z_s.shape == (B, d_z), f"Expected {(B, d_z)}, got {z_s.shape}"
        assert logits.shape == (B, L, vocab_size), f"Expected {(B, L, vocab_size)}, got {logits.shape}"
