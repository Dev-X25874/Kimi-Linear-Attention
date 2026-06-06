"""Tests for Kimi Delta Attention."""

import torch
from kimi_linear.kda import KimiDeltaAttention


def test_kda_forward():
    """Test KDA forward pass."""
    kda = KimiDeltaAttention(
        hidden_size=256,
        num_heads=4,
        feature_dim=64,
        chunk_size=32,
    )
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, 256)
    
    output, state = kda(hidden_states)
    
    assert output.shape == (batch_size, seq_len, 256)
    print("✓ KDA forward pass works")


def test_kda_with_cache():
    """Test KDA with caching."""
    kda = KimiDeltaAttention(
        hidden_size=256,
        num_heads=4,
        feature_dim=64,
    )
    
    hidden_states = torch.randn(1, 10, 256)
    
    # First pass
    output1, state1 = kda(hidden_states, use_cache=True)
    
    # Second pass with cache
    output2, state2 = kda(hidden_states[:, :5], use_cache=True, past_state=state1)
    
    assert output1.shape == (1, 10, 256)
    assert output2.shape == (1, 5, 256)
    print("✓ KDA caching works")


def test_kda_gating():
    """Test KDA with and without gating."""
    kda_with_gate = KimiDeltaAttention(
        hidden_size=256,
        num_heads=4,
        feature_dim=64,
        use_gating=True,
    )
    
    kda_no_gate = KimiDeltaAttention(
        hidden_size=256,
        num_heads=4,
        feature_dim=64,
        use_gating=False,
    )
    
    hidden_states = torch.randn(2, 10, 256)
    
    output1, _ = kda_with_gate(hidden_states)
    output2, _ = kda_no_gate(hidden_states)
    
    assert output1.shape == output2.shape
    print("✓ KDA gating options work")


if __name__ == "__main__":
    test_kda_forward()
    test_kda_with_cache()
    test_kda_gating()
    print("\n✓✓✓ All KDA tests passed! ✓✓✓")
