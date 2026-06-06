"""Tests for full Kimi Linear model."""

import torch
from kimi_linear import KimiLinearConfig, KimiLinearModel


def test_model_creation():
    """Test model creation."""
    config = KimiLinearConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
    )
    
    model = KimiLinearModel(config)
    
    num_params = model.get_num_params()
    assert num_params > 0
    print(f"✓ Model created with {num_params:,} parameters")


def test_forward_pass():
    """Test forward pass."""
    config = KimiLinearConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
    )
    
    model = KimiLinearModel(config)
    
    input_ids = torch.randint(0, 1000, (2, 10))
    outputs = model(input_ids)
    
    assert outputs.logits.shape == (2, 10, 1000)
    print("✓ Forward pass works")


def test_training_mode():
    """Test model in training mode."""
    config = KimiLinearConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
    )
    
    model = KimiLinearModel(config)
    model.train()
    
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    
    assert outputs.loss is not None
    assert outputs.loss.requires_grad
    print(f"✓ Training mode works, loss: {outputs.loss.item():.4f}")


def test_generation():
    """Test text generation."""
    config = KimiLinearConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
    )
    
    model = KimiLinearModel(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 5))
    generated = model.generate(input_ids, max_length=15)
    
    assert generated.shape[1] == 20  # 5 + 15
    print(f"✓ Generation works, generated {generated.shape[1]} tokens")


def test_save_load():
    """Test saving and loading."""
    config = KimiLinearConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_kda_layers=3,
        num_mla_layers=1,
    )
    
    model = KimiLinearModel(config)
    
    # Save
    model.save_pretrained("test_checkpoint")
    
    # Load
    loaded_model = KimiLinearModel.from_pretrained("test_checkpoint")
    
    # Compare
    input_ids = torch.randint(0, 1000, (1, 10))
    
    with torch.no_grad():
        out1 = model(input_ids).logits
        out2 = loaded_model(input_ids).logits
    
    assert torch.allclose(out1, out2, atol=1e-5)
    print("✓ Save/load works")


if __name__ == "__main__":
    test_model_creation()
    test_forward_pass()
    test_training_mode()
    test_generation()
    test_save_load()
    print("\n✓✓✓ All model tests passed! ✓✓✓")
