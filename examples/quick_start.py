"""Quick start example for Kimi Linear."""

import torch
from kimi_linear import KimiLinearConfig, KimiLinearModel


def main():
    print("=" * 50)
    print("KIMI LINEAR - QUICK START")
    print("=" * 50)
    
    # Create configuration
    print("\n1. Creating model configuration...")
    config = KimiLinearConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=12,
        num_kda_layers=9,   # 3:1 ratio
        num_mla_layers=3,
        num_heads=8,
        intermediate_size=2048,
    )
    print(f"   ✓ Config created: {config.num_layers} layers (9 KDA + 3 MLA)")
    
    # Create model
    print("\n2. Initializing model...")
    model = KimiLinearModel(config)
    num_params = model.get_num_params()
    print(f"   ✓ Model created: {num_params:,} parameters")
    
    # Forward pass
    print("\n3. Testing forward pass...")
    input_ids = torch.randint(0, 10000, (2, 32))
    outputs = model(input_ids, labels=input_ids)
    print(f"   ✓ Forward pass complete")
    print(f"   - Loss: {outputs.loss.item():.4f}")
    print(f"   - Logits shape: {outputs.logits.shape}")
    
    # Generation
    print("\n4. Testing generation...")
    model.eval()
    prompt = torch.randint(0, 10000, (1, 5))
    
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=20,
            temperature=1.0,
            top_k=50,
        )
    
    print(f"   ✓ Generated {generated.shape[1]} tokens")
    print(f"   - Input length: {prompt.shape[1]}")
    print(f"   - Output length: {generated.shape[1]}")
    
    # Training step
    print("\n5. Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for step in range(3):
        batch = torch.randint(0, 10000, (2, 32))
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}/3 - Loss: {loss.item():.4f}")
    
    print(f"   ✓ Training step complete")
    
    # Save model
    print("\n6. Saving model...")
    model.save_pretrained("quick_start_checkpoint")
    print(f"   ✓ Model saved to quick_start_checkpoint/")
    
    # Load model
    print("\n7. Loading model...")
    loaded_model = KimiLinearModel.from_pretrained("quick_start_checkpoint")
    print(f"   ✓ Model loaded successfully")
    
    print("\n" + "=" * 50)
    print("✓✓✓ ALL STEPS COMPLETED SUCCESSFULLY! ✓✓✓")
    print("=" * 50)
    
    print("\n📚 Next steps:")
    print("   - Modify config for your use case")
    print("   - Train on your dataset")
    print("   - Run scripts/train.py for full training")
    print("   - Check tests/ for more examples")


if __name__ == "__main__":
    main()
