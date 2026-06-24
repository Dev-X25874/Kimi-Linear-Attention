"""Training script for Kimi Linear models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from kimi_linear import KimiLinearConfig, KimiLinearModel
import argparse
import yaml
from tqdm import tqdm
import os


class DummyDataset(Dataset):
    """Dummy dataset for demonstration."""
    
    def __init__(self, vocab_size, seq_length, num_samples):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_length,))


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = KimiLinearConfig(**config_dict)
    
    # Create model
    model = KimiLinearModel(config)
    model = model.to(args.device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dataset
    dataset = DummyDataset(
        vocab_size=config.vocab_size,
        seq_length=128,
        num_samples=1000,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch.to(args.device)
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    train()
