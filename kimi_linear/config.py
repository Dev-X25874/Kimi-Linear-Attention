"""Configuration for Kimi Linear models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KimiLinearConfig:
    """Configuration class for Kimi Linear models.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Dimension of hidden states
        num_layers: Total number of layers
        num_kda_layers: Number of KDA layers (should be 3x num_mla_layers)
        num_mla_layers: Number of MLA layers (global attention)
        num_heads: Number of attention heads
        intermediate_size: Size of FFN intermediate layer
        
        # KDA specific
        kda_chunk_size: Chunk size for parallel KDA processing
        kda_feature_dim: Feature dimension for KDA (d_k, d_v)
        kda_use_gating: Whether to use fine-grained gating
        kda_decay_init: Initial decay value for alpha
        
        # MLA specific
        mla_kv_lora_rank: Low-rank dimension for KV compression in MLA
        mla_q_lora_rank: Low-rank dimension for Q compression in MLA
        mla_qk_rope_head_dim: Dimension for RoPE in MLA (set to 0 for NoPE)
        
        # Position encoding
        max_position_embeddings: Maximum sequence length
        use_rope_in_mla: Whether to use RoPE in MLA layers (False for NoPE)
        
        # Normalization
        rms_norm_eps: Epsilon for RMSNorm
        
        # Training
        initializer_range: Standard deviation for initialization
        use_cache: Whether to use KV cache during inference
    """
    
    # Model architecture
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 24
    num_kda_layers: int = 18  # 3:1 ratio
    num_mla_layers: int = 6
    num_heads: int = 16
    intermediate_size: int = 8192
    
    # KDA configuration
    kda_chunk_size: int = 128
    kda_feature_dim: int = 128
    kda_use_gating: bool = True
    kda_decay_init: float = 0.9
    kda_beta_init: float = 0.1
    
    # MLA configuration
    mla_kv_lora_rank: int = 512
    mla_q_lora_rank: int = 1536
    mla_qk_rope_head_dim: int = 0  # 0 means NoPE
    
    # Position encoding
    max_position_embeddings: int = 131072  # 128k
    use_rope_in_mla: bool = False  # NoPE for MLA
    rope_theta: float = 10000.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Initialization
    initializer_range: float = 0.02
    
    # Generation
    use_cache: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate layer ratio
        if self.num_kda_layers + self.num_mla_layers != self.num_layers:
            raise ValueError(
                f"num_kda_layers ({self.num_kda_layers}) + "
                f"num_mla_layers ({self.num_mla_layers}) must equal "
                f"num_layers ({self.num_layers})"
            )
        
        # Recommended 3:1 ratio
        expected_kda = self.num_layers * 3 // 4
        if abs(self.num_kda_layers - expected_kda) > 1:
            print(
                f"Warning: Recommended 3:1 KDA:MLA ratio. "
                f"Expected ~{expected_kda} KDA layers, got {self.num_kda_layers}"
            )
        
        # NoPE configuration
        if not self.use_rope_in_mla and self.mla_qk_rope_head_dim != 0:
            print(
                "Warning: use_rope_in_mla=False but mla_qk_rope_head_dim!=0. "
                "Setting mla_qk_rope_head_dim=0 for NoPE."
            )
            self.mla_qk_rope_head_dim = 0
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls, name_or_path: str):
        """Load configuration from pretrained model."""
        import torch
        import os
        
        config_path = os.path.join(name_or_path, "config.pt")
        if os.path.exists(config_path):
            config_dict = torch.load(config_path)
            return cls.from_dict(config_dict)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")


@dataclass
class KimiLinearTrainingConfig:
    """Training configuration for Kimi Linear models.
    
    Args:
        output_dir: Directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Peak learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
        lr_scheduler_type: Learning rate scheduler
        logging_steps: Logging frequency
        save_steps: Checkpoint save frequency
        eval_steps: Evaluation frequency
        max_grad_norm: Gradient clipping threshold
        bf16: Use BF16 mixed precision
        fp16: Use FP16 mixed precision
    """
    
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"
    
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    
    max_grad_norm: float = 1.0
    
    bf16: bool = True
    fp16: bool = False
    
    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
