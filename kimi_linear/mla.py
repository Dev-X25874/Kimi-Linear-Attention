"""Multi-Head Latent Attention (MLA) implementation.

MLA uses low-rank projections to compress KV cache while maintaining
full expressivity. In Kimi Linear, MLA layers use NoPE (No Position Encoding)
as positional information is handled by KDA layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with low-rank KV compression.
    
    MLA uses low-rank projections to compress K and V before caching:
    - Standard attention: O(seq_len * num_heads * head_dim) cache
    - MLA: O(seq_len * kv_lora_rank) cache
    
    In Kimi Linear configuration:
    - NoPE (No Position Encoding) for MLA layers
    - Positional information delegated to KDA layers
    - Can be converted to MQA during inference for efficiency
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        kv_lora_rank: Low-rank dimension for KV compression
        q_lora_rank: Low-rank dimension for Q compression (if used)
        qk_rope_head_dim: Dimension for RoPE (0 for NoPE)
        rope_theta: RoPE theta parameter
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 0,  # 0 = NoPE
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_rope = qk_rope_head_dim > 0
        
        # Query projection (potentially with low-rank)
        if q_lora_rank > 0:
            # Low-rank Q projection: hidden -> q_lora_rank -> num_heads * head_dim
            self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
        else:
            # Direct projection
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        
        # Low-rank KV projection
        # Compress: hidden -> kv_lora_rank
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        
        # Decompress: kv_lora_rank -> K and V
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Layer norm
        self.norm = RMSNorm(hidden_size, eps=eps)
        
        # RoPE (if used)
        if self.use_rope:
            self.rope_theta = rope_theta
            # Note: Kimi Linear uses NoPE for MLA, so this is typically not used
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through MLA.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Causal mask (batch, 1, seq_len, seq_len)
            use_cache: Whether to cache K, V for generation
            past_key_value: Cached (K, V) from previous steps
            
        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply normalization
        normed = self.norm(hidden_states)
        
        # Project to Q
        if hasattr(self, 'q_down_proj'):
            # Low-rank Q
            q_compressed = self.q_down_proj(normed)
            q = self.q_up_proj(q_compressed)
        else:
            # Direct Q
            q = self.q_proj(normed)
        
        # Reshape Q for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # Project to low-rank KV representation
        kv_compressed = self.kv_down_proj(normed)  # (batch, seq_len, kv_lora_rank)
        
        # If using cache, concatenate with past
        if past_key_value is not None:
            past_kv_compressed = past_key_value[0]
            kv_compressed = torch.cat([past_kv_compressed, kv_compressed], dim=1)
        
        # Cache the compressed representation (not K and V separately!)
        # This is the key memory saving of MLA
        new_cache = (kv_compressed,) if use_cache else None
        
        # Decompress to K and V
        k = self.k_up_proj(kv_compressed)
        v = self.v_up_proj(kv_compressed)
        
        # Reshape K, V for multi-head
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, kv_seq_len, head_dim)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to V
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)
        
        return output, new_cache


class MLP(nn.Module):
    """Feed-Forward Network with SwiGLU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * up(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
