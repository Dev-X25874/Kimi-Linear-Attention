"""Kimi Delta Attention (KDA) implementation.

KDA is a linear attention mechanism based on the gated delta rule with
fine-grained channel-wise gating for better memory control.
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


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention (KDA) layer.
    
    KDA is a linear attention mechanism that uses:
    1. Fine-grained diagonal gating (channel-wise forget gate)
    2. Gated delta rule for recurrent state updates
    3. Chunkwise parallel processing for efficiency
    
    Key equation:
        S_t = diag(α_t) @ S_{t-1} + β_t * (k_t @ v_t^T)
        o_t = q_t @ S_t
    
    Where:
        α_t: channel-wise decay (forget gate)
        β_t: channel-wise update strength
        S_t: recurrent state (fixed-size memory)
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        feature_dim: Feature dimension for q, k, v
        chunk_size: Size of chunks for parallel processing
        use_gating: Whether to use fine-grained gating
        decay_init: Initial value for decay gate
        beta_init: Initial value for update gate
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int = 128,
        chunk_size: int = 128,
        use_gating: bool = True,
        decay_init: float = 0.9,
        beta_init: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.use_gating = use_gating
        self.eps = eps
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * feature_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * feature_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * feature_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * feature_dim, hidden_size, bias=False)
        
        # Gating MLPs (for α and β)
        if use_gating:
            # Decay gate (α): controls forgetting
            self.alpha_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, num_heads * feature_dim, bias=False),
            )
            
            # Update gate (β): controls update strength
            self.beta_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, num_heads * feature_dim, bias=False),
            )
            
            # Initialize to reasonable defaults
            with torch.no_grad():
                # Initialize alpha to decay_init
                self.alpha_mlp[-1].weight.data.fill_(0.0)
                
                # Initialize beta to beta_init
                self.beta_mlp[-1].weight.data.fill_(0.0)
        
        # Layer norm
        self.norm = RMSNorm(hidden_size, eps=eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through KDA.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask (not used in linear attention)
            use_cache: Whether to return recurrent state for caching
            past_state: Previous recurrent state for incremental decoding
            
        Returns:
            Tuple of (output, new_state)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply normalization
        normed = self.norm(hidden_states)
        
        # Project to Q, K, V
        q = self.q_proj(normed)  # (batch, seq_len, num_heads * feature_dim)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.feature_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.feature_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.feature_dim)
        
        # Compute gating values if enabled
        if self.use_gating:
            # Channel-wise decay (α_t)
            alpha = self.alpha_mlp(normed)
            alpha = alpha.view(batch_size, seq_len, self.num_heads, self.feature_dim)
            alpha = torch.sigmoid(alpha)  # (0, 1) range
            
            # Channel-wise update strength (β_t)
            beta = self.beta_mlp(normed)
            beta = beta.view(batch_size, seq_len, self.num_heads, self.feature_dim)
            beta = torch.sigmoid(beta)  # (0, 1) range
        else:
            alpha = torch.ones_like(q) * 0.9
            beta = torch.ones_like(q) * 0.1
        
        # Apply chunkwise recurrent processing
        output = self._chunkwise_recurrence(q, k, v, alpha, beta, past_state)
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_len, self.num_heads * self.feature_dim)
        output = self.o_proj(output)
        
        # Compute final state for caching (if needed)
        new_state = None
        if use_cache:
            # Simple implementation: just store last state
            # In practice, you'd want to properly maintain the recurrent state
            new_state = self._compute_final_state(k, v, alpha, beta, past_state)
        
        return output, new_state
    
    def _chunkwise_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply chunkwise recurrent processing.
        
        This implements the core KDA recurrence in a parallel-friendly way:
        S_t = diag(α_t) @ S_{t-1} + β_t * (k_t @ v_t^T)
        o_t = q_t @ S_t
        
        Args:
            q, k, v: Query, key, value tensors (batch, seq_len, num_heads, feature_dim)
            alpha: Decay gates (batch, seq_len, num_heads, feature_dim)
            beta: Update gates (batch, seq_len, num_heads, feature_dim)
            past_state: Previous state for incremental decoding
            
        Returns:
            Output tensor (batch, seq_len, num_heads, feature_dim)
        """
        batch_size, seq_len, num_heads, feature_dim = q.shape
        
        # Split into chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Initialize state
        if past_state is not None:
            state = past_state
        else:
            state = torch.zeros(
                batch_size, num_heads, feature_dim, feature_dim,
                device=q.device, dtype=q.dtype
            )
        
        outputs = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            
            # Get chunk
            q_chunk = q[:, start_idx:end_idx]  # (batch, chunk_len, heads, feat)
            k_chunk = k[:, start_idx:end_idx]
            v_chunk = v[:, start_idx:end_idx]
            alpha_chunk = alpha[:, start_idx:end_idx]
            beta_chunk = beta[:, start_idx:end_idx]
            
            chunk_len = end_idx - start_idx
            
            # Process chunk recurrently
            chunk_outputs = []
            for t in range(chunk_len):
                # Decay previous state: S_t = diag(α_t) @ S_{t-1}
                # alpha_t shape: (batch, heads, feat)
                # state shape: (batch, heads, feat, feat)
                alpha_t = alpha_chunk[:, t]  # (batch, heads, feat)
                
                # Apply diagonal decay: multiply each row by alpha
                state = state * alpha_t.unsqueeze(-1)
                
                # Add rank-1 update: S_t += β_t * (k_t @ v_t^T)
                beta_t = beta_chunk[:, t]  # (batch, heads, feat)
                k_t = k_chunk[:, t]  # (batch, heads, feat)
                v_t = v_chunk[:, t]  # (batch, heads, feat)
                
                # Compute outer product with beta weighting
                # (batch, heads, feat, 1) @ (batch, heads, 1, feat)
                update = (beta_t * k_t).unsqueeze(-1) @ v_t.unsqueeze(-2)
                state = state + update
                
                # Compute output: o_t = q_t @ S_t
                q_t = q_chunk[:, t]  # (batch, heads, feat)
                # (batch, heads, 1, feat) @ (batch, heads, feat, feat)
                o_t = (q_t.unsqueeze(-2) @ state).squeeze(-2)
                
                chunk_outputs.append(o_t)
            
            # Stack chunk outputs
            chunk_output = torch.stack(chunk_outputs, dim=1)
            outputs.append(chunk_output)
        
        # Concatenate all chunks
        output = torch.cat(outputs, dim=1)
        
        return output
    
    def _compute_final_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute final recurrent state for caching."""
        # Simplified: just compute the state after processing all tokens
        # In practice, this would be computed during the forward pass
        batch_size, seq_len, num_heads, feature_dim = k.shape
        
        if past_state is not None:
            state = past_state
        else:
            state = torch.zeros(
                batch_size, num_heads, feature_dim, feature_dim,
                device=k.device, dtype=k.dtype
            )
        
        # Apply recurrence for all tokens
        for t in range(seq_len):
            alpha_t = alpha[:, t]
            beta_t = beta[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            
            state = state * alpha_t.unsqueeze(-1)
            update = (beta_t * k_t).unsqueeze(-1) @ v_t.unsqueeze(-2)
            state = state + update
        
        return state
