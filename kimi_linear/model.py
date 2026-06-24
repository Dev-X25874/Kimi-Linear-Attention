"""Kimi Linear model implementation.

This module implements the hybrid architecture that interleaves
KDA and MLA layers in a 3:1 ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import KimiLinearConfig
from .kda import KimiDeltaAttention, RMSNorm
from .mla import MultiHeadLatentAttention, MLP


@dataclass
class ModelOutput:
    """Model output structure."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple] = None
    past_key_values: Optional[List] = None


class KimiLinearBlock(nn.Module):
    """Single block in Kimi Linear architecture.
    
    Each block contains either:
    - KDA layer + MLP, or
    - MLA layer + MLP
    
    Args:
        config: Model configuration
        layer_idx: Index of this layer
        is_kda: Whether this is a KDA layer (vs MLA)
    """
    
    def __init__(self, config: KimiLinearConfig, layer_idx: int, is_kda: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = is_kda
        
        # Attention layer (KDA or MLA)
        if is_kda:
            self.attention = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                feature_dim=config.kda_feature_dim,
                chunk_size=config.kda_chunk_size,
                use_gating=config.kda_use_gating,
                decay_init=config.kda_decay_init,
                beta_init=config.kda_beta_init,
                eps=config.rms_norm_eps,
            )
        else:
            self.attention = MultiHeadLatentAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                kv_lora_rank=config.mla_kv_lora_rank,
                q_lora_rank=config.mla_q_lora_rank,
                qk_rope_head_dim=config.mla_qk_rope_head_dim,
                rope_theta=config.rope_theta,
                eps=config.rms_norm_eps,
            )
        
        # MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        
        # Layer norms
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass through block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            use_cache: Whether to cache states
            past_key_value: Cached states
            
        Returns:
            Tuple of (output, new_cache)
        """
        # Attention with residual
        residual = hidden_states
        if self.is_kda:
            attn_output, new_cache = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_state=past_key_value,
            )
        else:
            attn_output, new_cache = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
        hidden_states = residual + attn_output
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, new_cache


class KimiLinearModel(nn.Module):
    """Kimi Linear: Hybrid Linear Attention Architecture.
    
    Architecture:
        [KDA → KDA → KDA → MLA] × N blocks
        
    Key features:
    - 3:1 KDA to MLA ratio
    - NoPE (No Position Encoding) for MLA layers
    - Positional information handled by KDA layers
    - Low-rank KV compression in MLA
    - Fine-grained gating in KDA
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Create layer sequence following 3:1 ratio
        # Pattern: KDA, KDA, KDA, MLA, KDA, KDA, KDA, MLA, ...
        self.layers = nn.ModuleList()
        
        kda_count = 0
        mla_count = 0
        
        for layer_idx in range(config.num_layers):
            # Determine if this should be KDA or MLA
            # Use 3:1 ratio: every 4th layer is MLA
            is_kda = (layer_idx % 4) != 3
            
            if is_kda and kda_count < config.num_kda_layers:
                self.layers.append(KimiLinearBlock(config, layer_idx, is_kda=True))
                kda_count += 1
            elif not is_kda and mla_count < config.num_mla_layers:
                self.layers.append(KimiLinearBlock(config, layer_idx, is_kda=False))
                mla_count += 1
            else:
                # Fallback to maintain total layer count
                is_kda = kda_count < config.num_kda_layers
                self.layers.append(KimiLinearBlock(config, layer_idx, is_kda=is_kda))
                if is_kda:
                    kda_count += 1
                else:
                    mla_count += 1
        
        # Final layer norm and output
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # The generic init above zeros every Linear bias and randomizes every
        # Linear weight, which would clobber the KDA gate biases that encode
        # decay_init/beta_init. Re-apply that gate-specific init afterwards.
        for module in self.modules():
            if isinstance(module, KimiDeltaAttention):
                module._reset_gate_parameters()
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List] = None,
    ) -> ModelOutput:
        """Forward pass through Kimi Linear model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (optional)
            labels: Labels for language modeling loss
            use_cache: Whether to cache states for generation
            past_key_values: Cached states from previous forward pass
            
        Returns:
            ModelOutput with loss, logits, and cached states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask for MLA layers
        if attention_mask is None:
            # Causal mask: prevent attending to future tokens
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        
        # Initialize cache list
        new_key_values = [] if use_cache else None
        
        # Apply layers
        for layer_idx, layer in enumerate(self.layers):
            # Get cached state for this layer
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            
            # Forward through layer
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_kv,
            )
            
            # Store new cache
            if use_cache:
                new_key_values.append(new_kv)
        
        # Final norm and output projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=new_key_values,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Initial token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        past_key_values = None
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass
                outputs = self.forward(
                    input_ids if past_key_values is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Update cache
                past_key_values = outputs.past_key_values
        
        return input_ids
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load pretrained model."""
        config = KimiLinearConfig.from_pretrained(path)
        model = cls(config)
        model.load_state_dict(torch.load(f"{path}/model.pt", map_location='cpu'))
        return model
    
    def save_pretrained(self, path: str):
        """Save model."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.config.to_dict(), f"{path}/config.pt")
        torch.save(self.state_dict(), f"{path}/model.pt")
    
    def get_num_params(self, non_embedding: bool = True):
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
