"""
Polarity-Aware Asymmetric Attention (PAAA)

Novel contribution: ON and OFF events encode fundamentally different motion semantics
(leading vs trailing edges). PAAA processes them through separate attention pathways
before asymmetric cross-attention fusion, enabling explicit motion direction reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention module."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_k, D]
            attn_mask: Optional [B, N_q, N_k] boolean mask (True = masked)
            attn_bias: Optional [B, N_q, N_k] or [B, H, N_q, N_k] additive bias
            
        Returns:
            output: [B, N_q, D]
            attn_weights: [B, H, N_q, N_k]
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # Project and reshape
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, N, head_dim]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_k]
        
        # Add attention bias if provided
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [B, 1, N_q, N_k]
            attn = attn + attn_bias
        
        # Apply mask
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, N_q, head_dim]
        output = output.transpose(1, 2).reshape(B, N_q, D)
        output = self.out_proj(output)
        
        return output, attn_weights


class PolaritySelfAttention(nn.Module):
    """
    Self-attention within a single polarity group (ON or OFF events).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_temporal_bias: bool = True
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.use_temporal_bias = use_temporal_bias
        
        # Learnable temperature for temporal bias
        if use_temporal_bias:
            self.temporal_temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        x: torch.Tensor,
        temporal_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] event features
            temporal_bias: [B, N, N] temporal attention bias
            
        Returns:
            output: [B, N, D]
            attn_weights: [B, H, N, N]
        """
        residual = x
        x = self.norm(x)
        
        if self.use_temporal_bias and temporal_bias is not None:
            temporal_bias = temporal_bias * self.temporal_temperature
        
        output, attn_weights = self.attention(x, x, x, attn_bias=temporal_bias)
        output = output + residual
        
        return output, attn_weights


class AsymmetricCrossAttention(nn.Module):
    """
    Asymmetric cross-attention between ON and OFF event streams.
    
    Key insight: The interaction ON→OFF and OFF→ON capture different
    motion relationships and should be modeled separately.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # ON queries attending to OFF keys/values
        self.on_to_off_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # OFF queries attending to ON keys/values  
        self.off_to_on_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm_on = nn.LayerNorm(embed_dim)
        self.norm_off = nn.LayerNorm(embed_dim)
        
        # Learnable fusion weights
        self.on_fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.off_fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(
        self,
        on_features: torch.Tensor,
        off_features: torch.Tensor,
        on_off_bias: Optional[torch.Tensor] = None,
        off_on_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            on_features: [B, N_on, D] ON event features
            off_features: [B, N_off, D] OFF event features
            on_off_bias: Optional attention bias from ON to OFF
            off_on_bias: Optional attention bias from OFF to ON
            
        Returns:
            fused_on: [B, N_on, D] ON features enriched with OFF context
            fused_off: [B, N_off, D] OFF features enriched with ON context
        """
        on_normed = self.norm_on(on_features)
        off_normed = self.norm_off(off_features)
        
        # ON attends to OFF (captures trailing edge context for leading edges)
        on_cross, _ = self.on_to_off_attn(
            query=on_normed,
            key=off_normed,
            value=off_normed,
            attn_bias=on_off_bias
        )
        
        # OFF attends to ON (captures leading edge context for trailing edges)
        off_cross, _ = self.off_to_on_attn(
            query=off_normed,
            key=on_normed,
            value=on_normed,
            attn_bias=off_on_bias
        )
        
        # Fuse with residual connections
        alpha_on = torch.sigmoid(self.on_fusion_weight)
        alpha_off = torch.sigmoid(self.off_fusion_weight)
        
        fused_on = on_features + alpha_on * on_cross
        fused_off = off_features + alpha_off * off_cross
        
        return fused_on, fused_off


class PolarityAwareAsymmetricAttention(nn.Module):
    """
    Polarity-Aware Asymmetric Attention (PAAA) Module.
    
    Complete PAAA block that:
    1. Separates events by polarity
    2. Applies self-attention within each polarity group
    3. Applies asymmetric cross-attention between groups
    4. Recombines into unified representation
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_temporal_bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Self-attention for each polarity
        self.on_self_attn = PolaritySelfAttention(
            embed_dim, num_heads, dropout, use_temporal_bias
        )
        self.off_self_attn = PolaritySelfAttention(
            embed_dim, num_heads, dropout, use_temporal_bias
        )
        
        # Asymmetric cross-attention
        self.cross_attn = AsymmetricCrossAttention(embed_dim, num_heads, dropout)
        
        # FFN for final processing
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        polarity: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_separate: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] event features
            polarity: [B, N] polarity values in {-1, +1}
            timestamps: Optional [B, N] timestamps for temporal bias
            return_separate: If True, return ON and OFF features separately
            
        Returns:
            output: [B, N, D] processed features (or tuple if return_separate)
        """
        B, N, D = features.shape
        device = features.device
        
        # Separate by polarity
        on_mask = polarity > 0  # [B, N]
        off_mask = polarity < 0
        
        # Get indices for each polarity per batch
        # This is tricky because different batches may have different counts
        # We'll use a padded approach for efficiency
        
        max_on = on_mask.sum(dim=1).max().item()
        max_off = off_mask.sum(dim=1).max().item()
        
        if max_on == 0 or max_off == 0:
            # Edge case: all events have same polarity
            output, _ = self.on_self_attn(features)
            output = output + self.ffn(output)
            return output
        
        # Gather ON and OFF features
        on_features_list = []
        off_features_list = []
        on_times_list = []
        off_times_list = []
        
        for b in range(B):
            on_idx = torch.where(on_mask[b])[0]
            off_idx = torch.where(off_mask[b])[0]
            
            # Pad to max length
            on_feat = features[b, on_idx]  # [N_on, D]
            off_feat = features[b, off_idx]  # [N_off, D]
            
            on_padded = F.pad(on_feat, (0, 0, 0, max_on - len(on_idx)))
            off_padded = F.pad(off_feat, (0, 0, 0, max_off - len(off_idx)))
            
            on_features_list.append(on_padded)
            off_features_list.append(off_padded)
            
            if timestamps is not None:
                on_t = timestamps[b, on_idx]
                off_t = timestamps[b, off_idx]
                on_t_padded = F.pad(on_t, (0, max_on - len(on_idx)))
                off_t_padded = F.pad(off_t, (0, max_off - len(off_idx)))
                on_times_list.append(on_t_padded)
                off_times_list.append(off_t_padded)
        
        on_features = torch.stack(on_features_list)  # [B, max_on, D]
        off_features = torch.stack(off_features_list)  # [B, max_off, D]
        
        # Compute temporal bias if timestamps provided
        on_temporal_bias = None
        off_temporal_bias = None
        
        if timestamps is not None:
            on_times = torch.stack(on_times_list)
            off_times = torch.stack(off_times_list)
            
            # Compute pairwise temporal differences for bias
            on_temporal_bias = self._compute_temporal_bias(on_times)
            off_temporal_bias = self._compute_temporal_bias(off_times)
        
        # Apply self-attention within each polarity
        on_features, on_attn = self.on_self_attn(on_features, on_temporal_bias)
        off_features, off_attn = self.off_self_attn(off_features, off_temporal_bias)
        
        # Apply asymmetric cross-attention
        on_features, off_features = self.cross_attn(on_features, off_features)
        
        if return_separate:
            return on_features, off_features
        
        # Reconstruct full feature tensor
        output = torch.zeros_like(features)
        
        for b in range(B):
            on_idx = torch.where(on_mask[b])[0]
            off_idx = torch.where(off_mask[b])[0]
            
            output[b, on_idx] = on_features[b, :len(on_idx)]
            output[b, off_idx] = off_features[b, :len(off_idx)]
        
        # Apply FFN
        output = output + self.ffn(output)
        
        return output
    
    def _compute_temporal_bias(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute temporal attention bias from timestamps."""
        # timestamps: [B, N]
        t1 = timestamps.unsqueeze(-1)  # [B, N, 1]
        t2 = timestamps.unsqueeze(-2)  # [B, 1, N]
        time_diff = torch.abs(t1 - t2)  # [B, N, N]
        
        # Normalize and convert to bias (closer = higher)
        max_diff = time_diff.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8
        bias = -time_diff / max_diff
        
        return bias


class PolarityAwareFusion(nn.Module):
    """
    Alternative fusion module that keeps polarity separation throughout
    and only fuses at the final stage.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Polarity-specific projections
        self.on_proj = nn.Linear(embed_dim, embed_dim)
        self.off_proj = nn.Linear(embed_dim, embed_dim)
        
        # Motion-aware fusion
        self.motion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(
        self,
        on_features: torch.Tensor,
        off_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse ON and OFF features with motion-aware gating.
        
        Args:
            on_features: [B, D] aggregated ON features
            off_features: [B, D] aggregated OFF features
            
        Returns:
            fused: [B, D] motion-aware fused features
        """
        on_proj = self.on_proj(on_features)
        off_proj = self.off_proj(off_features)
        
        # Concatenate for gating
        concat = torch.cat([on_proj, off_proj], dim=-1)
        
        # Compute motion gate
        gate = self.motion_gate(concat)
        
        # Gated combination
        gated_on = gate * on_proj
        gated_off = (1 - gate) * off_proj
        
        # Final projection
        fused = self.output_proj(torch.cat([gated_on, gated_off], dim=-1))
        
        return fused


def test_paaa():
    """Test PAAA module."""
    print("Testing Polarity-Aware Asymmetric Attention...")
    
    B, N, D = 2, 1000, 256
    num_heads = 8
    
    paaa = PolarityAwareAsymmetricAttention(D, num_heads)
    
    # Create random features and polarities
    features = torch.randn(B, N, D)
    polarity = torch.randint(0, 2, (B, N)) * 2 - 1  # Random -1 or +1
    timestamps = torch.rand(B, N).sort(dim=1)[0]  # Sorted timestamps
    
    # Forward pass
    output = paaa(features, polarity, timestamps)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ON events: {(polarity > 0).sum().item()}")
    print(f"  OFF events: {(polarity < 0).sum().item()}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    print("  ✓ PAAA test passed!\n")


if __name__ == "__main__":
    test_paaa()
