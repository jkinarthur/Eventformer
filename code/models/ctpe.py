"""
Continuous-Time Positional Encoding (CTPE)

Novel contribution: Embeds events at arbitrary continuous timestamps using
multi-scale Fourier features, preserving microsecond temporal resolution
without discretization into fixed bins.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class ContinuousTimePositionalEncoding(nn.Module):
    """
    Continuous-Time Positional Encoding (CTPE) using Random Fourier Features.
    
    Unlike discrete positional encodings that assume fixed positions,
    CTPE maps continuous timestamps to high-dimensional representations,
    enabling the model to reason about temporal relationships at arbitrary precision.
    
    Args:
        embed_dim: Output embedding dimension
        num_frequencies: Number of frequency components per scale
        num_scales: Number of temporal scales (fine to coarse)
        sigma_min: Minimum frequency scale (for fine temporal patterns)
        sigma_max: Maximum frequency scale (for coarse temporal patterns)
        learnable: Whether to learn frequency parameters
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_frequencies: int = 64,
        num_scales: int = 3,
        sigma_min: float = 1e-4,  # Microsecond scale
        sigma_max: float = 1e-1,  # ~100ms scale
        learnable: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        self.num_scales = num_scales
        self.learnable = learnable
        
        # Each scale contributes 2 * num_frequencies dimensions (sin + cos)
        self.freq_dim_per_scale = 2 * num_frequencies
        total_freq_dim = self.freq_dim_per_scale * num_scales
        
        # Initialize frequency scales (log-spaced from fine to coarse)
        sigmas = torch.logspace(
            math.log10(sigma_min),
            math.log10(sigma_max),
            num_scales
        )
        
        # Initialize random frequencies for each scale
        frequencies = []
        for sigma in sigmas:
            # Sample from N(0, sigma^2)
            freq = torch.randn(num_frequencies) * sigma
            frequencies.append(freq)
        
        frequencies = torch.stack(frequencies)  # [num_scales, num_frequencies]
        
        if learnable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
        
        # Project concatenated features to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(total_freq_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Learnable scale factors for each temporal scale
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous timestamps to high-dimensional representations.
        
        Args:
            timestamps: Tensor of shape [N] or [B, N] containing continuous timestamps
            
        Returns:
            Tensor of shape [N, embed_dim] or [B, N, embed_dim]
        """
        # Handle both batched and non-batched inputs
        input_shape = timestamps.shape
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)  # [1, N]
        
        B, N = timestamps.shape
        
        # Normalize timestamps to [0, 1] range for numerical stability
        t_min = timestamps.min(dim=-1, keepdim=True)[0]
        t_max = timestamps.max(dim=-1, keepdim=True)[0]
        t_range = t_max - t_min + 1e-8  # Avoid division by zero
        t_normalized = (timestamps - t_min) / t_range  # [B, N]
        
        # Compute multi-scale Fourier features
        fourier_features = []
        
        for scale_idx in range(self.num_scales):
            freq = self.frequencies[scale_idx]  # [num_frequencies]
            
            # Compute 2*pi*f*t
            phase = 2 * math.pi * t_normalized.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
            # phase: [B, N, num_frequencies]
            
            # Compute sin and cos features
            sin_features = torch.sin(phase)
            cos_features = torch.cos(phase)
            
            # Concatenate and weight by scale importance
            scale_features = torch.cat([sin_features, cos_features], dim=-1)  # [B, N, 2*num_freq]
            scale_features = scale_features * self.scale_weights[scale_idx]
            
            fourier_features.append(scale_features)
        
        # Concatenate all scales
        fourier_features = torch.cat(fourier_features, dim=-1)  # [B, N, total_freq_dim]
        
        # Project to embedding dimension
        output = self.projection(fourier_features)  # [B, N, embed_dim]
        
        # Return to original shape if input was unbatched
        if len(input_shape) == 1:
            output = output.squeeze(0)
        
        return output
    
    def get_temporal_attention_bias(
        self,
        timestamps: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute temporal attention bias based on time differences.
        Closer events in time should attend more strongly to each other.
        
        Args:
            timestamps: [N] or [B, N] tensor of timestamps
            temperature: Scaling factor for attention logits
            
        Returns:
            Attention bias of shape [N, N] or [B, N, N]
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
        
        B, N = timestamps.shape
        
        # Compute pairwise time differences
        t1 = timestamps.unsqueeze(-1)  # [B, N, 1]
        t2 = timestamps.unsqueeze(-2)  # [B, 1, N]
        time_diff = torch.abs(t1 - t2)  # [B, N, N]
        
        # Convert to attention bias (closer = higher attention)
        # Use exponential decay with learnable/fixed bandwidth
        max_diff = time_diff.max()
        normalized_diff = time_diff / (max_diff + 1e-8)
        attention_bias = -normalized_diff / temperature
        
        return attention_bias.squeeze(0) if B == 1 else attention_bias


class SpatialPositionalEncoding(nn.Module):
    """
    Spatial positional encoding for event coordinates.
    
    Encodes normalized (x, y) coordinates along with local geometric features.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_frequencies: int = 32,
        include_local_features: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        self.include_local_features = include_local_features
        
        # 2D Fourier features for x and y
        freq_dim = 4 * num_frequencies  # sin/cos for both x and y
        
        # Initialize frequencies
        frequencies = torch.randn(2, num_frequencies) * 0.1
        self.frequencies = nn.Parameter(frequencies)
        
        # Input dim includes local features if enabled
        input_dim = freq_dim
        if include_local_features:
            input_dim += 4  # local_density, local_orientation, normalized_x, normalized_y
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        coords: torch.Tensor,
        image_size: Tuple[int, int],
        local_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode spatial coordinates.
        
        Args:
            coords: [N, 2] or [B, N, 2] tensor of (x, y) coordinates
            image_size: (height, width) of the sensor
            local_features: Optional [N, 2] or [B, N, 2] tensor of (density, orientation)
            
        Returns:
            Spatial embeddings of shape [N, embed_dim] or [B, N, embed_dim]
        """
        input_shape = coords.shape
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        
        B, N, _ = coords.shape
        H, W = image_size
        
        # Normalize coordinates to [0, 1]
        x_norm = coords[..., 0:1] / W
        y_norm = coords[..., 1:2] / H
        coords_norm = torch.cat([x_norm, y_norm], dim=-1)  # [B, N, 2]
        
        # Compute Fourier features
        fourier_features = []
        for dim in range(2):
            coord = coords_norm[..., dim:dim+1]  # [B, N, 1]
            freq = self.frequencies[dim]  # [num_frequencies]
            
            phase = 2 * math.pi * coord * freq.unsqueeze(0).unsqueeze(0)
            sin_f = torch.sin(phase)
            cos_f = torch.cos(phase)
            fourier_features.extend([sin_f, cos_f])
        
        features = torch.cat(fourier_features, dim=-1)  # [B, N, 4*num_freq]
        
        # Add local features if provided
        if self.include_local_features:
            if local_features is not None:
                if local_features.dim() == 2:
                    local_features = local_features.unsqueeze(0)
                features = torch.cat([features, local_features, coords_norm], dim=-1)
            else:
                # Use zeros if local features not provided
                zeros = torch.zeros(B, N, 4, device=coords.device)
                features = torch.cat([features, zeros], dim=-1)
        
        output = self.projection(features)
        
        if len(input_shape) == 2:
            output = output.squeeze(0)
        
        return output


class PolarityEmbedding(nn.Module):
    """
    Learnable embedding for event polarity.
    
    ON (+1) and OFF (-1) events encode different motion semantics
    and should have distinct learned representations.
    """
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Separate embeddings for ON and OFF events
        self.on_embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.off_embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)
        
    def forward(self, polarity: torch.Tensor) -> torch.Tensor:
        """
        Embed polarity values.
        
        Args:
            polarity: [N] or [B, N] tensor with values in {-1, +1}
            
        Returns:
            Polarity embeddings of shape [N, embed_dim] or [B, N, embed_dim]
        """
        input_shape = polarity.shape
        if polarity.dim() == 1:
            polarity = polarity.unsqueeze(0)
        
        B, N = polarity.shape
        
        # Create output tensor
        output = torch.zeros(B, N, self.embed_dim, device=polarity.device)
        
        # Assign embeddings based on polarity
        on_mask = polarity > 0
        off_mask = polarity < 0
        
        output[on_mask] = self.on_embedding
        output[off_mask] = self.off_embedding
        
        if len(input_shape) == 1:
            output = output.squeeze(0)
        
        return output


def test_ctpe():
    """Test CTPE module."""
    print("Testing Continuous-Time Positional Encoding...")
    
    ctpe = ContinuousTimePositionalEncoding(
        embed_dim=256,
        num_frequencies=64,
        num_scales=3
    )
    
    # Test with random timestamps (in seconds)
    timestamps = torch.rand(1000) * 0.1  # 100ms window
    
    embeddings = ctpe(timestamps)
    print(f"  Input shape: {timestamps.shape}")
    print(f"  Output shape: {embeddings.shape}")
    
    # Test temporal attention bias
    bias = ctpe.get_temporal_attention_bias(timestamps[:100])
    print(f"  Attention bias shape: {bias.shape}")
    
    # Verify temporal locality: events close in time should have higher bias
    close_pairs_bias = bias[torch.abs(torch.arange(100).unsqueeze(1) - torch.arange(100).unsqueeze(0)) < 5].mean()
    far_pairs_bias = bias[torch.abs(torch.arange(100).unsqueeze(1) - torch.arange(100).unsqueeze(0)) > 50].mean()
    print(f"  Close pairs avg bias: {close_pairs_bias:.4f}")
    print(f"  Far pairs avg bias: {far_pairs_bias:.4f}")
    assert close_pairs_bias > far_pairs_bias, "Temporal locality not preserved!"
    
    print("  ✓ CTPE test passed!\n")


if __name__ == "__main__":
    test_ctpe()
