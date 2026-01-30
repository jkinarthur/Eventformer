"""
Adaptive Spatiotemporal Neighborhood Attention (ASNA)

Novel contribution: Local attention mechanism where each event attends to its
k-nearest neighbors in spatiotemporal space, with k adapting based on local
event density. Sparse regions use larger neighborhoods; dense regions use smaller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from pytorch3d.ops import knn_points
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("Warning: pytorch3d not available. Using fallback kNN implementation.")


class SpatiotemporalDistance(nn.Module):
    """
    Compute spatiotemporal distances between events.
    
    Distance metric: d = sqrt(α_s * ((x1-x2)² + (y1-y2)²) + α_t * (t1-t2)²)
    
    α_s and α_t are learnable parameters that balance spatial and temporal proximity.
    """
    
    def __init__(
        self,
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        learnable: bool = True
    ):
        super().__init__()
        
        if learnable:
            self.spatial_weight = nn.Parameter(torch.tensor(spatial_weight))
            self.temporal_weight = nn.Parameter(torch.tensor(temporal_weight))
        else:
            self.register_buffer('spatial_weight', torch.tensor(spatial_weight))
            self.register_buffer('temporal_weight', torch.tensor(temporal_weight))
    
    def forward(
        self,
        coords1: torch.Tensor,
        times1: torch.Tensor,
        coords2: Optional[torch.Tensor] = None,
        times2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise spatiotemporal distances.
        
        Args:
            coords1: [B, N1, 2] spatial coordinates (x, y)
            times1: [B, N1] timestamps
            coords2: Optional [B, N2, 2] (if None, computes self-distances)
            times2: Optional [B, N2]
            
        Returns:
            distances: [B, N1, N2] pairwise distances
        """
        if coords2 is None:
            coords2 = coords1
            times2 = times1
        
        B, N1, _ = coords1.shape
        N2 = coords2.shape[1]
        
        # Spatial distances
        spatial_diff = coords1.unsqueeze(2) - coords2.unsqueeze(1)  # [B, N1, N2, 2]
        spatial_dist_sq = (spatial_diff ** 2).sum(dim=-1)  # [B, N1, N2]
        
        # Temporal distances
        temporal_diff = times1.unsqueeze(2) - times2.unsqueeze(1)  # [B, N1, N2]
        temporal_dist_sq = temporal_diff ** 2
        
        # Combined distance
        alpha_s = F.softplus(self.spatial_weight)  # Ensure positive
        alpha_t = F.softplus(self.temporal_weight)
        
        distance = torch.sqrt(
            alpha_s * spatial_dist_sq + alpha_t * temporal_dist_sq + 1e-8
        )
        
        return distance


def knn_graph_fallback(
    points: torch.Tensor,
    k: int,
    distance_matrix: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback kNN implementation without pytorch3d.
    
    Args:
        points: [B, N, D] point coordinates
        k: number of neighbors
        distance_matrix: Optional precomputed [B, N, N] distances
        
    Returns:
        indices: [B, N, k] neighbor indices
        distances: [B, N, k] neighbor distances
    """
    B, N, D = points.shape
    
    if distance_matrix is None:
        # Compute pairwise distances
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # [B, N, N, D]
        distance_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, N, N]
    
    # Get k nearest (excluding self)
    distance_matrix = distance_matrix.clone()
    distance_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))  # Exclude self
    
    k = min(k, N - 1)
    distances, indices = torch.topk(distance_matrix, k, dim=-1, largest=False)
    
    return indices, distances


class AdaptiveKNN(nn.Module):
    """
    Adaptive k-Nearest Neighbors based on local event density.
    
    k_i = k_base * (mean_density / local_density)^γ
    
    Sparse regions (low density) → larger k
    Dense regions (high density) → smaller k
    """
    
    def __init__(
        self,
        k_base: int = 32,
        gamma: float = 0.5,
        k_min: int = 8,
        k_max: int = 128
    ):
        super().__init__()
        
        self.k_base = k_base
        self.gamma = gamma
        self.k_min = k_min
        self.k_max = k_max
        
        # Learnable gamma
        self.gamma_param = nn.Parameter(torch.tensor(gamma))
        
    def compute_local_density(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        radius: float = 0.05
    ) -> torch.Tensor:
        """
        Estimate local event density around each event.
        
        Args:
            coords: [B, N, 2] normalized spatial coordinates
            times: [B, N] normalized timestamps
            radius: radius for density estimation
            
        Returns:
            density: [B, N] local density values
        """
        B, N, _ = coords.shape
        
        # Combine into 3D points (x, y, t)
        points = torch.cat([coords, times.unsqueeze(-1)], dim=-1)  # [B, N, 3]
        
        # Compute pairwise distances
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # [B, N, N, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, N, N]
        
        # Count neighbors within radius
        neighbor_count = (distances < radius).sum(dim=-1).float()  # [B, N]
        
        # Density = count / volume (normalized)
        density = neighbor_count / (N * radius ** 3 + 1e-8)
        
        return density
    
    def compute_adaptive_k(self, density: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive k values based on local density.
        
        Args:
            density: [B, N] local density values
            
        Returns:
            k_values: [B, N] integer k values for each event
        """
        mean_density = density.mean(dim=-1, keepdim=True)  # [B, 1]
        
        # k = k_base * (mean/local)^gamma
        gamma = torch.sigmoid(self.gamma_param)  # Keep in (0, 1)
        ratio = (mean_density / (density + 1e-8)) ** gamma
        k_values = self.k_base * ratio
        
        # Clamp to valid range
        k_values = k_values.clamp(self.k_min, self.k_max).long()
        
        return k_values
    
    def forward(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        features: torch.Tensor,
        distance_module: Optional[SpatiotemporalDistance] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adaptive kNN graph.
        
        Args:
            coords: [B, N, 2] spatial coordinates
            times: [B, N] timestamps
            features: [B, N, D] event features
            distance_module: Optional distance computation module
            
        Returns:
            neighbor_indices: [B, N, k_max] neighbor indices (padded)
            neighbor_mask: [B, N, k_max] valid neighbor mask
            neighbor_distances: [B, N, k_max] distances to neighbors
        """
        B, N, _ = coords.shape
        device = coords.device
        
        # Compute local density
        density = self.compute_local_density(coords, times)
        
        # Compute adaptive k values
        k_values = self.compute_adaptive_k(density)
        
        # Compute distance matrix
        if distance_module is not None:
            distances = distance_module(coords, times)
        else:
            # Simple Euclidean in (x, y, t) space
            points = torch.cat([coords, times.unsqueeze(-1)], dim=-1)
            diff = points.unsqueeze(2) - points.unsqueeze(1)
            distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        
        # Get k_max nearest neighbors for all
        k_max = min(self.k_max, N - 1)
        neighbor_indices, neighbor_distances = knn_graph_fallback(
            torch.cat([coords, times.unsqueeze(-1)], dim=-1),
            k_max,
            distances
        )
        
        # Create mask based on adaptive k
        # neighbor_mask[b, i, j] = True if j < k_values[b, i]
        k_range = torch.arange(k_max, device=device).unsqueeze(0).unsqueeze(0)
        neighbor_mask = k_range < k_values.unsqueeze(-1)
        
        return neighbor_indices, neighbor_mask, neighbor_distances


class ASNAAttention(nn.Module):
    """
    Adaptive Spatiotemporal Neighborhood Attention.
    
    Each event attends only to its k-nearest neighbors in spatiotemporal space,
    with k adapting based on local event density.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        k_base: int = 32,
        gamma: float = 0.5,
        dropout: float = 0.0,
        use_relative_pos: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Adaptive kNN
        self.adaptive_knn = AdaptiveKNN(k_base=k_base, gamma=gamma)
        
        # Distance computation
        self.distance = SpatiotemporalDistance()
        
        # Relative position encoding
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.rel_pos_mlp = nn.Sequential(
                nn.Linear(3, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_heads)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] event features
            coords: [B, N, 2] spatial coordinates
            times: [B, N] timestamps
            
        Returns:
            output: [B, N, D] attended features
        """
        B, N, D = features.shape
        H = self.num_heads
        
        # Get adaptive kNN graph
        neighbor_idx, neighbor_mask, neighbor_dist = self.adaptive_knn(
            coords, times, features, self.distance
        )
        # neighbor_idx: [B, N, K]
        # neighbor_mask: [B, N, K]
        # neighbor_dist: [B, N, K]
        
        K = neighbor_idx.shape[-1]
        
        # Project features
        q = self.q_proj(features)  # [B, N, D]
        k = self.k_proj(features)  # [B, N, D]
        v = self.v_proj(features)  # [B, N, D]
        
        # Gather neighbor features
        # Expand indices for gathering
        neighbor_idx_expanded = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        k_neighbors = torch.gather(
            k.unsqueeze(2).expand(-1, -1, K, -1),
            dim=1,
            index=neighbor_idx_expanded.transpose(1, 2).reshape(B, K, N, D).transpose(1, 2)
        ).reshape(B, N, K, D)
        v_neighbors = torch.gather(
            v.unsqueeze(2).expand(-1, -1, K, -1),
            dim=1,
            index=neighbor_idx_expanded.transpose(1, 2).reshape(B, K, N, D).transpose(1, 2)
        ).reshape(B, N, K, D)
        
        # Actually, let's do this more carefully
        # For each point i, we need features of its K neighbors
        k_neighbors = torch.zeros(B, N, K, D, device=features.device)
        v_neighbors = torch.zeros(B, N, K, D, device=features.device)
        
        for b in range(B):
            k_neighbors[b] = k[b, neighbor_idx[b]]  # [N, K, D]
            v_neighbors[b] = v[b, neighbor_idx[b]]
        
        # Reshape for multi-head attention
        q = q.view(B, N, H, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        k_neighbors = k_neighbors.view(B, N, K, H, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, H, N, K, head_dim]
        v_neighbors = v_neighbors.view(B, N, K, H, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, H, N, K, head_dim]
        
        # Compute attention scores
        # q: [B, H, N, head_dim], k: [B, H, N, K, head_dim]
        attn = torch.einsum('bhnd,bhnkd->bhnk', q, k_neighbors) * self.scale  # [B, H, N, K]
        
        # Add relative position bias if enabled
        if self.use_relative_pos:
            # Get relative positions
            rel_coords = torch.zeros(B, N, K, 2, device=features.device)
            rel_times = torch.zeros(B, N, K, device=features.device)
            
            for b in range(B):
                neighbor_coords = coords[b, neighbor_idx[b]]  # [N, K, 2]
                neighbor_times = times[b, neighbor_idx[b]]  # [N, K]
                rel_coords[b] = neighbor_coords - coords[b].unsqueeze(1)
                rel_times[b] = neighbor_times - times[b].unsqueeze(1)
            
            rel_pos = torch.cat([rel_coords, rel_times.unsqueeze(-1)], dim=-1)  # [B, N, K, 3]
            rel_pos_bias = self.rel_pos_mlp(rel_pos)  # [B, N, K, H]
            rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)  # [B, H, N, K]
            
            attn = attn + rel_pos_bias
        
        # Apply mask (invalid neighbors get -inf)
        attn_mask = ~neighbor_mask.unsqueeze(1)  # [B, 1, N, K]
        attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Handle NaN from all-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Apply attention to values
        # attn: [B, H, N, K], v: [B, H, N, K, head_dim]
        output = torch.einsum('bhnk,bhnkd->bhnd', attn, v_neighbors)  # [B, H, N, head_dim]
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        output = self.out_proj(output)
        
        return output


class ASNABlock(nn.Module):
    """
    Complete ASNA transformer block with attention, normalization, and FFN.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        k_base: int = 32,
        gamma: float = 0.5,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ASNAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_base=k_base,
            gamma=gamma,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D]
            coords: [B, N, 2]
            times: [B, N]
            
        Returns:
            output: [B, N, D]
        """
        # Attention with residual
        features = features + self.attn(self.norm1(features), coords, times)
        
        # FFN with residual
        features = features + self.ffn(self.norm2(features))
        
        return features


def test_asna():
    """Test ASNA module."""
    print("Testing Adaptive Spatiotemporal Neighborhood Attention...")
    
    B, N, D = 2, 500, 256
    H = 8
    k_base = 32
    
    asna = ASNABlock(embed_dim=D, num_heads=H, k_base=k_base)
    
    # Create test data
    features = torch.randn(B, N, D)
    coords = torch.rand(B, N, 2)  # Normalized coordinates
    times = torch.rand(B, N).sort(dim=1)[0]  # Sorted timestamps
    
    # Forward pass
    output = asna(features, coords, times)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  k_base: {k_base}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    print("  ✓ ASNA test passed!\n")


if __name__ == "__main__":
    test_asna()
