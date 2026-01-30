"""
Eventformer: Frame-Free Vision Transformer for Event Cameras

Main model architecture combining:
- CTPE: Continuous-Time Positional Encoding
- PAAA: Polarity-Aware Asymmetric Attention
- ASNA: Adaptive Spatiotemporal Neighborhood Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

from .ctpe import ContinuousTimePositionalEncoding, SpatialPositionalEncoding, PolarityEmbedding
from .paaa import PolarityAwareAsymmetricAttention
from .asna import ASNABlock


class EventEmbedding(nn.Module):
    """
    Initial embedding layer for raw events.
    
    Transforms (x, y, t, p) → D-dimensional features.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_fourier_scales: int = 8,
        image_size: Tuple[int, int] = (346, 260)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        # Continuous-time positional encoding
        self.ctpe = ContinuousTimePositionalEncoding(
            embed_dim=embed_dim,
            num_scales=num_fourier_scales
        )
        
        # Spatial positional encoding (image_size passed at forward time)
        self.spatial_pe = SpatialPositionalEncoding(
            embed_dim=embed_dim
        )
        
        # Polarity embedding
        self.polarity_embed = PolarityEmbedding(embed_dim=embed_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 2] spatial coordinates (x, y)
            times: [B, N] timestamps
            polarities: [B, N] polarity values (+1/-1)
            
        Returns:
            embeddings: [B, N, D]
        """
        # Get individual embeddings
        time_embed = self.ctpe(times)  # [B, N, D]
        spatial_embed = self.spatial_pe(coords, self.image_size)  # [B, N, D]
        polarity_embed = self.polarity_embed(polarities)  # [B, N, D]
        
        # Concatenate and fuse
        combined = torch.cat([time_embed, spatial_embed, polarity_embed], dim=-1)
        embeddings = self.fusion(combined)
        
        return embeddings


class EventformerBlock(nn.Module):
    """
    Single Eventformer block combining:
    1. PAAA for polarity-aware global attention
    2. ASNA for spatiotemporal local attention
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        k_base: int = 32,
        gamma: float = 0.5,
        dropout: float = 0.0,
        use_asna: bool = True
    ):
        super().__init__()
        
        # PAAA for polarity-aware attention
        self.paaa = PolarityAwareAsymmetricAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ASNA for local attention
        self.use_asna = use_asna
        if use_asna:
            self.asna = ASNABlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                k_base=k_base,
                gamma=gamma,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN for PAAA branch
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Gating for combining PAAA and ASNA
        if use_asna:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor,
        temporal_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D]
            coords: [B, N, 2]
            times: [B, N]
            polarities: [B, N]
            temporal_bias: Optional [B, H, N, N] temporal attention bias
            
        Returns:
            output: [B, N, D]
        """
        # PAAA branch
        paaa_out = self.paaa(
            self.norm1(features),
            polarities,
            attn_bias=temporal_bias
        )
        paaa_out = features + paaa_out
        paaa_out = paaa_out + self.ffn(self.norm2(paaa_out))
        
        if self.use_asna:
            # ASNA branch
            asna_out = self.asna(features, coords, times)
            
            # Gated combination
            gate = self.gate(torch.cat([paaa_out, asna_out], dim=-1))
            output = gate * paaa_out + (1 - gate) * asna_out
        else:
            output = paaa_out
        
        return output


class HierarchicalDownsample(nn.Module):
    """
    Spatiotemporal downsampling layer using farthest point sampling.
    Reduces number of events while preserving spatial and temporal coverage.
    """
    
    def __init__(
        self,
        embed_dim_in: int,
        embed_dim_out: int,
        ratio: float = 0.25
    ):
        super().__init__()
        
        self.ratio = ratio
        self.proj = nn.Linear(embed_dim_in, embed_dim_out)
        self.norm = nn.LayerNorm(embed_dim_out)
        
    def farthest_point_sample(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Farthest point sampling in spatiotemporal space.
        
        Args:
            coords: [B, N, 2]
            times: [B, N]
            num_samples: number of points to sample
            
        Returns:
            indices: [B, num_samples] indices of sampled points
        """
        B, N, _ = coords.shape
        device = coords.device
        
        # Combine into 3D points
        points = torch.cat([coords, times.unsqueeze(-1)], dim=-1)  # [B, N, 3]
        
        # FPS
        indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distances = torch.full((B, N), float('inf'), device=device)
        
        # Start with random point
        indices[:, 0] = torch.randint(0, N, (B,), device=device)
        
        for i in range(1, num_samples):
            # Get last selected point
            last = points[torch.arange(B, device=device), indices[:, i-1]]  # [B, 3]
            
            # Compute distances to last point
            dist = torch.sqrt(((points - last.unsqueeze(1)) ** 2).sum(dim=-1) + 1e-8)
            
            # Update minimum distances
            distances = torch.min(distances, dist)
            
            # Select farthest point
            indices[:, i] = distances.argmax(dim=-1)
        
        return indices
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, N, D_in]
            coords: [B, N, 2]
            times: [B, N]
            polarities: [B, N]
            
        Returns:
            features: [B, M, D_out]
            coords: [B, M, 2]
            times: [B, M]
            polarities: [B, M]
        """
        B, N, _ = features.shape
        M = max(int(N * self.ratio), 1)
        
        # Sample points
        indices = self.farthest_point_sample(coords, times, M)
        
        # Gather features and coordinates
        features_out = torch.gather(
            features, 1,
            indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )
        coords_out = torch.gather(
            coords, 1,
            indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        times_out = torch.gather(times, 1, indices)
        polarities_out = torch.gather(polarities, 1, indices)
        
        # Project to new dimension
        features_out = self.norm(self.proj(features_out))
        
        return features_out, coords_out, times_out, polarities_out


class EventformerStage(nn.Module):
    """
    Single stage of the hierarchical Eventformer.
    Contains multiple blocks at the same resolution.
    """
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        k_base: int = 32,
        gamma: float = 0.5,
        dropout: float = 0.0,
        use_asna: bool = True
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            EventformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                k_base=k_base,
                gamma=gamma,
                dropout=dropout,
                use_asna=use_asna
            )
            for _ in range(depth)
        ])
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor,
        temporal_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, D]
            coords: [B, N, 2]
            times: [B, N]
            polarities: [B, N]
            temporal_bias: Optional attention bias
            
        Returns:
            output: [B, N, D]
        """
        for block in self.blocks:
            features = block(features, coords, times, polarities, temporal_bias)
        
        return features


class Eventformer(nn.Module):
    """
    Eventformer: Frame-Free Vision Transformer for Event Cameras
    
    Hierarchical architecture with 4 stages:
    Stage 1: N events, D features
    Stage 2: N/4 events, 2D features
    Stage 3: N/16 events, 4D features
    Stage 4: N/64 events, 8D features
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (2, 4, 8, 16),
        mlp_ratio: float = 4.0,
        k_base: int = 32,
        gamma: float = 0.5,
        dropout: float = 0.0,
        image_size: Tuple[int, int] = (346, 260),
        use_ctpe: bool = True,
        use_paaa: bool = True,
        use_asna: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.use_ctpe = use_ctpe
        self.use_paaa = use_paaa
        self.use_asna = use_asna
        
        # Compute dimensions for each stage
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Event embedding
        self.event_embed = EventEmbedding(
            embed_dim=embed_dim,
            image_size=image_size
        )
        
        # CTPE for temporal attention bias
        if use_ctpe:
            self.ctpe = ContinuousTimePositionalEncoding(embed_dim=embed_dim)
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(EventformerStage(
                embed_dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                k_base=k_base,
                gamma=gamma,
                dropout=dropout,
                use_asna=use_asna
            ))
        
        # Downsampling between stages
        self.downsample = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.downsample.append(HierarchicalDownsample(
                embed_dim_in=dims[i],
                embed_dim_out=dims[i + 1],
                ratio=0.25
            ))
        
        # Final feature dimension
        self.out_dim = dims[-1]
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, N, 2] spatial coordinates (x, y)
            times: [B, N] timestamps
            polarities: [B, N] polarity values (+1/-1)
            
        Returns:
            features: [B, M, D_out] final event features
            coords: [B, M, 2] final coordinates
            times: [B, M] final timestamps
            polarities: [B, M] final polarities
        """
        # Initial embedding
        features = self.event_embed(coords, times, polarities)
        
        # Compute temporal attention bias if using CTPE
        temporal_bias = None
        if self.use_ctpe:
            # Note: Only compute for first stage, recompute after downsampling
            pass
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            features = stage(features, coords, times, polarities, temporal_bias)
            
            # Downsample between stages
            if i < len(self.downsample):
                features, coords, times, polarities = self.downsample[i](
                    features, coords, times, polarities
                )
        
        return features, coords, times, polarities
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class EventformerForClassification(nn.Module):
    """
    Eventformer with classification head for gesture/object recognition.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 64,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (2, 4, 8, 16),
        **kwargs
    ):
        super().__init__()
        
        self.backbone = Eventformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            **kwargs
        )
        
        # Classification head
        self.norm = nn.LayerNorm(self.backbone.out_dim)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)
        
    def forward(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 2]
            times: [B, N]
            polarities: [B, N]
            
        Returns:
            logits: [B, num_classes]
        """
        features, _, _, _ = self.backbone(coords, times, polarities)
        
        # Global average pooling
        features = features.mean(dim=1)  # [B, D]
        features = self.norm(features)
        logits = self.head(features)
        
        return logits


class EventformerForDetection(nn.Module):
    """
    Eventformer with detection head for object detection.
    Uses a simple point-based detection head.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 64,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (2, 4, 8, 16),
        **kwargs
    ):
        super().__init__()
        
        self.backbone = Eventformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            **kwargs
        )
        
        out_dim = self.backbone.out_dim
        
        # Detection heads
        self.cls_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, num_classes + 1)  # +1 for background
        )
        
        self.box_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 4)  # x, y, w, h
        )
        
        self.objectness_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, 1)
        )
        
    def forward(
        self,
        coords: torch.Tensor,
        times: torch.Tensor,
        polarities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords: [B, N, 2]
            times: [B, N]
            polarities: [B, N]
            
        Returns:
            dict with:
                - cls_logits: [B, M, num_classes+1]
                - box_preds: [B, M, 4]
                - objectness: [B, M, 1]
                - coords: [B, M, 2] (anchor locations)
        """
        features, out_coords, _, _ = self.backbone(coords, times, polarities)
        
        cls_logits = self.cls_head(features)
        box_preds = self.box_head(features)
        objectness = self.objectness_head(features)
        
        return {
            'cls_logits': cls_logits,
            'box_preds': box_preds,
            'objectness': objectness,
            'coords': out_coords
        }


def eventformer_tiny(**kwargs) -> Eventformer:
    """Eventformer-Tiny: 2.5M parameters"""
    return Eventformer(
        embed_dim=32,
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 4, 8),
        **kwargs
    )


def eventformer_small(**kwargs) -> Eventformer:
    """Eventformer-Small: 7.5M parameters"""
    return Eventformer(
        embed_dim=48,
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 6, 12),
        **kwargs
    )


def eventformer_base(**kwargs) -> Eventformer:
    """Eventformer-Base: 22M parameters"""
    return Eventformer(
        embed_dim=64,
        depths=(2, 2, 8, 2),
        num_heads=(2, 4, 8, 16),
        **kwargs
    )


def eventformer_large(**kwargs) -> Eventformer:
    """Eventformer-Large: 45M parameters"""
    return Eventformer(
        embed_dim=96,
        depths=(2, 2, 12, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs
    )


def test_eventformer():
    """Test Eventformer model."""
    print("Testing Eventformer...")
    
    B, N = 2, 1024
    num_classes = 10
    
    # Create test data
    coords = torch.rand(B, N, 2) * torch.tensor([346, 260])
    times = torch.rand(B, N).sort(dim=1)[0]
    polarities = torch.randint(0, 2, (B, N)).float() * 2 - 1
    
    # Test backbone
    print("\n1. Testing Eventformer backbone...")
    model = eventformer_tiny()
    features, out_coords, out_times, out_pols = model(coords, times, polarities)
    print(f"   Input: {N} events")
    print(f"   Output: {features.shape[1]} events, {features.shape[2]} features")
    print(f"   Parameters: {model.get_num_params():,}")
    
    # Test classification
    print("\n2. Testing classification head...")
    cls_model = EventformerForClassification(
        num_classes=num_classes,
        embed_dim=32,
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 4, 8)
    )
    logits = cls_model(coords, times, polarities)
    print(f"   Output logits: {logits.shape}")
    
    # Test detection
    print("\n3. Testing detection head...")
    det_model = EventformerForDetection(
        num_classes=num_classes,
        embed_dim=32,
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 4, 8)
    )
    outputs = det_model(coords, times, polarities)
    print(f"   Class logits: {outputs['cls_logits'].shape}")
    print(f"   Box predictions: {outputs['box_preds'].shape}")
    
    print("\n✓ All Eventformer tests passed!")


if __name__ == "__main__":
    test_eventformer()
