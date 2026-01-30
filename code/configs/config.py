"""
Configuration file for Eventformer experiments.

Defines model, training, and dataset configurations.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Architecture
    embed_dim: int = 64
    depths: Tuple[int, ...] = (2, 2, 8, 2)
    num_heads: Tuple[int, ...] = (2, 4, 8, 16)
    mlp_ratio: float = 4.0
    
    # Components
    use_ctpe: bool = True  # Continuous-Time Positional Encoding
    use_paaa: bool = True  # Polarity-Aware Asymmetric Attention
    use_asna: bool = True  # Adaptive Spatiotemporal Neighborhood Attention
    
    # ASNA parameters
    k_base: int = 32
    gamma: float = 0.5
    
    # Regularization
    dropout: float = 0.0
    drop_path: float = 0.1
    
    # Input
    num_events: int = 4096
    image_size: Tuple[int, int] = (346, 260)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    
    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'linear'
    
    # Gradient clipping
    clip_grad: float = 1.0
    
    # Data augmentation
    augment: bool = True
    flip_horizontal: bool = True
    flip_vertical: bool = False
    flip_polarity: bool = True
    time_jitter: float = 0.01
    spatial_jitter: float = 0.01
    drop_rate: float = 0.1
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    
    # Checkpointing
    save_interval: int = 10
    log_interval: int = 50
    eval_interval: int = 1


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    name: str = 'ncaltech101'
    root: str = './data'
    
    # Event processing
    num_events: int = 4096
    sample_method: str = 'uniform'  # 'uniform', 'random', 'voxel'
    
    # For detection datasets
    time_window: float = 50000  # microseconds
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


# Predefined configurations
CONFIGS = {
    'eventformer_tiny': ModelConfig(
        embed_dim=32,
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 4, 8),
    ),
    'eventformer_small': ModelConfig(
        embed_dim=48,
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 6, 12),
    ),
    'eventformer_base': ModelConfig(
        embed_dim=64,
        depths=(2, 2, 8, 2),
        num_heads=(2, 4, 8, 16),
    ),
    'eventformer_large': ModelConfig(
        embed_dim=96,
        depths=(2, 2, 12, 2),
        num_heads=(3, 6, 12, 24),
    )
}


# Dataset-specific configurations
DATASET_CONFIGS = {
    'gen1': DatasetConfig(
        name='gen1',
        root='./data/gen1',
        num_events=8192,
        time_window=50000,
    ),
    'ncaltech101': DatasetConfig(
        name='ncaltech101',
        root='./data/ncaltech101',
        num_events=4096,
    ),
    'dvs128_gesture': DatasetConfig(
        name='dvs128_gesture',
        root='./data/dvs128_gesture',
        num_events=4096,
    ),
}


def get_config(model_name: str = 'eventformer_base', dataset_name: str = 'ncaltech101'):
    """Get configuration for experiment."""
    model_config = CONFIGS.get(model_name, CONFIGS['eventformer_base'])
    dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['ncaltech101'])
    training_config = TrainingConfig()
    
    return {
        'model': model_config,
        'dataset': dataset_config,
        'training': training_config
    }
