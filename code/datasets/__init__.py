"""
Datasets package for Eventformer.
"""

from .event_utils import (
    EventData,
    EventCollator,
    load_events_from_npy,
    load_events_from_h5,
    normalize_events,
    random_sample_events,
    uniform_temporal_sample,
    voxel_grid_sample,
    augment_events,
    events_to_frame,
    events_to_voxel_grid
)

from .gen1 import GEN1Dataset, GEN1DetectionDataset
from .ncaltech101 import NCaltech101Dataset, NCaltech101Classification
from .dvs_gesture import DVS128GestureDataset, DVS128GestureClassification


def get_dataset(name: str, root: str, split: str = 'train', **kwargs):
    """
    Factory function to get a dataset by name.
    
    Args:
        name: Dataset name ('gen1', 'ncaltech101', 'dvs128_gesture')
        root: Path to dataset root
        split: 'train', 'val', or 'test'
        **kwargs: Additional arguments for the dataset
        
    Returns:
        Dataset instance
    """
    datasets = {
        'gen1': GEN1DetectionDataset,
        'gen1_detection': GEN1DetectionDataset,
        'ncaltech101': NCaltech101Classification,
        'n-caltech101': NCaltech101Classification,
        'dvs128_gesture': DVS128GestureClassification,
        'dvs_gesture': DVS128GestureClassification,
        'dvs128': DVS128GestureClassification
    }
    
    name_lower = name.lower().replace('-', '_')
    
    if name_lower not in datasets:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(datasets.keys())}"
        )
    
    return datasets[name_lower](root=root, split=split, **kwargs)


__all__ = [
    # Utils
    'EventData',
    'EventCollator',
    'load_events_from_npy',
    'load_events_from_h5',
    'normalize_events',
    'random_sample_events',
    'uniform_temporal_sample',
    'voxel_grid_sample',
    'augment_events',
    'events_to_frame',
    'events_to_voxel_grid',
    
    # Datasets
    'GEN1Dataset',
    'GEN1DetectionDataset',
    'NCaltech101Dataset',
    'NCaltech101Classification',
    'DVS128GestureDataset',
    'DVS128GestureClassification',
    
    # Factory
    'get_dataset'
]
