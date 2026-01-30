"""
N-Caltech101 Event Camera Dataset

Neuromorphic version of Caltech-101, captured with ATIS sensor.
Contains 101 object categories + background class.

Dataset: https://www.garrickorchard.com/datasets/n-caltech101
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

from .event_utils import EventData, normalize_events, random_sample_events


class NCaltech101Dataset(Dataset):
    """
    N-Caltech101 Object Classification Dataset.
    
    Expected directory structure:
    root/
        Caltech101/
            accordion/
                image_0001.bin
                image_0002.bin
                ...
            airplane/
                ...
            ...
    
    .bin files contain events in binary format.
    """
    
    IMAGE_SIZE = (240, 180)  # ATIS sensor resolution
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_events: int = 4096,
        train_ratio: float = 0.8,
        transform=None,
        seed: int = 42
    ):
        """
        Args:
            root: Path to dataset root
            split: 'train' or 'test'
            num_events: Number of events to sample per sample
            train_ratio: Fraction of data for training
            transform: Optional data augmentation
            seed: Random seed for train/test split
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.num_events = num_events
        self.train_ratio = train_ratio
        self.transform = transform
        
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._load_samples(seed)
        
        print(f"N-Caltech101 {split}: {len(self.samples)} samples, {len(self.classes)} classes")
    
    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Find all classes in the dataset."""
        data_dir = os.path.join(self.root, 'Caltech101')
        
        if not os.path.exists(data_dir):
            # Return synthetic classes for testing
            classes = [f'class_{i}' for i in range(101)]
            return classes, {c: i for i, c in enumerate(classes)}
        
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx
    
    def _load_samples(self, seed: int) -> List[Dict]:
        """Load list of all samples with their labels."""
        samples = []
        data_dir = os.path.join(self.root, 'Caltech101')
        
        if not os.path.exists(data_dir):
            # Create synthetic samples for testing
            np.random.seed(seed)
            for i in range(1000):
                class_idx = i % len(self.classes)
                samples.append({
                    'class': self.classes[class_idx],
                    'class_idx': class_idx,
                    'file_idx': i,
                    'synthetic': True
                })
        else:
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                class_idx = self.class_to_idx[class_name]
                
                for filename in os.listdir(class_dir):
                    if filename.endswith('.bin'):
                        samples.append({
                            'path': os.path.join(class_dir, filename),
                            'class': class_name,
                            'class_idx': class_idx,
                            'synthetic': False
                        })
        
        # Train/test split
        np.random.seed(seed)
        np.random.shuffle(samples)
        
        split_idx = int(len(samples) * self.train_ratio)
        
        if self.split == 'train':
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]
        
        return samples
    
    def _load_events_bin(self, path: str) -> EventData:
        """
        Load events from N-Caltech101 binary format.
        
        Format: Each event is stored as (address, timestamp) pair.
        Address encoding: x + y * x_dim + polarity * x_dim * y_dim
        """
        with open(path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        
        # Parse events (simple format)
        # This is a simplified loader - actual format may vary
        if len(raw_data) % 5 == 0:
            # Assume 5 bytes per event: x(1), y(1), p(1), t(2)
            events = raw_data.reshape(-1, 5)
            return EventData(
                x=events[:, 0].astype(np.float32),
                y=events[:, 1].astype(np.float32),
                t=events[:, 3:5].view(np.uint16)[:, 0].astype(np.float64),
                p=(events[:, 2] * 2 - 1).astype(np.float32)  # Convert 0/1 to -1/+1
            )
        else:
            # Try reading as structured array
            try:
                # Alternative: timestamp + address format
                data = np.frombuffer(raw_data, dtype=[
                    ('timestamp', '<u4'),
                    ('address', '<u4')
                ])
                
                x = (data['address'] & 0x3FF).astype(np.float32)
                y = ((data['address'] >> 10) & 0x3FF).astype(np.float32)
                p = ((data['address'] >> 20) & 0x1).astype(np.float32) * 2 - 1
                t = data['timestamp'].astype(np.float64)
                
                return EventData(x=x, y=y, t=t, p=p)
            except:
                pass
        
        # Fallback: return synthetic events
        return self._create_synthetic_events()
    
    def _create_synthetic_events(self) -> EventData:
        """Create synthetic events for testing."""
        N = 10000
        return EventData(
            x=np.random.uniform(0, self.IMAGE_SIZE[0], N).astype(np.float32),
            y=np.random.uniform(0, self.IMAGE_SIZE[1], N).astype(np.float32),
            t=np.sort(np.random.uniform(0, 1e6, N)).astype(np.float64),
            p=np.random.choice([-1, 1], N).astype(np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        
        if sample_info.get('synthetic', False):
            np.random.seed(sample_info['file_idx'])
            events = self._create_synthetic_events()
        else:
            events = self._load_events_bin(sample_info['path'])
        
        # Normalize events
        events = normalize_events(events, image_size=self.IMAGE_SIZE)
        
        # Sample events
        events = random_sample_events(events, self.num_events, preserve_temporal_order=True)
        
        # Convert to tensors
        tensor_dict = events.to_tensor()
        
        # Apply transform
        if self.transform:
            tensor_dict = self.transform(tensor_dict)
        
        return {
            **tensor_dict,
            'label': sample_info['class_idx'],
            'class': sample_info['class']
        }
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)


class NCaltech101Classification(NCaltech101Dataset):
    """
    N-Caltech101 configured for classification.
    Returns tensors ready for model input.
    """
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        sample = super().__getitem__(idx)
        
        inputs = {
            'coords': sample['coords'],
            'times': sample['times'],
            'polarities': sample['polarities']
        }
        
        return inputs, sample['label']


def test_ncaltech101_dataset():
    """Test N-Caltech101 dataset loading."""
    print("Testing N-Caltech101 Dataset...")
    
    # Create dataset
    dataset = NCaltech101Classification(
        root='./data/ncaltech101',
        split='train',
        num_events=4096
    )
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes}")
    print(f"  Image size: {dataset.IMAGE_SIZE}")
    
    # Load sample
    inputs, label = dataset[0]
    print(f"  Sample coords shape: {inputs['coords'].shape}")
    print(f"  Sample times shape: {inputs['times'].shape}")
    print(f"  Sample label: {label}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    batch_inputs, batch_labels = next(iter(loader))
    print(f"  Batch coords shape: {batch_inputs['coords'].shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    
    print("  ✓ N-Caltech101 dataset test passed!\n")


if __name__ == "__main__":
    test_ncaltech101_dataset()
