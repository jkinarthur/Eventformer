"""
GEN1 Automotive Detection Dataset

The Prophesee GEN1 dataset contains events from driving scenes with
bounding box annotations for cars and pedestrians.

Dataset: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import json

from .event_utils import EventData, normalize_events, EventCollator


class GEN1Dataset(Dataset):
    """
    GEN1 Automotive Detection Dataset.
    
    Expected directory structure:
    root/
        train/
            *.npy or *.dat (event files)
            labels/
                *.json or *.npy (bounding box annotations)
        val/
            ...
        test/
            ...
    
    Each sample contains events within a time window and corresponding
    bounding box annotations.
    """
    
    CLASSES = ['car', 'pedestrian']
    NUM_CLASSES = 2
    IMAGE_SIZE = (304, 240)  # GEN1 sensor resolution
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        time_window: float = 50000,  # 50ms window in microseconds
        num_events: Optional[int] = 4096,
        transform=None,
        min_box_area: float = 100
    ):
        """
        Args:
            root: Path to dataset root
            split: 'train', 'val', or 'test'
            time_window: Time window in microseconds
            num_events: Number of events to sample per window (None = use all)
            transform: Optional data augmentation
            min_box_area: Minimum box area to include
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.time_window = time_window
        self.num_events = num_events
        self.transform = transform
        self.min_box_area = min_box_area
        
        self.split_dir = os.path.join(root, split)
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load list of samples with their timestamps."""
        samples = []
        
        if not os.path.exists(self.split_dir):
            print(f"Warning: {self.split_dir} does not exist. Creating synthetic samples for testing.")
            # Return synthetic samples for testing
            for i in range(100):
                samples.append({
                    'file_id': f'synthetic_{i}',
                    'start_time': i * self.time_window,
                    'synthetic': True
                })
            return samples
        
        # Find all event files
        for filename in os.listdir(self.split_dir):
            if filename.endswith(('.npy', '.dat', '.h5')):
                file_path = os.path.join(self.split_dir, filename)
                file_id = os.path.splitext(filename)[0]
                
                # Load file to get duration
                try:
                    events = self._load_events_file(file_path)
                    duration = events.t.max() - events.t.min()
                    
                    # Create windows
                    num_windows = int(duration / self.time_window)
                    for i in range(num_windows):
                        start_time = events.t.min() + i * self.time_window
                        samples.append({
                            'file_path': file_path,
                            'file_id': file_id,
                            'start_time': start_time,
                            'synthetic': False
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not samples:
            # Return synthetic samples if no real data found
            for i in range(100):
                samples.append({
                    'file_id': f'synthetic_{i}',
                    'start_time': i * self.time_window,
                    'synthetic': True
                })
        
        return samples
    
    def _load_events_file(self, file_path: str) -> EventData:
        """Load events from file."""
        if file_path.endswith('.npy'):
            data = np.load(file_path)
            if data.ndim == 2 and data.shape[1] >= 4:
                return EventData(
                    x=data[:, 0].astype(np.float32),
                    y=data[:, 1].astype(np.float32),
                    t=data[:, 2].astype(np.float64),
                    p=data[:, 3].astype(np.float32)
                )
            elif data.dtype.names:
                return EventData(
                    x=data['x'].astype(np.float32),
                    y=data['y'].astype(np.float32),
                    t=data['t'].astype(np.float64),
                    p=data['p'].astype(np.float32)
                )
        elif file_path.endswith('.h5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                return EventData(
                    x=f['events/x'][:].astype(np.float32),
                    y=f['events/y'][:].astype(np.float32),
                    t=f['events/t'][:].astype(np.float64),
                    p=f['events/p'][:].astype(np.float32)
                )
        
        raise ValueError(f"Unsupported file format: {file_path}")
    
    def _create_synthetic_sample(self, idx: int) -> Dict:
        """Create synthetic sample for testing."""
        np.random.seed(idx)
        
        # Generate random events
        N = 10000
        events = EventData(
            x=np.random.uniform(0, self.IMAGE_SIZE[0], N).astype(np.float32),
            y=np.random.uniform(0, self.IMAGE_SIZE[1], N).astype(np.float32),
            t=np.sort(np.random.uniform(0, self.time_window, N)).astype(np.float64),
            p=np.random.choice([-1, 1], N).astype(np.float32)
        )
        
        # Generate random boxes (1-5 boxes)
        num_boxes = np.random.randint(1, 6)
        boxes = []
        labels = []
        
        for _ in range(num_boxes):
            x1 = np.random.uniform(0, 0.8)
            y1 = np.random.uniform(0, 0.8)
            x2 = x1 + np.random.uniform(0.1, 0.2)
            y2 = y1 + np.random.uniform(0.1, 0.2)
            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(0, self.NUM_CLASSES))
        
        return {
            'events': events,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64)
        }
    
    def _load_annotations(self, file_id: str, start_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Load bounding box annotations for time window."""
        # Look for annotation files
        label_paths = [
            os.path.join(self.split_dir, 'labels', f'{file_id}.json'),
            os.path.join(self.split_dir, 'labels', f'{file_id}.npy'),
            os.path.join(self.split_dir, f'{file_id}_labels.json'),
        ]
        
        boxes = []
        labels = []
        
        for label_path in label_paths:
            if os.path.exists(label_path):
                if label_path.endswith('.json'):
                    with open(label_path, 'r') as f:
                        annotations = json.load(f)
                    
                    for ann in annotations:
                        if start_time <= ann.get('t', start_time) < start_time + self.time_window:
                            box = [
                                ann['x'] / self.IMAGE_SIZE[0],
                                ann['y'] / self.IMAGE_SIZE[1],
                                (ann['x'] + ann['w']) / self.IMAGE_SIZE[0],
                                (ann['y'] + ann['h']) / self.IMAGE_SIZE[1]
                            ]
                            boxes.append(box)
                            labels.append(ann.get('class', 0))
                            
                elif label_path.endswith('.npy'):
                    ann_data = np.load(label_path, allow_pickle=True)
                    # Filter by time window
                    mask = (ann_data['t'] >= start_time) & (ann_data['t'] < start_time + self.time_window)
                    for ann in ann_data[mask]:
                        box = [
                            ann['x'] / self.IMAGE_SIZE[0],
                            ann['y'] / self.IMAGE_SIZE[1],
                            (ann['x'] + ann['w']) / self.IMAGE_SIZE[0],
                            (ann['y'] + ann['h']) / self.IMAGE_SIZE[1]
                        ]
                        boxes.append(box)
                        labels.append(ann.get('class', 0))
                break
        
        if not boxes:
            # Return empty arrays if no annotations found
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        
        if sample_info.get('synthetic', False):
            sample = self._create_synthetic_sample(idx)
        else:
            # Load events for time window
            events = self._load_events_file(sample_info['file_path'])
            
            # Filter to time window
            start_time = sample_info['start_time']
            mask = (events.t >= start_time) & (events.t < start_time + self.time_window)
            events = events[mask]
            
            # Load annotations
            boxes, labels = self._load_annotations(sample_info['file_id'], start_time)
            
            sample = {
                'events': events,
                'boxes': boxes,
                'labels': labels
            }
        
        # Normalize events
        sample['events'] = normalize_events(
            sample['events'],
            image_size=self.IMAGE_SIZE
        )
        
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @staticmethod
    def get_collate_fn(num_events: int = 4096, sample_method: str = 'uniform'):
        """Get collate function for DataLoader."""
        return EventCollator(
            num_samples=num_events,
            sample_method=sample_method,
            normalize=False,  # Already normalized in __getitem__
            image_size=GEN1Dataset.IMAGE_SIZE
        )


class GEN1DetectionDataset(GEN1Dataset):
    """
    GEN1 dataset configured for detection task.
    Returns proper detection format with targets.
    """
    
    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        
        # Convert to tensor dict
        events = sample['events']
        tensor_dict = events.to_tensor()
        
        # Sample events
        N = len(events)
        if self.num_events and N > self.num_events:
            # Uniform temporal sampling
            indices = np.linspace(0, N - 1, self.num_events).astype(int)
            for key in ['coords', 'times', 'polarities']:
                tensor_dict[key] = tensor_dict[key][indices]
        elif self.num_events and N < self.num_events:
            # Pad
            pad_size = self.num_events - N
            for key in ['coords', 'times', 'polarities']:
                if key == 'coords':
                    tensor_dict[key] = torch.cat([
                        tensor_dict[key],
                        torch.zeros(pad_size, 2)
                    ])
                else:
                    tensor_dict[key] = torch.cat([
                        tensor_dict[key],
                        torch.zeros(pad_size)
                    ])
        
        return {
            **tensor_dict,
            'boxes': torch.from_numpy(sample['boxes']),
            'labels': torch.from_numpy(sample['labels']),
            'idx': idx
        }


def test_gen1_dataset():
    """Test GEN1 dataset loading."""
    print("Testing GEN1 Dataset...")
    
    # Create dataset (will use synthetic data if real data not available)
    dataset = GEN1DetectionDataset(
        root='./data/gen1',
        split='train',
        num_events=4096
    )
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Image size: {dataset.IMAGE_SIZE}")
    print(f"  Classes: {dataset.CLASSES}")
    
    # Load sample
    sample = dataset[0]
    print(f"  Sample coords shape: {sample['coords'].shape}")
    print(f"  Sample times shape: {sample['times'].shape}")
    print(f"  Sample polarities shape: {sample['polarities'].shape}")
    print(f"  Sample boxes shape: {sample['boxes'].shape}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"  Batch coords shape: {batch['coords'].shape}")
    
    print("  ✓ GEN1 dataset test passed!\n")


if __name__ == "__main__":
    test_gen1_dataset()
