"""
Event data utilities for loading and preprocessing event camera data.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class EventData:
    """Container for event camera data."""
    x: np.ndarray  # [N] x coordinates
    y: np.ndarray  # [N] y coordinates
    t: np.ndarray  # [N] timestamps (microseconds)
    p: np.ndarray  # [N] polarities (+1/-1)
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx) -> 'EventData':
        return EventData(
            x=self.x[idx],
            y=self.y[idx],
            t=self.t[idx],
            p=self.p[idx]
        )
    
    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return {
            'coords': torch.stack([
                torch.from_numpy(self.x.astype(np.float32)),
                torch.from_numpy(self.y.astype(np.float32))
            ], dim=-1),
            'times': torch.from_numpy(self.t.astype(np.float32)),
            'polarities': torch.from_numpy(self.p.astype(np.float32))
        }


def load_events_from_npy(filepath: str) -> EventData:
    """
    Load events from numpy file.
    
    Expected format: [N, 4] array with columns (x, y, t, p)
    """
    data = np.load(filepath)
    
    if data.ndim == 2 and data.shape[1] == 4:
        return EventData(
            x=data[:, 0],
            y=data[:, 1],
            t=data[:, 2],
            p=data[:, 3]
        )
    else:
        raise ValueError(f"Expected shape [N, 4], got {data.shape}")


def load_events_from_h5(filepath: str, key: str = 'events') -> EventData:
    """
    Load events from HDF5 file.
    """
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        events = f[key]
        return EventData(
            x=events['x'][:].astype(np.float32),
            y=events['y'][:].astype(np.float32),
            t=events['t'][:].astype(np.float64),
            p=events['p'][:].astype(np.float32)
        )


def normalize_events(
    events: EventData,
    image_size: Tuple[int, int] = (346, 260),
    time_window: Optional[float] = None
) -> EventData:
    """
    Normalize event coordinates and timestamps.
    
    Args:
        events: Raw event data
        image_size: (width, height) for spatial normalization
        time_window: If provided, normalize time to [0, 1] within window
        
    Returns:
        Normalized events with:
        - x, y in [0, 1]
        - t in [0, 1] (if time_window provided) or seconds (if not)
        - p in {-1, +1}
    """
    x_norm = events.x / image_size[0]
    y_norm = events.y / image_size[1]
    
    if time_window is not None:
        t_norm = (events.t - events.t.min()) / (time_window + 1e-9)
    else:
        # Convert to seconds and normalize to [0, 1]
        t_norm = (events.t - events.t.min()) / (events.t.max() - events.t.min() + 1e-9)
    
    # Ensure polarity is in {-1, +1}
    p_norm = events.p.copy()
    if p_norm.min() >= 0:
        p_norm = 2 * p_norm - 1  # Convert from {0, 1} to {-1, +1}
    
    return EventData(
        x=x_norm.astype(np.float32),
        y=y_norm.astype(np.float32),
        t=t_norm.astype(np.float32),
        p=p_norm.astype(np.float32)
    )


def random_sample_events(
    events: EventData,
    num_samples: int,
    preserve_temporal_order: bool = True
) -> EventData:
    """
    Randomly sample events while optionally preserving temporal order.
    
    Args:
        events: Input events
        num_samples: Number of events to sample
        preserve_temporal_order: If True, sort by time after sampling
        
    Returns:
        Sampled events
    """
    N = len(events)
    
    if N <= num_samples:
        # Pad with zeros if not enough events
        pad_size = num_samples - N
        return EventData(
            x=np.pad(events.x, (0, pad_size), mode='constant'),
            y=np.pad(events.y, (0, pad_size), mode='constant'),
            t=np.pad(events.t, (0, pad_size), mode='constant'),
            p=np.pad(events.p, (0, pad_size), mode='constant')
        )
    
    indices = np.random.choice(N, num_samples, replace=False)
    
    if preserve_temporal_order:
        indices = np.sort(indices)
    
    return events[indices]


def uniform_temporal_sample(
    events: EventData,
    num_samples: int
) -> EventData:
    """
    Sample events uniformly across time.
    Ensures good temporal coverage of the event stream.
    
    Args:
        events: Input events
        num_samples: Number of events to sample
        
    Returns:
        Sampled events with good temporal coverage
    """
    N = len(events)
    
    if N <= num_samples:
        pad_size = num_samples - N
        return EventData(
            x=np.pad(events.x, (0, pad_size), mode='constant'),
            y=np.pad(events.y, (0, pad_size), mode='constant'),
            t=np.pad(events.t, (0, pad_size), mode='constant'),
            p=np.pad(events.p, (0, pad_size), mode='constant')
        )
    
    # Sort by time
    time_order = np.argsort(events.t)
    
    # Sample uniformly from sorted indices
    sample_indices = np.linspace(0, N - 1, num_samples).astype(int)
    indices = time_order[sample_indices]
    
    return events[indices]


def voxel_grid_sample(
    events: EventData,
    num_samples: int,
    num_bins: int = 10
) -> EventData:
    """
    Sample events using voxel grid (spatiotemporal binning).
    Ensures coverage across space and time.
    
    Args:
        events: Input events
        num_samples: Number of events to sample
        num_bins: Number of bins per dimension
        
    Returns:
        Sampled events
    """
    N = len(events)
    
    if N <= num_samples:
        pad_size = num_samples - N
        return EventData(
            x=np.pad(events.x, (0, pad_size), mode='constant'),
            y=np.pad(events.y, (0, pad_size), mode='constant'),
            t=np.pad(events.t, (0, pad_size), mode='constant'),
            p=np.pad(events.p, (0, pad_size), mode='constant')
        )
    
    # Normalize to [0, 1]
    x_norm = (events.x - events.x.min()) / (events.x.max() - events.x.min() + 1e-9)
    y_norm = (events.y - events.y.min()) / (events.y.max() - events.y.min() + 1e-9)
    t_norm = (events.t - events.t.min()) / (events.t.max() - events.t.min() + 1e-9)
    
    # Compute voxel indices
    x_bin = np.clip((x_norm * num_bins).astype(int), 0, num_bins - 1)
    y_bin = np.clip((y_norm * num_bins).astype(int), 0, num_bins - 1)
    t_bin = np.clip((t_norm * num_bins).astype(int), 0, num_bins - 1)
    
    voxel_idx = x_bin * num_bins * num_bins + y_bin * num_bins + t_bin
    
    # Sample from each voxel
    unique_voxels = np.unique(voxel_idx)
    samples_per_voxel = max(1, num_samples // len(unique_voxels))
    
    selected_indices = []
    for v in unique_voxels:
        mask = voxel_idx == v
        voxel_events = np.where(mask)[0]
        n_select = min(samples_per_voxel, len(voxel_events))
        selected_indices.extend(
            np.random.choice(voxel_events, n_select, replace=False).tolist()
        )
    
    # Trim or pad to exact size
    selected_indices = np.array(selected_indices)
    if len(selected_indices) > num_samples:
        selected_indices = np.random.choice(selected_indices, num_samples, replace=False)
    elif len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        available = np.setdiff1d(np.arange(N), selected_indices)
        selected_indices = np.concatenate([
            selected_indices,
            np.random.choice(available, remaining, replace=False)
        ])
    
    selected_indices = np.sort(selected_indices)
    
    return events[selected_indices]


def augment_events(
    events: EventData,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    flip_polarity: bool = False,
    time_jitter: float = 0.0,
    spatial_jitter: float = 0.0,
    drop_rate: float = 0.0,
    image_size: Tuple[int, int] = (346, 260)
) -> EventData:
    """
    Apply augmentations to events.
    
    Args:
        events: Input events
        flip_horizontal: Flip x coordinates
        flip_vertical: Flip y coordinates
        flip_polarity: Flip polarity values
        time_jitter: Standard deviation of temporal noise
        spatial_jitter: Standard deviation of spatial noise
        drop_rate: Probability of dropping each event
        image_size: Original image size for normalization
        
    Returns:
        Augmented events
    """
    x, y, t, p = events.x.copy(), events.y.copy(), events.t.copy(), events.p.copy()
    
    # Random event dropping
    if drop_rate > 0:
        mask = np.random.random(len(events)) > drop_rate
        x, y, t, p = x[mask], y[mask], t[mask], p[mask]
    
    # Spatial flips
    if flip_horizontal:
        x = image_size[0] - x - 1 if x.max() > 1 else 1 - x
    
    if flip_vertical:
        y = image_size[1] - y - 1 if y.max() > 1 else 1 - y
    
    # Polarity flip
    if flip_polarity:
        p = -p
    
    # Add jitter
    if time_jitter > 0:
        t = t + np.random.normal(0, time_jitter, len(t))
        t = np.clip(t, 0, None)
    
    if spatial_jitter > 0:
        x = x + np.random.normal(0, spatial_jitter, len(x))
        y = y + np.random.normal(0, spatial_jitter, len(y))
        x = np.clip(x, 0, image_size[0] if x.max() > 1 else 1)
        y = np.clip(y, 0, image_size[1] if y.max() > 1 else 1)
    
    return EventData(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        t=t.astype(np.float32),
        p=p.astype(np.float32)
    )


def events_to_frame(
    events: EventData,
    image_size: Tuple[int, int] = (346, 260),
    num_channels: int = 2
) -> np.ndarray:
    """
    Convert events to frame representation (for visualization/comparison).
    
    Args:
        events: Input events
        image_size: (width, height) of output frame
        num_channels: 2 for [ON, OFF], 1 for combined
        
    Returns:
        frame: [H, W, C] event frame
    """
    W, H = image_size
    
    # Scale coordinates if normalized
    x = events.x
    y = events.y
    if x.max() <= 1:
        x = (x * W).astype(int)
        y = (y * H).astype(int)
    else:
        x = x.astype(int)
        y = y.astype(int)
    
    # Clip to valid range
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    
    if num_channels == 2:
        frame = np.zeros((H, W, 2), dtype=np.float32)
        
        # ON events (positive polarity)
        on_mask = events.p > 0
        np.add.at(frame[:, :, 0], (y[on_mask], x[on_mask]), 1)
        
        # OFF events (negative polarity)
        off_mask = events.p < 0
        np.add.at(frame[:, :, 1], (y[off_mask], x[off_mask]), 1)
    else:
        frame = np.zeros((H, W, 1), dtype=np.float32)
        np.add.at(frame[:, :, 0], (y, x), events.p)
    
    return frame


def events_to_voxel_grid(
    events: EventData,
    image_size: Tuple[int, int] = (346, 260),
    num_bins: int = 5
) -> np.ndarray:
    """
    Convert events to voxel grid representation.
    
    Args:
        events: Input events
        image_size: (width, height)
        num_bins: Number of temporal bins
        
    Returns:
        voxel_grid: [num_bins, H, W] voxel grid
    """
    W, H = image_size
    
    x = events.x
    y = events.y
    t = events.t
    p = events.p
    
    # Normalize if needed
    if x.max() <= 1:
        x = (x * W).astype(int)
        y = (y * H).astype(int)
    else:
        x = x.astype(int)
        y = y.astype(int)
    
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    
    # Normalize time to [0, num_bins-1]
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    t_bin = np.clip((t_norm * (num_bins - 1)).astype(int), 0, num_bins - 1)
    
    voxel_grid = np.zeros((num_bins, H, W), dtype=np.float32)
    np.add.at(voxel_grid, (t_bin, y, x), p)
    
    return voxel_grid


class EventCollator:
    """
    Collate function for batching variable-length event sequences.
    """
    
    def __init__(
        self,
        num_samples: int = 4096,
        sample_method: str = 'uniform',  # 'random', 'uniform', 'voxel'
        normalize: bool = True,
        image_size: Tuple[int, int] = (346, 260)
    ):
        self.num_samples = num_samples
        self.sample_method = sample_method
        self.normalize = normalize
        self.image_size = image_size
        
    def __call__(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of event samples.
        
        Args:
            batch: List of dicts with 'events' key (EventData) and optional 'label', 'boxes'
            
        Returns:
            Batched tensors
        """
        coords_list = []
        times_list = []
        polarities_list = []
        labels = []
        boxes = []
        
        for sample in batch:
            events = sample['events']
            
            # Normalize
            if self.normalize:
                events = normalize_events(events, self.image_size)
            
            # Sample
            if self.sample_method == 'random':
                events = random_sample_events(events, self.num_samples)
            elif self.sample_method == 'uniform':
                events = uniform_temporal_sample(events, self.num_samples)
            elif self.sample_method == 'voxel':
                events = voxel_grid_sample(events, self.num_samples)
            
            tensor_dict = events.to_tensor()
            coords_list.append(tensor_dict['coords'])
            times_list.append(tensor_dict['times'])
            polarities_list.append(tensor_dict['polarities'])
            
            if 'label' in sample:
                labels.append(sample['label'])
            if 'boxes' in sample:
                boxes.append(sample['boxes'])
        
        result = {
            'coords': torch.stack(coords_list),
            'times': torch.stack(times_list),
            'polarities': torch.stack(polarities_list)
        }
        
        if labels:
            result['labels'] = torch.tensor(labels)
        if boxes:
            # Pad boxes to same length
            max_boxes = max(len(b) for b in boxes)
            padded_boxes = []
            for b in boxes:
                if len(b) < max_boxes:
                    b = np.pad(b, ((0, max_boxes - len(b)), (0, 0)), mode='constant')
                padded_boxes.append(torch.from_numpy(b))
            result['boxes'] = torch.stack(padded_boxes)
        
        return result


def test_event_utils():
    """Test event utilities."""
    print("Testing event utilities...")
    
    # Create synthetic events
    N = 10000
    events = EventData(
        x=np.random.randint(0, 346, N).astype(np.float32),
        y=np.random.randint(0, 260, N).astype(np.float32),
        t=np.sort(np.random.uniform(0, 1e6, N)).astype(np.float64),
        p=np.random.choice([-1, 1], N).astype(np.float32)
    )
    
    print(f"  Original events: {len(events)}")
    
    # Test normalization
    norm_events = normalize_events(events)
    print(f"  Normalized x range: [{norm_events.x.min():.3f}, {norm_events.x.max():.3f}]")
    print(f"  Normalized t range: [{norm_events.t.min():.3f}, {norm_events.t.max():.3f}]")
    
    # Test sampling
    sampled = random_sample_events(events, 4096)
    print(f"  Random sampled: {len(sampled)}")
    
    uniform = uniform_temporal_sample(events, 4096)
    print(f"  Uniform sampled: {len(uniform)}")
    
    voxel = voxel_grid_sample(events, 4096)
    print(f"  Voxel sampled: {len(voxel)}")
    
    # Test augmentation
    aug_events = augment_events(
        events,
        flip_horizontal=True,
        time_jitter=100.0,
        drop_rate=0.1
    )
    print(f"  Augmented events: {len(aug_events)}")
    
    # Test frame conversion
    frame = events_to_frame(events)
    print(f"  Frame shape: {frame.shape}")
    
    # Test voxel grid
    voxel_grid = events_to_voxel_grid(events)
    print(f"  Voxel grid shape: {voxel_grid.shape}")
    
    print("  ✓ All event utility tests passed!\n")


if __name__ == "__main__":
    test_event_utils()
