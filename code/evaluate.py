"""
Evaluation script for Eventformer.

Provides comprehensive evaluation including:
- Standard metrics (accuracy, mAP)
- Per-class performance
- Confusion matrices
- Speed benchmarking
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EventformerForClassification, EventformerForDetection
from datasets import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Eventformer')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='ncaltech101',
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_events', type=int, default=4096,
                        help='Number of events per sample')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='./eval_results.json',
                        help='Output path for results')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--benchmark_iters', type=int, default=100,
                        help='Number of iterations for benchmark')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    args = checkpoint.get('args', {})
    model_configs = {
        'tiny': {'embed_dim': 32, 'depths': (2, 2, 4, 2), 'num_heads': (1, 2, 4, 8)},
        'small': {'embed_dim': 48, 'depths': (2, 2, 6, 2), 'num_heads': (2, 4, 6, 12)},
        'base': {'embed_dim': 64, 'depths': (2, 2, 8, 2), 'num_heads': (2, 4, 8, 16)}
    }
    
    model_size = args.get('model', 'tiny')
    config = model_configs.get(model_size, model_configs['tiny'])
    
    model = EventformerForClassification(
        num_classes=num_classes,
        use_ctpe=args.get('use_ctpe', True),
        use_paaa=args.get('use_paaa', True),
        use_asna=args.get('use_asna', True),
        **config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate classification model.
    
    Returns:
        Dict with accuracy, per-class accuracy, confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_time = 0
    num_samples = 0
    
    for inputs, labels in loader:
        coords = inputs['coords'].to(device)
        times = inputs['times'].to(device)
        polarities = inputs['polarities'].to(device)
        labels_np = labels.numpy()
        
        # Time the forward pass
        start = time.perf_counter()
        logits = model(coords, times, polarities)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        total_time += time.perf_counter() - start
        
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        all_probs.extend(probs.cpu().numpy())
        num_samples += len(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall accuracy
    accuracy = 100.0 * (all_preds == all_labels).mean()
    
    # Per-class accuracy
    num_classes = all_probs.shape[1]
    per_class_acc = {}
    per_class_count = {}
    
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = 100.0 * (all_preds[mask] == all_labels[mask]).mean()
            class_name = class_names[c] if class_names and c < len(class_names) else f'class_{c}'
            per_class_acc[class_name] = float(class_acc)
            per_class_count[class_name] = int(mask.sum())
    
    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label, pred] += 1
    
    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0
    
    for probs, label in zip(all_probs, all_labels):
        top_k = np.argsort(probs)[::-1]
        if label in top_k[:3]:
            top3_correct += 1
        if label in top_k[:5]:
            top5_correct += 1
    
    top3_acc = 100.0 * top3_correct / len(all_labels)
    top5_acc = 100.0 * top5_correct / len(all_labels)
    
    # Timing
    avg_time_per_sample = total_time / num_samples * 1000  # ms
    throughput = num_samples / total_time
    
    return {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'per_class_accuracy': per_class_acc,
        'per_class_count': per_class_count,
        'confusion_matrix': confusion_matrix.tolist(),
        'num_samples': num_samples,
        'avg_inference_time_ms': float(avg_time_per_sample),
        'throughput_samples_per_sec': float(throughput)
    }


@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    num_events: int,
    device: torch.device,
    batch_size: int = 1,
    num_iters: int = 100,
    warmup_iters: int = 10
) -> Dict:
    """
    Benchmark model inference speed.
    
    Returns:
        Dict with timing statistics
    """
    model.eval()
    
    # Create dummy input
    coords = torch.rand(batch_size, num_events, 2, device=device)
    times = torch.rand(batch_size, num_events, device=device).sort(dim=1)[0]
    polarities = torch.randint(0, 2, (batch_size, num_events), device=device).float() * 2 - 1
    
    # Warmup
    for _ in range(warmup_iters):
        _ = model(coords, times, polarities)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times_list = []
    
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = model(coords, times, polarities)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_list.append(time.perf_counter() - start)
    
    times_arr = np.array(times_list) * 1000  # Convert to ms
    
    return {
        'batch_size': batch_size,
        'num_events': num_events,
        'num_iterations': num_iters,
        'mean_time_ms': float(times_arr.mean()),
        'std_time_ms': float(times_arr.std()),
        'min_time_ms': float(times_arr.min()),
        'max_time_ms': float(times_arr.max()),
        'median_time_ms': float(np.median(times_arr)),
        'throughput_samples_per_sec': float(batch_size / (times_arr.mean() / 1000))
    }


def count_parameters(model: nn.Module) -> Dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by module
    by_module = {}
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            by_module[name] = params
    
    return {
        'total': total,
        'trainable': trainable,
        'by_module': by_module
    }


def print_results(results: Dict, class_names: Optional[List[str]] = None):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Accuracy
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    if 'top3_accuracy' in results:
        print(f"Top-3 Accuracy: {results['top3_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    
    # Timing
    if 'avg_inference_time_ms' in results:
        print(f"\nInference Time: {results['avg_inference_time_ms']:.2f} ms/sample")
        print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Per-class accuracy (top 5 best and worst)
    if 'per_class_accuracy' in results:
        print("\nPer-Class Accuracy:")
        sorted_classes = sorted(results['per_class_accuracy'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        print("  Best:")
        for name, acc in sorted_classes[:5]:
            count = results['per_class_count'].get(name, 0)
            print(f"    {name}: {acc:.1f}% (n={count})")
        
        print("  Worst:")
        for name, acc in sorted_classes[-5:]:
            count = results['per_class_count'].get(name, 0)
            print(f"    {name}: {acc:.1f}% (n={count})")
    
    print("\n" + "="*60)


def main():
    args = get_args()
    
    print("\n" + "="*60)
    print("Eventformer Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Device: CPU")
    
    # Load dataset
    print("\nLoading dataset...")
    data_path = os.path.join(args.data_root, args.dataset)
    
    test_dataset = get_dataset(
        args.dataset,
        data_path,
        split='test',
        num_events=args.num_events
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    num_classes = test_dataset.num_classes if hasattr(test_dataset, 'num_classes') else len(test_dataset.classes)
    class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else None
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, num_classes, device)
    
    # Count parameters
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_classification(model, test_loader, device, class_names)
    results['parameters'] = param_info
    
    # Speed benchmark
    if args.benchmark:
        print("\nRunning speed benchmark...")
        benchmark_results = benchmark_speed(
            model, args.num_events, device,
            batch_size=1, num_iters=args.benchmark_iters
        )
        results['benchmark'] = benchmark_results
        
        # Also benchmark with larger batch
        benchmark_batch = benchmark_speed(
            model, args.num_events, device,
            batch_size=args.batch_size, num_iters=args.benchmark_iters
        )
        results['benchmark_batched'] = benchmark_batch
    
    # Print results
    print_results(results, class_names)
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    main()
