"""
Ablation study script for Eventformer.

Systematically evaluates the contribution of each component:
1. Full model (CTPE + PAAA + ASNA)
2. w/o CTPE (remove continuous-time positional encoding)
3. w/o PAAA (remove polarity-aware attention)
4. w/o ASNA (remove adaptive spatiotemporal neighborhood attention)
5. w/o Hierarchical (single-scale architecture)
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EventformerForClassification
from datasets import get_dataset


# Ablation configurations
ABLATION_CONFIGS = {
    'full': {
        'name': 'Full Model',
        'use_ctpe': True,
        'use_paaa': True,
        'use_asna': True,
        'description': 'Complete Eventformer with all components'
    },
    'no_ctpe': {
        'name': 'w/o CTPE',
        'use_ctpe': False,
        'use_paaa': True,
        'use_asna': True,
        'description': 'Without Continuous-Time Positional Encoding'
    },
    'no_paaa': {
        'name': 'w/o PAAA',
        'use_ctpe': True,
        'use_paaa': False,
        'use_asna': True,
        'description': 'Without Polarity-Aware Asymmetric Attention'
    },
    'no_asna': {
        'name': 'w/o ASNA',
        'use_ctpe': True,
        'use_paaa': True,
        'use_asna': False,
        'description': 'Without Adaptive Spatiotemporal Neighborhood Attention'
    },
    'minimal': {
        'name': 'Minimal',
        'use_ctpe': False,
        'use_paaa': False,
        'use_asna': False,
        'description': 'Basic transformer without novel components'
    }
}


def get_args():
    parser = argparse.ArgumentParser(description='Eventformer Ablation Study')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ncaltech101',
                        choices=['ncaltech101', 'dvs128_gesture'],
                        help='Dataset for ablation')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root')
    parser.add_argument('--num_events', type=int, default=4096,
                        help='Number of events per sample')
    
    # Model
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Model size for ablation')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per configuration')
    
    # Ablation selection
    parser.add_argument('--configs', type=str, nargs='+',
                        default=['full', 'no_ctpe', 'no_paaa', 'no_asna'],
                        choices=list(ABLATION_CONFIGS.keys()),
                        help='Ablation configurations to run')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed_base', type=int, default=42,
                        help='Base random seed')
    
    return parser.parse_args()


def get_model_config(model_size: str) -> Dict:
    """Get model configuration."""
    configs = {
        'tiny': {'embed_dim': 32, 'depths': (2, 2, 4, 2), 'num_heads': (1, 2, 4, 8)},
        'small': {'embed_dim': 48, 'depths': (2, 2, 6, 2), 'num_heads': (2, 4, 6, 12)},
        'base': {'embed_dim': 64, 'depths': (2, 2, 8, 2), 'num_heads': (2, 4, 8, 16)}
    }
    return configs[model_size]


def create_model(num_classes: int, model_size: str, ablation_config: Dict) -> nn.Module:
    """Create model with specific ablation configuration."""
    model_config = get_model_config(model_size)
    
    model = EventformerForClassification(
        num_classes=num_classes,
        use_ctpe=ablation_config['use_ctpe'],
        use_paaa=ablation_config['use_paaa'],
        use_asna=ablation_config['use_asna'],
        **model_config
    )
    
    return model


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    """Train for one epoch, return loss."""
    model.train()
    total_loss = 0
    
    for inputs, labels in loader:
        coords = inputs['coords'].to(device)
        times = inputs['times'].to(device)
        polarities = inputs['polarities'].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(coords, times, polarities)
        loss = criterion(logits, labels)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in loader:
        coords = inputs['coords'].to(device)
        times = inputs['times'].to(device)
        polarities = inputs['polarities'].to(device)
        labels = labels.to(device)
        
        logits = model(coords, times, polarities)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def run_single_experiment(
    config_name: str,
    ablation_config: Dict,
    args,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    run_idx: int
) -> Dict:
    """Run a single ablation experiment."""
    print(f"\n  Run {run_idx + 1}/{args.num_runs}")
    
    # Set seed for reproducibility
    seed = args.seed_base + run_idx
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = create_model(num_classes, args.model_size, ablation_config)
    model = model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_accs = []
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = evaluate(model, val_loader, args.device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Val Acc = {val_acc:.2f}% (Best: {best_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    return {
        'config_name': config_name,
        'run_idx': run_idx,
        'num_params': num_params,
        'best_accuracy': best_acc,
        'final_accuracy': val_accs[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'training_time': training_time
    }


def run_ablation_study(args):
    """Run complete ablation study."""
    print(f"\n{'='*60}")
    print("Eventformer Ablation Study")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model size: {args.model_size}")
    print(f"Configurations: {args.configs}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"{'='*60}\n")
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    data_path = os.path.join(args.data_root, args.dataset)
    
    train_dataset = get_dataset(args.dataset, data_path, split='train', num_events=args.num_events)
    val_dataset = get_dataset(args.dataset, data_path, split='test', num_events=args.num_events)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    num_classes = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else len(train_dataset.classes)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num classes: {num_classes}")
    
    # Run ablation experiments
    all_results = {}
    
    for config_name in args.configs:
        ablation_config = ABLATION_CONFIGS[config_name]
        print(f"\n{'='*40}")
        print(f"Configuration: {ablation_config['name']}")
        print(f"Description: {ablation_config['description']}")
        print(f"CTPE: {ablation_config['use_ctpe']}")
        print(f"PAAA: {ablation_config['use_paaa']}")
        print(f"ASNA: {ablation_config['use_asna']}")
        print(f"{'='*40}")
        
        run_results = []
        
        for run_idx in range(args.num_runs):
            result = run_single_experiment(
                config_name, ablation_config, args,
                train_loader, val_loader, num_classes, run_idx
            )
            run_results.append(result)
        
        # Aggregate results
        accuracies = [r['best_accuracy'] for r in run_results]
        times = [r['training_time'] for r in run_results]
        
        all_results[config_name] = {
            'config': ablation_config,
            'runs': run_results,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_time': np.mean(times),
            'num_params': run_results[0]['num_params']
        }
        
        print(f"\n  Results: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<20} {'Accuracy':<20} {'Δ vs Full':<15} {'Params':<10}")
    print(f"{'-'*60}")
    
    full_acc = all_results.get('full', {}).get('mean_accuracy', 0)
    
    for config_name in args.configs:
        result = all_results[config_name]
        acc_str = f"{result['mean_accuracy']:.2f}% ± {result['std_accuracy']:.2f}%"
        delta = result['mean_accuracy'] - full_acc
        delta_str = f"{delta:+.2f}%" if config_name != 'full' else "-"
        params_str = f"{result['num_params']/1e6:.2f}M"
        
        print(f"{result['config']['name']:<20} {acc_str:<20} {delta_str:<15} {params_str:<10}")
    
    print(f"{'='*60}\n")
    
    # Save results
    save_path = os.path.join(output_dir, 'ablation_results.json')
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(save_path, 'w') as f:
        json.dump(convert_for_json({
            'args': vars(args),
            'results': all_results
        }), f, indent=2)
    
    print(f"Results saved to: {save_path}")
    
    return all_results


def generate_ablation_table_latex(results: Dict) -> str:
    """Generate LaTeX table for ablation results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on component contributions.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & CTPE & PAAA & ASNA & Accuracy (\%) \\",
        r"\midrule"
    ]
    
    for config_name, result in results.items():
        config = result['config']
        ctpe = r"\checkmark" if config['use_ctpe'] else ""
        paaa = r"\checkmark" if config['use_paaa'] else ""
        asna = r"\checkmark" if config['use_asna'] else ""
        acc = f"{result['mean_accuracy']:.2f} ± {result['std_accuracy']:.2f}"
        
        lines.append(f"{config['name']} & {ctpe} & {paaa} & {asna} & {acc} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


if __name__ == '__main__':
    args = get_args()
    results = run_ablation_study(args)
    
    # Generate LaTeX table
    latex_table = generate_ablation_table_latex(results)
    print("\nLaTeX Table:")
    print(latex_table)
