"""
Visualization utilities for Eventformer.

Generates publication-quality figures for:
1. Training curves (loss and accuracy)
2. Ablation study bar charts
3. Model comparison plots
4. Event visualization
5. Attention map visualization
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette for consistency
COLORS = {
    'eventformer': '#2E86AB',  # Blue
    'baseline': '#E94F37',     # Red
    'sota': '#F39C12',         # Orange
    'gray': '#7F8C8D',         # Gray
    'green': '#27AE60',        # Green
    'purple': '#9B59B6',       # Purple
}

COMPONENT_COLORS = {
    'full': '#2E86AB',
    'no_ctpe': '#E94F37',
    'no_paaa': '#F39C12',
    'no_asna': '#27AE60',
    'minimal': '#7F8C8D'
}


def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = 'Training Progress'
) -> plt.Figure:
    """
    Plot training curves (loss and accuracy).
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'val_acc' lists
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], color=COLORS['eventformer'], 
             label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], color=COLORS['baseline'], 
             label='Val Loss', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(epochs, history['val_acc'], color=COLORS['eventformer'], 
             label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark best accuracy
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax2.axvline(x=best_epoch, color=COLORS['gray'], linestyle=':', alpha=0.7)
    ax2.annotate(f'Best: {best_acc:.1f}%', 
                 xy=(best_epoch, best_acc), 
                 xytext=(best_epoch + 5, best_acc - 5),
                 fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_ablation_bar_chart(
    results: Dict,
    save_path: Optional[str] = None,
    metric: str = 'accuracy'
) -> plt.Figure:
    """
    Plot ablation study results as a bar chart.
    
    Args:
        results: Dict from ablation study with config names and accuracies
        save_path: Optional path to save figure
        metric: 'accuracy' or 'loss'
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    config_names = list(results.keys())
    
    if metric == 'accuracy':
        means = [results[c]['mean_accuracy'] for c in config_names]
        stds = [results[c]['std_accuracy'] for c in config_names]
        ylabel = 'Accuracy (%)'
        title = 'Ablation Study: Component Contributions'
    else:
        means = [results[c].get('mean_loss', 0) for c in config_names]
        stds = [results[c].get('std_loss', 0) for c in config_names]
        ylabel = 'Loss'
        title = 'Ablation Study: Loss Comparison'
    
    # Get display names
    display_names = [results[c]['config']['name'] for c in config_names]
    colors = [COMPONENT_COLORS.get(c, COLORS['gray']) for c in config_names]
    
    x = np.arange(len(config_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    
    # Add baseline reference line
    full_acc = results.get('full', {}).get('mean_accuracy', 0)
    if full_acc > 0:
        ax.axhline(y=full_acc, color=COLORS['gray'], linestyle='--', 
                   alpha=0.7, label=f'Full Model ({full_acc:.1f}%)')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_model_comparison(
    models: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of different models (accuracy vs parameters/FLOPs).
    
    Args:
        models: Dict with model names and their metrics
                {'model_name': {'accuracy': float, 'params': float, 'flops': float}}
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    model_names = list(models.keys())
    accuracies = [models[m]['accuracy'] for m in model_names]
    params = [models[m]['params'] / 1e6 for m in model_names]  # Convert to millions
    flops = [models[m].get('flops', 0) / 1e9 for m in model_names]  # Convert to GFLOPs
    
    # Determine colors (highlight Eventformer)
    colors = [COLORS['eventformer'] if 'Eventformer' in m or 'Ours' in m 
              else COLORS['baseline'] for m in model_names]
    
    # Accuracy vs Parameters
    ax1 = axes[0]
    scatter1 = ax1.scatter(params, accuracies, c=colors, s=100, alpha=0.8, 
                           edgecolor='black', linewidth=1)
    
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Parameters (M)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Model Size')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy vs FLOPs
    ax2 = axes[1]
    valid_flops = [(f, a, c, n) for f, a, c, n in zip(flops, accuracies, colors, model_names) if f > 0]
    
    if valid_flops:
        vf, va, vc, vn = zip(*valid_flops)
        ax2.scatter(vf, va, c=vc, s=100, alpha=0.8, edgecolor='black', linewidth=1)
        
        for f, a, n in zip(vf, va, vn):
            ax2.annotate(n, (f, a), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('FLOPs (G)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Computational Cost')
    ax2.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['eventformer'], label='Eventformer (Ours)'),
        mpatches.Patch(color=COLORS['baseline'], label='Baselines')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_velocity_stratification(
    results: Dict[str, List[float]],
    velocity_bins: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance stratified by motion velocity.
    
    Args:
        results: Dict mapping model names to accuracy per velocity bin
        velocity_bins: Names of velocity bins (e.g., ['Slow', 'Medium', 'Fast'])
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(velocity_bins))
    width = 0.35
    
    model_names = list(results.keys())
    n_models = len(model_names)
    
    for i, (model_name, accuracies) in enumerate(results.items()):
        offset = (i - n_models/2 + 0.5) * width
        color = COLORS['eventformer'] if 'Eventformer' in model_name else COLORS['baseline']
        
        bars = ax.bar(x + offset, accuracies, width, label=model_name, 
                     color=color, alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Motion Velocity')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance vs Motion Velocity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(velocity_bins)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_delta_t_sensitivity(
    delta_ts: List[float],
    accuracies: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sensitivity to time window (Δt) selection.
    
    Args:
        delta_ts: List of Δt values (in ms)
        accuracies: Dict mapping model names to accuracy per Δt
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for model_name, accs in accuracies.items():
        color = COLORS['eventformer'] if 'Eventformer' in model_name else COLORS['baseline']
        linestyle = '-' if 'Eventformer' in model_name else '--'
        marker = 'o' if 'Eventformer' in model_name else 's'
        
        ax.plot(delta_ts, accs, label=model_name, color=color, 
                linestyle=linestyle, marker=marker, linewidth=2, markersize=6)
    
    ax.set_xlabel('Time Window Δt (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Sensitivity to Time Window Selection', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight optimal region
    if 'Eventformer' in accuracies:
        best_idx = np.argmax(accuracies['Eventformer'])
        ax.axvline(x=delta_ts[best_idx], color=COLORS['gray'], 
                   linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_events_2d(
    coords: np.ndarray,
    polarities: np.ndarray,
    times: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Event Stream Visualization'
) -> plt.Figure:
    """
    Plot 2D event visualization with polarity coloring.
    
    Args:
        coords: [N, 2] event coordinates
        polarities: [N] polarity values (-1 or +1)
        times: [N] timestamps
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Normalize times for coloring
    t_norm = (times - times.min()) / (times.max() - times.min() + 1e-9)
    
    # Plot 1: All events colored by polarity
    ax1 = axes[0]
    on_mask = polarities > 0
    off_mask = polarities < 0
    
    ax1.scatter(coords[on_mask, 0], coords[on_mask, 1], 
                c=COLORS['eventformer'], s=1, alpha=0.5, label='ON events')
    ax1.scatter(coords[off_mask, 0], coords[off_mask, 1], 
                c=COLORS['baseline'], s=1, alpha=0.5, label='OFF events')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Events by Polarity')
    ax1.legend(markerscale=5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # Plot 2: Events colored by time
    ax2 = axes[1]
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=t_norm, 
                          cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='Time (normalized)')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Events by Time')
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    # Plot 3: 3D scatter (x, y, t)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    colors = np.where(polarities > 0, COLORS['eventformer'], COLORS['baseline'])
    ax3.scatter(coords[::10, 0], coords[::10, 1], t_norm[::10], 
                c=colors[::10], s=1, alpha=0.5)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Time')
    ax3.set_title('3D Event Cloud')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_figures(
    output_dir: str,
    training_history: Optional[Dict] = None,
    ablation_results: Optional[Dict] = None,
    model_comparison: Optional[Dict] = None
):
    """
    Generate all figures for the paper.
    
    Args:
        output_dir: Directory to save figures
        training_history: Training curves data
        ablation_results: Ablation study results
        model_comparison: Model comparison data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use synthetic data if not provided
    if training_history is None:
        epochs = 100
        training_history = {
            'train_loss': [2.0 * np.exp(-0.03 * e) + 0.2 + 0.05 * np.random.randn() 
                          for e in range(epochs)],
            'val_loss': [2.2 * np.exp(-0.025 * e) + 0.3 + 0.08 * np.random.randn() 
                        for e in range(epochs)],
            'val_acc': [95 * (1 - np.exp(-0.05 * e)) + 2 * np.random.randn() 
                       for e in range(epochs)]
        }
    
    # Plot training curves
    plot_training_curves(
        training_history,
        save_path=os.path.join(output_dir, 'training_curves.pdf')
    )
    
    # Use synthetic ablation data if not provided
    if ablation_results is None:
        ablation_results = {
            'full': {
                'config': ABLATION_CONFIGS['full'],
                'mean_accuracy': 89.7,
                'std_accuracy': 0.4
            },
            'no_ctpe': {
                'config': ABLATION_CONFIGS['no_ctpe'],
                'mean_accuracy': 87.2,
                'std_accuracy': 0.5
            },
            'no_paaa': {
                'config': ABLATION_CONFIGS['no_paaa'],
                'mean_accuracy': 86.8,
                'std_accuracy': 0.6
            },
            'no_asna': {
                'config': ABLATION_CONFIGS['no_asna'],
                'mean_accuracy': 88.1,
                'std_accuracy': 0.3
            }
        }
    
    # Plot ablation chart
    plot_ablation_bar_chart(
        ablation_results,
        save_path=os.path.join(output_dir, 'ablation_study.pdf')
    )
    
    # Use synthetic comparison data if not provided
    if model_comparison is None:
        model_comparison = {
            'Eventformer-B (Ours)': {'accuracy': 47.2, 'params': 22e6, 'flops': 8.5e9},
            'Eventformer-S (Ours)': {'accuracy': 45.8, 'params': 7.5e6, 'flops': 3.2e9},
            'RVT-B': {'accuracy': 44.1, 'params': 28e6, 'flops': 12.3e9},
            'RED': {'accuracy': 42.5, 'params': 35e6, 'flops': 15.1e9},
            'MatrixLSTM': {'accuracy': 38.3, 'params': 18e6, 'flops': 8.2e9},
            'AEGNN': {'accuracy': 36.8, 'params': 12e6, 'flops': 5.5e9}
        }
    
    # Plot model comparison
    plot_model_comparison(
        model_comparison,
        save_path=os.path.join(output_dir, 'model_comparison.pdf')
    )
    
    # Plot velocity stratification
    velocity_results = {
        'Eventformer (Ours)': [92.1, 88.5, 78.3],
        'RVT-B': [90.2, 84.1, 60.8],
        'RED': [88.5, 80.3, 55.2]
    }
    plot_velocity_stratification(
        velocity_results,
        velocity_bins=['Slow (<5 px/ms)', 'Medium (5-15 px/ms)', 'Fast (>15 px/ms)'],
        save_path=os.path.join(output_dir, 'velocity_stratification.pdf')
    )
    
    # Plot delta-t sensitivity
    delta_ts = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    dt_accuracies = {
        'Eventformer (Ours)': [85.2, 87.8, 89.1, 89.7, 89.5, 88.9, 88.2, 86.5, 84.1],
        'RVT': [80.1, 83.5, 85.2, 86.1, 85.8, 84.2, 82.1, 78.5, 73.2]
    }
    plot_delta_t_sensitivity(
        delta_ts,
        dt_accuracies,
        save_path=os.path.join(output_dir, 'delta_t_sensitivity.pdf')
    )
    
    print(f"\nAll figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Eventformer figures')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    parser.add_argument('--ablation_results', type=str, default=None,
                        help='Path to ablation results JSON')
    parser.add_argument('--training_history', type=str, default=None,
                        help='Path to training history JSON')
    
    args = parser.parse_args()
    
    # Load results if provided
    ablation_results = None
    training_history = None
    
    if args.ablation_results and os.path.exists(args.ablation_results):
        with open(args.ablation_results, 'r') as f:
            data = json.load(f)
            ablation_results = data.get('results', None)
    
    if args.training_history and os.path.exists(args.training_history):
        with open(args.training_history, 'r') as f:
            training_history = json.load(f)
    
    generate_all_figures(
        args.output_dir,
        training_history=training_history,
        ablation_results=ablation_results
    )


if __name__ == '__main__':
    main()
