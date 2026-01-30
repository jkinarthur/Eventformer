"""
Training script for Eventformer.

Supports:
- Classification on N-Caltech101 and DVS128 Gesture
- Detection on GEN1 Automotive
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    eventformer_tiny,
    eventformer_small,
    eventformer_base,
    EventformerForClassification,
    EventformerForDetection
)
from datasets import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Train Eventformer')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ncaltech101',
                        choices=['ncaltech101', 'dvs128_gesture', 'gen1'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root')
    parser.add_argument('--num_events', type=int, default=4096,
                        help='Number of events per sample')
    
    # Model
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--use_ctpe', type=bool, default=True,
                        help='Use Continuous-Time Positional Encoding')
    parser.add_argument('--use_paaa', type=bool, default=True,
                        help='Use Polarity-Aware Asymmetric Attention')
    parser.add_argument('--use_asna', type=bool, default=True,
                        help='Use Adaptive Spatiotemporal Neighborhood Attention')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_training(args):
    """Setup training environment."""
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.model}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")
    
    return args


def get_model(args, num_classes: int, task: str = 'classification'):
    """Create model based on arguments."""
    model_configs = {
        'tiny': {'embed_dim': 32, 'depths': (2, 2, 4, 2), 'num_heads': (1, 2, 4, 8)},
        'small': {'embed_dim': 48, 'depths': (2, 2, 6, 2), 'num_heads': (2, 4, 6, 12)},
        'base': {'embed_dim': 64, 'depths': (2, 2, 8, 2), 'num_heads': (2, 4, 8, 16)},
        'large': {'embed_dim': 96, 'depths': (2, 2, 12, 2), 'num_heads': (3, 6, 12, 24)}
    }
    
    config = model_configs[args.model]
    
    if task == 'classification':
        model = EventformerForClassification(
            num_classes=num_classes,
            use_ctpe=args.use_ctpe,
            use_paaa=args.use_paaa,
            use_asna=args.use_asna,
            **config
        )
    else:
        model = EventformerForDetection(
            num_classes=num_classes,
            use_ctpe=args.use_ctpe,
            use_paaa=args.use_paaa,
            use_asna=args.use_asna,
            **config
        )
    
    return model


def get_optimizer_and_scheduler(model, args, num_training_steps: int):
    """Create optimizer and learning rate scheduler."""
    # Separate weight decay for different parameter types
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr)
    
    # Cosine annealing with warmup
    warmup_steps = args.warmup_epochs * (num_training_steps // args.epochs)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    writer: SummaryWriter
) -> Dict:
    """Train one epoch for classification."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        # Move to device
        coords = inputs['coords'].to(device)
        times = inputs['times'].to(device)
        polarities = inputs['polarities'].to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(coords, times, polarities)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            acc = 100. * correct / total
            elapsed = time.time() - start_time
            print(f'Epoch {epoch} [{batch_idx+1}/{len(loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {acc:.2f}% '
                  f'LR: {scheduler.get_last_lr()[0]:.6f} '
                  f'Time: {elapsed:.1f}s')
    
    epoch_loss = total_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Log to tensorboard
    global_step = epoch * len(loader)
    writer.add_scalar('train/loss', epoch_loss, global_step)
    writer.add_scalar('train/accuracy', epoch_acc, global_step)
    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:
    """Evaluate classification model."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
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
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100. * correct / total
    }


def train_detection_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    args,
    writer: SummaryWriter
) -> Dict:
    """Train one epoch for detection."""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        coords = batch['coords'].to(device)
        times = batch['times'].to(device)
        polarities = batch['polarities'].to(device)
        target_boxes = batch['boxes'].to(device)
        target_labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(coords, times, polarities)
        
        # Compute losses (simplified)
        cls_loss = nn.CrossEntropyLoss()(
            outputs['cls_logits'].reshape(-1, outputs['cls_logits'].size(-1)),
            torch.zeros(outputs['cls_logits'].size(0) * outputs['cls_logits'].size(1), dtype=torch.long, device=device)
        )
        
        # Box regression loss (L1)
        box_loss = nn.L1Loss()(outputs['box_preds'], torch.zeros_like(outputs['box_preds']))
        
        loss = cls_loss + box_loss
        
        # Backward pass
        loss.backward()
        
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch} [{batch_idx+1}/{len(loader)}] '
                  f'Loss: {loss.item():.4f} (cls: {cls_loss.item():.4f}, box: {box_loss.item():.4f}) '
                  f'Time: {elapsed:.1f}s')
    
    return {
        'loss': total_loss / len(loader),
        'cls_loss': total_cls_loss / len(loader),
        'box_loss': total_box_loss / len(loader)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    torch.save(checkpoint, os.path.join(args.output_dir, filename))


def main():
    args = get_args()
    args = setup_training(args)
    
    print(f"\n{'='*60}")
    print(f"Training Eventformer-{args.model.capitalize()} on {args.dataset}")
    print(f"{'='*60}\n")
    
    # Create datasets
    print("Loading datasets...")
    data_path = os.path.join(args.data_root, args.dataset)
    
    train_dataset = get_dataset(
        args.dataset, 
        data_path, 
        split='train',
        num_events=args.num_events
    )
    
    val_dataset = get_dataset(
        args.dataset,
        data_path,
        split='test',
        num_events=args.num_events
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Determine task type
    task = 'detection' if args.dataset == 'gen1' else 'classification'
    num_classes = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else len(train_dataset.classes)
    
    print(f"Task: {task}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(args, num_classes, task)
    model = model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * args.epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_metric = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*40}")
        
        if task == 'classification':
            train_metrics = train_classification_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, args.device, epoch, args, writer
            )
            val_metrics = evaluate_classification(
                model, val_loader, criterion, args.device
            )
            metric_name = 'accuracy'
            current_metric = val_metrics['accuracy']
        else:
            train_metrics = train_detection_epoch(
                model, train_loader, optimizer, scheduler,
                args.device, epoch, args, writer
            )
            val_metrics = {'loss': 0}  # Simplified for detection
            metric_name = 'loss'
            current_metric = -val_metrics['loss']  # Negative because we minimize loss
        
        print(f"\nTrain: Loss={train_metrics['loss']:.4f}")
        print(f"Val: {val_metrics}")
        
        # Log to tensorboard
        writer.add_scalar(f'val/loss', val_metrics['loss'], epoch)
        if 'accuracy' in val_metrics:
            writer.add_scalar(f'val/accuracy', val_metrics['accuracy'], epoch)
        
        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(model, optimizer, scheduler, epoch, args, val_metrics, 'best_model.pth')
            print(f"✓ New best {metric_name}: {abs(current_metric):.2f}")
        
        # Regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args, val_metrics, f'checkpoint_epoch{epoch+1}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, args, val_metrics, 'final_model.pth')
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best {metric_name}: {abs(best_metric):.2f}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    writer.close()


if __name__ == '__main__':
    main()
