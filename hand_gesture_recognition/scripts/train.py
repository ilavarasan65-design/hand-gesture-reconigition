"""
train.py — Training script for the hybrid gesture recognition model.

Usage:
    python scripts/train.py --config configs/asl_baseline.yaml
    python scripts/train.py --config configs/multilingual.yaml --wandb
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import GestureDataset
from src.models.fusion_model import GestureClassifier
from src.utils.metrics import compute_metrics, ConfusionTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device, use_image):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in tqdm(loader, desc='Train', leave=False):
        graph = batch['graph'].to(device)
        labels = batch['label'].to(device)
        lang_ids = batch['language_id'].to(device)
        images = batch['image'].to(device) if use_image else None

        optimizer.zero_grad()
        logits, _ = model(graph, images, lang_ids)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_image, num_classes):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    tracker = ConfusionTracker(num_classes)

    for batch in tqdm(loader, desc='Eval', leave=False):
        graph = batch['graph'].to(device)
        labels = batch['label'].to(device)
        lang_ids = batch['language_id'].to(device)
        images = batch['image'].to(device) if use_image else None

        logits, _ = model(graph, images, lang_ids)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        tracker.update(preds.cpu(), labels.cpu())

    metrics = compute_metrics(tracker)
    metrics['loss'] = total_loss / total_samples
    metrics['accuracy'] = total_correct / total_samples
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = GestureDataset(
        cfg['data']['train_csv'],
        augment=True,
        use_image=cfg['model'].get('use_image', True),
        languages=cfg['data'].get('languages'),
    )
    val_ds = GestureDataset(
        cfg['data']['val_csv'],
        augment=False,
        use_image=cfg['model'].get('use_image', True),
        languages=cfg['data'].get('languages'),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training'].get('num_workers', 4),
        collate_fn=GestureDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training'].get('num_workers', 4),
        collate_fn=GestureDataset.collate_fn,
    )

    print(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')
    print(f'Classes: {train_ds.num_classes}')

    # ── Model ──────────────────────────────────────────────────────────────
    model = GestureClassifier(
        num_classes=train_ds.num_classes,
        embed_dim=cfg['model'].get('embed_dim', 256),
        graph_type=cfg['model'].get('graph_type', 'gcn'),
        cnn_backbone=cfg['model'].get('cnn_backbone', 'mobilenet_v3'),
        use_image=cfg['model'].get('use_image', True),
        languages=cfg['data'].get('languages'),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params:,}')

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=cfg['training']['lr'] * 0.01,
    )

    # Class-weighted CE for imbalanced datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f'Resumed from epoch {start_epoch}')

    # W&B logging
    if args.wandb:
        import wandb
        wandb.init(project='hand-gesture-recognition', config=cfg)

    os.makedirs('checkpoints', exist_ok=True)
    use_image = cfg['model'].get('use_image', True)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['training']['epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_image)

        val_metrics = evaluate(
            model, val_loader, criterion, device, use_image, train_ds.num_classes)

        scheduler.step()

        print(
            f'Epoch {epoch+1:03d}/{cfg["training"]["epochs"]} | '
            f'Train loss: {train_loss:.4f} acc: {train_acc:.4f} | '
            f'Val loss: {val_metrics["loss"]:.4f} acc: {val_metrics["accuracy"]:.4f} | '
            f'F1: {val_metrics.get("macro_f1", 0):.4f}'
        )

        if args.wandb:
            import wandb
            wandb.log({'epoch': epoch + 1,
                       'train/loss': train_loss, 'train/acc': train_acc,
                       **{f'val/{k}': v for k, v in val_metrics.items()}})

        # Save best checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': cfg,
                'label_to_idx': train_ds.label_to_idx,
            }, 'checkpoints/best_model.pt')
            print(f'  ✓ Saved best model (val acc: {best_val_acc:.4f})')

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       f'checkpoints/epoch_{epoch+1:03d}.pt')

    print(f'\nTraining complete. Best val accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
