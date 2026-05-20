"""
src/model/train_model.py

Trains DeepfakeDetector on your unified_dataset:

  unified_dataset/
    train/
      real/           <- real face images
      ai_generated/   <- AI-generated images (CIFAKE)
      deepfake/       <- deepfake / manipulated faces

Labels:
  0 = Real Image
  1 = AI Generated
  2 = Deepfake

Usage:
  python src/model/train_model.py
  python src/model/train_model.py --epochs 25 --batch_size 16 --lr 0.0002
"""

import os
import sys
import argparse
import time
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────── PROJECT ROOT ──────────────────────
# train_model.py is at: src/model/train_model.py
# .parent = src/model   .parent.parent = src   .parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import get_model, save_model

# ─────────────────────────── CONFIG ────────────────────────────
# unified_dataset/train/ contains: real/  ai_generated/  deepfake/
DATASET_ROOT   = PROJECT_ROOT / 'unified_dataset' / 'train'
MODEL_SAVE_DIR = PROJECT_ROOT / 'models'
MODEL_SAVE_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

CLASS_NAMES = ['Real Image', 'AI Generated', 'Deepfake']


# ─────────────────────────── DATASET ───────────────────────────
class DeepfakeDataset(Dataset):
    """
    Reads images from labelled subfolders inside DATASET_ROOT.
    FOLDER_MAP maps label index to a list of subfolder names.
    Uses rglob so images inside nested subfolders are found too.
    """

    FOLDER_MAP = {
        0: ['real'],          # Real images
        1: ['ai_generated'],  # AI-generated images
        2: ['deepfake'],      # Deepfake / manipulated faces
    }

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root, split='train', val_split=0.15, seed=42):
        self.root      = Path(root)
        self.split     = split
        self.transform = self._get_transforms(split)
        self.samples   = []   # list of (path_str, label_int)

        rng = np.random.default_rng(seed)

        for label, folder_names in self.FOLDER_MAP.items():
            paths = []
            for folder in folder_names:
                folder_path = self.root / folder
                if folder_path.exists():
                    imgs = [
                        p for p in folder_path.rglob('*')
                        if p.suffix.lower() in self.EXTENSIONS
                    ]
                    paths.extend(imgs)
                else:
                    print(f"  [warn] Folder not found: {folder_path}")

            if not paths:
                print(f"WARNING: No images for class {label} ({CLASS_NAMES[label]})")
                print(f"         Looked in: {[str(self.root / f) for f in folder_names]}")
                continue

            # Deterministic shuffle then split
            paths  = sorted(paths)
            rng.shuffle(paths)
            n_val  = max(1, int(len(paths) * val_split))

            chosen = paths[:n_val] if split == 'val' else paths[n_val:]
            self.samples.extend([(str(p), label) for p in chosen])
            print(f"  Class {label} ({CLASS_NAMES[label]:>12}): {len(chosen):>7} images  [{split}]")

        if not self.samples:
            raise ValueError(
                f"\nNo images found in: {root}\n"
                f"Expected subfolders: real/  ai_generated/  deepfake/\n"
                f"Check that DATASET_ROOT points to the right place.\n"
                f"Current DATASET_ROOT = {DATASET_ROOT}"
            )

    @staticmethod
    def _get_transforms(split):
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), label
        except Exception:
            # Corrupted image fallback
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            return self.transform(img), label

    def get_class_weights(self):
        """Per-sample weights for WeightedRandomSampler to balance classes."""
        labels            = [s[1] for s in self.samples]
        counts            = np.bincount(labels, minlength=3)
        counts            = np.where(counts == 0, 1, counts)
        weights_per_class = 1.0 / counts
        sample_weights    = [weights_per_class[l] for l in labels]
        return torch.FloatTensor(sample_weights)


# ─────────────────────────── TRAINING ──────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1:>4}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100. * correct / total:.1f}%")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        loss           = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────── MAIN TRAIN FN ─────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  AURORA DEEPFAKE DETECTOR - TRAINING")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"{'='*60}\n")

    # ── Datasets ────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = DeepfakeDataset(DATASET_ROOT, split='train')
    val_ds   = DeepfakeDataset(DATASET_ROOT, split='val')
    print(f"\n  Total train: {len(train_ds):,} | Total val: {len(val_ds):,}\n")

    sampler = WeightedRandomSampler(
        weights     = train_ds.get_class_weights(),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = sampler,
        num_workers = args.workers,
        pin_memory  = (device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = (device.type == 'cuda'),
    )

    # ── Model ───────────────────────────────────────────────────
    print("Loading EfficientNet-B4 (pretrained ImageNet weights)...")
    model = get_model(num_classes=3, pretrained=True)
    model = model.to(device)

    # ── Loss / Optimizer / Scheduler ────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Training Loop ────────────────────────────────────────────
    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'─'*50}")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\n  Train -> Loss: {train_loss:.4f}  Acc: {100.*train_acc:.2f}%")
        print(f"  Val   -> Loss: {val_loss:.4f}  Acc: {100.*val_acc:.2f}%   [{elapsed:.0f}s]")
        print(f"  LR    : {scheduler.get_last_lr()[0]:.2e}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model whenever val accuracy improves
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(
                model,
                MODEL_SAVE_DIR / 'best_model.pth',
                optimizer = optimizer,
                epoch     = epoch,
                val_acc   = val_acc,
            )
            print(f"  *** New best model saved!  Val Acc: {100.*val_acc:.2f}%")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt = MODEL_SAVE_DIR / f'checkpoint_epoch{epoch}.pth'
            save_model(model, ckpt, epoch=epoch, val_acc=val_acc)
            print(f"  Checkpoint: {ckpt.name}")

    # ── Final evaluation & report ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!  Best Val Acc: {100.*best_val_acc:.2f}%")
    print(f"{'='*60}\n")

    model.load_state_dict(best_model_wts)
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    print("Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix  (rows = True class | cols = Predicted):")
    cm = confusion_matrix(final_labels, final_preds)
    header = "  ".join(f"{n[:9]:>9}" for n in CLASS_NAMES)
    print(f"{'':>14}  {header}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>14}  " + "  ".join(f"{v:>9}" for v in row))

    # Save history
    history_path = MODEL_SAVE_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved : {history_path}")

    # Save final model
    final_path = MODEL_SAVE_DIR / 'final_model.pth'
    save_model(model, final_path, epoch=args.epochs, val_acc=best_val_acc)
    print(f"Final model   : {final_path}")


# ─────────────────────────── ENTRY POINT ───────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Aurora Deepfake Detector')
    parser.add_argument('--epochs',     type=int,   default=25,
                        help='Training epochs (default: 25)')
    parser.add_argument('--batch_size', type=int,   default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr',         type=float, default=2e-4,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--workers',    type=int,   default=0,
                        help='DataLoader workers — keep 0 on Windows (default: 0)')
    args = parser.parse_args()

    train(args)