"""
train_unified.py

One training script that matches the model app.py actually loads, so the
accuracy you measure here is the accuracy you'll get in production —
no more mismatch between a Colab notebook, train_real_model.py, and the
deployed checkpoint.

- Architecture: EfficientNet-B0 + the exact classifier head in app.py's
  DeepfakeDetector (so state_dict keys line up with load_model()).
- Classes: ['ai_generated', 'deepfake', 'real'] — same order app.py expects.
- Data: reads datasets/organized_v2/{train,val,test}/<class>/ produced by
  prepare_dataset_v2.py (grouped split — no source-video leakage).
- Saves models/best_model.pth in the checkpoint format app.py's
  load_model() reads: {'model_state_dict', 'epoch', 'val_accuracy',
  'model_arch', 'class_names'}.
- Reports final metrics on `test/`, which the model never saw during
  training or model selection — that number is your real one.

Usage:
    python train_unified.py --epochs 25 --batch_size 32
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "organized_v2"
MODEL_OUT = PROJECT_ROOT / "models" / "best_model.pth"

CLASS_NAMES = ["ai_generated", "deepfake", "real"]  # must match app.py
IMG_SIZE = 224
RESIZE_TO = 232
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Model — copied to match app.py's DeepfakeDetector exactly
# ------------------------------------------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_f = self.backbone.classifier[1].in_features  # 1280 for B0
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2, inplace=True),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = Path(root) / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((p, idx))
        self.transform = transform
        counts = {c: sum(1 for _, l in self.samples if l == i) for i, c in enumerate(CLASS_NAMES)}
        print(f"  {root.name}: {len(self.samples)} images — {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def class_counts(self):
        counts = np.zeros(len(CLASS_NAMES))
        for _, label in self.samples:
            counts[label] += 1
        return counts


train_transform = transforms.Compose([
    transforms.Resize(RESIZE_TO),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    # Random JPEG-like compression / blur augmentation helps the model not
    # keying off compression artifacts as its only signal.
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(RESIZE_TO),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc="train" if is_train else "eval")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()
            pbar.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience on val loss")
    args = ap.parse_args()

    print(f"Using device: {device}\n")
    print("Loading datasets...")
    train_ds = ImageFolderDataset(DATA_ROOT / "train", train_transform)
    val_ds = ImageFolderDataset(DATA_ROOT / "val", eval_transform)
    test_ds = ImageFolderDataset(DATA_ROOT / "test", eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class-weighted loss in case classes are imbalanced after mixing sources
    counts = train_ds.class_counts()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    print(f"Class weights (imbalance correction): {dict(zip(CLASS_NAMES, weights.tolist()))}\n")

    model = DeepfakeDetector(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_acc / 100.0,
                "model_arch": "efficientnet_b0",
                "class_names": CLASS_NAMES,
            }, MODEL_OUT)
            print(f"  Saved new best checkpoint -> {MODEL_OUT} (val_acc={val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nNo val improvement for {args.patience} epochs — stopping early.")
                break

    # Final honest check: load the BEST checkpoint and evaluate on test/,
    # which was never used for training or model selection.
    print("\n" + "=" * 60)
    print("Held-out TEST evaluation (never seen during training/tuning)")
    print("=" * 60)
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    print(f"\nTest accuracy: {test_acc:.2f}%\n")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASS_NAMES)
    print(confusion_matrix(test_labels, test_preds))
    print(f"\nBest checkpoint saved to: {MODEL_OUT}")
    print("Restart the Flask app (or redeploy) to pick up the new checkpoint.")


if __name__ == "__main__":
    main()