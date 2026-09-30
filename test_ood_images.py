"""
test_ood_images.py

Runs your trained checkpoint against a folder of images it has NEVER seen
during training or the held-out test split -- real photos, AI-generated
faces from generators it wasn't trained on, deepfakes from a different
source dataset. This is the test that actually tells you whether the
99.11% held-out number reflects real generalization or just familiarity
with your specific training sources.

SETUP:
  1. Create a folder, e.g. ood_test_images/
  2. Drop in your test images. Optionally name them with a true-class
     prefix so this script can also compute accuracy:
        real_anything.jpg          -> true class 'real'
        ai_generated_anything.png  -> true class 'ai_generated'
        deepfake_anything.jpg      -> true class 'deepfake'
     Any filename that doesn't match a known prefix is still scored and
     shown, just without a right/wrong marker.

USAGE:
    python test_ood_images.py --model models/deepfake_detector_export.pth --folder ood_test_images
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DeepfakeDetector(nn.Module):
    """Must match the architecture used in train_unified.py / the Colab notebook
    exactly, so the state_dict loads cleanly."""
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def guess_true_label(filename: str, class_names):
    lower = filename.lower()
    # Check longer/more specific names first so "ai_generated" doesn't get
    # accidentally matched by a shorter unrelated prefix.
    for name in sorted(class_names, key=len, reverse=True):
        if lower.startswith(name.lower()):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/deepfake_detector_export.pth")
    ap.add_argument("--folder", type=str, default="ood_test_images")
    args = ap.parse_args()

    ckpt_path = Path(args.model)
    folder = Path(args.folder)

    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
    if not folder.exists():
        print(f"Image folder not found at {folder} -- create it and add some test images first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    print(f"Classes      : {class_names}")
    print(f"Held-out acc reported at export time: "
          f"val={ckpt.get('val_accuracy', 0)*100:.2f}%  test={ckpt.get('test_accuracy', 0)*100:.2f}%\n")

    model = DeepfakeDetector(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 232 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS])
    if not images:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(images)} images in {folder}\n")
    print(f"{'FILE':<35} {'PREDICTED':<15} {'CONF':<8} {'TRUE':<15} {'RESULT'}")
    print("-" * 90)

    correct, scored = 0, 0

    for path in images:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"{path.name:<35} -- could not open image: {e}")
            continue

        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax())
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        true_label = guess_true_label(path.name, class_names)
        result = ""
        if true_label:
            scored += 1
            is_correct = (true_label == pred_label)
            correct += int(is_correct)
            result = "correct" if is_correct else "WRONG"

        print(f"{path.name:<35} {pred_label:<15} {confidence*100:>5.1f}%  "
              f"{(true_label or '-'):<15} {result}")

        # Show full per-class probability breakdown for transparency
        breakdown = "  ".join(f"{c}={float(probs[i])*100:.1f}%" for i, c in enumerate(class_names))
        print(f"{'':<35} {breakdown}")

    print("-" * 90)
    if scored:
        print(f"\nAccuracy on {scored} labeled OOD images: {correct}/{scored} "
              f"({100*correct/scored:.1f}%)")
        print("Compare this to the 99.11% held-out test accuracy -- a big gap here")
        print("confirms the model isn't generalizing beyond its training sources.")
    else:
        print("\nNo filenames matched a known class prefix -- rename files as")
        print("real_*, ai_generated_*, or deepfake_* to get an accuracy summary,")
        print("or just eyeball the per-image predictions above.")


if __name__ == "__main__":
    main()