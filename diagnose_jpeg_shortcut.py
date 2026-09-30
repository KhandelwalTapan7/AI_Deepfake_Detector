"""
diagnose_jpeg_shortcut.py

Tests a specific theory: fetch_diverse_ai_faces.py saved every diverse
AI-generated image as JPEG at a uniform quality=95. If that shared
recompression fingerprint -- rather than actual image content -- is what
the model learned to associate with "ai_generated", then re-saving a
REAL photo at that same quality=95 (with zero other change) should push
the model's prediction toward ai_generated, the same way the earlier
resolution-degradation test proved the resolution theory one way or the
other.

Usage:
    python diagnose_jpeg_shortcut.py --model models/deepfake_detector_export.pth --folder ood_test_images
"""

import io
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DeepfakeDetector(nn.Module):
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


def recompress_jpeg_q95(img: Image.Image) -> Image.Image:
    """Re-save through an in-memory JPEG at quality=95, exactly matching
    what fetch_diverse_ai_faces.py did to every diverse AI image."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf)


def predict(model, transform, img, device, class_names):
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx]), probs.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/deepfake_detector_export.pth")
    ap.add_argument("--folder", type=str, default="ood_test_images")
    args = ap.parse_args()

    ckpt_path = Path(args.model)
    folder = Path(args.folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    model = DeepfakeDetector(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 232 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Only test images whose filename marks them as real -- these are the
    # ones that flipped to ai_generated after the retrain.
    images = sorted([p for p in folder.iterdir()
                      if p.suffix.lower() in EXTENSIONS and "real" in p.stem.lower()])

    if not images:
        print(f"No filenames containing 'real' found in {folder} -- "
              "point this at your real-labeled OOD photos.")
        return

    print(f"Testing {len(images)} real-labeled images: original vs. re-encoded at JPEG q95\n")
    print(f"{'FILE':<35} {'ORIGINAL':<20} {'RE-ENCODED q95':<20} {'SHIFT'}")
    print("-" * 95)

    flips_to_ai = 0
    for path in images:
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"{path.name}: could not open ({e})")
            continue

        orig_label, orig_conf, orig_probs = predict(model, transform, img, device, class_names)
        recompressed = recompress_jpeg_q95(img)
        new_label, new_conf, new_probs = predict(model, transform, recompressed, device, class_names)

        shift = ""
        if orig_label != "ai_generated" and new_label == "ai_generated":
            shift = "FLIPPED TO ai_generated"
            flips_to_ai += 1
        elif orig_label != new_label:
            shift = f"changed ({orig_label} -> {new_label})"
        else:
            shift = "no change"

        print(f"{path.name:<35} {orig_label + f' {orig_conf*100:.0f}%':<20} "
              f"{new_label + f' {new_conf*100:.0f}%':<20} {shift}")

    print("-" * 95)
    print(f"\nImages that flipped to 'ai_generated' purely from JPEG q95 re-encoding: "
          f"{flips_to_ai}/{len(images)}")
    print("If this is high, it confirms the JPEG-recompression-shortcut theory --")
    print("the model is keying off how the diverse training images were saved,")
    print("not their actual content.")


if __name__ == "__main__":
    main()