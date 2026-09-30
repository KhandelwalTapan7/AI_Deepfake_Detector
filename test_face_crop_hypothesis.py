"""
test_face_crop_hypothesis.py

Your real phone photos get called 'ai_generated' at 93-95% confidence, but
the model was trained only on TIGHT, UPRIGHT face crops (about 150-256 px).
Two things could make a normal phone photo look nothing like that to the
model, and both are about how the photo is FED to the model, not what the
model learned:

  1. EXIF ROTATION. Phones often store a portrait photo sideways plus a
     small "rotate me" tag. Windows Explorer applies the tag (so thumbnails
     look upright), but PIL's Image.open() does NOT. The model then sees a
     sideways person.

  2. FRAMING. test_ood_images.py (and probably app.py) does Resize + a
     center crop of the WHOLE photo. On a wide phone shot where the face
     is small, the center crop can be torso or background, not a face.

For each image this script prints the model's prediction under five
conditions, so you can see which fix (if any) changes the answer:

    AS-IS        exactly what test_ood_images.py does today
    EXIF-FIXED   same, but with the rotation tag applied first
    CROP x1.4    EXIF-fixed, then crop around the detected face (tight)
    CROP x1.8    ... medium margin (closest to the 256px training crops)
    CROP x2.2    ... loose margin

It also saves every face crop to ood_crops_debug/ so you can open them and
check that the detector found a real face and the crop looks like your
training images.

Usage:
    python test_face_crop_hypothesis.py --model models/deepfake_detector_export.pth --folder ood_test_images
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_SCALES = [1.4, 1.8, 2.2]      # crop side = detected face size x this
DETECT_MAX_SIDE = 1600             # downscale huge phone photos for detection
SHORT = {"ai_generated": "AI", "deepfake": "FAKE", "real": "REAL"}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ----------------------------------------------------------------------
# Image helpers (no torch needed)
# ----------------------------------------------------------------------
def load_image_both_ways(path):
    """Return (as_is, upright, orientation_tag).

    as_is  : what Image.open(...).convert('RGB') gives -- rotation tag ignored
    upright: same image with the EXIF rotation tag applied
    orientation_tag: raw EXIF value (None/1 = already upright, 6/8 = sideways,
                     3 = upside down)
    """
    raw = Image.open(path)
    orientation = raw.getexif().get(0x0112)
    upright = ImageOps.exif_transpose(raw).convert("RGB")
    as_is = raw.convert("RGB")
    return as_is, upright, orientation


def detect_largest_face(img):
    """Haar face detection on a downscaled copy. Returns (x, y, w, h) in the
    ORIGINAL image's pixel coordinates, or None if no face was found."""
    w, h = img.size
    scale = min(1.0, DETECT_MAX_SIDE / max(w, h))
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else img
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (x / scale, y / scale, fw / scale, fh / scale)


def crop_box(face, scale, img_w, img_h):
    """Square box centered on the face, side = face size x scale, shifted
    (never padded) so it stays inside the image."""
    x, y, fw, fh = face
    cx, cy = x + fw / 2, y + fh / 2
    side = min(max(fw, fh) * scale, img_w, img_h)
    left = min(max(cx - side / 2, 0), img_w - side)
    top = min(max(cy - side / 2, 0), img_h - side)
    return (int(left), int(top), int(left + side), int(top + side))


def guess_true_label(filename, class_names):
    """Class prefix at a word boundary only ('real_x.jpg', 'real x.jpg'),
    so 'realistic-...' does not count as 'real'."""
    stem = Path(filename).stem.lower()
    for name in sorted(class_names, key=len, reverse=True):
        n = name.lower()
        if stem == n:
            return name
        if stem.startswith(n) and len(stem) > len(n) and stem[len(n)] in "_- ":
            return name
    return None


# ----------------------------------------------------------------------
# Model helpers (torch imported lazily)
# ----------------------------------------------------------------------
def load_model(ckpt_path, device):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

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
    return model, transform, class_names


def predict(model, transform, img, device, class_names):
    import torch
    import torch.nn.functional as F
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])


def fmt(result):
    if result is None:
        return "no face"
    label, conf = result
    return f"{SHORT.get(label, label)} {conf * 100:.0f}%"


# ----------------------------------------------------------------------
def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/deepfake_detector_export.pth")
    ap.add_argument("--folder", type=str, default="ood_test_images")
    ap.add_argument("--debug_dir", type=str, default="ood_crops_debug")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not Path(args.model).exists() or not folder.exists():
        print("Check --model and --folder paths.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, class_names = load_model(args.model, device)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(exist_ok=True)

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not images:
        print(f"No images in {folder}")
        return

    cols = ["as-is", "exif-fixed"] + [f"crop x{s}" for s in CROP_SCALES]
    print(f"\n{'FILE':<32} {'EXIF':<5} " + " ".join(f"{c:<11}" for c in cols) + " TRUE")
    print("-" * (32 + 6 + 12 * len(cols) + 6))

    tally = {c: [0, 0, 0] for c in cols}   # correct, scored, no_face  (labeled files only)

    for path in images:
        try:
            as_is, upright, orientation = load_image_both_ways(path)
        except Exception as e:
            print(f"{path.name[:32]:<32} could not open: {e}")
            continue

        results = {
            "as-is": predict(model, transform, as_is, device, class_names),
            "exif-fixed": predict(model, transform, upright, device, class_names),
        }

        face = detect_largest_face(upright)
        for s in CROP_SCALES:
            key = f"crop x{s}"
            if face is None:
                results[key] = None
                continue
            crop = upright.crop(crop_box(face, s, *upright.size))
            crop.save(debug_dir / f"{path.stem}_crop{s}.jpg", quality=95)
            results[key] = predict(model, transform, crop, device, class_names)

        true_label = guess_true_label(path.name, class_names)
        if true_label:
            for c in cols:
                r = results[c]
                if r is None:
                    tally[c][2] += 1
                else:
                    tally[c][1] += 1
                    tally[c][0] += int(r[0] == true_label)

        exif_txt = str(orientation) if orientation not in (None, 1) else "-"
        print(f"{path.name[:32]:<32} {exif_txt:<5} "
              + " ".join(f"{fmt(results[c]):<11}" for c in cols)
              + f" {SHORT.get(true_label, '?') if true_label else '?'}")

    print("-" * (32 + 6 + 12 * len(cols) + 6))
    print("EXIF column: '-' = no rotation tag. 6 or 8 = photo is stored sideways,")
    print("3 = upside down. 'no face' = the detector found no face to crop.\n")

    if any(v[1] for v in tally.values()):
        print("Accuracy on files whose name starts with real / ai_generated / deepfake:")
        for c in cols:
            ok, scored, nf = tally[c]
            extra = f"  ({nf} had no face)" if nf else ""
            print(f"  {c:<11}: {ok}/{scored}{extra}")
        print()

    print("HOW TO READ THIS:")
    print(" - EXIF-fixed fixes the phone photos  -> rotation was the main culprit.")
    print("   app.py needs ImageOps.exif_transpose too (this hits real users' photos).")
    print(" - Only the crop columns fix them     -> framing was the culprit; the app")
    print("   should detect and crop the face before classifying.")
    print(" - Neither changes anything           -> the model itself needs more diverse")
    print("   real photos (then the FFHQ/CelebA plan is worth doing).")
    print(f"Open {debug_dir}/ to confirm the crops really are faces.")


if __name__ == "__main__":
    main()