"""
fetch_diverse_ai_faces.py

Easier alternative to manually hunting down and downloading a Kaggle
dataset: pulls from a Hugging Face dataset that already combines Stable
Diffusion, MidJourney, and DALL-E generated images (KarmaLeo/Xiz9Dataset),
streams it (no full-dataset download needed), keeps only images that pass
a face-detection check (since the source dataset is general images, not
faces-only -- same issue as CIFAKE), and saves the result as a local
folder ready to feed into:

    python prepare_dataset_v2.py --extra_ai "diverse_ai_faces"

Usage:
    pip install datasets --break-system-packages
    python fetch_diverse_ai_faces.py --target_count 2000
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "diverse_ai_faces"

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def decode_image(img):
    """Handle both auto-decoded PIL Images and raw {'bytes':..., 'path':...} dicts
    that some HF datasets return in streaming mode."""
    if isinstance(img, dict) and "bytes" in img:
        import io
        return Image.open(io.BytesIO(img["bytes"]))
    return img


def has_face(img) -> bool:
    img = decode_image(img)
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    return len(faces) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_count", type=int, default=2000,
                    help="Stop once this many face-containing fake images are saved")
    ap.add_argument("--max_scan", type=int, default=20000,
                    help="Safety cap on how many source images to look through")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("The 'datasets' package isn't installed. Run:")
        print("  pip install datasets --break-system-packages")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to KarmaLeo/Xiz9Dataset on Hugging Face (streaming, no full download)...")
    ds = load_dataset("KarmaLeo/Xiz9Dataset", split="train", streaming=True)

    saved = 0
    scanned = 0
    fake_seen = 0
    no_face_skipped = 0
    error_count = 0

    print(f"Scanning for AI-generated (fake) images with a detectable face, "
          f"target={args.target_count}...\n", flush=True)

    for example in ds:
        scanned += 1

        if scanned == 1:
            # One-time debug print so we can see the real field names/values
            # this dataset uses, in case the label check below needs adjusting.
            debug_fields = {k: (type(v).__name__ if k == "image" else v) for k, v in example.items()}
            print(f"[debug] first row fields: {debug_fields}\n", flush=True)

        if scanned % 200 == 0:
            print(f"  ...scanned {scanned} rows so far "
                  f"(fake-labeled seen: {fake_seen}, no-face skipped: {no_face_skipped}, "
                  f"errors: {error_count}, saved: {saved})", flush=True)

        if scanned > args.max_scan:
            print(f"\nHit max_scan={args.max_scan} without reaching target -- stopping.")
            break

        label = example.get("label")
        # Dataset docs: label is 'fake' or 'real' (or 0/1 depending on version) -- handle both.
        is_fake = label in ("fake", 1, "1") if not isinstance(label, bool) else label

        if not is_fake:
            continue
        fake_seen += 1

        img = example.get("image")
        if img is None:
            continue

        try:
            if not has_face(img):
                no_face_skipped += 1
                continue
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"[debug] exception on row {scanned} (type={type(img).__name__}): "
                      f"{type(e).__name__}: {e}", flush=True)
            continue

        out_path = OUT_DIR / f"diverse_ai_{saved:06d}.jpg"
        decode_image(img).convert("RGB").save(out_path, quality=95)
        saved += 1

        if saved % 50 == 0:
            print(f"  Saved {saved}/{args.target_count} (scanned {scanned} source images)", flush=True)

        if saved >= args.target_count:
            break

    print(f"\nDone. Saved {saved} face-containing AI-generated images to {OUT_DIR}")
    print(f"(Scanned {scanned} source images total to find them.)")
    print("\nNext step:")
    print(f'  python prepare_dataset_v2.py --extra_ai "{OUT_DIR.name}"')


if __name__ == "__main__":
    main()