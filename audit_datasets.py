"""
audit_datasets.py

Run this first. It answers "what do I actually have?" without you having
to remember or screenshot folders.

For each known dataset root it reports:
  - total image count
  - whether it's GROUPED (subfolders containing images -> e.g. one folder
    per source video/identity) or FLAT (images directly in the folder,
    no grouping info)
  - if grouped: how many groups, and min/max/avg images per group
  - a face-likelihood check: opens a small random sample of images and
    runs OpenCV's Haar face detector + checks image size, to flag any
    folder that looks like it's NOT human faces (e.g. the CIFAKE
    animals/vehicles set we found in `AI Generated dataset`)

Nothing is copied or modified — this is read-only.

Usage:
    python audit_datasets.py
"""

import random
from pathlib import Path
from collections import defaultdict

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Known roots from what we've found so far — edit/add paths as needed.
# Each entry: (label, path relative to project root)
CANDIDATE_ROOTS = [
    ("unified_dataset/train/ai_generated", "unified_dataset/train/ai_generated"),
    ("unified_dataset/train/deepfake", "unified_dataset/train/deepfake"),
    ("unified_dataset/train/real", "unified_dataset/train/real"),
    ("Deepfake dataset/cropped_images", "Deepfake dataset/cropped_images"),
    ("AI Generated dataset/train", "AI Generated dataset/train"),
    ("AI Generated dataset/test", "AI Generated dataset/test"),
    ("real_img", "real_img"),
]

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SAMPLE_SIZE = 40  # images sampled per root for the face-likelihood check


def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS]


def is_grouped(root: Path, images):
    """Grouped = images live inside subfolders of root, not directly in root."""
    direct = [p for p in images if p.parent == root]
    return len(direct) < len(images) * 0.5  # majority live in subfolders


def group_stats(root: Path, images):
    groups = defaultdict(int)
    for p in images:
        groups[p.parent] += 1
    counts = list(groups.values())
    return len(groups), min(counts), max(counts), sum(counts) / len(counts)


def face_likelihood(images):
    """Sample images, check average resolution and Haar face-detection hit rate."""
    if not images:
        return None
    sample = random.sample(images, min(SAMPLE_SIZE, len(images)))
    face_hits = 0
    sizes = []
    checked = 0
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            face_hits += 1
        checked += 1
    if checked == 0:
        return None
    avg_w = sum(w for w, h in sizes) / len(sizes)
    avg_h = sum(h for w, h in sizes) / len(sizes)
    return {
        "checked": checked,
        "face_hit_rate": face_hits / checked,
        "avg_resolution": (round(avg_w), round(avg_h)),
    }


def main():
    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)

    for label, rel_path in CANDIDATE_ROOTS:
        root = PROJECT_ROOT / rel_path
        if not root.exists():
            print(f"\n[{label}] -- not found at {root}, skipping")
            continue

        images = collect_images(root)
        if not images:
            print(f"\n[{label}] -- 0 images found under {root}")
            continue

        grouped = is_grouped(root, images)
        print(f"\n[{label}]")
        print(f"  path        : {root}")
        print(f"  total images: {len(images)}")

        if grouped:
            n_groups, mn, mx, avg = group_stats(root, images)
            print(f"  structure   : GROUPED — {n_groups} subfolders "
                  f"(min={mn}, max={mx}, avg={avg:.1f} images/group)")
            print("  -> safe to split by subfolder name to avoid leakage")
        else:
            print("  structure   : FLAT — no subfolder grouping detected")
            print("  -> no grouping info available; treat each image as its own group,")
            print("     or reconstruct grouping via near-duplicate detection if frames")
            print("     from the same source might be present")

        fl = face_likelihood(images)
        if fl:
            verdict = "LIKELY FACES" if fl["face_hit_rate"] >= 0.3 else "LIKELY *NOT* FACES"
            print(f"  face check  : {verdict} — Haar face hit rate "
                  f"{fl['face_hit_rate']*100:.0f}% on {fl['checked']} sampled images, "
                  f"avg resolution {fl['avg_resolution']}")
            if fl["face_hit_rate"] < 0.3:
                print("  ⚠️  WARNING: this folder does not look like human face data.")
                print("      Do not feed it into a face-forgery classifier's classes.")

    print("\n" + "=" * 70)
    print("Read the ⚠️ warnings above before running prepare_dataset_v2.py.")
    print("Any FLAT source with no grouping and high per-image similarity risk")
    print("(e.g. burst-extracted video frames) should be treated cautiously.")
    print("=" * 70)


if __name__ == "__main__":
    main()