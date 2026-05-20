"""
prepare_datasets.py  (run from project root)

Organizes downloaded Kaggle datasets into the folder structure
that train_model.py expects:

  datasets/
    real_faces/          ← FaceForensics original frames
    deepfake_faces/      ← FaceForensics manipulated frames
    ai_generated/        ← CIFAKE FAKE images
    real_general/        ← CIFAKE REAL images

Usage:
  python prepare_datasets.py \
      --ff_root   path/to/faceforensics_dataset \
      --cifake_root path/to/cifake_dataset \
      --max_per_class 5000
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / 'datasets'
EXTENSIONS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(folder, max_count=None):
    """Recursively collect image paths from a folder"""
    paths = [
        p for p in Path(folder).rglob('*')
        if p.suffix.lower() in EXTENSIONS
    ]
    random.shuffle(paths)
    if max_count:
        paths = paths[:max_count]
    return paths


def copy_images(src_paths, dest_folder, prefix=''):
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, src in enumerate(src_paths):
        dest = dest_folder / f"{prefix}{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        copied += 1
        if (i + 1) % 500 == 0:
            print(f"  Copied {i+1}/{len(src_paths)}...")
    return copied


def prepare_faceforensics(ff_root, max_per_class, seed):
    """
    FaceForensics++ structure varies by download.
    Common structures:
      ff_root/original_sequences/  → real faces
      ff_root/manipulated_sequences/ → deepfakes
    Or flat:
      ff_root/real/
      ff_root/fake/
    """
    ff_root = Path(ff_root)
    random.seed(seed)

    # Try to detect structure
    real_candidates = [
        ff_root / 'original_sequences',
        ff_root / 'real',
        ff_root / 'Real',
        ff_root / 'original',
    ]
    fake_candidates = [
        ff_root / 'manipulated_sequences',
        ff_root / 'fake',
        ff_root / 'Fake',
        ff_root / 'manipulated',
        ff_root / 'Deepfakes',
    ]

    real_folder = next((p for p in real_candidates if p.exists()), None)
    fake_folder = next((p for p in fake_candidates if p.exists()), None)

    if real_folder is None or fake_folder is None:
        print(f"⚠️  Could not auto-detect FaceForensics structure in {ff_root}")
        print(f"    Found subdirs: {[d.name for d in ff_root.iterdir() if d.is_dir()]}")
        print(f"    Edit this script to point real_folder and fake_folder correctly.")
        return 0, 0

    print(f"\n📁 FaceForensics real:  {real_folder}")
    print(f"📁 FaceForensics fake:  {fake_folder}")

    real_imgs  = collect_images(real_folder, max_per_class)
    fake_imgs  = collect_images(fake_folder, max_per_class)

    r = copy_images(real_imgs, DATASET_DIR / 'real_faces', prefix='ff_real_')
    f = copy_images(fake_imgs, DATASET_DIR / 'deepfake_faces', prefix='ff_fake_')
    print(f"  ✅ Copied {r} real faces, {f} deepfake faces")
    return r, f


def prepare_cifake(cifake_root, max_per_class, seed):
    """
    CIFAKE structure:
      cifake_root/train/REAL/
      cifake_root/train/FAKE/
      cifake_root/test/REAL/
      cifake_root/test/FAKE/
    """
    cifake_root = Path(cifake_root)
    random.seed(seed + 1)

    real_paths = []
    fake_paths = []

    for split in ['train', 'test']:
        rp = cifake_root / split / 'REAL'
        fp = cifake_root / split / 'FAKE'
        if rp.exists():
            real_paths.extend(collect_images(rp))
        if fp.exists():
            fake_paths.extend(collect_images(fp))

    # Also try flat structure
    if not real_paths:
        for candidate in ['REAL', 'real', 'Real']:
            p = cifake_root / candidate
            if p.exists():
                real_paths = collect_images(p)
                break
    if not fake_paths:
        for candidate in ['FAKE', 'fake', 'Fake']:
            p = cifake_root / candidate
            if p.exists():
                fake_paths = collect_images(p)
                break

    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:max_per_class]
    fake_paths = fake_paths[:max_per_class]

    print(f"\n📁 CIFAKE real found:  {len(real_paths)} images")
    print(f"📁 CIFAKE fake found:  {len(fake_paths)} images")

    r = copy_images(real_paths, DATASET_DIR / 'real_general', prefix='cifake_real_')
    f = copy_images(fake_paths, DATASET_DIR / 'ai_generated', prefix='cifake_fake_')
    print(f"  ✅ Copied {r} real images, {f} AI-generated images")
    return r, f


def print_summary():
    print(f"\n{'='*55}")
    print(f"📊 DATASET SUMMARY")
    print(f"{'='*55}")
    folders = {
        'real_faces':     'Real (FaceForensics)',
        'deepfake_faces': 'Deepfake (FaceForensics)',
        'ai_generated':   'AI Generated (CIFAKE)',
        'real_general':   'Real General (CIFAKE)',
    }
    total = 0
    for folder, label in folders.items():
        p = DATASET_DIR / folder
        count = len(list(p.glob('*'))) if p.exists() else 0
        total += count
        print(f"  {label:<30} {count:>6} images")
    print(f"  {'─'*38}")
    print(f"  {'TOTAL':<30} {total:>6} images")
    print(f"{'='*55}\n")
    print("✅ Ready to train! Run:")
    print("   python src/model/train_model.py --epochs 25 --batch_size 16")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff_root',      type=str, default=None,
                        help='Path to FaceForensics++ dataset root')
    parser.add_argument('--cifake_root',  type=str, default=None,
                        help='Path to CIFAKE dataset root')
    parser.add_argument('--max_per_class', type=int, default=5000,
                        help='Max images per class (default 5000)')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    DATASET_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {DATASET_DIR}")

    if args.ff_root:
        prepare_faceforensics(args.ff_root, args.max_per_class, args.seed)
    else:
        print("⚠️  --ff_root not provided, skipping FaceForensics")

    if args.cifake_root:
        prepare_cifake(args.cifake_root, args.max_per_class, args.seed)
    else:
        print("⚠️  --cifake_root not provided, skipping CIFAKE")

    print_summary()