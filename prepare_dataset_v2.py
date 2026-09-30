"""
prepare_dataset_v2.py  (v2 — matches your actual folder layout)

Builds a leakage-safe train/val/test split from your real data, based on
what audit_datasets.py found:

  - Deepfake dataset/cropped_images/<video_id>/*.png
        GROUPED — one folder per source video/identity. Split by folder
        name so all frames of one video/identity land in exactly one of
        train/val/test.

  - unified_dataset/train/ai_generated/*.jpg
        FLAT — single synthetic face portraits, no subfolders. Each image
        treated as its own group (no known shared source to leak across).

  - unified_dataset/train/real/*.jpg
        FLAT — same treatment as ai_generated.

  - real_img/  (optional additional real-face source, if you point --real_img_extra at it)

  - "AI Generated dataset" (CIFAKE) is EXCLUDED BY DEFAULT. It's animals/
    vehicles/objects, not faces — audit_datasets.py confirmed this. Mixing
    it into the ai_generated class teaches the model to distinguish photo
    vs. GAN-object textures, which doesn't transfer to face forgery
    detection and can actively hurt the face classes. Pass
    --include_cifake_as_sanity_only to copy it into a SEPARATE folder
    (sanity_check_nonface/) that is never used for training, only as an
    optional later check that the model doesn't misbehave on wildly
    out-of-domain input.

Any source folder is auto-detected as GROUPED (has subfolders containing
the images) or FLAT (images directly inside it) — so if your layout
shifts again, you mainly need to update the ROOTS list below, not the
splitting logic.

Usage:
    python prepare_dataset_v2.py --max_per_class 12000
    python prepare_dataset_v2.py --real_img_extra "real_img" --include_cifake_as_sanity_only
"""

import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "datasets" / "organized_v2"
SANITY_DIR = PROJECT_ROOT / "datasets" / "sanity_check_nonface"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = ["ai_generated", "deepfake", "real"]  # must match app.py

# (label_for_logging, relative_path, class)
ROOTS = [
    ("Deepfake cropped frames", "Deepfake dataset/cropped_images", "deepfake"),
    ("Unified AI-generated faces", "unified_dataset/train/ai_generated", "ai_generated"),
    ("Unified real faces", "unified_dataset/train/real", "real"),
]

CIFAKE_ROOTS = [
    ("CIFAKE train", "AI Generated dataset/train"),
    ("CIFAKE test", "AI Generated dataset/test"),
]


def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS]


def is_grouped(root: Path, images):
    direct = [p for p in images if p.parent == root]
    return len(direct) < len(images) * 0.5


def load_source(label, rel_path, label_class, records):
    root = PROJECT_ROOT / rel_path
    if not root.exists():
        print(f"  [{label}] not found at {root} — skipping")
        return
    images = collect_images(root)
    if not images:
        print(f"  [{label}] 0 images found — skipping")
        return
    grouped = is_grouped(root, images)
    for p in images:
        group = f"grp_{p.parent.name}" if grouped else f"img_{p.name}"
        records.append({"path": p, "label": label_class, "group": group})
    mode = "GROUPED (by subfolder)" if grouped else "FLAT (per-image groups)"
    print(f"  [{label}] {len(images)} images, {mode} -> class '{label_class}'")


def load_cifake_sanity(records_sanity):
    for label, rel_path in CIFAKE_ROOTS:
        root = PROJECT_ROOT / rel_path
        if not root.exists():
            continue
        images = collect_images(root)
        for p in images:
            records_sanity.append(p)
        print(f"  [{label}] {len(images)} images -> sanity_check_nonface/ (not used for training)")


def add_extra_folder(folder, label_class, records):
    if not folder:
        return
    folder = Path(folder)
    if not folder.exists():
        print(f"  Extra folder not found, skipping: {folder}")
        return
    images = collect_images(folder)
    grouped = is_grouped(folder, images)
    for p in images:
        group = f"extra_{p.parent.name}" if grouped else f"extra_img_{p.name}"
        # priority=True guarantees these survive cap_per_class's random cut
        # instead of being diluted proportionally with the much larger
        # original sources -- the whole point of adding them is lost if a
        # 2,000-image addition gets cut down to ~3% of a 12,000 cap.
        records.append({"path": p, "label": label_class, "group": group, "priority": True})
    print(f"  [extra: {folder.name}] {len(images)} images -> class '{label_class}' (protected from capping)")


def cap_per_class(records, max_per_class, seed):
    random.seed(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)
    capped = []
    for c in CLASS_NAMES:
        items = by_class.get(c, [])
        priority_items = [r for r in items if r.get("priority")]
        normal_items = [r for r in items if not r.get("priority")]
        random.shuffle(priority_items)
        random.shuffle(normal_items)
        if max_per_class:
            # Priority (--extra_*) items are kept in full first, up to the
            # cap; only the remaining budget is filled from the original,
            # much larger sources. This guarantees deliberately-added data
            # isn't randomly diluted away.
            keep_priority = priority_items[:max_per_class]
            remaining_budget = max(max_per_class - len(keep_priority), 0)
            keep_normal = normal_items[:remaining_budget]
            capped += keep_priority + keep_normal
            if len(priority_items) > len(keep_priority):
                print(f"  Note: {c} priority items ({len(priority_items)}) exceeded "
                      f"max_per_class ({max_per_class}) -- some were still cut. "
                      f"Consider raising --max_per_class.")
        else:
            capped += priority_items + normal_items
    return capped


def grouped_split(records, seed):
    """70/15/15 split by group — no group straddles two splits."""
    groups = [r["group"] for r in records]
    labels = [r["label"] for r in records]

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss1.split(records, labels, groups))

    rest_records = [records[i] for i in rest_idx]
    rest_groups = [r["group"] for r in rest_records]
    rest_labels = [r["label"] for r in rest_records]

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_records, rest_labels, rest_groups))

    train = [records[i] for i in train_idx]
    val = [rest_records[i] for i in val_idx]
    test = [rest_records[i] for i in test_idx]
    return train, val, test


def write_split(name, records):
    for c in CLASS_NAMES:
        (OUT_DIR / name / c).mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for r in records:
        dest = OUT_DIR / name / r["label"] / f"{counts[r['label']]:07d}{r['path'].suffix.lower()}"
        shutil.copy2(r["path"], dest)
        counts[r["label"]] += 1
    print(f"  {name}: " + ", ".join(f"{c}={counts.get(c, 0)}" for c in CLASS_NAMES))


def write_sanity(paths):
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        shutil.copy2(p, SANITY_DIR / f"{i:07d}{p.suffix.lower()}")
    print(f"  sanity_check_nonface/: {len(paths)} images (never used for training)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_img_extra", type=str, default=None,
                    help="Optional extra real-face folder, e.g. real_img/")
    ap.add_argument("--extra_ai", type=str, default=None,
                    help="Optional extra ai_generated-face folder")
    ap.add_argument("--extra_deepfake", type=str, default=None,
                    help="Optional extra deepfake folder")
    ap.add_argument("--include_cifake_as_sanity_only", action="store_true",
                    help="Copy CIFAKE images to datasets/sanity_check_nonface/ "
                         "(never used for training — see module docstring)")
    ap.add_argument("--max_per_class", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading known face sources...")
    records = []
    for label, rel_path, cls in ROOTS:
        load_source(label, rel_path, cls, records)

    add_extra_folder(args.real_img_extra, "real", records)
    add_extra_folder(args.extra_ai, "ai_generated", records)
    add_extra_folder(args.extra_deepfake, "deepfake", records)

    print("\nCIFAKE ('AI Generated dataset') is excluded from training classes")
    print("(confirmed non-face content — animals/vehicles/objects).")
    if args.include_cifake_as_sanity_only:
        sanity_paths = []
        load_cifake_sanity(sanity_paths)
        write_sanity(sanity_paths)

    if not records:
        print("\nNo source data found — check the ROOTS paths at the top of this script.")
        return

    print(f"\nBefore capping: " +
          ", ".join(f"{c}={sum(1 for r in records if r['label']==c)}" for c in CLASS_NAMES))
    priority_before = sum(1 for r in records if r.get("priority"))
    if priority_before:
        print(f"  (of which {priority_before} are protected --extra_* additions)")
    records = cap_per_class(records, args.max_per_class, args.seed)
    print(f"After capping to max {args.max_per_class}/class: " +
          ", ".join(f"{c}={sum(1 for r in records if r['label']==c)}" for c in CLASS_NAMES))
    priority_after = sum(1 for r in records if r.get("priority"))
    if priority_before:
        print(f"  (of which {priority_after} are the protected --extra_* additions -- "
              f"should equal {priority_before} unless a note above says otherwise)")

    train, val, test = grouped_split(records, args.seed)

    print(f"\nWriting grouped split to {OUT_DIR} ...")
    write_split("train", train)
    write_split("val", val)
    write_split("test", test)

    print("\nDone. `test/` is your held-out honesty check — never look at it")
    print("while tuning, and evaluate on it only once at the end.")
    print("Next: python train_unified.py")


if __name__ == "__main__":
    main()