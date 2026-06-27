r"""
merge_smoking_to_smoking4.py
============================
Merge smoking2, Smoking, and smoking3 datasets into a unified smoking4 dataset
in standard YOLO format with a single class: cigarette.

Source datasets:
  - smoking2:   YOLO format (images/labels), 1 class  ['cigarette'], splits: train/valid/test
  - Smoking:    YOLO format (images/labels), 2 classes ['face', 'smoking'], splits: train/valid/test
  - smoking3:   Flat format (images+labels mixed), 1 class (cigarette), splits: train/test (no valid)

Processing:
  - smoking2:  direct copy, class 0 stays 0.  Prefix: s2_
  - Smoking:   filter class 1 (smoking) only, remap 1->0, skip face-only images.  Prefix: s1_
  - smoking3:  restructure to YOLO format, split ~20% train->valid.  Prefix: s3_

Output: smoking4/ with standard YOLO structure + data.yaml

Usage:
    cd C:/Users/admin/Desktop/work/smoke
    python utils/merge_smoking_to_smoking4.py
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

# ── Configuration ────────────────────────────────────────────────────────────

BASE = Path(r"C:\Users\admin\Desktop\work\smoke")
OUTPUT = BASE / "smoking4"
VALID_SPLIT_RATIO = 0.20   # 20% of smoking3 train → valid
RANDOM_SEED = 42

SOURCES = [
    {
        "name": "smoking2",
        "path": BASE / "smoking2",
        "prefix": "s2_",
        "structure": "yolo",           # images/ + labels/ subdirectories
        "splits": ["train", "valid", "test"],
        "class_filter": None,          # keep all classes as-is
        "class_remap": None,           # no remapping
    },
    {
        "name": "Smoking",
        "path": BASE / "Smoking",
        "prefix": "s1_",
        "structure": "yolo",
        "splits": ["train", "valid", "test"],
        "class_filter": {1},           # keep only class 1 (smoking)
        "class_remap": {1: 0},         # remap 1 → 0 (cigarette)
    },
    {
        "name": "smoking3",
        "path": BASE / "smoking3",
        "prefix": "s3_",
        "structure": "flat",           # images + labels mixed in same dir
        "splits": ["train", "test"],   # no valid — we create it from train
        "class_filter": None,
        "class_remap": None,
    },
]

OUTPUT_YAML = """\
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['cigarette']
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create output directory tree."""
    for split in ["train", "valid", "test"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def find_image_label_pairs_flat(directory: Path) -> List[Tuple[Path, Path]]:
    """
    In a flat directory, pair .jpg images with their corresponding .txt labels.
    Returns list of (image_path, label_path) tuples.
    """
    pairs = []
    for img in sorted(directory.glob("*.jpg")):
        label = directory / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def find_image_label_pairs_yolo(directory: Path) -> List[Tuple[Path, Path]]:
    """
    In a YOLO-structured directory (containing images/ and labels/),
    pair images with their labels by matching filename stems.
    Returns list of (image_path, label_path) tuples.
    """
    pairs = []
    images_dir = directory / "images"
    labels_dir = directory / "labels"
    for img in sorted(images_dir.glob("*.jpg")):
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def filter_and_remap_labels(
    label_path: Path,
    keep_classes: set | None,
    remap: dict | None,
) -> list[str] | None:
    """
    Read a YOLO label file, optionally keep only certain class ids,
    optionally remap class ids.
    Returns list of label lines, or None if no labels remain (skip this image).
    """
    with open(label_path, "r") as f:
        lines = f.readlines()

    kept = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])

        if keep_classes is not None and cls not in keep_classes:
            continue  # drop this annotation

        if remap is not None and cls in remap:
            cls = remap[cls]

        parts[0] = str(cls)
        kept.append(" ".join(parts))

    return kept if kept else None


def process_dataset(cfg: dict, stats: dict) -> None:
    """
    Process one source dataset according to its config.
    Populates stats dict with counts.
    """
    name = cfg["name"]
    src = cfg["path"]
    prefix = cfg["prefix"]
    structure = cfg["structure"]
    splits = cfg["splits"]
    keep_classes = cfg["class_filter"]
    remap = cfg["class_remap"]

    stats[name] = {"train": 0, "valid": 0, "test": 0, "skipped": 0}

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  Structure: {structure}")
    print(f"  Keep classes: {keep_classes}")
    print(f"  Remap: {remap}")
    print(f"  Prefix: '{prefix}'")

    # Process original splits
    for split in splits:
        if structure == "yolo":
            src_split_dir = src / split
            pairs = find_image_label_pairs_yolo(src_split_dir)
        else:
            src_split_dir = src / split
            pairs = find_image_label_pairs_flat(src_split_dir)

        print(f"  {split}: found {len(pairs)} pairs")

        # If this is smoking3 train, we need to split some to valid
        if name == "smoking3" and split == "train":
            random.shuffle(pairs)
            n_valid = int(len(pairs) * VALID_SPLIT_RATIO)
            train_pairs = pairs[n_valid:]  # remaining → train
            valid_pairs = pairs[:n_valid]  # first N → valid

            print(f"    -> keeping {len(train_pairs)} for train, "
                  f"moving {len(valid_pairs)} to valid")

            _copy_pairs(train_pairs, prefix, "train", keep_classes, remap, stats, name)
            _copy_pairs(valid_pairs, prefix, "valid", keep_classes, remap, stats, name)
        else:
            _copy_pairs(pairs, prefix, split, keep_classes, remap, stats, name)

    # Print summary
    s = stats[name]
    total_kept = s["train"] + s["valid"] + s["test"]
    print(f"  Summary: {total_kept} kept, {s['skipped']} skipped (face-only)")


def _copy_pairs(
    pairs: list,
    prefix: str,
    split: str,
    keep_classes: set | None,
    remap: dict | None,
    stats: dict,
    src_name: str,
) -> None:
    """
    Copy image+label pairs to the output directory for a given split.
    Applies class filtering and remapping on labels.
    """
    out_img_dir = OUTPUT / split / "images"
    out_lbl_dir = OUTPUT / split / "labels"

    for img_path, lbl_path in pairs:
        # Filter & remap
        new_lines = filter_and_remap_labels(lbl_path, keep_classes, remap)
        if new_lines is None:
            stats[src_name]["skipped"] += 1
            continue  # no annotations left after filtering

        new_name = prefix + img_path.stem

        # Copy image
        shutil.copy2(img_path, out_img_dir / f"{new_name}.jpg")

        # Write filtered label
        with open(out_lbl_dir / f"{new_name}.txt", "w") as f:
            f.write("\n".join(new_lines) + "\n")

        stats[src_name][split] += 1


def write_data_yaml() -> None:
    """Write the data.yaml file for the merged dataset."""
    yaml_path = OUTPUT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(OUTPUT_YAML)
    print(f"\nWrote: {yaml_path}")


def print_grand_total(stats: dict) -> None:
    """Print final summary."""
    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    grand = {"train": 0, "valid": 0, "test": 0, "skipped": 0}
    for name, s in stats.items():
        kept = s["train"] + s["valid"] + s["test"]
        print(f"  {name:12s}: train={s['train']:5d}  valid={s['valid']:5d}  "
              f"test={s['test']:5d}  kept={kept:5d}  skipped={s['skipped']:5d}")
        for k in grand:
            grand[k] += s[k]
    total = grand["train"] + grand["valid"] + grand["test"]
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':12s}: train={grand['train']:5d}  valid={grand['valid']:5d}  "
          f"test={grand['test']:5d}  kept={total:5d}  skipped={grand['skipped']:5d}")
    print(f"\nOutput: {OUTPUT}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Clean output if exists
    if OUTPUT.exists():
        print(f"Removing existing: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    ensure_dirs()

    stats = {}
    for cfg in SOURCES:
        process_dataset(cfg, stats)

    write_data_yaml()
    print_grand_total(stats)


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    main()
