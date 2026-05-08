"""
build_real_dataset.py — GWSat v4  (REGION-AWARE, SCENE-LEVEL SPLIT)
---------------------------------------------------------------------
Builds train/val/test .pt files from real Sentinel-2 scenes.

Folder layout (put scenes here manually):
    processed/
        train/
            telangana/
                may2020critical/
                nov2021stable/
                sept2023moderate/
            maharashtra/          ← future region, just drop it in
                jan2023critical/
        test/                     ← leave empty for now, add later
            telangana/
                jan2022critical/

Rules:
    - Folder name must contain 'stable', 'moderate', or 'critical'
      to get a label automatically. No manual label config needed.
    - All scenes under processed/train/ go into training pool.
    - All scenes under processed/test/ become the true holdout test set.
    - If processed/test/ is empty or missing, the script falls back to
      a random 15% patch split from training data as a temporary val set.
    - Region name (e.g. telangana) is saved as metadata in .pt files.
    - Adding a new region = just create processed/train/newregion/ and
      drop scene folders inside. Zero code changes needed.

Usage:
    python build_real_dataset.py
    python build_real_dataset.py --processed_dir /path/to/processed
    python build_real_dataset.py --stride 32 --min_ndvi 0.08
    python build_real_dataset.py --preview
    python build_real_dataset.py --no_balance
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import torch
from pathlib import Path

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

# ── Band config ────────────────────────────────────────────────────────────────

BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

BAND_PATTERNS = {
    "B8A": [r"B8A", r"B08A"],
    "B4":  [r"_B04[_.]", r"_B4[_.]",  r"\bB04\b", r"\bB4\b"],
    "B5":  [r"_B05[_.]", r"_B5[_.]",  r"\bB05\b", r"\bB5\b"],
    "B6":  [r"_B06[_.]", r"_B6[_.]",  r"\bB06\b", r"\bB6\b"],
    "B7":  [r"_B07[_.]", r"_B7[_.]",  r"\bB07\b", r"\bB7\b"],
    "B8":  [r"_B08[_.]", r"_B8[_.]",  r"\bB08\b", r"(?<![A8])B8(?!A)\b"],
    "B11": [r"_B11[_.]", r"\bB11\b"],
    "B12": [r"_B12[_.]", r"\bB12\b"],
}

LABEL_MAP   = {"stable": 0, "moderate": 1, "critical": 2}
LABEL_NAMES = ["Stable", "Moderate", "Critical"]


# ── Label detection ────────────────────────────────────────────────────────────

def infer_label(name: str) -> int:
    """Infer stress label from folder name. Returns -1 if no match."""
    name_lower = name.lower()
    for key, lbl in LABEL_MAP.items():
        if key in name_lower:
            return lbl
    return -1


# ── Band detection ─────────────────────────────────────────────────────────────

def detect_bands(folder: str) -> dict:
    tif_files = (
        glob.glob(os.path.join(folder, "*.tif"))
        + glob.glob(os.path.join(folder, "*.TIF"))
        + glob.glob(os.path.join(folder, "*.tiff"))
    )
    if not tif_files:
        return {}

    detected, used = {}, set()
    priority = ["B8A", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

    for band in priority:
        for tif in tif_files:
            if tif in used:
                continue
            fname = os.path.basename(tif)
            for pat in BAND_PATTERNS.get(band, []):
                if re.search(pat, fname, re.IGNORECASE):
                    detected[band] = tif
                    used.add(tif)
                    break
            if band in detected:
                break

    return detected


# ── Band loader ────────────────────────────────────────────────────────────────

def load_band(path: str, ref_shape=None) -> np.ndarray:
    with rasterio.open(path) as src:
        if ref_shape and (src.height, src.width) != ref_shape:
            arr = src.read(
                1,
                out_shape=(1, ref_shape[0], ref_shape[1]),
                resampling=Resampling.bilinear,
            ).astype(np.float32)[0]
        else:
            arr = src.read(1).astype(np.float32)

    if arr.max() > 2.0:
        arr /= 10000.0
    return np.clip(arr, 0.0, 1.0)


# ── Scene → patches ────────────────────────────────────────────────────────────

def scene_to_patches(
    folder: str,
    label: int,
    region: str,
    patch_size: int = 64,
    stride: int = 64,
    min_ndvi: float = 0.08,
    verbose: bool = True,
) -> tuple:
    """
    Load all bands from a scene folder, tile into patches, filter by NDVI.
    Returns (patches, labels, regions) — all same length.
    """
    band_paths = detect_bands(folder)

    if "B4" not in band_paths:
        if verbose:
            print(f"    SKIP (no B4 found): {folder}")
        return [], [], []
    if "B8" not in band_paths:
        if verbose:
            print(f"    SKIP (no B8 found): {folder}")
        return [], [], []

    b4_arr    = load_band(band_paths["B4"])
    ref_shape = b4_arr.shape
    H, W      = ref_shape

    arrays = {"B4": b4_arr}
    for band in BAND_ORDER:
        if band == "B4":
            continue
        arrays[band] = (
            load_band(band_paths[band], ref_shape)
            if band in band_paths
            else np.zeros(ref_shape, dtype=np.float32)
        )

    patches, labels, regions = [], [], []
    n_skipped_veg   = 0
    n_skipped_cloud = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = np.stack(
                [arrays[b][y:y+patch_size, x:x+patch_size] for b in BAND_ORDER],
                axis=0,
            )

            b4_mean  = patch[0].mean()
            b8_mean  = patch[4].mean()
            b11_mean = patch[6].mean()

            ndvi = (b8_mean - b4_mean) / (b8_mean + b4_mean + 1e-8)
            lswi = (b8_mean - b11_mean) / (b8_mean + b11_mean + 1e-8)

            if ndvi < min_ndvi:
                n_skipped_veg += 1
                continue
            if lswi < -0.70:
                n_skipped_cloud += 1
                continue

            patches.append(patch)
            labels.append(label)
            regions.append(region)

    if verbose:
        kept    = len(patches)
        skipped = n_skipped_veg + n_skipped_cloud
        print(
            f"    {Path(folder).name:<45}  "
            f"region={region:<12}  "
            f"label={LABEL_NAMES[label]:<10}  "
            f"{kept:>4} patches kept  "
            f"({skipped} skipped: {n_skipped_veg} non-veg, "
            f"{n_skipped_cloud} cloud/water)"
        )

    return patches, labels, regions


# ── Collect scenes from a split folder ────────────────────────────────────────

def collect_scenes_from_split(split_dir: Path, verbose: bool = True) -> list:
    """
    Collect (scene_path, label, region) from:
        split_dir/
            region_name/
                scene_folder_with_label_in_name/

    Also handles flat layout (no region subfolder) by using 'unknown' as region.
    Returns list of (Path, int, str) tuples.
    """
    if not split_dir.exists():
        return []

    scenes = []

    for item in sorted(split_dir.iterdir()):
        if not item.is_dir():
            continue

        # Check if item itself is a scene (flat layout — no region folder)
        label = infer_label(item.name)
        if label != -1:
            scenes.append((item, label, "unknown"))
            continue

        # Otherwise treat item as a region folder
        region = item.name
        for scene_dir in sorted(item.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_label = infer_label(scene_dir.name)
            if scene_label == -1:
                if verbose:
                    print(f"  ⚠️  Cannot infer label from: {scene_dir.name} — skipping")
                continue
            scenes.append((scene_dir, scene_label, region))

    return scenes


# ── Band profile stats ─────────────────────────────────────────────────────────

def print_band_stats(X: np.ndarray, y: np.ndarray, regions: list):
    print("\n── Band Profile Stats ──────────────────────────────────────────────")
    print(f"  {'Class':<12} {'B4':>6} {'B8':>6} {'B11':>6} {'NDVI':>7} {'LSWI':>7} {'N':>6}")
    print("  " + "─" * 56)
    for cls, name in enumerate(LABEL_NAMES):
        mask = y == cls
        if mask.sum() == 0:
            print(f"  {name:<12}  NO DATA")
            continue
        Xc   = X[mask]
        b4   = Xc[:, 0].mean()
        b8   = Xc[:, 4].mean()
        b11  = Xc[:, 6].mean()
        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
        lswi = (b8 - b11) / (b8 + b11 + 1e-8)
        print(
            f"  {name:<12} {b4:>6.3f} {b8:>6.3f} {b11:>6.3f} "
            f"{ndvi:>+7.3f} {lswi:>+7.3f} {mask.sum():>6}"
        )

    # Per-region summary
    unique_regions = sorted(set(regions))
    if len(unique_regions) > 1 or unique_regions[0] != "unknown":
        print(f"\n── Per-Region Patch Counts ─────────────────────────────────────────")
        for region in unique_regions:
            region_mask = np.array([r == region for r in regions])
            region_y    = y[region_mask]
            counts      = np.bincount(region_y, minlength=3)
            print(f"  {region:<20}  S:{counts[0]:>4}  M:{counts[1]:>4}  C:{counts[2]:>4}  total:{counts.sum():>5}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    processed_dir = Path(args.processed_dir)
    train_dir     = processed_dir / "train"
    test_dir      = processed_dir / "test"

    if not processed_dir.exists():
        print(f"ERROR: {processed_dir} does not exist.")
        print("\nCreate this structure:")
        print("  processed/")
        print("    train/")
        print("      telangana/")
        print("        may2020critical/")
        print("        nov2021stable/")
        print("    test/             ← leave empty for now")
        sys.exit(1)

    # Support old flat layout (no train/ subfolder) for backwards compat
    has_train_folder = train_dir.exists()
    if not has_train_folder:
        print("⚠️  No processed/train/ folder found.")
        print("   Falling back to old flat layout (processed/ directly).")
        print("   Recommended: move scenes into processed/train/telangana/")
        train_dir = processed_dir

    Path("data").mkdir(exist_ok=True)

    # ── Collect training scenes ───────────────────────────────────────────────
    print(f"\n── Collecting training scenes from: {train_dir} ────────────────────")
    train_scenes = collect_scenes_from_split(train_dir, verbose=True)

    if not train_scenes:
        print(f"ERROR: No labelled scenes found in {train_dir}")
        print("Scene folder names must contain 'stable', 'moderate', or 'critical'.")
        sys.exit(1)

    # Group by label for display
    by_label = {0: [], 1: [], 2: []}
    for scene_dir, label, region in train_scenes:
        by_label[label].append((scene_dir, region))

    all_patches, all_labels, all_regions = [], [], []

    for cls, name in enumerate(LABEL_NAMES):
        print(f"\n  [{name.upper()}]  {len(by_label[cls])} scene(s):")
        for scene_dir, region in by_label[cls]:
            p, l, r = scene_to_patches(
                str(scene_dir), cls, region,
                patch_size=args.patch_size,
                stride=args.stride,
                min_ndvi=args.min_ndvi,
                verbose=True,
            )
            all_patches.extend(p)
            all_labels.extend(l)
            all_regions.extend(r)

    if not all_patches:
        print("\nERROR: No patches extracted from training scenes.")
        sys.exit(1)

    X_train = torch.tensor(np.stack(all_patches), dtype=torch.float32)
    y_train = torch.tensor(all_labels, dtype=torch.long)

    counts = y_train.bincount(minlength=3)
    print(f"\n── Training raw totals ─────────────────────────────────────────────")
    print(f"  Total : {len(X_train)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<12}: {counts[i].item()}")

    print_band_stats(X_train.numpy(), y_train.numpy(), all_regions)

    if args.preview:
        print("Preview mode — nothing saved.")
        return

    if counts.min() == 0:
        print("ERROR: At least one class has zero patches.")
        sys.exit(1)

    # ── Class balancing ───────────────────────────────────────────────────────
    if not args.no_balance:
        min_count = counts.min().item()
        print(f"── Balancing classes (cap per class: {min_count}) ──────────────────")
        balanced_X, balanced_y, balanced_r = [], [], []
        for cls in range(3):
            mask = (y_train == cls).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(mask))[:min_count]
            sel  = mask[perm]
            balanced_X.append(X_train[sel])
            balanced_y.append(y_train[sel])
            balanced_r.extend([all_regions[i] for i in sel.tolist()])
            print(f"  {LABEL_NAMES[cls]:<12}: {len(mask)} → {min_count}")
        X_train    = torch.cat(balanced_X)
        y_train    = torch.cat(balanced_y)
        all_regions = balanced_r

    if y_train.bincount(minlength=3).min() < 50:
        print(f"\n  ⚠️  Smallest class has only {y_train.bincount(minlength=3).min().item()} patches.")
        print("     Consider collecting more data.")

    # ── Shuffle and split 70/15/15 from training pool ────────────────────────
    idx = torch.randperm(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    all_regions = [all_regions[i] for i in idx.tolist()]

    n = len(X_train)
    t = int(0.70 * n)
    v = int(0.85 * n)

    torch.save({
        "X":       X_train[:t],
        "y":       y_train[:t],
        "regions": all_regions[:t],
        "source":  "real",
    }, "data/train.pt")

    torch.save({
        "X":       X_train[t:v],
        "y":       y_train[t:v],
        "regions": all_regions[t:v],
        "source":  "real",
    }, "data/val.pt")

    # ── True holdout test set from processed/test/ ───────────────────────────
    test_scenes = collect_scenes_from_split(test_dir, verbose=True) if test_dir.exists() else []

    if test_scenes:
        print(f"\n── Collecting true holdout test scenes from: {test_dir} ──────────")
        test_patches, test_labels, test_regions = [], [], []
        for scene_dir, label, region in test_scenes:
            p, l, r = scene_to_patches(
                str(scene_dir), label, region,
                patch_size=args.patch_size,
                stride=args.stride,
                min_ndvi=args.min_ndvi,
                verbose=True,
            )
            test_patches.extend(p)
            test_labels.extend(l)
            test_regions.extend(r)

        if test_patches:
            X_test = torch.tensor(np.stack(test_patches), dtype=torch.float32)
            y_test = torch.tensor(test_labels, dtype=torch.long)
            torch.save({
                "X":       X_test,
                "y":       y_test,
                "regions": test_regions,
                "source":  "real_holdout",
            }, "data/test.pt")
            print(f"\n  ✅ True holdout test.pt: {len(X_test)} patches  "
                  f"{y_test.bincount(minlength=3).tolist()}")
        else:
            print("  ⚠️  test/ folder exists but no patches extracted.")
            _save_fallback_test(X_train, y_train, all_regions, v, n)
    else:
        print(f"\n  ⚠️  No test scenes in {test_dir} yet.")
        print("     Using random 15% of training patches as temporary test set.")
        print("     Add real holdout scenes to processed/test/<region>/ later.")
        _save_fallback_test(X_train, y_train, all_regions, v, n)

    print(f"\n── Saved ───────────────────────────────────────────────────────────")
    print(f"  data/train.pt  {t} patches  {y_train[:t].bincount(minlength=3).tolist()}")
    print(f"  data/val.pt    {v-t} patches  {y_train[t:v].bincount(minlength=3).tolist()}")
    print(f"  data/test.pt   (see above)")
    print(f"\n  [Stable, Moderate, Critical]")
    print(f"\nNext step:  python train_moderate_fix.py")


def _save_fallback_test(X, y, regions, v, n):
    torch.save({
        "X":       X[v:],
        "y":       y[v:],
        "regions": regions[v:],
        "source":  "real_patch_split",  # not true holdout — same scenes as train
    }, "data/test.pt")
    print(f"  Fallback test.pt: {n-v} patches  {y[v:].bincount(minlength=3).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build GWSat dataset from real Sentinel-2 TIF scenes"
    )
    p.add_argument(
        "--processed_dir", default="processed",
        help="Root folder containing train/ and test/ subfolders (default: processed/)",
    )
    p.add_argument("--patch_size", type=int,   default=64)
    p.add_argument("--stride",     type=int,   default=64,
                   help="Patch stride (64=no overlap, 32=50%% overlap)")
    p.add_argument("--min_ndvi",   type=float, default=0.08,
                   help="Minimum patch NDVI to keep")
    p.add_argument("--preview",    action="store_true",
                   help="Print stats only, do not save")
    p.add_argument("--no_balance", action="store_true",
                   help="Skip class balancing")
    main(p.parse_args())