"""
build_real_dataset.py — GWSat
------------------------------
Builds train/val/test from real GEE Sentinel-2 scenes.
Auto-labels each patch from LSWI (not folder name).
Balances classes before saving.

Usage:
    python build_real_dataset.py
    python build_real_dataset.py --stride 32
    python build_real_dataset.py --preview
"""

import argparse
import glob
import os
import numpy as np
import torch
from pathlib import Path

try:
    import rasterio
except ImportError:
    print("pip install rasterio")
    exit()

BANDS = ['B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

SCENES = [
    ('gee_scenes/stable_nizam_2022', 0),
    ('gee_scenes/stable_nizam_2023', 0),
    ('gee_scenes/stable_nizam_2024', 0),
    ('gee_scenes/stable_poch_2022',  0),
    ('gee_scenes/stable_poch_2023',  0),
    ('gee_scenes/stable_poch_2024',  0),
    ('gee_scenes/moderate_2022',     1),
    ('gee_scenes/moderate_2023',     1),
    ('gee_scenes/moderate_2024',     1),
    ('gee_scenes/critical_2022',     2),
    ('gee_scenes/critical_2023',     2),
    ('gee_scenes/critical_2024',     2),
    ('gee_scenes/stable_2023',       0),
    ('Telangana_Instant_Data',       1),
    ('Telangana_Instant_Data_post_monsoon', 0),  # post-monsoon = Stable (high soil moisture)
]


def load_folder(folder):
    arrays = {}
    for b in BANDS:
        matches = glob.glob(os.path.join(folder, f'*{b}*'))
        if matches:
            with rasterio.open(matches[0]) as src:
                d = src.read(1).astype(np.float32)
                if d.max() > 2.0:
                    d /= 10000.0
                arrays[b] = np.clip(d, 0, 1)
    return arrays


def extract_patches(arrays, patch_size=64, stride=64, folder_label=None):
    """
    Tile scene into patches and auto-label each from spectral indices.

    Uses LSWI as primary signal (physically: leaf water status).
    folder_label is used as a prior for ambiguous moderate-zone patches
    so that known Stable scenes (post-monsoon) aren't mislabelled.

    Skips bare-soil/cloud patches (NDVI < 0.08).
    """
    if 'B4' not in arrays:
        return [], []
    H, W = arrays['B4'].shape
    patches, labels = [], []
    for y in range(0, H - patch_size, stride):
        for x in range(0, W - patch_size, stride):
            patch = np.stack([
                arrays.get(b, np.zeros((H, W)))[y:y+patch_size, x:x+patch_size]
                for b in BANDS
            ], axis=0)

            b4  = patch[0].mean()
            b8  = patch[4].mean()
            b11 = patch[6].mean()
            b12 = patch[7].mean()

            # Skip bare-soil / cloud patches
            ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
            if ndvi < 0.08:
                continue

            lswi  = (b8  - b11) / (b8  + b11  + 1e-8)
            swir  = b11 / (b12 + 1e-8)

            # Primary labelling from LSWI + SWIR (physically grounded)
            if lswi > 0.15:
                label = 0   # Stable: high leaf water
            elif lswi < -0.05 or swir > 1.8:
                label = 2   # Critical: severe water deficit
            else:
                # Ambiguous moderate zone — use folder_label as prior
                if folder_label is not None:
                    label = folder_label
                else:
                    label = 1   # Default: Moderate

            patches.append(patch)
            labels.append(label)
    return patches, labels


def print_band_stats(X, y, label_names=('Stable', 'Moderate', 'Critical')):
    print("\n── Real Band Profile Stats (verification) ─────────────")
    print(f"  {'Class':<12} {'B4':>6} {'B8':>6} {'B11':>6} "
          f"{'NDVI':>7} {'LSWI':>7} {'N':>5}")
    print("  " + "-" * 58)
    for cls, name in enumerate(label_names):
        mask = y == cls
        if mask.sum() == 0:
            print(f"  {name:<12} NO DATA")
            continue
        Xc   = X[mask]
        b4   = Xc[:, 0].mean()
        b8   = Xc[:, 4].mean()
        b11  = Xc[:, 6].mean()
        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
        lswi = (b8 - b11) / (b8 + b11 + 1e-8)
        print(f"  {name:<12} {b4:>6.3f} {b8:>6.3f} {b11:>6.3f} "
              f"{ndvi:>+7.3f} {lswi:>+7.3f} {mask.sum():>5}")
    print()
    print("  Expected pattern (physically correct):")
    print("    Stable   → LSWI most positive  (B8 >> B11)")
    print("    Moderate → LSWI near zero")
    print("    Critical → LSWI most negative  (B11 > B8)")
    print()


def main(args):
    Path('data').mkdir(exist_ok=True)
    all_patches, all_labels = [], []

    print("\n── Loading scenes ──────────────────────────────────────")
    for folder, folder_label in SCENES:
        if not Path(folder).exists():
            print(f"  SKIP (not found): {folder}")
            continue
        arrays  = load_folder(folder)
        patches, labels = extract_patches(arrays, stride=args.stride,
                                          folder_label=folder_label)
        if not patches:
            print(f"  SKIP (no B4):    {folder}")
            continue
        all_patches.extend(patches)
        all_labels.extend(labels)
        label_name = ['Stable', 'Moderate', 'Critical']
        dist = [labels.count(i) for i in range(3)]
        print(f"  {folder:<45} {len(patches):>4} patches  "
              f"S:{dist[0]} M:{dist[1]} C:{dist[2]}")

    if not all_patches:
        print("ERROR: No patches extracted.")
        return

    X = torch.tensor(np.stack(all_patches), dtype=torch.float32)
    y = torch.tensor(all_labels,            dtype=torch.long)

    print(f"\n  Total patches : {len(X)}")
    print(f"  Class dist    : {y.bincount().tolist()}  "
          f"[Stable, Moderate, Critical]")

    print_band_stats(X.numpy(), y.numpy())

    if args.preview:
        print("Preview mode — not saved.")
        return

    # ── Balance classes ────────────────────────────────────
    counts    = y.bincount()
    min_count = counts.min().item()
    cap       = min_count * 2
    print(f"  Balancing: capping each class to {cap} patches")

    balanced_X, balanced_y = [], []
    for cls in range(3):
        mask = (y == cls).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(mask))[:cap]
        mask = mask[perm]
        balanced_X.append(X[mask])
        balanced_y.append(y[mask])

    X = torch.cat(balanced_X)
    y = torch.cat(balanced_y)
    idx = torch.randperm(len(X))
    X, y = X[idx], y[idx]
    print(f"  Balanced dist : {y.bincount().tolist()}")

    # Warn if any class is still small
    if y.bincount().min() < 50:
        print(f"  WARNING: smallest class has < 50 patches")

    # ── Save ──────────────────────────────────────────────
    n = len(X)
    t = int(0.70 * n)
    v = int(0.85 * n)

    torch.save({'X': X[:t],  'y': y[:t]},  'data/train.pt')
    torch.save({'X': X[t:v], 'y': y[t:v]}, 'data/val.pt')
    torch.save({'X': X[v:],  'y': y[v:]},  'data/test.pt')
    print(f"  Saved: data/train.pt  val.pt  test.pt")
    print(f"  Train:{t}  Val:{v-t}  Test:{n-v}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stride',  type=int, default=64)
    p.add_argument('--preview', action='store_true')
    main(p.parse_args())