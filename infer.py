"""
infer.py — GWSat v2
--------------------
Single entry point for all inference modes.

THE FIX: --scene mode uses tiled inference (no resize aliasing).
  Instead of squishing a 1114×1115 image to 224×224,
  we tile it into 64×64 patches, run the model on each patch,
  and majority-vote. Dry-soil SWIR signal is preserved at full resolution.

Usage:
    # Telangana TIF scene (tiled, no resize)
    python infer.py --scene \
        Telangana_Instant_Data_B4.tif \
        Telangana_Instant_Data_B5.tif \
        Telangana_Instant_Data_B6.tif \
        Telangana_Instant_Data_B7.tif \
        Telangana_Instant_Data_B8.tif \
        Telangana_Instant_Data_B8A.tif \
        Telangana_Instant_Data_B11.tif \
        Telangana_Instant_Data_B12.tif

    # Demo on synthetic tiles
    python infer.py --demo

    # Single .pt tile
    python infer.py --tile sample_input/sample_class2.pt

    # GEE batch tensor
    python infer.py --tensor data/inference/telangana_demo.pt
"""

import sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import GWSatModel, compute_spectral_indices
from data_pipeline import synthetic_tile


LABELS     = ["Stable", "Moderate", "Critical"]
ALERT_TEXT = {
    0: ("✅", "STABLE",   "Standard monitoring. Groundwater within normal range."),
    1: ("⚠️", "MODERATE", "Spectral stress detected weeks before NDVI drops.\n"
                          "     Recommend field inspection within 14 days."),
    2: ("🚨", "CRITICAL", "SEVERE PRE-DROUGHT SIGNAL DETECTED.\n"
                          "     IR pressure index elevated — stomatal closure confirmed.\n"
                          "     Alert NDMA / insurance. Crop failure risk: 2-4 weeks."),
}


# ───────────────────────────────────────────────────────────
# TIF loader — reads raw Sentinel-2 TIF, NO resize
# ───────────────────────────────────────────────────────────

def load_tif_scene(tif_paths: list) -> torch.Tensor:
    """
    Load 8 Sentinel-2 TIF files into [8, H, W] float32 tensor.
    Values normalised to [0,1] (Sentinel-2 reflectance / 10000).
    The scene is NOT resized — full resolution is preserved for tiled inference.

    tif_paths: list of 8 paths in order [B4, B5, B6, B7, B8, B8A, B11, B12]
    """
    try:
        import rasterio
    except ImportError:
        print("pip install rasterio")
        sys.exit(1)

    bands = []
    validity_mask = None

    for path in tif_paths:
        with rasterio.open(path) as src:
            refl = src.read(1).astype(np.float32)
            # Band 2 is the validity mask in Sentinel-2 L2A exports
            if validity_mask is None and src.count >= 2:
                validity_mask = src.read(2).astype(bool)
        bands.append(refl)

    stack = np.stack(bands, axis=0)   # [8, H, W]

    # Apply validity mask
    if validity_mask is not None:
        stack[:, ~validity_mask] = 0.0

    # Normalize: Sentinel-2 surface reflectance is in [0, 10000]
    # Detect range automatically
    vmax = stack.max()
    if vmax > 2.0:
        stack /= 10000.0

    stack = np.clip(stack, 0.0, 1.0)
    return torch.from_numpy(stack)


# ───────────────────────────────────────────────────────────
# Print helpers
# ───────────────────────────────────────────────────────────

def print_result(result: dict, mode: str = "tile"):
    cls   = result["stress_class"]
    emoji, short, detail = ALERT_TEXT[cls]
    si    = result["spectral_indices"]
    conf  = result["confidence"]
    probs = result["probabilities"]
    lat   = result.get("latency_ms", 0)

    print("\n" + "═" * 64)
    print(f"  GWSat v2 — GROUNDWATER STRESS ALERT  [{mode.upper()}]")
    print("═" * 64)
    print(f"  Status     : {emoji}  {short}")
    print(f"  Confidence : {conf:.1%}")
    if lat:
        print(f"  Latency    : {lat:.1f} ms")
    if "n_patches" in result:
        pd      = result.get("patch_distribution", {})
        skipped = result.get("skipped_patches", 0)
        total   = result["n_patches"] + skipped
        print(f"  Patches    : {result['n_patches']} vegetated / "
              f"{total} total  "
              f"({skipped} urban/bare-soil skipped)")
        print(f"  Veg stress : Stable:{pd.get('Stable',0)}  "
              f"Moderate:{pd.get('Moderate',0)}  "
              f"Critical:{pd.get('Critical',0)}")
    print()
    print(f"  {detail}")
    print()
    print("  Spectral Indices:")
    print(f"    NDVI (traditional greenness)  : {si['NDVI']:+.4f}"
          f"  → {'HEALTHY (misleading)' if si['NDVI'] > 0.4 else 'stressed'}")
    print(f"    Red-Edge Index                : {si['RedEdge_Index']:+.4f}"
          f"  → {'OK' if si['RedEdge_Index'] > 0.05 else 'chlorophyll degrading'}")
    print(f"    Leaf Water Stress (LSWI)      : {si['LeafWaterStress_LSWI']:+.4f}"
          f"  → {'adequate' if si['LeafWaterStress_LSWI'] > 0.20 else 'WATER DEFICIT'}")
    print(f"    IR Pressure Index             : {si['IR_Pressure_Index']:+.4f}"
          f"  → {'elevated' if si['IR_Pressure_Index'] > 0.10 else 'normal'}")
    print()
    print("  Class Probabilities:")
    for label, prob in probs.items():
        bar = "█" * int(prob * 32)
        print(f"    {label:<12} {prob:.3f}  {bar}")
    print()
    raw_mb = result["raw_tile_bytes"] / 1e6
    print(f"  Bandwidth: {raw_mb:.2f} MB raw → {result['alert_bytes']} bytes alert "
          f"({result['raw_tile_bytes'] // result['alert_bytes']:,}× reduction)")
    print("═" * 64)

    if cls == 2:
        print("\n  🚨 DOWNLINK ALERT PAYLOAD (64 bytes):")
        print(f"     CLASS=CRITICAL  CONF={conf:.2f}  "
              f"LSWI={si['LeafWaterStress_LSWI']:.3f}  "
              f"IRP={si['IR_Pressure_Index']:.3f}")
    print()


# ───────────────────────────────────────────────────────────
# Build model
# ───────────────────────────────────────────────────────────

def build_model(head_path: str) -> GWSatModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = GWSatModel(device=device)
    if Path(head_path).exists():
        model.load_head(head_path)
        print(f"✅ Loaded checkpoint: {head_path}")
    else:
        print(f"⚠️  No checkpoint at {head_path}. Using random weights (run train.py).")
    model.eval()
    return model


# ───────────────────────────────────────────────────────────
# Modes
# ───────────────────────────────────────────────────────────

def mode_demo(model):
    sample_dir = Path("sample_input")
    sample_dir.mkdir(exist_ok=True)
    for cls in range(3):
        p = sample_dir / f"sample_class{cls}.pt"
        if not p.exists():
            torch.save(torch.from_numpy(synthetic_tile(cls)), p)

    print("\n  GWSat v2 DEMO — Synthetic Tiles")
    print("=" * 64)
    print(f"  {'File':<24} {'True':^10} {'Predicted':^12} {'Conf':^8}")
    print("-" * 64)

    for path in sorted(sample_dir.glob("*.pt")):
        tile  = torch.load(str(path), weights_only=False)
        if isinstance(tile, dict):
            tile = tile["X"][0]
        if tile.dim() == 4:
            tile = tile[0]
        tile = tile.float()

        t0 = time.perf_counter()
        result = model.predict(tile)
        lat = (time.perf_counter() - t0) * 1000

        true_cls = "?"
        for i in range(3):
            if f"class{i}" in path.stem or f"cls{i}" in path.stem:
                true_cls = LABELS[i]
        pred = LABELS[result["stress_class"]]
        mark = "✅" if true_cls == pred else "❌"
        print(f"  {mark} {path.stem:<24} {true_cls:^10} {pred:^12} "
              f"{result['confidence']:.1%}  ({lat:.0f}ms)")

    print("=" * 64)


def mode_scene(args, model):
    """TILED INFERENCE — the fix for the resize aliasing bug."""
    if len(args.scene) != 8:
        print("ERROR: --scene requires exactly 8 TIF paths "
              "(B4 B5 B6 B7 B8 B8A B11 B12)")
        sys.exit(1)

    print(f"\nLoading scene ({len(args.scene)} bands)…")
    scene = load_tif_scene(args.scene)
    C, H, W = scene.shape
    print(f"  Scene shape : {C} bands × {H} × {W} pixels")
    print(f"  Coverage    : ~{H*10/1000:.0f} × {W*10/1000:.0f} km "
          f"(at 10m/px)")
    print(f"  Running tiled inference (64×64 patches, stride=32)…")

    t0 = time.perf_counter()
    result = model.predict_scene(scene, patch_size=64, stride=32)
    lat = (time.perf_counter() - t0) * 1000

    result["latency_ms"] = round(lat, 1)
    print_result(result, mode="tiled scene")

    # Save heatmap if matplotlib available
    if "heatmap" in result:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            hm = result["heatmap"].numpy()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                     facecolor="#111")
            fig.suptitle("GWSat v2 — Pre-Drought Stress Heatmap",
                         color="white", fontsize=14, fontweight="bold")

            # False-colour composite: NIR(B8)→R, RedEdge(B6)→G, Red(B4)→B
            # This is the standard 8-4-3 false colour for vegetation stress
            # Healthy vegetation = bright red; stressed = orange/yellow
            ax = axes[0]
            rgb = np.stack([scene[4].numpy(),   # B8  NIR → R channel
                            scene[2].numpy(),   # B6  RedEdge → G channel
                            scene[0].numpy()],  # B4  Red → B channel
                           axis=2)
            rgb = np.clip(rgb * 2.5, 0, 1)
            ax.imshow(rgb)
            ax.set_title("False Colour (NIR-RedEdge-Red)\nHealthy veg=red, stressed=orange/yellow",
                         color="white", fontsize=9)
            ax.axis("off")

            # Stress heatmap
            ax = axes[1]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "stress", ["#2ecc71", "#f39c12", "#e74c3c"])
            im = ax.imshow(hm, cmap=cmap, vmin=0, vmax=1)
            ax.set_title("GWSat Stress Heatmap\nGreen=Stable  Orange=Moderate  Red=Critical",
                         color="white", fontsize=9)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label("Stress Score", color="white")
            cb.ax.yaxis.set_tick_params(color="white")
            cb.ax.tick_params(colors="white")

            out = "stress_heatmap.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150, facecolor="#111")
            print(f"  Heatmap saved: {out}")
        except Exception as e:
            print(f"  (Heatmap save failed: {e})")


def mode_tile(path: str, model: GWSatModel):
    p = Path(path)
    if p.suffix in (".tif", ".tiff"):
        # Single-band tif → single patch
        tile = load_tif_scene([path])  # treat as 1-band, repeat
    elif p.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        tile = torch.from_numpy(arr)
    else:
        obj = torch.load(path, weights_only=False, map_location="cpu")
        tile = obj["X"][0] if isinstance(obj, dict) else obj
    tile = tile.float()
    if tile.dim() == 4:
        tile = tile[0]

    t0     = time.perf_counter()
    result = model.predict(tile)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    print_result(result, mode="tile")


def mode_tensor(args, model: GWSatModel):
    data   = torch.load(args.tensor, weights_only=False)
    X      = data["X"]
    coords = data.get("coords", [{}] * len(X))

    print(f"\n  Batch: {len(X)} patches from {data.get('region','?')}")
    counts = [0, 0, 0]
    rows   = []

    for i in range(len(X)):
        r = model.predict(X[i].float())
        counts[r["stress_class"]] += 1
        c  = coords[i] if i < len(coords) else {}
        si = r["spectral_indices"]
        rows.append({
            "lat": c.get("lat", ""), "lon": c.get("lon", ""),
            "prediction": LABELS[r["stress_class"]],
            "confidence": round(r["confidence"], 4),
            **{f"prob_{LABELS[j].lower()}": round(v, 4)
               for j, v in enumerate(r["probabilities"].values())},
            **{k: round(v, 4) for k, v in si.items()},
        })

    total = len(X)
    print(f"\n  {'Class':<12} {'Count':>6} {'%':>7}  Bar")
    for i, name in enumerate(LABELS):
        bar = "█" * int(20 * counts[i] / (total or 1))
        pct = 100 * counts[i] / (total or 1)
        print(f"  {name:<12} {counts[i]:>6} {pct:>6.1f}%  {bar}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n  CSV: {args.out_csv}")


# ───────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GWSat v2 — Pre-Drought Inference")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--demo",   action="store_true")
    grp.add_argument("--scene",  nargs=8, metavar="TIF",
                     help="8 Sentinel-2 TIFs: B4 B5 B6 B7 B8 B8A B11 B12")
    grp.add_argument("--tile",   help="Single .pt/.npy tile")
    grp.add_argument("--tensor", help="GEE batch .pt tensor")
    p.add_argument("--head",    default="checkpoints/best_head.pth")
    p.add_argument("--json",    action="store_true")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    model = build_model(args.head)

    if args.demo:
        mode_demo(model)
    elif args.scene:
        mode_scene(args, model)
    elif args.tile:
        mode_tile(args.tile, model)
    elif args.tensor:
        mode_tensor(args, model)


if __name__ == "__main__":
    main()