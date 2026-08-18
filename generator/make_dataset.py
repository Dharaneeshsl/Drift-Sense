"""Generate reproducible local diagnostics from continuous synthetic scenes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .degrade import degrade
from .noise import apply_noise
from .scene import make_scene


def _fast_case(out_dir: Path, ident: str, seed: int, anchor: bool, hard: bool, ood: bool) -> dict[str, float | int | str | bool]:
    rng = np.random.default_rng(seed)
    pitch_x = int(rng.integers(42, 68) if not ood else rng.integers(70, 100))
    pitch_y = int(rng.integers(38, 64) if not ood else rng.integers(66, 95))
    base = make_scene(1000, pitch_x=pitch_x, pitch_y=pitch_y, line_width=int(rng.integers(5, 11)), contact_size=int(rng.integers(7, 13)), seed=seed, anchor=False)
    x0, y0 = int(rng.integers(120, 780)), int(rng.integers(120, 780))
    patch = base[y0:y0 + 100, x0:x0 + 100].copy()
    if anchor:
        cv2.rectangle(patch, (15, 18), (35, 42), 252, -1)
        cv2.circle(patch, (77, 71), 8, 12, -1)
    if hard:
        patch = (np.floor(patch / 24.0) * 24.0).astype(np.float32)
        base[y0:y0 + 100, x0:x0 + 100] = patch
        dx, dy = max(100, x0 - pitch_x * 2), max(100, y0 - pitch_y * 2)
        base[dy:dy + 100, dx:dx + 100] = patch
        x0, y0 = dx, dy
    else:
        base[y0:y0 + 100, x0:x0 + 100] = patch
    reference = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    reference = apply_noise(degrade(reference, seed + 11, blur_sigma=0.35, rotation_deg=float(rng.uniform(-0.25, 0.25)), contrast=float(rng.uniform(0.97, 1.03)), edge_gain=float(rng.uniform(0.98, 1.02))), seed + 101, shot_scale=45, detector_sigma=0.7)
    search = apply_noise(degrade(base, seed + 22, blur_sigma=0.25, rotation_deg=float(rng.uniform(-0.15, 0.15)), contrast=float(rng.uniform(0.98, 1.02)), edge_gain=float(rng.uniform(0.99, 1.01)),), seed + 202, shot_scale=50, detector_sigma=0.7)
    np.save(out_dir / f"{ident}_reference.npy", reference.astype(np.float32))
    np.save(out_dir / f"{ident}_search.npy", search.astype(np.float32))
    return {"id": ident, "x_true": x0 + 50.0, "y_true": y0 + 50.0, "seed": seed, "pitch_x": pitch_x, "pitch_y": pitch_y, "mode": "fast_fixture", "anchor": anchor, "hard": hard}


def _full_case(out_dir: Path, ident: str, seed: int, anchor: bool, hard: bool, ood: bool) -> dict[str, float | int | str | bool]:
    """Generate one physically scaled pair from one 10000×10000 latent scene."""
    rng = np.random.default_rng(seed)
    pitch_x = int(rng.integers(420, 680) if not ood else rng.integers(700, 980))
    pitch_y = int(rng.integers(380, 640) if not ood else rng.integers(660, 940))
    latent = make_scene(10000, pitch_x=pitch_x, pitch_y=pitch_y, line_width=int(rng.integers(35, 75)), contact_size=int(rng.integers(50, 105)), seed=seed, anchor=False)
    x0, y0 = int(rng.integers(1000, 8800)), int(rng.integers(1000, 8800))
    if anchor:
        cv2.rectangle(latent, (x0 + 140, y0 + 160), (x0 + 330, y0 + 420), 252, -1)
        cv2.circle(latent, (x0 + 730, y0 + 710), 75, 12, -1)
    ref_raw = latent[y0:y0 + 1000, x0:x0 + 1000].copy()
    if hard:
        latent[y0:y0 + 1000, x0:x0 + 1000] = ref_raw
        dx, dy = max(1000, x0 - pitch_x * 2), max(1000, y0 - pitch_y * 2)
        latent[dy:dy + 1000, dx:dx + 1000] = ref_raw
        x0, y0 = dx, dy
    search_raw = cv2.resize(latent, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = apply_noise(degrade(ref_raw, seed + 11, blur_sigma=1.0, rotation_deg=float(rng.uniform(-0.12, 0.12)), contrast=float(rng.uniform(0.98, 1.02)), edge_gain=float(rng.uniform(0.99, 1.01))), seed + 101, shot_scale=55, detector_sigma=0.8)
    search = apply_noise(degrade(search_raw, seed + 22, blur_sigma=0.35, rotation_deg=float(rng.uniform(-0.08, 0.08)), contrast=float(rng.uniform(0.99, 1.01)), edge_gain=float(rng.uniform(0.995, 1.005)),), seed + 202, shot_scale=55, detector_sigma=0.8)
    np.save(out_dir / f"{ident}_reference.npy", reference.astype(np.float32))
    np.save(out_dir / f"{ident}_search.npy", search.astype(np.float32))
    return {"id": ident, "x_true": x0 / 10.0 + 50.0, "y_true": y0 / 10.0 + 50.0, "seed": seed, "pitch_x_latent": pitch_x, "pitch_y_latent": pitch_y, "mode": "continuous_10000_latent", "anchor": anchor, "hard": hard}


def make_split(root: Path, split: str, count: int, seed_offset: int, anchor: bool, hard: bool, ood: bool, full: bool) -> None:
    out = root / split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = out / "configs"
    configs.mkdir(exist_ok=True)
    case_fn = _full_case if full else _fast_case
    for i in range(count):
        ident = f"{split}_{i:02d}"
        row = case_fn(out, ident, seed_offset + i, anchor=anchor, hard=hard, ood=ood)
        rows.append(row)
        (configs / f"{ident}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference", "search"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "reference": f'{row["id"]}_reference.npy', "search": f'{row["id"]}_search.npy'})
    with (out / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x_true", "y_true"])
        writer.writeheader()
        writer.writerows([{k: row[k] for k in ("id", "x_true", "y_true")} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DriftSense-FM local diagnostics")
    parser.add_argument("--output", type=Path, default=Path("data/diagnostic_30case"))
    parser.add_argument("--fast", action="store_true", help="generate compact 1000x1000 fixtures quickly")
    parser.add_argument("--full", action="store_true", help="generate true 10000x10000 latent scenes before downsampling; CPU/memory intensive")
    args = parser.parse_args()
    if args.fast and args.full:
        parser.error("choose only one of --fast or --full")
    full = bool(args.full)
    if not args.fast and not args.full:
        print("No mode selected; using --fast compact fixtures. Pass --full for continuous 10000x10000 latent scenes.")
    args.output.mkdir(parents=True, exist_ok=True)
    make_split(args.output, "id_visible_anchor", 20, 1000, True, False, False, full)
    make_split(args.output, "ood_visible_anchor", 10, 2000, True, False, True, full)
    make_split(args.output, "hard_ambiguous", 5, 3000, False, True, False, full)


if __name__ == "__main__":
    main()
