"""Profile-specific ablation harness for the deterministic pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run import predict
from driftsense.io_utils import load_array

PROFILES = ["intensity_only", "+gradient", "+high_pass", "+squared_error_agreement", "+fourier_agreement", "driftsense_fm_full"]
SPLITS = [("ID", "id_visible_anchor"), ("OOD", "ood_visible_anchor"), ("Hard periodic", "hard_ambiguous")]


def evaluate(split_dir: Path, profile: str) -> tuple[int, int, float]:
    truth = {r["id"]: (float(r["x_true"]), float(r["y_true"])) for r in csv.DictReader((split_dir / "ground_truth.csv").open())}
    strict = 0
    errors = []
    for row in csv.DictReader((split_dir / "pairs.csv").open()):
        ref = load_array(split_dir / row["reference"], "reference")
        search = load_array(split_dir / row["search"], "search")
        canonical_profile = "full" if profile == "driftsense_fm_full" else profile
        pred, _ = predict(ref, search, profile=canonical_profile, top_k=16)
        tx, ty = truth[row["id"]]
        dx, dy = pred["x"] - tx, pred["y"] - ty
        strict += int(abs(dx) <= 0.5 and abs(dy) <= 0.5)
        errors.append(float(np.hypot(dx, dy)))
    return strict, len(errors), float(np.mean(errors)) if errors else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/diagnostic_30case")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmark/reports/ablation_table.md")
    args = parser.parse_args()
    rows = []
    for profile in PROFILES:
        values = {}
        for label, split in SPLITS:
            strict, count, mean = evaluate(args.data / split, profile)
            values[label] = (strict, count, mean)
        rows.append((profile, values))
    lines = ["# Ablation", "", "Each profile recomputes candidates and scoring. Visible-anchor splits are already discriminative; therefore these rows do not isolate periodicity-aware value. The real test is a future adversarial near-decoy set.", "", "| Profile | ID strict | OOD strict | Hard periodic strict | ID mean error | OOD mean error |", "|---|---:|---:|---:|---:|---:|"]
    for profile, values in rows:
        i, o, h = values["ID"], values["OOD"], values["Hard periodic"]
        lines.append(f"| {profile} | {i[0]}/{i[1]} | {o[0]}/{o[1]} | {h[0]}/{h[1]} | {i[2]:.3f} px | {o[2]:.3f} px |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_rows = []
    for profile, values in rows:
        json_rows.append({"profile": profile, "id": {"strict": values["ID"][0], "cases": values["ID"][1], "mean_error": values["ID"][2]}, "ood": {"strict": values["OOD"][0], "cases": values["OOD"][1], "mean_error": values["OOD"][2]}, "hard_periodic": {"strict": values["Hard periodic"][0], "cases": values["Hard periodic"][1], "mean_error": values["Hard periodic"][2]}})
    args.report.with_suffix(".json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
