"""Measured local diagnostic benchmark; never fabricates results."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = [("ID visible-anchor", "id_visible_anchor"), ("OOD visible-anchor", "ood_visible_anchor"), ("Hard ambiguous periodic", "hard_ambiguous")]


def run_split(split_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="driftsense_bench_") as tmp:
        output = Path(tmp) / "output"
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, str(ROOT / "run.py"), str(split_dir), str(output)], check=False, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "no subprocess output").strip()
            raise RuntimeError(f"benchmark CLI failed for {split_dir.name} with exit code {completed.returncode}: {details}")
        truth = {r["id"]: (float(r["x_true"]), float(r["y_true"])) for r in csv.DictReader((split_dir / "ground_truth.csv").open())}
        rows = list(csv.DictReader((output / "predictions.csv").open()))
        errors = []
        strict = relaxed = 0
        for row in rows:
            x, y = float(row["x"]), float(row["y"])
            tx, ty = truth[row["id"]]
            dx, dy = x - tx, y - ty
            err = float(np.hypot(dx, dy))
            errors.append(err)
            strict += int(abs(dx) <= 0.5 and abs(dy) <= 0.5)
            relaxed += int(err <= 1.0)
        count = len(rows)
        return {"cases": count, "strict": strict, "relaxed": relaxed, "mean_error": float(np.mean(errors)) if errors else float("nan"), "runtime": elapsed / max(count, 1), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/diagnostic_30case")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmark/reports/benchmark_results")
    args = parser.parse_args()
    results = {}
    for label, dirname in SPLITS:
        results[dirname] = run_split(args.data / dirname)
    args.report.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Local diagnostic benchmark", "", "These are measured local synthetic results, not official challenge results.", "", "| Split | Cases | Strict ±0.5px | Relaxed ≤1px | Mean Euclidean error | Mean runtime/pair |", "|---|---:|---:|---:|---:|---:|"]
    for label, dirname in SPLITS:
        r = results[dirname]
        lines.append(f"| {label} | {r['cases']} | {r['strict']}/{r['cases']} | {r['relaxed']}/{r['cases']} | {r['mean_error']:.3f} px | {r['runtime']:.3f} s |")
    args.report.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
