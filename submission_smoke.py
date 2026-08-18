#!/usr/bin/env python3
"""Run a clean local submission-contract smoke test without network access."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def make_pair(directory: Path, ident: str, x: int, y: int) -> None:
    patch = np.zeros((100, 100), dtype=np.float32)
    cv2.rectangle(patch, (10, 10), (35, 40), 220, -1)
    cv2.circle(patch, (72, 70), 9, 90, -1)
    search = np.zeros((1000, 1000), dtype=np.float32)
    search[y:y + 100, x:x + 100] = patch
    reference = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    np.save(directory / f"{ident}_reference.npy", reference)
    np.save(directory / f"{ident}_search.npy", search)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="driftsense_submission_smoke_") as tmp:
        root = Path(tmp)
        input_dir, output_dir = root / "input", root / "output"
        input_dir.mkdir()
        make_pair(input_dir, "smoke_a", 140, 180)
        make_pair(input_dir, "smoke_b", 520, 610)
        (input_dir / "pairs.csv").write_text(
            "id,reference,search\nsmoke_a,smoke_a_reference.npy,smoke_a_search.npy\nsmoke_b,smoke_b_reference.npy,smoke_b_search.npy\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), str(input_dir), str(output_dir)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout or "CLI smoke test failed")
        with (output_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {"smoke_a", "smoke_b"}
        if {row["id"] for row in rows} != expected or len(rows) != len(expected):
            raise RuntimeError(f"prediction IDs do not match input IDs: {rows}")
        for row in rows:
            x, y, confidence = float(row["x"]), float(row["y"]), float(row["confidence"])
            if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and 0.0 <= confidence <= 1.0):
                raise RuntimeError(f"invalid prediction row: {row}")
            diagnostics_path = output_dir / f"{row['id']}.json"
            if not diagnostics_path.exists():
                raise RuntimeError(f"missing diagnostics file: {diagnostics_path}")
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if "top_candidates" not in diagnostics or "runtime_seconds" not in diagnostics:
                raise RuntimeError(f"incomplete diagnostics for {row['id']}")
    print("submission smoke test: PASS (offline, official-sized, multi-pair contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
