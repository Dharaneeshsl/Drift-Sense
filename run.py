#!/usr/bin/env python3
"""DriftSense-FM command line entry point."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from driftsense.calibration import calibrate_reference
from driftsense.candidates import generate_candidates
from driftsense.diagnostics import build_diagnostics
from driftsense.io_utils import discover_pairs, load_array, write_outputs
from driftsense.scoring import score_candidate, estimate_pitch
from driftsense.subpixel import refine_peak
from driftsense.tiebreak import choose_candidate


def predict(reference, search, top_k: int = 16, min_distance: int = 12, score_margin: float = 0.015, profile: str = "full", official: bool = True):
    canonical_profile = "full" if profile == "driftsense_fm_full" else profile
    template = calibrate_reference(reference, official=official)
    bundle = generate_candidates(template, search, top_k=top_k, min_distance=min_distance, profile=canonical_profile)
    if not bundle["points"]:
        raise RuntimeError("candidate generation produced no valid placements")
    scored = []
    pitches = {}
    for y, x in bundle["points"]:
        refined_x, refined_y = refine_peak(bundle["coarse_map"], y, x)
        base = {
            "intensity": float(bundle["intensity_map"][y, x]),
            "squared_error_agreement": float(bundle["squared_map"][y, x]),
            "gradient": float(bundle["gradient_map"][y, x]),
            "high_pass": float(bundle["high_pass_map"][y, x]),
        }
        candidate, pitch = score_candidate(bundle["template"], bundle["search"], y, x, base, profile=canonical_profile)
        candidate["center_x"] = float(refined_x + template.shape[1] / 2.0)
        candidate["center_y"] = float(refined_y + template.shape[0] / 2.0)
        scored.append(candidate)
        pitches[id(candidate)] = pitch
    selected, margin, ambiguous = choose_candidate(scored, search.shape, margin=score_margin)
    pitch = pitches.get(id(selected), estimate_pitch(bundle["search"]))
    diag = build_diagnostics(selected, scored, margin, ambiguous, pitch)
    return {"x": float(selected["center_x"]), "y": float(selected["center_y"]), "confidence": float(diag["confidence"])}, diag


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline fixed-scale DriftSense-FM localization")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dev", action="store_true", help="allow non-1000x1000 arrays for local tests")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--score-margin", type=float, default=0.015)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.min_distance < 0:
        parser.error("--min-distance must be non-negative")
    if args.score_margin < 0:
        parser.error("--score-margin must be non-negative")
    pairs = discover_pairs(args.input_dir, official=not args.dev)
    predictions, diagnostics = [], {}
    for pair in pairs:
        pair_started = time.perf_counter()
        reference = load_array(pair["reference"], "reference", official=not args.dev)
        search = load_array(pair["search"], "search", official=not args.dev)
        prediction, diag = predict(reference, search, top_k=args.top_k, min_distance=args.min_distance, score_margin=args.score_margin, official=not args.dev)
        prediction["id"] = pair["id"]
        predictions.append({"id": pair["id"], "x": prediction["x"], "y": prediction["y"], "confidence": prediction["confidence"]})
        diagnostics[pair["id"]] = {**diag, "id": pair["id"], "runtime_seconds": time.perf_counter() - pair_started}
    write_outputs(args.output_dir, predictions, diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
