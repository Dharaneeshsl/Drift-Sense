"""Deterministic score-margin tie-breaking."""
from __future__ import annotations

import math
from typing import Any

DEFAULT_SCORE_MARGIN = 0.015


def choose_candidate(candidates: list[dict[str, Any]], image_shape: tuple[int, int], margin: float = DEFAULT_SCORE_MARGIN) -> tuple[dict[str, Any], float, bool]:
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError(f"image_shape must contain two positive dimensions; received {image_shape}")
    if not np_is_finite_nonnegative(margin):
        raise ValueError(f"score margin must be finite and non-negative; received {margin}")
    ranked = sorted(candidates, key=lambda c: float(c["final_score"]), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    score_margin = float(best["final_score"] - second["final_score"]) if second else 1.0
    ambiguous = bool(second and score_margin <= margin)
    if ambiguous:
        center_y, center_x = image_shape[0] / 2.0, image_shape[1] / 2.0
        eligible = (c for c in ranked if float(ranked[0]["final_score"] - c["final_score"]) <= margin)
        best = min(eligible, key=lambda c: math.hypot(float(c["center_x"]) - center_x, float(c["center_y"]) - center_y))
    return best, score_margin, ambiguous


def np_is_finite_nonnegative(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) >= 0.0
