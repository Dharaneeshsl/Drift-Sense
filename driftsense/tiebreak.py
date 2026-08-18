"""Deterministic score-margin tie-breaking."""
from __future__ import annotations

import math
from typing import Any

DEFAULT_SCORE_MARGIN = 0.015


def choose_candidate(candidates: list[dict[str, Any]], image_shape: tuple[int, int], margin: float = DEFAULT_SCORE_MARGIN) -> tuple[dict[str, Any], float, bool]:
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    ranked = sorted(candidates, key=lambda c: float(c["final_score"]), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    score_margin = float(best["final_score"] - second["final_score"]) if second else 1.0
    ambiguous = bool(second and score_margin <= margin)
    if ambiguous:
        cy, cx = (np_center := (image_shape[0] / 2.0, image_shape[1] / 2.0))
        best = min((c for c in ranked if float(ranked[0]["final_score"] - c["final_score"]) <= margin), key=lambda c: math.hypot(float(c["center_x"]) - cx, float(c["center_y"]) - cy))
    return best, score_margin, ambiguous
