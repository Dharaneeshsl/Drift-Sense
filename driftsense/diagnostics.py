"""Human- and machine-readable diagnostics for every prediction."""
from __future__ import annotations

from typing import Any


def build_diagnostics(selected: dict[str, Any], candidates: list[dict[str, Any]], score_margin: float, ambiguous: bool, dominant_pitch: tuple[float, float]) -> dict[str, Any]:
    top = []
    for c in sorted(candidates, key=lambda x: float(x["final_score"]), reverse=True)[:10]:
        top.append({"x": float(c["center_x"]), "y": float(c["center_y"]), "final_score": float(c["final_score"])})
    score = float(selected.get("final_score", 0.0))
    confidence = max(0.0, min(1.0, 0.5 * max(0.0, score) + 0.5 * min(1.0, score_margin / 0.05)))
    spatially_separated_tie = False
    if ambiguous and len(top) > 1:
        spatially_separated_tie = any(((c["x"] - top[0]["x"]) ** 2 + (c["y"] - top[0]["y"]) ** 2) ** 0.5 >= 25.0 for c in top[1:])
    label = "non_identifiable_periodic" if spatially_separated_tie else ("periodic_ambiguity" if ambiguous else "high_confidence")
    return {
        "x": float(selected["center_x"]),
        "y": float(selected["center_y"]),
        "confidence": confidence,
        "score_margin": float(score_margin),
        "top_candidates": top,
        "dominant_pitch": {"x": float(dominant_pitch[0]), "y": float(dominant_pitch[1])},
        "component_scores": {k: float(selected.get(k, 0.0)) for k in ("intensity", "squared_error_agreement", "gradient", "high_pass", "fourier_agreement", "lattice_consistency", "boundary_agreement")},
        "label": label,
        "identifiability_warning": bool(spatially_separated_tie),
    }
