"""Human- and machine-readable diagnostics for every prediction."""
from __future__ import annotations

from typing import Any


def build_diagnostics(selected: dict[str, Any], candidates: list[dict[str, Any]], score_margin: float, ambiguous: bool, dominant_pitch: tuple[float, float], image_shape: tuple[int, int]) -> dict[str, Any]:
    top = []
    for c in sorted(candidates, key=lambda x: float(x["final_score"]), reverse=True)[:10]:
        top.append({"x": float(c["center_x"]), "y": float(c["center_y"]), "final_score": float(c["final_score"])})
    score = float(selected.get("final_score", 0.0))
    confidence = max(0.0, min(1.0, 0.5 * max(0.0, score) + 0.5 * min(1.0, score_margin / 0.05)))
    return {
        "x": float(selected["center_x"]),
        "y": float(selected["center_y"]),
        "confidence": confidence,
        "score_margin": float(score_margin),
        "top_candidates": top,
        "dominant_pitch": {"x": float(dominant_pitch[0]), "y": float(dominant_pitch[1])},
        "component_scores": {k: float(selected.get(k, 0.0)) for k in ("intensity", "squared_error_agreement", "gradient", "high_pass", "fourier_agreement", "lattice_consistency", "boundary_agreement")},
        "label": "periodic_ambiguity" if ambiguous else "high_confidence",
    }
