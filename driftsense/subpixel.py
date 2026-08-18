"""Subpixel refinement for a response-map peak."""
from __future__ import annotations

import numpy as np


def _quadratic_delta(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def refine_peak(response: np.ndarray, y: int, x: int) -> tuple[float, float]:
    h, w = response.shape
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)
    dx = _quadratic_delta(float(response[y, x - 1]), float(response[y, x]), float(response[y, x + 1]))
    dy = _quadratic_delta(float(response[y - 1, x]), float(response[y, x]), float(response[y + 1, x]))
    return float(x + dx), float(y + dy)
