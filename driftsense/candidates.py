"""Candidate generation for periodic scenes."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .calibration import normalize_image


def _match(template: np.ndarray, image: np.ndarray, method: int) -> np.ndarray:
    return cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32), method)


def _to_unit(response: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip((response.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0), nan=0.0)


def _local_maxima(score: np.ndarray, count: int, min_distance: int) -> list[tuple[int, int]]:
    work = score.copy()
    points: list[tuple[int, int]] = []
    for _ in range(max(count * 4, count)):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        value = float(work[y, x])
        if not np.isfinite(value) or value <= -1e5:
            break
        if all((x - px) ** 2 + (y - py) ** 2 >= min_distance ** 2 for py, px in points):
            points.append((int(y), int(x)))
            if len(points) >= count:
                break
        yy0, yy1 = max(0, y - min_distance), min(work.shape[0], y + min_distance + 1)
        xx0, xx1 = max(0, x - min_distance), min(work.shape[1], x + min_distance + 1)
        work[yy0:yy1, xx0:xx1] = -1e6
    return points


def generate_candidates(template: np.ndarray, search: np.ndarray, top_k: int = 16, min_distance: int = 12, profile: str = "full") -> dict[str, Any]:
    t = normalize_image(template)
    s = normalize_image(search)
    if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
        raise ValueError("calibrated template must not exceed search dimensions")
    gradient_t = cv2.magnitude(cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3))
    gradient_s = cv2.magnitude(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3))
    high_t = cv2.Laplacian(t, cv2.CV_32F)
    high_s = cv2.Laplacian(s, cv2.CV_32F)
    intensity = _to_unit(_match(t, s, cv2.TM_CCOEFF_NORMED))
    squared = 1.0 - np.clip(_match(t, s, cv2.TM_SQDIFF_NORMED), 0.0, 1.0)
    gradient = _to_unit(_match(gradient_t, gradient_s, cv2.TM_CCOEFF_NORMED))
    high_pass = _to_unit(_match(high_t, high_s, cv2.TM_CCOEFF_NORMED))
    if profile == "intensity_only":
        coarse = intensity
    elif profile == "+gradient":
        coarse = 0.70 * intensity + 0.30 * gradient
    elif profile == "+high_pass":
        coarse = 0.55 * intensity + 0.25 * gradient + 0.20 * high_pass
    elif profile == "+squared_error_agreement":
        coarse = 0.40 * intensity + 0.20 * gradient + 0.15 * high_pass + 0.25 * squared
    else:
        coarse = 0.35 * intensity + 0.20 * squared + 0.20 * gradient + 0.15 * high_pass
    points = _local_maxima(coarse, top_k, min_distance)
    return {
        "template": t, "search": s, "intensity_map": intensity, "squared_map": squared,
        "gradient_map": gradient, "high_pass_map": high_pass, "coarse_map": coarse,
        "points": points,
    }
