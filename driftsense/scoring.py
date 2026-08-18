"""Final deterministic scoring for candidate crops."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    aa, bb = aa - aa.mean(), bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-8 else 0.0


def estimate_pitch(image: np.ndarray) -> tuple[float, float]:
    gray = image.astype(np.float32)
    gx = np.abs(np.diff(gray.mean(axis=0)))
    gy = np.abs(np.diff(gray.mean(axis=1)))
    def one(values: np.ndarray) -> float:
        if values.size < 4 or float(values.max()) < 1e-8:
            return 0.0
        values = values - values.mean()
        corr = np.correlate(values, values, mode="full")[values.size - 1:]
        corr[:2] = -np.inf
        max_lag = min(values.size - 1, max(3, values.size // 2))
        return float(np.argmax(corr[:max_lag]))
    return one(gx), one(gy)


def _fourier_agreement(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a))))
    fb = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(b))))
    return float(np.clip((_cosine(fa, fb) + 1.0) * 0.5, 0.0, 1.0))


def _shift_similarity(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> float:
    h, w = a.shape
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    return float(np.clip((_cosine(a[y0:y1, x0:x1], b[y0-dy:y1-dy, x0-dx:x1-dx]) + 1.0) * 0.5, 0.0, 1.0))


def score_candidate(template: np.ndarray, search: np.ndarray, origin_y: int, origin_x: int, base: dict[str, float], profile: str = "full") -> tuple[dict[str, Any], tuple[float, float]]:
    h, w = template.shape
    crop = search[origin_y:origin_y + h, origin_x:origin_x + w]
    if crop.shape != template.shape:
        raise ValueError("candidate crop falls outside the valid search-placement map")
    template_n = template.astype(np.float32)
    crop_n = crop.astype(np.float32)
    fourier = _fourier_agreement(template_n, crop_n)
    pitch_x, pitch_y = estimate_pitch(crop_n)
    gt = cv2.magnitude(cv2.Sobel(template_n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(template_n, cv2.CV_32F, 0, 1, ksize=3))
    gc = cv2.magnitude(cv2.Sobel(crop_n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(crop_n, cv2.CV_32F, 0, 1, ksize=3))
    phase_scores = []
    for dx, dy in ((int(round(pitch_x)), 0), (-int(round(pitch_x)), 0), (0, int(round(pitch_y))), (0, -int(round(pitch_y)))):
        if dx or dy:
            phase_scores.append(_shift_similarity(gt, gc, dx, dy))
    shifted = float(np.mean(phase_scores)) if phase_scores else 0.5
    direct = float(np.clip((_cosine(gt, gc) + 1.0) * 0.5, 0.0, 1.0))
    lattice = float(np.clip(0.65 * direct + 0.35 * shifted, 0.0, 1.0))
    edge = np.concatenate([template_n[0, :], template_n[-1, :], template_n[:, 0], template_n[:, -1]])
    edge_crop = np.concatenate([crop_n[0, :], crop_n[-1, :], crop_n[:, 0], crop_n[:, -1]])
    boundary = float(np.clip(1.0 - np.mean(np.abs(edge - edge_crop)) / (np.std(template_n) + 1e-3), 0.0, 1.0))
    intensity = float(base.get("intensity", 0.0))
    squared = float(base.get("squared_error_agreement", 0.0))
    gradient = float(base.get("gradient", 0.0))
    high = float(base.get("high_pass", 0.0))
    if profile == "intensity_only":
        final = intensity
    elif profile == "+gradient":
        final = 0.70 * intensity + 0.30 * gradient
    elif profile == "+high_pass":
        final = 0.55 * intensity + 0.25 * gradient + 0.20 * high
    elif profile == "+squared_error_agreement":
        final = 0.40 * intensity + 0.20 * gradient + 0.15 * high + 0.25 * squared
    elif profile == "+fourier_agreement":
        final = 0.32 * intensity + 0.16 * gradient + 0.12 * high + 0.20 * squared + 0.20 * fourier
    else:
        final = 0.30 * intensity + 0.15 * squared + 0.18 * gradient + 0.12 * high + 0.10 * fourier + 0.10 * lattice + 0.05 * boundary
    return {**base, "fourier_agreement": fourier, "lattice_consistency": lattice, "boundary_agreement": boundary, "final_score": float(final), "origin_x": int(origin_x), "origin_y": int(origin_y), "center_x": float(origin_x + w / 2.0), "center_y": float(origin_y + h / 2.0)}, (pitch_x, pitch_y)
