"""Capture noise and scan-distortion utilities."""
from __future__ import annotations

import cv2
import numpy as np


def apply_noise(image: np.ndarray, seed: int, shot_scale: float = 35.0, detector_sigma: float = 1.2, row_drift: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.clip(image.astype(np.float32), 0.0, 255.0)
    lam = np.clip(x / 255.0 * shot_scale, 0.0, None)
    shot = rng.poisson(lam).astype(np.float32) / max(shot_scale, 1e-6) * 255.0
    out = 0.92 * x + 0.08 * shot + rng.normal(0.0, detector_sigma, x.shape).astype(np.float32)
    if row_drift > 0:
        drift = cv2.GaussianBlur(rng.normal(0.0, row_drift, (x.shape[0], 1)).astype(np.float32), (1, 0), 8)
        out += drift
    return np.clip(out, 0.0, 255.0).astype(np.float32)
