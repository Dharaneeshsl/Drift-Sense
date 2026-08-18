"""Imaging degradation transforms."""
from __future__ import annotations

import cv2
import numpy as np


def degrade(image: np.ndarray, seed: int, blur_sigma: float = 0.7, rotation_deg: float = 0.0, contrast: float = 1.0, edge_gain: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if abs(rotation_deg) > 1e-6:
        h, w = out.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_deg, 1.0)
        out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mean = float(out.mean())
    out = (out - mean) * contrast + mean
    if abs(edge_gain - 1.0) > 1e-6:
        lap = cv2.Laplacian(out, cv2.CV_32F)
        out = out + (edge_gain - 1.0) * lap
    out += rng.normal(0.0, 0.15, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.float32)
