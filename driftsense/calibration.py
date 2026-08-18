"""Known-scale calibration from 1000x1000 reference to 100x100 search-scale template."""
from __future__ import annotations

import cv2
import numpy as np

REFERENCE_SIZE = (1000, 1000)
TEMPLATE_SIZE = (100, 100)
SCALE_RATIO = 10


def calibrate_reference(reference: np.ndarray, official: bool = True) -> np.ndarray:
    arr = np.asarray(reference)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("reference must be a numeric 2-D array or (H,W,1)")
    if official and tuple(arr.shape) != REFERENCE_SIZE:
        raise ValueError(f"reference must have shape {REFERENCE_SIZE} in official mode; received {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("reference contains non-finite values")
    h, w = arr.shape
    target = (w // SCALE_RATIO, h // SCALE_RATIO)
    if target[0] < 2 or target[1] < 2:
        raise ValueError("reference is too small for fixed 10x calibration")
    return cv2.resize(arr.astype(np.float32), target, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
