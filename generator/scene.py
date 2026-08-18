"""Procedural grayscale semiconductor-like lattice scenes."""
from __future__ import annotations

import cv2
import numpy as np


def make_scene(size: int = 2000, pitch_x: int = 80, pitch_y: int = 72, line_width: int = 10, contact_size: int = 12, seed: int = 0, anchor: bool = True) -> np.ndarray:
    """Create a continuous scene. Use size=10000 for the official physical generator."""
    rng = np.random.default_rng(seed)
    scene = np.full((size, size), 28, dtype=np.float32)
    ox, oy = int(rng.integers(0, pitch_x)), int(rng.integers(0, pitch_y))
    for x in range(ox, size, pitch_x):
        cv2.line(scene, (x, 0), (x, size - 1), 150, max(1, line_width))
    for y in range(oy, size, pitch_y):
        cv2.line(scene, (0, y), (size - 1, y), 205, max(1, line_width))
    step_x, step_y = max(pitch_x, pitch_x * 2), max(pitch_y, pitch_y * 2)
    for y in range(oy, size, step_y):
        for x in range(ox, size, step_x):
            r = max(2, contact_size + int(rng.integers(-2, 3)))
            cv2.circle(scene, (x, y), r, 235, -1, lineType=cv2.LINE_AA)
    for _ in range(max(4, size // 250)):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        if rng.random() < 0.5:
            cv2.rectangle(scene, (x, y), (min(size - 1, x + int(rng.integers(3, 18))), min(size - 1, y + int(rng.integers(3, 18)))), int(rng.integers(60, 230)), -1)
    if anchor:
        ax, ay = int(size * 0.42), int(size * 0.57)
        cv2.rectangle(scene, (ax, ay), (min(size - 1, ax + pitch_x // 3), min(size - 1, ay + pitch_y // 3)), 252, -1)
        cv2.circle(scene, (min(size - 1, ax + pitch_x // 2), min(size - 1, ay + pitch_y // 2)), max(2, contact_size // 2), 8, -1)
    rough = rng.normal(0.0, 1.5, size=(size, size)).astype(np.float32)
    rough = cv2.GaussianBlur(rough, (0, 0), 1.2)
    return np.clip(scene + rough, 0.0, 255.0).astype(np.float32)
