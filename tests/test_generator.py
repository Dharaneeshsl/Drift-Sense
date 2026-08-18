import unittest

import numpy as np

from generator.degrade import degrade
from generator.noise import apply_noise
from generator.scene import make_scene


class GeneratorTests(unittest.TestCase):
    def test_scene_generation_is_seeded_and_finite(self):
        a = make_scene(128, pitch_x=24, pitch_y=20, line_width=3, contact_size=5, seed=7, anchor=True)
        b = make_scene(128, pitch_x=24, pitch_y=20, line_width=3, contact_size=5, seed=7, anchor=True)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape, (128, 128))
        self.assertTrue(np.isfinite(a).all())

    def test_degrade_and_noise_preserve_shape_and_seed(self):
        image = make_scene(128, pitch_x=24, pitch_y=20, line_width=3, contact_size=5, seed=8)
        degraded = degrade(image, seed=9, blur_sigma=0.4, rotation_deg=0.2, contrast=1.01, edge_gain=1.0)
        noisy_a = apply_noise(degraded, seed=10, shot_scale=40, detector_sigma=0.5)
        noisy_b = apply_noise(degraded, seed=10, shot_scale=40, detector_sigma=0.5)
        self.assertEqual(noisy_a.shape, image.shape)
        self.assertTrue(np.array_equal(noisy_a, noisy_b))
        self.assertTrue(np.isfinite(noisy_a).all())


if __name__ == "__main__":
    unittest.main()
