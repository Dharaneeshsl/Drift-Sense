import unittest

import numpy as np

from driftsense.scoring import estimate_pitch, score_candidate


class ScoringTests(unittest.TestCase):
    def test_pitch_is_finite(self):
        y, x = np.indices((100, 100))
        image = ((x % 12 < 3) | (y % 15 < 3)).astype(np.float32)
        px, py = estimate_pitch(image)
        self.assertTrue(np.isfinite(px))
        self.assertTrue(np.isfinite(py))

    def test_score_components_are_bounded(self):
        rng = np.random.default_rng(4)
        template = rng.random((20, 20), dtype=np.float32)
        search = rng.random((50, 50), dtype=np.float32)
        candidate, pitch = score_candidate(template, search, 10, 11, {"intensity": .8, "squared_error_agreement": .7, "gradient": .6, "high_pass": .5})
        for key in ("fourier_agreement", "lattice_consistency", "boundary_agreement", "final_score"):
            self.assertGreaterEqual(candidate[key], 0.0)
            self.assertLessEqual(candidate[key], 1.0)
        self.assertEqual(len(pitch), 2)


if __name__ == "__main__":
    unittest.main()
