import unittest

import numpy as np

from driftsense.subpixel import refine_peak
from driftsense.tiebreak import choose_candidate


class SubpixelTests(unittest.TestCase):
    def test_quadratic_peak_refines_toward_true_location(self):
        y, x = np.indices((9, 9))
        response = 1.0 - ((x - 4.25) ** 2 + (y - 3.7) ** 2) / 30.0
        rx, ry = refine_peak(response.astype(np.float32), 4, 4)
        self.assertAlmostEqual(rx, 4.25, delta=0.08)
        self.assertAlmostEqual(ry, 3.7, delta=0.08)

    def test_center_tie_break(self):
        candidates = [
            {"center_x": 100.0, "center_y": 100.0, "final_score": 0.900},
            {"center_x": 500.0, "center_y": 500.0, "final_score": 0.895},
        ]
        selected, margin, ambiguous = choose_candidate(candidates, (1000, 1000), margin=0.01)
        self.assertTrue(ambiguous)
        self.assertAlmostEqual(selected["center_x"], 500.0)
        self.assertAlmostEqual(selected["center_y"], 500.0)
        self.assertAlmostEqual(margin, 0.005)


if __name__ == "__main__":
    unittest.main()
