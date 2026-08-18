import unittest

import numpy as np

from driftsense.candidates import generate_candidates


class CandidateTests(unittest.TestCase):
    def test_valid_maps_and_points(self):
        rng = np.random.default_rng(7)
        template = rng.normal(size=(30, 30)).astype(np.float32)
        search = rng.normal(size=(80, 80)).astype(np.float32)
        result = generate_candidates(template, search, top_k=6, min_distance=5)
        self.assertEqual(result["intensity_map"].shape, (51, 51))
        self.assertEqual(result["coarse_map"].shape, (51, 51))
        self.assertLessEqual(len(result["points"]), 6)
        for y, x in result["points"]:
            self.assertGreaterEqual(y, 0)
            self.assertGreaterEqual(x, 0)
            self.assertLess(y, 51)
            self.assertLess(x, 51)


if __name__ == "__main__":
    unittest.main()
