import unittest

from driftsense.tiebreak import choose_candidate


class TieBreakTests(unittest.TestCase):
    def test_clear_margin_uses_best_score(self):
        candidates = [
            {"center_x": 100.0, "center_y": 100.0, "final_score": 0.90},
            {"center_x": 500.0, "center_y": 500.0, "final_score": 0.80},
        ]
        selected, margin, ambiguous = choose_candidate(candidates, (1000, 1000), margin=0.01)
        self.assertFalse(ambiguous)
        self.assertEqual(selected["center_x"], 100.0)
        self.assertAlmostEqual(margin, 0.10)


if __name__ == "__main__":
    unittest.main()
