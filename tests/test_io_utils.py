import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from driftsense.candidates import generate_candidates
from driftsense.diagnostics import build_diagnostics
from driftsense.io_utils import discover_pairs, write_outputs
from driftsense.reranker import available, rerank


class IOAndRobustnessTests(unittest.TestCase):
    def test_pairs_csv_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pairs.csv").write_text("id,reference,search\na,missing.npy,missing2.npy\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                discover_pairs(root)

    def test_pairs_csv_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("r.npy", "s.npy", "r2.npy", "s2.npy"):
                np.save(root / name, np.zeros((1000, 1000), dtype=np.float32))
            (root / "pairs.csv").write_text("id,reference,search\na,r.npy,s.npy\na,r2.npy,s2.npy\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                discover_pairs(root)

    def test_invalid_candidate_parameters_are_rejected(self):
        template = np.ones((10, 10), dtype=np.float32)
        search = np.ones((20, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            generate_candidates(template, search, top_k=0)
        with self.assertRaises(ValueError):
            generate_candidates(template, search, profile="unknown")

    def test_pairs_csv_and_glob_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a_reference.npy", "a_search.npy", "b_reference.npy", "b_search.npy"):
                np.save(root / name, np.zeros((1000, 1000), dtype=np.float32))
            pairs = root / "pairs.csv"
            pairs.write_text("id,reference,search\na,a_reference.npy,a_search.npy\nb,b_reference.npy,b_search.npy\n", encoding="utf-8")
            discovered = discover_pairs(root)
            self.assertEqual([row["id"] for row in discovered], ["a", "b"])
            pairs.unlink()
            discovered = discover_pairs(root)
            self.assertEqual([row["id"] for row in discovered], ["a", "b"])

    def test_diagnostics_and_output_contract(self):
        selected = {"center_x": 10.0, "center_y": 20.0, "final_score": 0.8, "intensity": 0.8, "gradient": 0.7}
        diag = build_diagnostics(selected, [selected], 0.2, False, (8.0, 9.0))
        self.assertIn("component_scores", diag)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(out, [{"id": "a", "x": 10.0, "y": 20.0, "confidence": 0.9}], {"a": diag})
            self.assertTrue((out / "predictions.csv").is_file())
            self.assertTrue((out / "a.json").is_file())

    def test_periodic_tie_is_marked_non_identifiable(self):
        selected = {"center_x": 10.0, "center_y": 20.0, "final_score": 0.8}
        decoy = {"center_x": 100.0, "center_y": 120.0, "final_score": 0.79}
        diag = build_diagnostics(selected, [selected, decoy], 0.01, True, (8.0, 9.0))
        self.assertEqual(diag["label"], "non_identifiable_periodic")
        self.assertTrue(diag["identifiability_warning"])

    def test_optional_reranker_fails_clearly_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(available(Path(tmp) / "missing.pt"))
        with self.assertRaises(RuntimeError):
            rerank(None)


if __name__ == "__main__":
    unittest.main()
