import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class EndToEndTests(unittest.TestCase):
    def test_cli_outputs_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, out = root / "input", root / "output"
            inp.mkdir()
            search = np.zeros((1000, 1000), dtype=np.float32)
            patch = np.zeros((100, 100), dtype=np.float32)
            cv2.rectangle(patch, (10, 10), (35, 40), 220, -1)
            cv2.circle(patch, (72, 70), 9, 90, -1)
            search[430:530, 370:470] = patch
            reference = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)
            np.save(inp / "reference.npy", reference)
            np.save(inp / "search.npy", search)
            subprocess.run([sys.executable, "run.py", str(inp), str(out)], cwd=Path(__file__).resolve().parents[1], check=True)
            with (out / "predictions.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "pair")
            self.assertTrue(0 <= float(rows[0]["x"]) <= 1000)
            self.assertTrue(0 <= float(rows[0]["y"]) <= 1000)
            self.assertTrue((out / "pair.json").exists())


if __name__ == "__main__":
    unittest.main()
