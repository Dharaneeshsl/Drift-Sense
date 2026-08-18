import tempfile
import unittest
from pathlib import Path

import numpy as np

from driftsense.calibration import calibrate_reference
from driftsense.io_utils import load_array


class CalibrationTests(unittest.TestCase):
    def test_area_calibration_shape(self):
        arr = np.arange(1_000_000, dtype=np.float32).reshape(1000, 1000)
        out = calibrate_reference(arr)
        self.assertEqual(out.shape, (100, 100))
        self.assertTrue(np.isfinite(out).all())

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_reference(np.zeros((20, 20), dtype=np.float32))

    def test_loader_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.npy"
            np.save(p, np.full((1000, 1000), np.nan, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_array(p, "reference")


if __name__ == "__main__":
    unittest.main()
