import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliContractTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run([sys.executable, "run.py", "--help"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input_dir", result.stdout)

    def test_missing_input_directory_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            result = subprocess.run(
                [sys.executable, "run.py", str(missing), str(Path(tmp) / "out")],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input directory does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
