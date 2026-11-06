"""Prevent the saved-output runner from accepting incomplete output."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "regression_runner", Path(__file__).with_name("main.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RegressionValidationTests(unittest.TestCase):
    def test_missing_frames_cannot_broadcast(self):
        accepted, _ = runner.compare_arrays(np.ones((1, 3)), np.ones((5, 3)))
        self.assertFalse(accepted)

    def test_missing_observable_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "actual.out"
            expected = Path(directory) / "expected.out"
            actual.write_text("step\n0\n1\n")
            expected.write_text("step kinetic\n0 1\n1 2\n")
            with self.assertRaisesRegex(AssertionError, "columns"):
                runner.compare_output(actual, expected)

    def test_saved_fixtures_are_accepted(self):
        for path in Path(__file__).with_name("cases").glob("*/simulation.out"):
            with self.subTest(case=path.parent.name):
                self.assertTrue(runner.compare_output(path, path))


if __name__ == "__main__":
    unittest.main()
