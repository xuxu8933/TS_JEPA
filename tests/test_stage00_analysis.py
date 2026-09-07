import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.analyze_stage00_results import load_predictions


class Stage00AnalysisTest(unittest.TestCase):
    def test_saved_rmse_is_checked_against_all_prediction_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            target = np.zeros((2, 4), dtype=np.float32)
            prediction = target.copy()
            prediction[1, 3] = 8
            np.savez(path, predictions=prediction, targets=target)
            row = {"predictions_path": str(path), "rmse": math.sqrt(8),
                   "dataset": "SIN_COS", "num_test_values": 8}
            errors, loaded = load_predictions(row)
            np.testing.assert_array_equal(errors.mean(1), [0, 16])
            np.testing.assert_array_equal(loaded, target)
            with self.assertRaises(ValueError):
                load_predictions({**row, "rmse": 2.0})  # Wrong: mean of sample RMSEs.

    def test_nonfinite_or_broadcastable_mismatched_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            row = {"predictions_path": str(path), "rmse": 0.,
                   "dataset": "SIN_COS", "num_test_values": 8}
            for predictions in (np.full((2, 4), np.nan), np.zeros((2, 1))):
                np.savez(path, predictions=predictions, targets=np.zeros((2, 4)))
                with self.assertRaises(ValueError):
                    load_predictions(row)


if __name__ == "__main__":
    unittest.main()
