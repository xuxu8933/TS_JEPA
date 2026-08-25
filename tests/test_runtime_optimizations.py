import unittest

import torch

from main.utils import ordered_scalar_mean


class OrderedScalarMeanTest(unittest.TestCase):
    def test_matches_ordered_item_mean_exactly(self):
        values = [
            torch.tensor(value, dtype=torch.float32, requires_grad=True)
            for value in (0.1, 1000.25, -0.3, 1.0 / 7.0)
        ]
        expected = sum(value.item() for value in values) / len(values)

        actual = ordered_scalar_mean(values)

        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, float)
        self.assertTrue(all(value.grad is None for value in values))

    def test_rejects_empty_values(self):
        with self.assertRaisesRegex(ValueError, "at least one scalar"):
            ordered_scalar_mean([])

    def test_rejects_non_scalar_tensor(self):
        with self.assertRaisesRegex(ValueError, "scalar tensors"):
            ordered_scalar_mean([torch.tensor([1.0, 2.0])])


if __name__ == "__main__":
    unittest.main()
