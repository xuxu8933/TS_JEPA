import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.stage00_baselines import forecast, reconstruct_rows
from src.stage00_comparison import evaluate, fit, summarize, load_sine_splits, GRUModel, PretrainedModel
from src.stage00_legacy import _sin_cos_rows, _write_rows, _naive_last_predictions


class Stage00ComparisonTest(unittest.TestCase):
    def test_forecast_baselines_have_hand_calculated_multistep_outputs(self):
        context = torch.tensor([[[1., 10.], [3., 20.], [5., 30.]]])
        for method, expected in [
            ("Naive-last", [[[5., 30.], [5., 30.]]]),
            ("Mean-context", [[[3., 20.], [3., 20.]]]),
            ("Drift", [[[7., 40.], [9., 50.]]]),
        ]:
            torch.testing.assert_close(forecast(context, 2, method), torch.tensor(expected))
        torch.testing.assert_close(forecast(context[:, :1], 2, "Drift"), context[:, :1].expand(-1, 2, -1))

    def test_reconstruction_baselines_never_read_masked_rows(self):
        rows = torch.tensor([[[99.], [2.], [99.], [6.], [99.]]])
        masks, visible = torch.tensor([[0, 2, 4]]), torch.tensor([[1, 3]])
        # Previous *visible* row, earliest-visible fallback at the top boundary.
        torch.testing.assert_close(reconstruct_rows(rows, masks, visible, "Naive-last"), torch.tensor([[[2.], [2.], [6.]]]))
        torch.testing.assert_close(reconstruct_rows(rows, masks, visible, "Mean-context"), torch.tensor([[[4.], [4.], [4.]]]))
        # Drift uses actual row distances, with zero slope until two rows exist.
        torch.testing.assert_close(reconstruct_rows(rows, masks, visible, "Drift"), torch.tensor([[[2.], [2.], [8.]]]))
        changed = rows.clone()
        changed[:, masks[0]] = -999
        for method in ("Naive-last", "Mean-context", "Drift"):
            torch.testing.assert_close(reconstruct_rows(rows, masks, visible, method), reconstruct_rows(changed, masks, visible, method))

    def test_rmse_pools_all_values_and_rejects_nonfinite_predictions(self):
        loader = DataLoader(TensorDataset(torch.zeros(3, 1), torch.tensor([[0.], [0.], [3.]])), batch_size=2)
        result = evaluate(lambda x: x, loader)
        self.assertAlmostEqual(result["rmse"], 3 ** 0.5, places=6)
        with self.assertRaises(ValueError):
            evaluate(lambda x: x * float("nan"), loader)

    def test_learned_reconstruction_cannot_see_masked_pixels(self):
        from src.models.encoder import Encoder
        from src.models.predictor import Predictor

        config = dict(encoder_kernel_size=3, encoder_embed_dim=8, encoder_embed_bias=True,
                      encoder_nhead=2, encoder_num_layers=1, predictor_embed=8,
                      predictor_nhead=2, predictor_num_layers=1, decoder_type="mlp",
                      decoder_hidden_dim=8, decoder_num_layers=2, decoder_dropout=0.)
        checkpoint = {"config": config,
                      "encoder": Encoder(28, 28, 3, 8, True, 2, 1, jepa=True).state_dict(),
                      "predictor": Predictor(28, 8, 8, 2, 1).state_dict()}
        for model in (PretrainedModel(checkpoint, "MNIST_ROWS"), GRUModel("MNIST_ROWS")):
            model.eval()
            for batch_size in (1, 3):
                rows = torch.rand(batch_size, 28, 28)
                masks = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]).expand(batch_size, -1)
                visible = torch.tensor([[i for i in range(28) if i not in masks[0]]]).expand(batch_size, -1)
                changed = rows.clone()
                changed[:, masks[0]] = 1000
                predicted = model(rows, masks, visible)
                self.assertEqual(predicted.shape, (batch_size, 11, 28))
                torch.testing.assert_close(predicted, model(changed, masks, visible))
            if isinstance(model, PretrainedModel):
                predicted.sum().backward()
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()))

    def test_best_validation_state_is_restored(self):
        # Train toward +1; validation prefers the initial neighborhood of zero.
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        train = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)))
        val = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)))
        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, train, val, epochs=3, lr=0.1, checkpoint=Path(tmp) / "best.pt")
            self.assertEqual(result["selected_epoch"], 1)
            self.assertLess(result["best_validation_loss"], result["final_validation_loss"])
            self.assertAlmostEqual(evaluate(model, val)["rmse"], result["best_validation_loss"])

    def test_sine_split_targets_are_disjoint_and_naive_matches_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sin.csv"
            _write_rows(path, _sin_cos_rows())
            train, val, test, audit = load_sine_splits(path)
            self.assertEqual([len(x.dataset) for x in (train, val, test)], [96, 10, 76])
            self.assertTrue(audit["target_splits_disjoint"])
            old_pred, old_target = _naive_last_predictions(test)
            result = evaluate(lambda x: forecast(x.reshape(-1, 20, 2)[:, :, :1], 4, "Naive-last").squeeze(-1), test)
            torch.testing.assert_close(torch.from_numpy(result["predictions"]), old_pred)
            torch.testing.assert_close(torch.from_numpy(result["targets"]), old_target)

    def test_summary_reports_sample_std_and_single_seed_null(self):
        rows = [{"dataset": "MNIST_ROWS", "model": "GRU", "seed": s, "rmse": r} for s, r in [(7, 1.), (8, 3.)]]
        result = summarize(rows)[0]
        self.assertEqual(result["mean_rmse"], 2.)
        self.assertAlmostEqual(result["std_rmse"], 2 ** 0.5)
        self.assertEqual(result["num_runs"], 2)
        self.assertIsNone(summarize(rows[:1])[0]["std_rmse"])
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])


if __name__ == "__main__":
    unittest.main()
