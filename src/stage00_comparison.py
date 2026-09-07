"""Stage 00 models, shared validation/test evaluation, and result aggregation."""

import hashlib
import json
import math
import statistics
from collections import defaultdict

import torch
from torch import nn

from main.utils import init_weights, rmse_loss
from src.data_loaders.data_loader_mnist_rows import get_mnist_row_loader
from src.models.decoder import MLPDecoder, build_reconstruction_decoder
from src.models.encoder import Encoder
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask
from src.stage00_legacy import _load_forecast_loaders

METHODS = ("JEPA-MAE", "JEPA-only", "MAE-only", "GRU", "Naive-last", "Drift", "Mean-context")
DATASETS = ("MNIST_ROWS", "SIN_COS")


def load_mnist_splits(root, seed):
    common = dict(root=str(root), batch_size=64, mask_ratio=0.4)
    train = get_mnist_row_loader(**common, train=True, sample_count=4000, seed=seed)
    val = get_mnist_row_loader(**common, train=True, sample_count=128, seed=seed,
                               sample_offset=4000, deterministic_masks=True)
    test = get_mnist_row_loader(**common, train=False, sample_count=128,
                                seed=seed + 10000, deterministic_masks=True)
    train_ids = train.dataset.sample_indices.tolist()
    val_ids = val.dataset.sample_indices.tolist()
    assert set(train_ids).isdisjoint(val_ids)
    audit = {
        "train_ids": train_ids, "validation_ids": val_ids,
        "test_ids": test.dataset.sample_indices.tolist(),
        "train_validation_source": "official MNIST train",
        "test_source": "official MNIST test",
        "train_validation_disjoint": True,
        "image_counts": [len(x.dataset) for x in (train, val, test)],
        "mask_ratio": 0.4, "masked_rows_per_image": 11,
    }
    return train, val, test, audit


def load_sine_splits(path):
    loaders = _load_forecast_loaders(path, splits=("train", "val", "test"))
    target_dates = []
    for loader in loaders:
        ds = loader.dataset
        dates = set()
        context_length = ds.context_size * ds.patch_size
        for start in ds.sample_starts:
            origin = start + context_length - 1
            assert ds.dates[origin] < ds.dates[origin + 1]
            dates.update(ds.dates[origin + 1:origin + 1 + ds.forecast_horizon])
        target_dates.append(dates)
    assert all(target_dates[i].isdisjoint(target_dates[j]) for i in range(3) for j in range(i))
    ds = loaders[0].dataset
    assert max(ds.train_dates) < min(ds.val_dates) <= max(ds.val_dates) < min(ds.test_dates)
    audit = {
        "target_splits_disjoint": True,
        "context_precedes_target": True,
        "row_counts": [len(x) for x in (ds.train_df, ds.val_df, ds.test_df)],
        "sample_counts": [len(x.dataset) for x in loaders],
        "date_ranges": [[str(min(x).date()), str(max(x).date())]
                        for x in (ds.train_dates, ds.val_dates, ds.test_dates)],
        "target_date_ranges": [[str(min(x).date()), str(max(x).date())] for x in target_dates],
        "context_length": 20, "forecast_horizon": 4,
        "normalization": ds.normalization, "warmup_report": ds.warmup_report,
    }
    return *loaders, audit


class PretrainedModel(nn.Module):
    def __init__(self, checkpoint, dataset):
        super().__init__()
        config = checkpoint["config"]
        self.mnist = dataset == "MNIST_ROWS"
        self.encoder = Encoder(
            num_patches=28 if self.mnist else 5, dim_in=28 if self.mnist else 8,
            kernel_size=config["encoder_kernel_size"], embed_dim=config["encoder_embed_dim"],
            embed_bias=config["encoder_embed_bias"], nhead=config["encoder_nhead"],
            num_layers=config["encoder_num_layers"], jepa=True,
        )
        self.encoder.load_state_dict(checkpoint["encoder"])
        if self.mnist:
            self.predictor = Predictor(
                num_patches=28, encoder_embed_dim=config["encoder_embed_dim"],
                predictor_embed_dim=config["predictor_embed"], nhead=config["predictor_nhead"],
                num_layers=config["predictor_num_layers"],
            )
            self.predictor.load_state_dict(checkpoint["predictor"])
            # All ablations learn a fresh identical pixel readout. JEPA-only's
            # unused pretraining decoder must never be evaluated as a predictor.
            self.encoder.requires_grad_(False)
            self.predictor.requires_grad_(False)
            self.decoder = build_reconstruction_decoder(
                decoder_type=config["decoder_type"], embedding_dim=config["encoder_embed_dim"],
                output_dim=28, hidden_dim=config["decoder_hidden_dim"],
                num_layers=config["decoder_num_layers"], dropout=config["decoder_dropout"],
            )
        else:
            self.decoder = MLPDecoder(emb_dim=16, patch_size=4, hidden_dim=64,
                                      num_layers=2, dropout=0.0)
        for module in self.decoder.modules():
            init_weights(module)

    def forward(self, context, masks=None, visible=None):
        if self.mnist:
            self.encoder.eval()
            self.predictor.eval()
            with torch.no_grad():
                encoded = self.encoder(context, mask=visible)
                encoded = self.predictor(encoded, mask=masks, non_masks=visible)
        else:
            encoded = self.encoder(context).mean(1)
        return self.decoder(encoded)


class GRUModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # Reuse the project's existing GRU implementation without changing the
        # financial experiment module or its defaults.
        from eval_forecast_prequential_with_baselines_gru_volume import GRUForecastModel

        self.mnist = dataset == "MNIST_ROWS"
        self.model = GRUForecastModel(input_size=29 if self.mnist else 2,
                                      hidden_size=64, num_layers=1,
                                      output_size=784 if self.mnist else 4, dropout=0.0)

    def forward(self, context, masks=None, visible=None):
        if not self.mnist:
            return self.model(context)
        # The indicator distinguishes unobserved rows from observed black rows.
        observed = torch.zeros_like(context[:, :, :1])
        observed.scatter_(1, visible[..., None], 1)
        inputs = torch.cat([context.masked_fill(observed == 0, 0), observed], dim=-1)
        return apply_mask(self.model(inputs).reshape(-1, 28, 28), masks)


def prediction_and_target(model, batch, device):
    batch = [x.to(device) for x in batch]
    if len(batch) == 3:
        rows, masks, visible = batch
        return model(rows, masks, visible), apply_mask(rows, masks)
    context, target = batch
    return model(context), target


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """One RMSE over every target scalar (never an average of batch RMSEs)."""
    if isinstance(model, nn.Module):
        model.eval()
    predictions, targets = [], []
    identity = hashlib.sha256()
    for batch in loader:
        for tensor in batch:
            identity.update(tensor.cpu().contiguous().numpy().tobytes())
        pred, target = prediction_and_target(model, batch, device)
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred.shape} != {target.shape}")
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite predictions or targets")
        predictions.append(pred.cpu())
        targets.append(target.cpu())
    if not targets:
        raise ValueError("Cannot evaluate an empty loader")
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    # Identical expression to the original sine-cosine Stage 00 metric;
    # equivalent pooled squared-error/count definition to MNIST reconstruction.
    score = (predictions - targets).square().mean().sqrt().item()
    if not math.isfinite(score):
        raise ValueError("Nonfinite RMSE")
    return {"rmse": score, "predictions": predictions.numpy(), "targets": targets.numpy(),
            "evaluation_sha256": identity.hexdigest(), "num_values": targets.numel()}


def fit(model, train, val, *, epochs, lr, checkpoint, device="cpu"):
    """Fixed training budget; select only by pooled validation RMSE, then restore."""
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    best = math.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train:
            optimizer.zero_grad()
            pred, target = prediction_and_target(model, batch, device)
            loss = rmse_loss(pred, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite training gradient")
            optimizer.step()
        # DataLoader iterator creation consumes the global torch RNG even when
        # shuffle=False. Validation must not change the next training shuffle.
        with torch.random.fork_rng(devices=[]):
            score = evaluate(model, val, device)["rmse"]
        history.append({"epoch": epoch, "validation_rmse": score})
        if score < best:
            best, selected = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation_rmse": score}, checkpoint)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"  epoch={epoch}/{epochs} validation_rmse={score:.8f}", flush=True)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"best_validation_loss": best, "final_validation_loss": history[-1]["validation_rmse"],
            "selected_epoch": selected, "selected_checkpoint": str(checkpoint)}


def summarize(rows):
    groups, identities = defaultdict(list), set()
    for row in rows:
        key = (row["dataset"], row["model"], row["seed"])
        if key in identities:
            raise ValueError(f"Duplicate dataset/model/seed: {key}")
        identities.add(key)
        value = float(row["rmse"])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite RMSE: {key}")
        groups[key[:2]].append(value)
    return [{"dataset": dataset, "model": method, "mean_rmse": statistics.mean(values),
             "std_rmse": statistics.stdev(values) if len(values) > 1 else None,
             "num_runs": len(values)}
            for dataset in DATASETS for method in METHODS
            if (values := groups.get((dataset, method)))]
