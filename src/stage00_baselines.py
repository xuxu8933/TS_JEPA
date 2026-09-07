"""Context-only Stage 00 baselines; no fitted parameters or target values."""

import torch

METHODS = ("Naive-last", "Drift", "Mean-context")


def forecast(context, horizon, method):
    """[B, L, F] -> [B, H, F], in the input's normalization coordinates."""
    if context.ndim != 3 or context.shape[1] == 0 or horizon < 1:
        raise ValueError("Expected nonempty [B, L, F] context and positive horizon")
    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    if method == "Mean-context":
        return context.mean(1, keepdim=True).expand(-1, horizon, -1)
    last = context[:, -1:]
    if method == "Naive-last" or context.shape[1] == 1:
        return last.expand(-1, horizon, -1)
    steps = torch.arange(1, horizon + 1, device=context.device, dtype=context.dtype)
    slope = (last - context[:, :1]) / (context.shape[1] - 1)
    return last + steps[None, :, None] * slope


def reconstruct_rows(rows, masks, visible, method):
    """Predict masked rows using only visible values and actual row positions.

    Naive-last copies the preceding visible row. Drift extrapolates from the
    first and last preceding visible rows. At the top boundary both use the
    first visible row; with one preceding row drift has zero slope. This is
    reconstruction, so that boundary fallback may use right-hand context.
    Mean-context averages all visible rows, as available to the learned models.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    visible = visible.sort(dim=1).values
    values = rows.gather(1, visible[..., None].expand(-1, -1, rows.shape[-1]))
    if method == "Mean-context":
        return values.mean(1, keepdim=True).expand(-1, masks.shape[1], -1)
    preceding = (visible[:, None, :] < masks[:, :, None]).sum(-1)
    last_slot = (preceding - 1).clamp_min(0)
    last = values.gather(1, last_slot[..., None].expand(-1, -1, rows.shape[-1]))
    if method == "Naive-last":
        return last
    last_position = visible.gather(1, last_slot)
    span = (last_position - visible[:, :1]).clamp_min(1)
    slope = (last - values[:, :1]) / span[..., None]
    return last + (masks - last_position)[..., None] * slope
