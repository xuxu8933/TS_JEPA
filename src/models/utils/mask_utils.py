"""
    Script containing utility functions related to the masking.

    ---
        - Contains the Apply mask function that is used for masking in the
          Encoder and the Predictor
"""

import torch


def _as_mask_list(masks):
    if isinstance(masks, torch.Tensor):
        if masks.dim() == 1:
            return [masks.unsqueeze(0)]
        if masks.dim() == 2:
            return [masks]
        raise ValueError(f"Expected mask tensor with 1 or 2 dims, got {masks.dim()}")
    return masks


def apply_mask(x, masks, concat=True, masked=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    
    all_x = []
    masks = _as_mask_list(masks)
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)
