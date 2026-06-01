"""
Visualize one real pretraining sample as an image.

This script loads the same pretraining data used by pretrain.py,
takes one sample from the JEPA pretrain dataloader,
and converts the numerical patch values into an image / heatmap.

Example:
    python visualize_pretrain_sample_as_image.py \
        --sample_idx 0 \
        --save_path ./results/pretrain_sample_image.png
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from config.config_pretrain import config
from main.utils import prepare_args, prepare_args_pretrain
from src.data_loaders.data_loader import get_jepa_loaders


def tensor_to_numpy(x):
    """
    Convert torch tensor to numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_sample_from_batch(batch, sample_idx=0):
    """
    Extract one sample from a pretraining batch.

    Different TS-JEPA versions may return different batch formats, for example:

        batch = x
        batch = (x, mask)
        batch = (context, target)
        batch = (context, target, mask)
        batch = {"x": ..., "mask": ...}

    This function tries to robustly find the tensor that represents
    the actual time-series patches.

    Expected output shape is usually:
        [num_patches, patch_size]
    or:
        [channels, num_patches, patch_size]
    """

    # Case 1: batch is a dict
    if isinstance(batch, dict):
        possible_keys = [
            "x",
            "sample",
            "samples",
            "data",
            "input",
            "inputs",
            "context",
            "context_patches",
        ]

        x = None
        for key in possible_keys:
            if key in batch:
                x = batch[key]
                break

        if x is None:
            raise ValueError(
                f"Cannot find data tensor in batch dict. Available keys: {batch.keys()}"
            )

    # Case 2: batch is tuple or list
    elif isinstance(batch, (tuple, list)):
        # Usually the first element is the actual data tensor
        x = batch[0]

    # Case 3: batch itself is tensor
    else:
        x = batch

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    # x may be:
    # [B, num_patches, patch_size]
    # [B, C, num_patches, patch_size]
    # [num_patches, patch_size]
    # [B, series_len]
    if x.dim() == 4:
        # [B, C, num_patches, patch_size]
        sample = x[sample_idx]

    elif x.dim() == 3:
        # [B, num_patches, patch_size]
        sample = x[sample_idx]

    elif x.dim() == 2:
        # Already [num_patches, patch_size]
        sample = x

    elif x.dim() == 1:
        # One-dimensional series
        sample = x

    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    return sample


def sample_to_image_matrix(sample):
    """
    Convert one sample into a 2D matrix for image visualization.

    Input can be:
        [num_patches, patch_size]
        [channels, num_patches, patch_size]
        [series_len]

    Output:
        2D numpy array
    """

    sample_np = tensor_to_numpy(sample)

    if sample_np.ndim == 3:
        # [channels, num_patches, patch_size]
        # If multivariate, average channels for one image.
        # You can also choose sample_np[0] to visualize only one channel.
        sample_img = sample_np.mean(axis=0)

    elif sample_np.ndim == 2:
        # [num_patches, patch_size]
        sample_img = sample_np

    elif sample_np.ndim == 1:
        # Try to reshape 1D series into patches.
        # You can change patch_size if needed.
        raise ValueError(
            "Sample is 1D. Please reshape manually using known patch_size, "
            "for example sample.reshape(num_patches, patch_size)."
        )

    else:
        raise ValueError(f"Unsupported sample ndim: {sample_np.ndim}")

    return sample_img


def plot_sample_image(
    sample_img,
    save_path="./results/pretrain_sample_image.png",
    cmap="viridis",
    title="Pretrain Sample as Image",
):
    """
    Plot sample matrix as image.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    im = plt.imshow(
        sample_img,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(im, label="value")

    plt.xlabel("time step inside patch")
    plt.ylabel("patch index")
    plt.title(title)

    # Show grid lines
    num_patches, patch_size = sample_img.shape

    plt.xticks(np.arange(patch_size))
    plt.yticks(np.arange(num_patches))

    plt.grid(
        which="both",
        color="white",
        linestyle="-",
        linewidth=0.5,
        alpha=0.5,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved image to: {save_path}")
    print(f"Image matrix shape: {sample_img.shape}")
    print(f"Value range: min={sample_img.min():.6f}, max={sample_img.max():.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample inside the first batch.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/pretrain_sample_image.png",
        help="Path to save the generated image.",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap, for example viridis, gray, plasma, coolwarm.",
    )

    args = parser.parse_args()

    # Load config exactly like pretrain.py
    cfg = prepare_args_pretrain(config)

    # Load JEPA pretraining dataloaders
    loaders = get_jepa_loaders(cfg)

    # Different implementations may return:
    # train_loader
    # train_loader, val_loader
    # {"train": train_loader, "val": val_loader}
    if isinstance(loaders, dict):
        train_loader = loaders.get("train", None)
        if train_loader is None:
            raise ValueError(f"Cannot find train loader in loaders dict: {loaders.keys()}")

    elif isinstance(loaders, (tuple, list)):
        train_loader = loaders[0]

    else:
        train_loader = loaders

    # Get first batch
    batch = next(iter(train_loader))

    print("Loaded one batch from pretrain dataloader.")

    if isinstance(batch, torch.Tensor):
        print("Batch tensor shape:", batch.shape)
    elif isinstance(batch, (tuple, list)):
        print("Batch is tuple/list:")
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"  batch[{i}] shape: {item.shape}")
            else:
                print(f"  batch[{i}] type: {type(item)}")
    elif isinstance(batch, dict):
        print("Batch is dict:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")

    sample = extract_sample_from_batch(batch, sample_idx=args.sample_idx)

    print("Extracted sample shape:", sample.shape)

    sample_img = sample_to_image_matrix(sample)

    plot_sample_image(
        sample_img=sample_img,
        save_path=args.save_path,
        cmap=args.cmap,
        title="Real Pretrain Sample as Image",
    )


if __name__ == "__main__":
    main()