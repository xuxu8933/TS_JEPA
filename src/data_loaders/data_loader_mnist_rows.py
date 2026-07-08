import random

import torch
from torch.utils.data import DataLoader, Dataset


class MNISTRowDataset(Dataset):
    """Represent each 28x28 MNIST image as 28 row tokens of 28 pixels."""

    def __init__(
        self,
        root,
        train,
        sample_count,
        mask_ratio,
        download=False,
        seed=0,
        deterministic_masks=False,
    ):
        from torchvision.datasets import MNIST

        dataset = MNIST(root=root, train=train, download=download)
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        sample_count = min(sample_count, len(dataset))

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:sample_count]
        self.images = dataset.data[indices].float().div_(255.0)
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)
        self.deterministic_masks = deterministic_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_rows = self.images[idx]
        num_rows = image_rows.shape[0]
        num_masked = max(1, int(num_rows * self.mask_ratio))

        if self.deterministic_masks:
            rng = random.Random(self.seed + idx)
            mask_indices = rng.sample(range(num_rows), num_masked)
        else:
            mask_indices = random.sample(range(num_rows), num_masked)

        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
        is_masked = torch.zeros(num_rows, dtype=torch.bool)
        is_masked[mask_indices] = True
        non_mask_indices = torch.arange(num_rows)[~is_masked]
        return image_rows, mask_indices, non_mask_indices


def get_mnist_row_loader(
    root,
    batch_size,
    mask_ratio,
    train,
    sample_count,
    download=False,
    seed=0,
    deterministic_masks=False,
):
    dataset = MNISTRowDataset(
        root=root,
        train=train,
        sample_count=sample_count,
        mask_ratio=mask_ratio,
        download=download,
        seed=seed,
        deterministic_masks=deterministic_masks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
    )
