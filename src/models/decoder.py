"""
    Script for the Decoder
    ---
        Class Decoder contains the decoder achitecture which is based on a
        simple Linear Layer.
"""

import torch.nn as nn


class LinearDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(emb_dim, patch_size)

    def forward(self, encoded_patch):
        return self.fc(encoded_patch)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        patch_size,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []
        dim_in = emb_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.LayerNorm(dim_in),
                    nn.Linear(dim_in, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim_in = hidden_dim

        layers.extend(
            [
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, patch_size),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, encoded_patch):
        return self.net(encoded_patch)


class ResidualMLPDecoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        patch_size,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.linear_head = nn.Linear(emb_dim, patch_size)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, patch_size),
        )

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, encoded_patch):
        return self.linear_head(encoded_patch) + self.residual_head(encoded_patch)
