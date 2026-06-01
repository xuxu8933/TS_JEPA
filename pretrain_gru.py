import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loaders.data_loader import get_evaluation_loaders


class GRUPretrainModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=5,
        dropout=0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        out, h = self.gru(x)

        # last hidden state
        last = out[:, -1, :]

        # predict next patch
        pred = self.head(last)

        return pred


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_data = "data/nike/nike.csv"
    batch_size = 32
    num_epochs = 100
    lr = 1e-3

    loader = get_evaluation_loaders(
        path=path_data,
        batch_size=batch_size,
    )

    model = GRUPretrainModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=5,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for context_patches, target_patch in loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)

            # context_patches: [batch, 12, 5]
            # flatten to [batch, 60, 1]
            x = context_patches.reshape(
                context_patches.size(0),
                -1,
                1
            )

            y = target_patch  # [batch, 5]

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch:03d} | GRU pretrain MSE: {avg_loss:.6f}")

    torch.save(
        {
            "gru": model.gru.state_dict(),
            "head": model.head.state_dict(),
            "model": model.state_dict(),
        },
        "gru_pretrained.pt"
    )

    print("Saved to gru_pretrained.pt")


if __name__ == "__main__":
    main()