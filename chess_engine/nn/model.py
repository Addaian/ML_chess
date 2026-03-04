"""Neural network model: PyTorch for training, numpy for runtime inference."""

import numpy as np


class NumpyChessNet:
    """Pure numpy forward pass — no PyTorch dependency at runtime."""

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w1 = data['w1']  # (768, 256)
        self.b1 = data['b1']  # (256,)
        self.w2 = data['w2']  # (256, 128)
        self.b2 = data['b2']  # (128,)
        self.w3 = data['w3']  # (128, 64)
        self.b3 = data['b3']  # (64,)
        self.w4 = data['w4']  # (64, 1)
        self.b4 = data['b4']  # (1,)

    def forward(self, features: np.ndarray) -> float:
        x = features @ self.w1 + self.b1
        x = np.maximum(x, 0)
        x = x @ self.w2 + self.b2
        x = np.maximum(x, 0)
        x = x @ self.w3 + self.b3
        x = np.maximum(x, 0)
        x = (x @ self.w4 + self.b4)[0]
        return float(np.tanh(x))


def get_chess_net():
    """Return a PyTorch ChessNet model instance. Imports torch lazily."""
    import torch
    import torch.nn as nn

    class ChessNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return ChessNet()


def export_weights(model, path: str):
    """Export PyTorch model weights to numpy .npz for runtime inference."""
    import numpy as np
    sd = model.state_dict()
    np.savez(
        path,
        w1=sd['net.0.weight'].detach().numpy().T,  # (in, out)
        b1=sd['net.0.bias'].detach().numpy(),
        w2=sd['net.2.weight'].detach().numpy().T,
        b2=sd['net.2.bias'].detach().numpy(),
        w3=sd['net.4.weight'].detach().numpy().T,
        b3=sd['net.4.bias'].detach().numpy(),
        w4=sd['net.6.weight'].detach().numpy().T,
        b4=sd['net.6.bias'].detach().numpy(),
    )
