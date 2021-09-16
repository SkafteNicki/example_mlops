from torch import Tensor, nn


class MyAwesomeModel(nn.Module):
    """
    Basic neural network
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),  # [N, 8, 20]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs inference on input x
        Args:
            x: tensor with shape [N, 1, 28, 28]

        Returns:
            log_probs: tensor with log probabilities with shape [N, 10]

        """
        return self.classifier(self.backbone(x))
