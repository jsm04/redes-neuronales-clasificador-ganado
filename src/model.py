import torch
from torch import nn


class CattleCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        height, width, channels = input_shape

        if channels != 3:
            raise ValueError(
                "The model expects RGB images with 3 channels."
            )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        feature_size = self._get_feature_size(
            height,
            width,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def _get_feature_size(
        self,
        height: int,
        width: int,
    ) -> int:
        with torch.no_grad():
            sample = torch.zeros(
                1,
                3,
                height,
                width,
            )

            output = self.features(sample)

        return output.flatten(1).shape[1]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    input_shape_config = config["model"]["input_shape"]

    if (
        not isinstance(input_shape_config, list)
        or len(input_shape_config) != 3
    ):
        raise ValueError(
            "model.input_shape must contain exactly "
            "three values: [height, width, channels]."
        )

    input_shape: tuple[int, int, int] = (
        int(input_shape_config[0]),
        int(input_shape_config[1]),
        int(input_shape_config[2]),
    )

    num_classes = int(
        config["model"]["num_classes"]
    )

    dropout_rate = float(
        config["model"]["regularization"]["dropout_rate"]
    )

    return CattleCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
