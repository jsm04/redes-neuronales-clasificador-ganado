import torch

from src.model import CattleCNN, build_model


def make_config() -> dict:
    return {
        "model": {
            "input_shape": [224, 224, 3],
            "num_classes": 2,
            "regularization": {
                "dropout_rate": 0.5,
            },
        },
        "training": {
            "learning_rate": 0.001,
        },
    }


def test_build_model() -> None:
    """CNN is constructed correctly."""
    model = build_model(
        make_config()
    )

    assert isinstance(
        model,
        CattleCNN,
    )


def test_model_has_trainable_parameters() -> None:
    """CNN contains trainable parameters."""
    model = build_model(
        make_config()
    )

    parameters = list(
        model.parameters()
    )

    assert len(parameters) > 0

    assert sum(
        parameter.numel()
        for parameter in parameters
    ) > 0


def test_model_can_run_inference() -> None:
    """CNN accepts image batches and produces class logits."""
    model = build_model(
        make_config()
    )

    model.eval()

    images = torch.rand(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        predictions = model(
            images
        )

    assert predictions.shape == (
        2,
        2,
    )

    assert torch.isfinite(
        predictions
    ).all()


def test_model_rejects_non_rgb_input() -> None:
    """CNN requires three-channel RGB input."""
    config = make_config()
    config["model"]["input_shape"] = [
        224,
        224,
        1,
    ]

    try:
        build_model(config)
        assert False, "Expected ValueError"
    except ValueError:
        pass
