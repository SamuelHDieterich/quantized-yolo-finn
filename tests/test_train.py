"""
Unit tests for qyf.stages.train.

Verifies that train() wires config.py's models to Ultralytics' API correctly
against a mocked YOLO(). No real training happens here.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# External
import pytest

# Internal
from qyf.stages.train import train


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture
def model_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text("model_name: yolov8n\n")
    return path


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text("train: images/train\nval: images/val\n")
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def _write_train_config(tmp_path: Path, **overrides: object) -> Path:
    values = {
        "epochs": 5,
        "batch_size": 8,
        "imgsz": 320,
        "device": "cpu",
        "pretrained": True,
        **overrides,
    }
    path = tmp_path / "train.yaml"
    path.write_text(
        "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n"
    )
    return path


def test_train_wires_config_to_ultralytics(
    tmp_path: Path, data_path: Path, model_config_path: Path
) -> None:
    train_config_path = _write_train_config(tmp_path)
    run_dir_path = tmp_path / "runs" / "train"

    with (
        patch("qyf.stages.train.YOLO") as mock_yolo_cls,
        patch("qyf.stages.train.run_dir", return_value=run_dir_path) as mock_run_dir,
    ):
        mock_model = mock_yolo_cls.return_value
        mock_model.train.return_value = MagicMock(save_dir=run_dir_path)

        best_weights = train(data_path, train_config_path, model_config_path)

    mock_yolo_cls.assert_called_once_with("yolov8n.pt")
    mock_run_dir.assert_called_once_with("train")

    call_kwargs = mock_model.train.call_args.kwargs
    assert call_kwargs["data"] == str(data_path.resolve())
    assert call_kwargs["epochs"] == 5
    assert call_kwargs["batch"] == 8
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["device"] == "cpu"
    assert call_kwargs["project"] == str(run_dir_path)

    assert best_weights == run_dir_path / "weights" / "best.pt"


def test_train_uses_yaml_weights_when_not_pretrained(
    tmp_path: Path, data_path: Path, model_config_path: Path
) -> None:
    train_config_path = _write_train_config(tmp_path, pretrained=False)
    run_dir_path = tmp_path / "runs" / "train"

    with (
        patch("qyf.stages.train.YOLO") as mock_yolo_cls,
        patch("qyf.stages.train.run_dir", return_value=run_dir_path),
    ):
        mock_model = mock_yolo_cls.return_value
        mock_model.train.return_value = MagicMock(save_dir=run_dir_path)

        train(data_path, train_config_path, model_config_path)

    mock_yolo_cls.assert_called_once_with("yolov8n.yaml")
