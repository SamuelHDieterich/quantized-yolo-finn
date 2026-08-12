"""
Unit tests for qyf.stages.qat.

Verifies QATDetectionTrainer patches the model Ultralytics builds --
Conv2d to QuantConv2d, SiLU to QuantHardTanh -- before training starts,
without exercising a real Ultralytics training run.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

from unittest.mock import patch

# External
import brevitas.nn as qnn
import torch.nn as nn

# Internal
from qyf.config import ModelConfig
from qyf.stages.qat import QATDetectionTrainer


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _TinyModel(nn.Module):
    """A stand-in for the DetectionModel Ultralytics' get_model() would return."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.act = nn.SiLU()


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def _model_cfg(**overrides: object) -> ModelConfig:
    values = {"model_name": "yolov8n", **overrides}
    return ModelConfig(**values)


def test_init_stores_model_cfg_and_delegates_to_super() -> None:
    model_cfg = _model_cfg()

    with patch(
        "qyf.stages.qat.DetectionTrainer.__init__", return_value=None
    ) as mock_init:
        trainer = QATDetectionTrainer(
            model_cfg, overrides={"model": "x.pt", "data": "d.yaml"}
        )

    assert trainer.model_cfg is model_cfg
    mock_init.assert_called_once()


def test_get_model_replaces_conv2d_and_silu() -> None:
    trainer = QATDetectionTrainer.__new__(QATDetectionTrainer)
    trainer.model_cfg = _model_cfg()

    with patch("qyf.stages.qat.DetectionTrainer.get_model", return_value=_TinyModel()):
        model = trainer.get_model()

    assert isinstance(model.conv, qnn.QuantConv2d)
    assert isinstance(model.act, qnn.QuantHardTanh)


def test_get_model_uses_configured_bit_widths() -> None:
    trainer = QATDetectionTrainer.__new__(QATDetectionTrainer)
    trainer.model_cfg = _model_cfg(weight_bit_width=3, act_bit_width=5)

    with patch("qyf.stages.qat.DetectionTrainer.get_model", return_value=_TinyModel()):
        model = trainer.get_model()

    assert model.conv.weight_quant.quant_injector.bit_width == 3
    assert model.act.act_quant.quant_injector.bit_width == 5
