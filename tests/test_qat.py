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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# External
import brevitas.nn as qnn
import torch
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


def _patched_trainer_for_save(tmp_path) -> QATDetectionTrainer:
    """A QATDetectionTrainer with just enough state for save_model()/final_eval(), no real Ultralytics init."""
    trainer = QATDetectionTrainer.__new__(QATDetectionTrainer)
    trainer.model_cfg = _model_cfg()
    with patch("qyf.stages.qat.DetectionTrainer.get_model", return_value=_TinyModel()):
        model = trainer.get_model()  # a real Brevitas-patched model, unpicklable whole

    trainer.ema = SimpleNamespace(ema=model, updates=0)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.args = SimpleNamespace(epochs=1, batch=2)
    trainer.metrics = {}
    trainer.fitness = 1.0
    trainer.best_fitness = 1.0
    trainer.epoch = 0
    trainer.save_period = -1
    trainer.wdir = tmp_path / "weights"
    trainer.last = trainer.wdir / "last.pt"
    trainer.best = trainer.wdir / "best.pt"
    return trainer


def test_save_model_writes_state_dict_not_whole_model(tmp_path) -> None:
    trainer = _patched_trainer_for_save(tmp_path)

    assert trainer.save_model() is True

    for ckpt_path in (trainer.last, trainer.best):
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in ckpt
        assert all(
            isinstance(v, torch.Tensor) for v in ckpt["model_state_dict"].values()
        )
        assert "model" not in ckpt
        assert "ema" not in ckpt


def test_save_model_only_writes_best_when_fitness_is_best(tmp_path) -> None:
    trainer = _patched_trainer_for_save(tmp_path)
    trainer.fitness = 0.5
    trainer.best_fitness = 1.0  # a prior epoch was better; this one isn't best

    trainer.save_model()

    assert trainer.last.exists()
    assert not trainer.best.exists()


def test_final_eval_validates_in_memory_when_best_exists(tmp_path) -> None:
    trainer = QATDetectionTrainer.__new__(QATDetectionTrainer)
    trainer.best = tmp_path / "best.pt"
    trainer.best.write_bytes(b"stub state_dict checkpoint")
    trainer.args = SimpleNamespace(plots=False)
    trainer.epoch = 3
    trainer.validator = MagicMock(return_value={"mAP50-95": 0.9, "fitness": 0.9})
    trainer.run_callbacks = MagicMock()

    trainer.final_eval()

    trainer.validator.assert_called_once_with(trainer=trainer)
    assert trainer.metrics == {"mAP50-95": 0.9}
    assert trainer.epoch == 3
    trainer.run_callbacks.assert_called_once_with("on_fit_epoch_end")


def test_final_eval_skips_validation_without_a_checkpoint(tmp_path) -> None:
    trainer = QATDetectionTrainer.__new__(QATDetectionTrainer)
    trainer.best = tmp_path / "best.pt"  # never written
    trainer.validator = MagicMock()

    trainer.final_eval()

    trainer.validator.assert_not_called()
