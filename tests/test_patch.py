"""
Unit tests for qyf.quantize.patch.

Verifies patch_conv2d() replaces every nn.Conv2d with a Brevitas
QuantConv2d, and patch_silu() replaces every nn.SiLU with a Brevitas
QuantHardTanh, both at the configured bit-width, without disturbing
other layers or the forward pass's output shape.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

# External
import brevitas.nn as qnn
import torch
import torch.nn as nn

# Internal
from qyf.config import ModelConfig
from qyf.quantize.patch import patch_conv2d, patch_silu


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


class _NestedBlock(nn.Module):
    """A stand-in for an Ultralytics block (e.g. C2f) nesting SiLU inside a Sequential."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.body = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        # Non-SiLU activation, which must survive untouched.
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        x = self.body(x)
        return self.relu(x)


class _NestedConvBlock(nn.Module):
    """A stand-in for an Ultralytics block nesting Conv2d inside a Sequential.

    Covers a grouped conv and a bias-less conv, since both need their
    Brevitas replacement configured to match rather than default.
    BatchNorm layers must survive untouched -- they get fused into the
    preceding convolution during ONNX export.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(6)
        self.body = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1, groups=6, bias=False),
            nn.BatchNorm2d(6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return self.body(x)


def _model_cfg(**overrides: object) -> ModelConfig:
    values = {"model_name": "yolov8n", **overrides}
    return ModelConfig(**values)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_patch_conv2d_replaces_all_conv_instances() -> None:
    model = _NestedConvBlock()

    patch_conv2d(model, _model_cfg())

    # QuantConv2d subclasses nn.Conv2d, so isinstance() alone can't tell a
    # patched layer from an unpatched one -- match the exact vanilla type.
    conv_count = sum(1 for m in model.modules() if type(m) is nn.Conv2d)
    quant_conv_count = sum(1 for m in model.modules() if isinstance(m, qnn.QuantConv2d))
    assert conv_count == 0
    assert quant_conv_count == 2


def test_patch_conv2d_leaves_other_layers_untouched() -> None:
    model = _NestedConvBlock()

    patch_conv2d(model, _model_cfg())

    assert isinstance(model.bn, nn.BatchNorm2d)
    assert isinstance(model.body[1], nn.BatchNorm2d)


def test_patch_conv2d_preserves_forward_shape() -> None:
    model = _NestedConvBlock()
    x = torch.randn(2, 3, 8, 8)
    expected_shape = model(x).shape

    patch_conv2d(model, _model_cfg())

    y = model(x)
    assert y.shape == expected_shape


def test_patch_conv2d_uses_configured_bit_width() -> None:
    model = _NestedConvBlock()

    patch_conv2d(model, _model_cfg(weight_bit_width=3))

    for m in model.modules():
        if isinstance(m, qnn.QuantConv2d):
            assert m.weight_quant.quant_injector.bit_width == 3


def test_patch_silu_replaces_all_silu_instances() -> None:
    model = _NestedBlock()

    patch_silu(model, _model_cfg())

    silu_count = sum(1 for m in model.modules() if isinstance(m, nn.SiLU))
    quant_hardtanh_count = sum(
        1 for m in model.modules() if isinstance(m, qnn.QuantHardTanh)
    )
    assert silu_count == 0
    assert quant_hardtanh_count == 2


def test_patch_silu_leaves_other_layers_untouched() -> None:
    model = _NestedBlock()

    patch_silu(model, _model_cfg())

    assert isinstance(model.conv, nn.Conv2d)
    assert isinstance(model.body[0], nn.Conv2d)
    assert isinstance(model.relu, nn.ReLU)


def test_patch_silu_preserves_forward_shape() -> None:
    model = _NestedBlock()
    x = torch.randn(2, 3, 8, 8)
    expected_shape = model(x).shape

    patch_silu(model, _model_cfg())

    y = model(x)
    assert y.shape == expected_shape


def test_patch_silu_uses_configured_bit_width() -> None:
    model = _NestedBlock()

    patch_silu(model, _model_cfg(act_bit_width=3))

    for m in model.modules():
        if isinstance(m, qnn.QuantHardTanh):
            assert m.act_quant.quant_injector.bit_width == 3
