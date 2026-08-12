"""
Replace Conv2d layers with Brevitas QuantConv2d equivalents.

Call `patch_conv2d` on a model to convert every nn.Conv2d in its module tree
to a weight-quantized Brevitas QuantConv2d, at the bit-width from ModelConfig.
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
from brevitas.quant import Int8WeightPerTensorFloat

# Internal
from qyf.config import ModelConfig


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _weight_quantizer(bit_width: int) -> type:
    """
    Build a per-tensor symmetric weight quantizer pinned to `bit_width`.

    Brevitas ships an 8-bit default as an importable class;
    any other bit-width is a thin subclass overriding `bit_width`,
    which is Brevitas' documented pattern for arbitrary bit-widths.

    Parameters
    ----------
    bit_width : int
        Target bit-width for the quantizer.

    Returns
    -------
    type
        A quantizer class pinned to `bit_width`.
    """

    if bit_width == 8:
        return Int8WeightPerTensorFloat

    return type(
        f"Int{bit_width}WeightPerTensorFloat",
        (Int8WeightPerTensorFloat,),
        {"bit_width": bit_width},
    )


def patch_conv2d(model: nn.Module, model_cfg: ModelConfig) -> nn.Module:
    """
    Recursively replace nn.Conv2d layers with Brevitas QuantConv2d, in-place.

    Pretrained weights (and biases) are copied onto each replacement so QAT
    starts from the source model's trained initialization rather than from scratch.
    BatchNorm layers are left untouched, as they get fused into the preceding convolution during ONNX export.

    Parameters
    ----------
    model : nn.Module
        Module tree to patch in-place.
    model_cfg : ModelConfig
        Model configuration; `weight_bit_width` sets the quantizer's
        bit-width.

    Returns
    -------
    nn.Module
        The patched model. Operates in-place and also returns the model for convenience.
    """

    # One quantizer class shared by every Conv2d in the tree,
    # built once here rather than per-layer inside _walk.
    weight_quant = _weight_quantizer(model_cfg.weight_bit_width)

    def _walk(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                quant_conv = qnn.QuantConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    # Preserve whether the source conv had a bias at all —
                    # the bias itself is copied, unquantized, below.
                    bias=child.bias is not None,
                    weight_quant=weight_quant,
                )
                # Start QAT from the pretrained weights
                with torch.no_grad():
                    # A Conv2d always has a weight tensor.
                    quant_conv.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        # Only copy if the source conv actually had one.
                        quant_conv.bias.data.copy_(child.bias.data)
                # Replace the child in its parent module, in-place.
                setattr(module, name, quant_conv)
            else:
                # Not a Conv2d itself — recurse into its children
                # (Sequential, C2f, etc.) to find the ones nested inside.
                _walk(child)

    # Kick off the recursion at the model's root.
    _walk(model)

    return model


def patch_silu(model: nn.Module, model_cfg: ModelConfig) -> nn.Module:
    """
    Recursively replace nn.SiLU activations with Brevitas QuantHardTanh, in-place.

    FINN has no native SiLU support, so the activation must be trained
    against the op FINN can actually synthesize. QuantHardTanh's [-1, 1]
    clamp matches nn.Hardtanh's default range, keeping this a like-for-like
    activation swap rather than a different nonlinearity.

    Parameters
    ----------
    model : nn.Module
        Module tree to patch in-place.
    model_cfg : ModelConfig
        Model configuration; `act_bit_width` sets the quantizer's bit-width.

    Returns
    -------
    nn.Module
        The patched model. Operates in-place and also returns the model for convenience.
    """

    def _walk(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.SiLU):
                setattr(
                    module,
                    name,
                    qnn.QuantHardTanh(
                        bit_width=model_cfg.act_bit_width,
                        min_val=-1.0,
                        max_val=1.0,
                    ),
                )
            else:
                # Not a SiLU itself — recurse into its children
                # (Sequential, C2f, etc.) to find the ones nested inside.
                _walk(child)

    # Kick off the recursion at the model's root.
    _walk(model)

    return model
