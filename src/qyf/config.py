"""
Pydantic models for the pipeline's YAML configs.

Centralizes config loading for model.yaml, train.yaml, and finn.yaml. An
unknown or mistyped key raises a pydantic ValidationError instead of being
silently ignored.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

from pathlib import Path

# External
import yaml
from pydantic import BaseModel, ConfigDict, Field


# ░█▄█░█▀█░█▀▄░█▀▀░█░░░█▀▀
# ░█░█░█░█░█░█░█▀▀░█░░░▀▀█
# ░▀░▀░▀▀▀░▀▀░░▀▀▀░▀▀▀░▀▀▀


class ModelConfig(BaseModel):
    """Model architecture and quantization configuration (model.yaml)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    # Brevitas weight quantizer bit-width for QuantConv2d layers.
    # 2-8 is the FINN-compatible range.
    weight_bit_width: int = Field(default=4, ge=2, le=8)
    # Brevitas activation quantizer bit-width for QuantHardTanh layers
    # (the SiLU replacement). 2-8 is the FINN-compatible range.
    act_bit_width: int = Field(default=4, ge=2, le=8)


class TrainConfig(BaseModel):
    """Full-precision training hyperparameters (train.yaml)."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: str = "cpu"
    pretrained: bool = True
    project: str = "runs/train"


class FinnConfig(BaseModel):
    """FINN dataflow compilation configuration (finn.yaml)."""

    model_config = ConfigDict(extra="forbid")

    board: str | None = None
    shell_flow_type: str = "vivado_zynq"
    stop_step: str = ""
    folding_config_file: str | None = None


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: Path) -> ModelConfig:
    """
    Load and validate model.yaml.

    Parameters
    ----------
    path : Path
        Path to the model configuration YAML.

    Returns
    -------
    ModelConfig
        Validated model configuration.
    """
    return ModelConfig(**_read_yaml(path))


def load_train_config(path: Path) -> TrainConfig:
    """
    Load and validate train.yaml.

    Parameters
    ----------
    path : Path
        Path to the training configuration YAML.

    Returns
    -------
    TrainConfig
        Validated training configuration.
    """
    return TrainConfig(**_read_yaml(path))


def load_finn_config(path: Path) -> FinnConfig:
    """
    Load and validate finn.yaml.

    Parameters
    ----------
    path : Path
        Path to the FINN configuration YAML.

    Returns
    -------
    FinnConfig
        Validated FINN configuration.
    """
    return FinnConfig(**_read_yaml(path))
