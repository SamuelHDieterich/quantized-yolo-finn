"""
Phase 2 — Quantization-aware YOLO fine-tuning via Brevitas.

Continues training from a full-precision checkpoint (train.py) with the
model's Conv2d and SiLU layers patched to Brevitas quantized equivalents
(quantize/patch.py), so the network is fine-tuned around the quantization
error rather than having it introduced after the fact.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

# External
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

# Internal
from qyf.config import ModelConfig, load_model_config, load_train_config
from qyf.paths import repo_root, run_dir
from qyf.quantize.patch import patch_conv2d, patch_silu


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


class QATDetectionTrainer(DetectionTrainer):
    """
    DetectionTrainer that quantizes the model Ultralytics builds, before training starts.

    Ultralytics constructs the DetectionModel inside get_model() (called
    from setup_model() at the start of _setup_train()), so the patch has to
    run there rather than on a model built ahead of time -- a model patched
    before being handed to the trainer would be discarded, since
    DetectionTrainer.get_model() builds its own from cfg/weights.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> Any:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        patch_conv2d(model, self.model_cfg)
        patch_silu(model, self.model_cfg)
        return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brevitas QAT fine-tuning")
    p.add_argument(
        "--weights",
        required=True,
        help="Path to the full-precision checkpoint produced by train.py",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (e.g. data/ships-in-aerial-images/data.yaml)",
    )
    p.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training hyperparameters YAML",
    )
    p.add_argument(
        "--model-config",
        default="configs/model.yaml",
        help="Model configuration YAML",
    )
    return p.parse_args()


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def qat(
    weights: Path,
    data: Path,
    train_config_path: Path,
    model_config_path: Path,
) -> Path:
    """
    Run Brevitas QAT fine-tuning, continuing from a full-precision checkpoint.

    Parameters
    ----------
    weights : Path
        Full-precision checkpoint from train.py.
    data : Path
        Path to the YOLO-format dataset YAML.
    train_config_path : Path
        Path to train.yaml.
    model_config_path : Path
        Path to model.yaml.

    Returns
    -------
    Path
        Path to the best checkpoint produced by QAT.
    """

    train_cfg = load_train_config(train_config_path)
    model_cfg = load_model_config(model_config_path)

    logger.info("==> QAT fine-tuning %s on %s", weights, data)
    logger.info("    Epochs: %s  |  Batch: %s", train_cfg.epochs, train_cfg.batch_size)

    trainer = QATDetectionTrainer(
        model_cfg,
        overrides={
            "model": str(weights),
            "data": str(Path(data).resolve()),
            "epochs": train_cfg.epochs,
            "batch": train_cfg.batch_size,
            "imgsz": train_cfg.imgsz,
            "device": train_cfg.device,
            "project": str(run_dir("qat")),
        },
    )
    trainer.train()

    best_weights = trainer.best
    logger.info("\n==> QAT complete. Best checkpoint: %s", best_weights)
    logger.info("    Next step: export --weights %s", best_weights)

    return best_weights


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    qat(
        weights=Path(args.weights),
        data=Path(args.data),
        train_config_path=repo_root() / args.config,
        model_config_path=repo_root() / args.model_config,
    )
