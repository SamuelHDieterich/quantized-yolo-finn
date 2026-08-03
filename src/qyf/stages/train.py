"""
Phase 1 — Full-precision YOLO training via Ultralytics.

This script trains a standard (non-quantized) YOLO model on the dataset.
The resulting checkpoint is the starting point for QAT fine-tuning (qat.py).
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

import argparse
import logging
from pathlib import Path

# External
from ultralytics import YOLO

# Internal
from qyf.config import load_model_config, load_train_config
from qyf.paths import repo_root, run_dir


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-precision YOLO training")
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


def train(data: Path, train_config_path: Path, model_config_path: Path) -> Path:
    """
    Run full-precision Ultralytics training.

    Parameters
    ----------
    data : Path
        Path to the YOLO-format dataset YAML.
    train_config_path : Path
        Path to train.yaml.
    model_config_path : Path
        Path to model.yaml.

    Returns
    -------
    Path
        Path to the best checkpoint produced by training.
    """

    train_cfg = load_train_config(train_config_path)
    model_cfg = load_model_config(model_config_path)

    weights = (
        f"{model_cfg.model_name}.pt"
        if train_cfg.pretrained
        else f"{model_cfg.model_name}.yaml"
    )
    model = YOLO(weights)

    logger.info("==> Training %s on %s", model_cfg.model_name, data)
    logger.info("    Epochs: %s  |  Batch: %s", train_cfg.epochs, train_cfg.batch_size)

    results = model.train(
        data=str(Path(data).resolve()),
        epochs=train_cfg.epochs,
        batch=train_cfg.batch_size,
        imgsz=train_cfg.imgsz,
        device=train_cfg.device,
        project=str(run_dir("train")),
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("\n==> Training complete. Best checkpoint: %s", best_weights)
    logger.info("    Next step: qat --weights %s --data %s", best_weights, data)

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
    train(
        data=Path(args.data),
        train_config_path=repo_root() / args.config,
        model_config_path=repo_root() / args.model_config,
    )
