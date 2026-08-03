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
import yaml
from ultralytics import YOLO


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


if __name__ == "__main__":
    args = parse_args()

    # Step 1: load hyperparameters and model configuration from YAML.
    with open(args.config) as f:
        train_cfg: dict = yaml.safe_load(f)

    with open(args.model_config) as f:
        model_cfg: dict = yaml.safe_load(f)

    model_name: str = model_cfg["model_name"]

    # Step 2: build the model — pretrained checkpoint, or random init from the
    # architecture YAML if the config asks for training from scratch.
    weights = (
        f"{model_name}.pt"
        if train_cfg.get("pretrained", True)
        else f"{model_name}.yaml"
    )
    model = YOLO(weights)

    logger.info("==> Training %s on %s", model_name, args.data)
    logger.info(
        "    Epochs: %s  |  Batch: %s",
        train_cfg.get("epochs", 100),
        train_cfg.get("batch_size", 16),
    )

    # Step 3: run full-precision training via Ultralytics.
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch_size", 16),
        imgsz=train_cfg.get("imgsz", 640),
        device=train_cfg.get("device", "cpu"),
        project=train_cfg.get("project", "runs/train"),
    )

    # Step 4: report where the best checkpoint landed.
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("\n==> Training complete. Best checkpoint: %s", best_weights)
    logger.info("    Next step: qat --weights %s --data %s", best_weights, args.data)
