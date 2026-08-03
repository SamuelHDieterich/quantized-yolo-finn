# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

from pathlib import Path

# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate repo root (no pyproject.toml above qyf/paths.py)"
    )


def run_dir(stage: str) -> Path:
    """
    Absolute output directory for a pipeline stage, e.g. run_dir("train").

    Must be absolute: Ultralytics combines a relative `project=` with its own
    global runs_dir setting, producing nested paths like
    runs/detect/runs/train/train/ instead of runs/train/.

    Parameters
    ----------
    stage : str
        Name of the pipeline stage, e.g. "train", "val", "test".

    Returns
    -------
    Path
        Absolute path to the output directory for the given stage.

    Raises
    ------
    RuntimeError
        If the repository root cannot be located (no pyproject.toml above qyf/paths.py).
    """
    path = repo_root() / "runs" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path
