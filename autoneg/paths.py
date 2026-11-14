"""Đường dẫn đọc từ biến môi trường — không hard-code."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def countgd_root() -> Path:
    return _env("COUNTGD_ROOT", REPO_ROOT / "third_party" / "CountGDPlusPlus")


def countse_root() -> Path:
    return _env("COUNTSE_ROOT", REPO_ROOT / "third_party" / "CountSE")


def data_root() -> Path:
    return _env("DATA_ROOT", REPO_ROOT / "data")


def fsc147_root() -> Path:
    return _env("FSC147_ROOT", data_root() / "FSC147_384_V2")


def ckpt_dir() -> Path:
    return _env("CKPT_DIR", REPO_ROOT / "checkpoints")


def results_dir() -> Path:
    return _env("RESULTS_DIR", REPO_ROOT / "results")
