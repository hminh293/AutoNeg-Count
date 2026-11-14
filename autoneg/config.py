"""Đọc config YAML: configs/base.yaml + file thí nghiệm (ghi đè sâu)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import REPO_ROOT


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Cfg(dict):
    """dict cho phép truy cập kiểu thuộc tính: cfg.cpe.alpha."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path: str | Path, base: str | Path | None = None) -> Cfg:
    base = Path(base) if base else REPO_ROOT / "configs" / "base.yaml"
    with open(base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return Cfg(cfg)
