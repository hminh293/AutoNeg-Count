"""Bộ đọc dữ liệu. Mỗi adapter trả iterator CountSample (autoneg.types) và có `python -m ... --check`.

[A – Minh]: fsc147, carpk, shanghaitech · [B – Bi Anh]: pairtally, omnicount, bloodcell
"""
from __future__ import annotations

import importlib

# tên dataset trong config -> (module, kwargs)
REGISTRY = {
    "fsc147_val": ("fsc147", {"split": "val"}),
    "fsc147_test": ("fsc147", {"split": "test"}),
    "carpk_test": ("carpk", {"split": "test"}),
    "shanghaitech_a_test": ("shanghaitech", {"part": "A", "split": "test"}),
    "shanghaitech_b_test": ("shanghaitech", {"part": "B", "split": "test"}),
    "pairtally": ("pairtally", {}),
    "omnicount_fruits": ("omnicount", {}),
    "bloodcell": ("bloodcell", {}),
}


def get_dataset(name: str, **kw):
    mod, base_kw = REGISTRY[name]
    m = importlib.import_module(f"autoneg.datasets.{mod}")
    return m.load(**{**base_kw, **kw})
