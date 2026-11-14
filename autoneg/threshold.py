"""[A – Minh] M3 – Ngưỡng thích nghi.  PLAN: P1.4 / S8.

- fixed(tau=0.23)
- otsu(scores, bins=64, lo=0.05, clip=(0.15, 0.5)): Otsu 1-D trên điểm > lo.
- valley(scores, clip=(0.15, 0.5)): điểm gãy lớn nhất của dãy điểm sắp giảm.
- pick_threshold(scores, cfg) với cfg.mode in {"fixed","otsu","valley"}; mảng rỗng → fixed.
"""
from __future__ import annotations

import numpy as np


def fixed(tau: float = 0.23) -> float:
    return float(tau)


def otsu(scores: np.ndarray, bins: int = 64, lo: float = 0.05, clip=(0.15, 0.5)) -> float:
    raise NotImplementedError("P1.4 otsu — Minh")


def valley(scores: np.ndarray, clip=(0.15, 0.5)) -> float:
    raise NotImplementedError("P1.4 valley — Minh")


def pick_threshold(scores: np.ndarray, cfg) -> float:
    raise NotImplementedError("P1.4 pick_threshold — Minh")
