"""[A – Minh] Chỉ số đếm + bootstrap.  PLAN: P1.5 / S10.

Ví dụ kiểm tra (Tài liệu 3): GT=[10,20,30,40], Pred=[12,18,36,40] →
MAE 2.5 · RMSE 3.317 · NAE 0.125 · SRE 0.671 · Bias +1.5.
"""
from __future__ import annotations

import numpy as np


def mae(pred, gt) -> float: raise NotImplementedError("P1.5 mae — Minh")
def rmse(pred, gt) -> float: raise NotImplementedError("P1.5 rmse — Minh")
def nae(pred, gt) -> float: raise NotImplementedError("P1.5 nae — Minh")
def sre(pred, gt) -> float: raise NotImplementedError("P1.5 sre — Minh")
def bias(pred, gt) -> float: raise NotImplementedError("P1.5 bias — Minh")
def overcount_rate(pred, gt) -> float: raise NotImplementedError("P1.5 overcount_rate — Minh")


def bootstrap_ci(values, B: int = 10000, alpha: float = 0.05, seed: int = 42, stat=np.mean):
    """Trả (stat, lo, hi) với CI (1-alpha) bằng bootstrap theo ảnh."""
    raise NotImplementedError("P1.5 bootstrap_ci — Minh")


def paired_bootstrap(pred_a, pred_b, gt, B: int = 10000, seed: int = 42):
    """CI 95% của MAE(a) - MAE(b) trên CÙNG ảnh. Trả dict(diff, lo, hi, significant)."""
    raise NotImplementedError("P1.5 paired_bootstrap — Minh")
