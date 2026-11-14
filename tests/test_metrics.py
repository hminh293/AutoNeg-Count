"""Spec P1.5 — autoneg/metrics.py (Minh). Ví dụ tính tay ở Tài liệu 3, mục 7."""
import numpy as np
import pytest

from autoneg import metrics as M

GT = [10, 20, 30, 40]
PRED = [12, 18, 36, 40]


@pytest.mark.parametrize("fn,val", [("mae", 2.5), ("rmse", 3.317), ("nae", 0.125),
                                    ("sre", 0.671), ("bias", 1.5), ("overcount_rate", 0.5)])
def test_hand_example(fn, val):
    assert getattr(M, fn)(PRED, GT) == pytest.approx(val, abs=1e-3)


def test_bootstrap_constant():
    stat, lo, hi = M.bootstrap_ci(np.full(50, 3.0), B=200)
    assert stat == lo == hi == 3.0


def test_bootstrap_contains_mean():
    v = np.random.default_rng(0).normal(5, 1, 200)
    stat, lo, hi = M.bootstrap_ci(v, B=1000)
    assert lo < stat < hi and lo < 5 < hi


def test_paired_identical_not_significant():
    r = M.paired_bootstrap(PRED, PRED, GT, B=500)
    assert r["diff"] == 0 and not r["significant"]


def test_paired_clear_improvement():
    gt = np.arange(10, 110)
    a = gt + 10                      # luôn sai 10
    b = gt + 1                       # luôn sai 1
    r = M.paired_bootstrap(a, b, gt, B=500)
    assert r["diff"] == pytest.approx(9) and r["significant"] and r["lo"] > 0
