"""Spec P1.4 — autoneg/threshold.py (Minh): M3."""
import numpy as np

from autoneg.config import Cfg
from autoneg.threshold import fixed, otsu, pick_threshold, valley


def _bimodal(seed=42):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.1, 0.03, 300), rng.normal(0.8, 0.05, 60)]).clip(0, 1)


def test_fixed():
    assert fixed() == 0.23


def test_otsu_bimodal_separates_modes():
    # Otsu không nhất thiết nằm giữa hai đỉnh khi một nhóm đông hơn nhiều,
    # nhưng phải TÁCH đúng hai nhóm.
    s = _bimodal()
    y = np.r_[np.zeros(300), np.ones(60)]
    t = otsu(s, clip=(0.0, 1.0))
    assert 0.1 < t < 0.7
    assert ((s > t) == y).mean() >= 0.98


def test_otsu_respects_clip():
    t = otsu(_bimodal(), clip=(0.15, 0.5))
    assert 0.15 <= t <= 0.5


def test_flat_scores_no_error():
    s = np.full(100, 0.3) + np.random.default_rng(0).normal(0, 0.005, 100)
    for f in (otsu, valley):
        t = f(s)
        assert 0.15 <= t <= 0.5


def test_valley_bimodal():
    t = valley(_bimodal())
    assert 0.15 <= t <= 0.5


def test_pick_threshold_modes():
    s = _bimodal()
    assert pick_threshold(s, Cfg(mode="fixed", tau=0.23)) == 0.23
    assert 0.15 <= pick_threshold(s, Cfg(mode="otsu", tau=0.23)) <= 0.5


def test_empty_scores_fallback_fixed():
    assert pick_threshold(np.array([]), Cfg(mode="otsu", tau=0.23)) == 0.23
