"""Spec S6 — autoneg/anm.py (Bi Anh): M2 tự khai thác mẫu âm."""
import numpy as np

from autoneg.anm import mine_negatives
from autoneg.config import Cfg


def _setup(seed=42, d=256):
    rng = np.random.default_rng(seed)
    c0, c1 = rng.normal(size=d), rng.normal(size=d)
    E = np.vstack([
        c0 + 0.1 * rng.normal(size=(6, d)),              # cụm 0: dương (best)
        c1 + 0.1 * rng.normal(size=(4, d)),              # cụm 1: nhiễu xa, điểm 0.4 → PHẢI thành mẫu âm
        rng.normal(size=(1, d)),                         # cụm 2: 1 phần tử → bỏ
        c0 + 0.1 * rng.normal(size=(3, d)),              # cụm 3: gần prototype dương → bỏ (d < delta)
    ])
    scores = np.array([0.8] * 6 + [0.4] * 4 + [0.5] + [0.35] * 3)
    labels = np.array([0] * 6 + [1] * 4 + [2] + [3] * 3)
    cands = dict(boxes_xyxy=np.zeros((14, 4)), scores=scores, emb=E)
    pos = dict(pos_idx=[0, 1, 2], labels=labels, best_cluster=0)
    cfg = Cfg(K_neg=2, m_neg=2, min_cluster=2, neg_score_min=0.20, delta=0.30, neg_text="empty")
    return cands, pos, cfg


def test_single_far_cluster_mined():
    cands, pos, cfg = _setup()
    negs = mine_negatives(cands, pos, cfg)
    assert len(negs) == 1
    assert set(negs[0]["idx"]) <= {6, 7, 8, 9} and len(negs[0]["idx"]) == 2
    assert negs[0]["text"] == ""


def test_k_neg_zero():
    cands, pos, cfg = _setup()
    cfg["K_neg"] = 0
    assert mine_negatives(cands, pos, cfg) == []
