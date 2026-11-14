"""Spec P1.3 — autoneg/cpe.py (Minh): M1 chọn mẫu dương sạch."""
import numpy as np

from autoneg.cpe import select_positive
from helpers import blobs, cpe_cfg, make_cands


def _chosen_true_cluster(sel, order, y, cls):
    return all(y[order[i]] == cls for i in sel["pos_idx"])


def test_alpha1_prefers_big_correct_cluster():
    # cụm đúng (0): 8 điểm x 0.5 ; cụm nhiễu (1): 4 điểm x 0.8  -> 8*0.5=4.0 > 4*0.8=3.2
    E, y, _ = blobs([8, 4])
    cands, order = make_cands(E, [0.5] * 8 + [0.8] * 4)
    sel = select_positive(cands, cpe_cfg(alpha=1.0))
    assert len(sel["pos_idx"]) == 3 and _chosen_true_cluster(sel, order, y, 0)
    assert len(sel["labels"]) == 12


def test_alpha0_behaves_like_top_score():
    E, y, _ = blobs([8, 4])
    cands, order = make_cands(E, [0.5] * 8 + [0.8] * 4)
    sel = select_positive(cands, cpe_cfg(alpha=0.0))
    assert _chosen_true_cluster(sel, order, y, 1)


def test_size_only_fails_when_distractor_is_more_numerous():
    # Ca minh hoạ cho slide: vật nhầm ĐÔNG hơn vật thật.
    # cụm đúng (0): 6 x 0.7 ; cụm nhiễu (1): 12 x 0.3
    E, y, _ = blobs([6, 12])
    cands, order = make_cands(E, [0.7] * 6 + [0.3] * 12)
    wrong = select_positive(cands, cpe_cfg(criterion="size_only"))        # kiểu CountSE / CountZES
    right = select_positive(cands, cpe_cfg(criterion="score_size", alpha=0.5))
    assert _chosen_true_cluster(wrong, order, y, 1)
    assert _chosen_true_cluster(right, order, y, 0)


def test_cluster_none_equals_baseline_topm():
    E, _, _ = blobs([5, 5])
    cands, _ = make_cands(E, np.linspace(0.9, 0.2, 10))
    sel = select_positive(cands, cpe_cfg(cluster="none"))
    assert sel["pos_idx"] == [0, 1, 2]


def test_few_candidates_returns_all():
    E, _, _ = blobs([2])
    cands, _ = make_cands(E, [0.6, 0.5])
    sel = select_positive(cands, cpe_cfg())
    assert sel["pos_idx"] == [0, 1]


def test_best_cluster_consistent():
    E, _, _ = blobs([8, 4])
    cands, _ = make_cands(E, [0.5] * 8 + [0.8] * 4)
    sel = select_positive(cands, cpe_cfg())
    assert all(sel["labels"][i] == sel["best_cluster"] for i in sel["pos_idx"])
    assert sel["best_cluster"] in sel["quality"]
