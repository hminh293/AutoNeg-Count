"""[AB] Pipeline đầy đủ.  PLAN: S7.  Viết CHUNG khi M1 + M2 đã có test xanh.

predict(model, transform, sample, cfg, cache=None) -> Prediction
  img_t = preprocess(sample) → p1 = cache hoặc run_pass1 → cands = build_candidates
  → pos = select_positive (M1) → negs = mine_negatives (M2, nếu bật)
  → out2 = forward_once(pos_boxes, neg_classes) → tau = pick_threshold (M3)
  → boxes, scores = filter_two_stage(out2, tau)
Nếu len(pos_boxes) == 0 → quay về đếm lượt 1 (B1).
"""
from __future__ import annotations


def predict(model, transform, sample, cfg, cache=None):
    raise NotImplementedError("S7 predict — làm chung")
