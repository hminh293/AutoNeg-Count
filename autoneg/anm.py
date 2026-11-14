"""[B – Bi Anh] M2 – Auto-Negative Mining.  PLAN: S6.

mine_negatives(cands, pos, cfg, confusers=None) -> list[NegClass]
- Bỏ cụm tốt nhất (pos["best_cluster"]); giữ cụm có >= min_cluster phần tử, điểm TB >= neg_score_min,
  khoảng cách cos tới prototype dương >= delta; sắp theo điểm, lấy tối đa K_neg; mỗi lớp lấy m_neg exemplar.
- neg_text: "empty" | "placeholder" | "confuser_llm" (configs/confusers.json, sinh sẵn) | "oracle".
"""
from __future__ import annotations

from .types import Candidates, NegClass, PosSelection


def mine_negatives(cands: Candidates, pos: PosSelection, cfg, confusers=None) -> list[NegClass]:
    raise NotImplementedError("S6 mine_negatives — Bi Anh")


def pick_negative_text(neg: NegClass, cfg, confusers=None, positive_text: str = "") -> str:
    raise NotImplementedError("S6 pick_negative_text — Bi Anh")
