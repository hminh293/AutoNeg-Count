"""[A – Minh] M1 – Clean Pseudo-Exemplar.  PLAN: P1.3 / S5.

select_positive(cands, cfg) -> PosSelection
- cfg: N, m (=3), alpha (=0.5), cluster ("spectral"|"kmeans_cos"|"none"),
       criterion ("score_size" | "size_only"), exemplar_mode ("topm" | "medoid" | "mean_token").
- criterion="score_size": q(c) = mean_score(c) * size(c) ** alpha   (đề xuất của nhóm)
- criterion="size_only" : q(c) = size(c)  → chọn CỤM ĐÔNG NHẤT (cách CountSE/CountZES) — đối chứng.
- cluster="none" hoặc N <= m → top-m theo điểm (= baseline B2).
"""
from __future__ import annotations

from .types import Candidates, PosSelection


def select_positive(cands: Candidates, cfg) -> PosSelection:
    raise NotImplementedError("P1.3 select_positive — Minh")
