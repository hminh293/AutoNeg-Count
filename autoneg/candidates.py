"""[AB] Pass1Output (cache) → Candidates cho M1/M2.  PLAN: S7 build_candidates."""
from __future__ import annotations

import numpy as np

from .types import Candidates, Pass1Output


def cxcywh_norm_to_xyxy_pixel(boxes: np.ndarray, hw) -> np.ndarray:
    """Hộp cxcywh chuẩn hoá → xyxy pixel theo (H,W) ảnh sau resize, có clamp (giống get_pseudo_exemplars)."""
    H, W = float(hw[0]), float(hw[1])
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, W)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, H)
    return out


def build_candidates(p1: Pass1Output, tau_low: float = 0.15, N: int = 30) -> Candidates:
    """Lọc điểm > tau_low trong topN_idx, giữ tối đa N, sắp giảm theo điểm."""
    raise NotImplementedError("S7 build_candidates — làm chung")
