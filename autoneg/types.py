"""HỢP ĐỒNG DỮ LIỆU giữa các module — đã chốt giữa A (Minh) và B (Bi Anh).

Muốn đổi file này: mở Pull Request, cả hai cùng duyệt (xem docs/INTERFACES.md).
Quy ước toạ độ: hộp `xyxy` tính bằng **pixel của ảnh SAU khi resize** (ảnh đưa vào model).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np


class CountSample(TypedDict):
    """Một mẫu đánh giá = (một ảnh, một lớp cần đếm)."""
    image_path: str
    image_id: str
    text: str                          # tên lớp cần đếm (positive prompt)
    gt_count: int
    gt_points: Optional[np.ndarray]    # [N,2] (x,y) pixel ảnh gốc, nếu có
    gt_boxes: Optional[np.ndarray]     # [N,4] xywh pixel ảnh gốc, nếu có
    distractor_texts: List[str]        # lớp gây nhầm có GT trong cùng ảnh; [] nếu không có
    meta: Dict[str, Any]


class Pass1Output(TypedDict):
    """Đầu ra lượt 1 (chỉ chữ), cũng là định dạng cache `.npz` (scripts/cache_pass1.py)."""
    boxes: np.ndarray        # [900,4] cxcywh chuẩn hoá 0..1
    probs_text: np.ndarray   # [900,Lt] sigmoid logit trên các token của cụm dương
    image_hw: np.ndarray     # [2] (H,W) ảnh sau resize
    topN_idx: np.ndarray     # [N] chỉ số query điểm cao (N<=50, điểm>=0.10)
    emb: np.ndarray          # [N,256] roi_align embedding của topN_idx


class Candidates(TypedDict):
    """Ứng viên đưa vào M1/M2 (A tạo từ Pass1Output bằng candidates.build_candidates)."""
    boxes_xyxy: np.ndarray   # [N,4] pixel ảnh sau resize
    scores: np.ndarray       # [N] sắp GIẢM dần, đã lọc > tau_low
    emb: np.ndarray          # [N,256]


class PosSelection(TypedDict, total=False):
    """Đầu ra M1 (cpe.select_positive) — M2 (anm.mine_negatives) đọc labels/best_cluster."""
    pos_idx: List[int]           # <= m chỉ số trong Candidates
    labels: np.ndarray           # [N] nhãn cụm của từng ứng viên
    best_cluster: int
    quality: Dict[int, float]    # điểm chất lượng từng cụm
    pos_tokens: np.ndarray       # [k,256] chỉ khi exemplar_mode == "mean_token"


class NegClass(TypedDict):
    """Một lớp âm do M2 khai thác."""
    text: str                # "" | "object" | tên vật dễ nhầm
    idx: List[int]           # chỉ số trong Candidates dùng làm exemplar âm
    score: float             # điểm trung bình cụm
    dist: float              # 1 - cos tới prototype dương


class Prediction(TypedDict, total=False):
    count: int
    boxes: np.ndarray        # [K,4] xyxy pixel ảnh sau resize
    scores: np.ndarray       # [K]
    pos_boxes: np.ndarray
    negs: List[Dict[str, Any]]
    tau: float
    debug: Dict[str, Any]


# Cột chuẩn của results/<dataset>/<method>/per_image.csv (A và B dùng chung)
PER_IMAGE_COLUMNS = [
    "image_id", "text", "gt_count", "pred_count", "abs_err", "signed_err",
    "n_cand", "time_s", "extra_json",
]
