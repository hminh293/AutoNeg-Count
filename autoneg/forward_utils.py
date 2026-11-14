"""[B – Bi Anh] Gọi model lượt 1 / lượt 2 (có mẫu dương, mẫu âm).  Cần GPU.  PLAN: S2.

Lưu ý đã xác minh trong code CountGD++: exemplar âm thứ i chèn vào CỤM THỨ i+1 của caption
→ mỗi lớp âm phải có một cụm text riêng ("" vẫn tạo cụm " . ").
"""
from __future__ import annotations


def forward_once(model, img_t, text, pos_boxes=None, neg_classes=None, device="cuda", return_feats=False):
    """img_t [3,H,W]; pos_boxes Tensor[k,4] xyxy pixel; neg_classes list[dict(text, boxes)].
    Trả (out: dict của model, caption: str)."""
    raise NotImplementedError("S2 forward_once — Bi Anh")


def run_pass1(model, img_t, text, n_top=50, min_score=0.10, device="cuda"):
    """Lượt 1 chỉ chữ + return_feats → Pass1Output (types.Pass1Output)."""
    raise NotImplementedError("S4 run_pass1 — làm chung")


def filter_two_stage(out, tau):
    """Sao chép NGUYÊN logic app.py::get_boxes_from_prediction: tách tại '.' đầu tiên của out['input_ids'];
    tầng 1 pos.max > tau; tầng 2 pos.max > neg.max (nếu có cụm âm). Trả (boxes, scores)."""
    raise NotImplementedError("S6 filter_two_stage — Bi Anh")
