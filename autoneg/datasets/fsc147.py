"""[A – Minh] Adapter fsc147.  PLAN: P1.6 / S1.

GT từ COCO JSON của CountSE: third_party/CountSE/data/fsc147_gdino/fsc147_coco/coco_{val,test}_exemplars.json
- Mỗi điểm GT là hộp 2x2 (area == 4) → điểm = (x+1, y+1). 3 annotation cuối mỗi ảnh (area > 4) là exemplar.
  gt_count = số annotation có area == 4 (= len(anns) - 3).
- Tên lớp: json["categories"] (id -> name), khớp val_label_list trong config CountSE.
- Đã kiểm tra: val = 1286 ảnh, test = 1190 ảnh.
- Ảnh: $FSC147_ROOT/images_384_VarV2/<file_name>.
"""
from __future__ import annotations

import argparse


def load(split: str = "val", json_path=None, image_root=None):
    """Trả list[CountSample] (autoneg.types.CountSample)."""
    raise NotImplementedError("P1.6 datasets.fsc147 — Minh")


def check(samples) -> None:
    """In số mẫu, 3 mẫu ngẫu nhiên (seed 42), phân bố gt_count."""
    import random
    import statistics
    print("số mẫu:", len(samples))
    for s in random.Random(42).sample(list(samples), min(3, len(samples))):
        print(" ", s["image_id"], s["text"], s["gt_count"])
    counts = [s["gt_count"] for s in samples]
    if counts:
        print("gt_count: min", min(counts), "median", statistics.median(counts), "max", max(counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a, rest = ap.parse_known_args()
    check(load())
