"""[B – Bi Anh] Adapter bloodcell.  PLAN: S1.

COCO trong CountGDPlusPlus/data/*/_annotations.coco.json; mỗi (ảnh, lớp) = 1 mẫu.
"""
from __future__ import annotations

import argparse


def load(root=None):
    """Trả list[CountSample] (autoneg.types.CountSample)."""
    raise NotImplementedError("S1 datasets.bloodcell — Bi Anh")


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
