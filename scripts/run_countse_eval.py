"""[A – Minh] Chạy CountSE (môi trường RIÊNG) trên FSC-147, ghi CSV cùng định dạng per_image.csv  PLAN: P3.2 (B3)."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy CountSE (môi trường RIÊNG) trên FSC-147, ghi CSV cùng định dạng per_image.csv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", action="store_true", help="mặc định TẮT để so sánh công bằng")
    a = ap.parse_args()
    raise NotImplementedError("P3.2 (B3) scripts/run_countse_eval.py — A – Minh")


if __name__ == "__main__":
    main()
