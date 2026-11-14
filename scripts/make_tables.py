"""[A – Minh] Gộp các per_image.csv thành bảng Markdown (MAE/RMSE/NAE/Bias + CI bootstrap)  PLAN: S10."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Gộp các per_image.csv thành bảng Markdown (MAE/RMSE/NAE/Bias + CI bootstrap)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    raise NotImplementedError("S10 scripts/make_tables.py — A – Minh")


if __name__ == "__main__":
    main()
