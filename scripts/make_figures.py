"""[B – Bi Anh] Lưới hình trước/sau (đỏ=bị loại, xanh=giữ, vàng=exemplar dương, tím=negative)  PLAN: S5/S6."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Lưới hình trước/sau (đỏ=bị loại, xanh=giữ, vàng=exemplar dương, tím=negative)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    raise NotImplementedError("S5/S6 scripts/make_figures.py — B – Bi Anh")


if __name__ == "__main__":
    main()
