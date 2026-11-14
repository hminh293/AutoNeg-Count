"""[AB] Quét một biến quanh O3 trên cache (vd --var cpe.alpha --values 0 0.5 1)  PLAN: S10."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Quét một biến quanh O3 trên cache (vd --var cpe.alpha --values 0 0.5 1)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--var", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    a = ap.parse_args()
    raise NotImplementedError("S10 scripts/run_ablation.py — AB")


if __name__ == "__main__":
    main()
