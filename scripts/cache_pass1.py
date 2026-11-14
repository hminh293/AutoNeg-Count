"""[AB] Cache đầu ra lượt 1 (.npz theo types.Pass1Output) để ablation M1/M2/M3 chạy CPU  PLAN: P4.1 / S4."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache đầu ra lượt 1 (.npz theo types.Pass1Output) để ablation M1/M2/M3 chạy CPU")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    raise NotImplementedError("P4.1 / S4 scripts/cache_pass1.py — AB")


if __name__ == "__main__":
    main()
