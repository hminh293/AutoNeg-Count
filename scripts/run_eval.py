"""[AB] Chạy 1 thí nghiệm (dataset, method, config) → results/<dataset>/<method>/per_image.csv.  PLAN: S3/S10.

Chạy trong MÔI TRƯỜNG CountGD++. Với --cache_dir, M1/M2/M3 đọc lượt 1 từ cache (chỉ lượt 2 cần GPU).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoneg.config import load_config  # noqa: E402
from autoneg.paths import results_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N mẫu đầu (debug)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    random.seed(cfg.seed)
    out = Path(a.out) if a.out else results_dir() / cfg.dataset / cfg.method / "per_image.csv"
    print(f"[run_eval] dataset={cfg.dataset} method={cfg.method} -> {out}")
    # TODO (S3/S7): model, transform = load_countgdpp(); for sample in get_dataset(cfg.dataset):
    #   pred = predict(model, transform, sample, cfg, cache); rows.append(make_row(...)); write_per_image(rows, out)
    raise NotImplementedError("S3/S7 run_eval — làm chung")


if __name__ == "__main__":
    main()
