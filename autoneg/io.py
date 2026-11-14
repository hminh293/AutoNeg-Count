"""Đọc/ghi CSV kết quả theo cột chuẩn PER_IMAGE_COLUMNS."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .types import PER_IMAGE_COLUMNS


def make_row(image_id, text, gt_count, pred_count, n_cand=0, time_s=0.0, **extra) -> Dict[str, Any]:
    err = int(pred_count) - int(gt_count)
    return dict(image_id=str(image_id), text=text, gt_count=int(gt_count), pred_count=int(pred_count),
                abs_err=abs(err), signed_err=err, n_cand=int(n_cand), time_s=round(float(time_s), 4),
                extra_json=json.dumps(extra, ensure_ascii=False))


def write_per_image(rows: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PER_IMAGE_COLUMNS})
    return path


def read_per_image(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("gt_count", "pred_count", "abs_err", "signed_err", "n_cand"):
            r[k] = int(r[k])
        r["time_s"] = float(r["time_s"])
    return rows
