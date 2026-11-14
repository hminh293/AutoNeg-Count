"""[A – Minh] Phân cụm embedding ứng viên.  PLAN: P1.2 / S5.

Yêu cầu:
- cos_sim(E): E đã L2-normalize; giá trị âm → 0.
- eigengap_k(S, kmax=4, gap_thr=0.1): Laplacian chuẩn hoá (scipy.sparse.csgraph.laplacian(normed=True)),
  trị riêng bằng np.linalg.eigvalsh (đã sắp tăng dần — sửa lỗi LA.eig của CountSE).
- cluster(E, method in {"none","spectral","kmeans_cos"}, k=None) -> np.ndarray[int] nhãn cụm.
- Ca biên: N < 3, embedding trùng nhau, hàng toàn 0 → không lỗi, trả 1 cụm.
"""
from __future__ import annotations

import numpy as np


def cos_sim(E: np.ndarray) -> np.ndarray:
    raise NotImplementedError("P1.2 cos_sim — Minh")


def eigengap_k(S: np.ndarray, kmax: int = 4, gap_thr: float = 0.1) -> int:
    raise NotImplementedError("P1.2 eigengap_k — Minh")


def cluster(E: np.ndarray, method: str = "spectral", k: int | None = None, seed: int = 0) -> np.ndarray:
    raise NotImplementedError("P1.2 cluster — Minh")
