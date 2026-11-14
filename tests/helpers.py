import numpy as np

from autoneg.config import Cfg


def blobs(sizes, d=256, noise=0.1, seed=42, centers=None):
    """Các cụm Gauss quanh tâm ngẫu nhiên (gần như trực giao trong 256 chiều)."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = [rng.normal(size=d) for _ in sizes]
    E, y = [], []
    for c, (ctr, n) in enumerate(zip(centers, sizes)):
        E.append(ctr + noise * rng.normal(size=(n, d)))
        y += [c] * n
    return np.vstack(E), np.array(y), centers


def make_cands(E, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    n = len(scores)
    boxes = np.tile(np.array([[0, 0, 10, 10]], float), (n, 1)) + np.arange(n)[:, None]
    return dict(boxes_xyxy=boxes[order], scores=np.asarray(scores, float)[order], emb=E[order]), order


def cpe_cfg(**kw):
    base = dict(N=30, m=3, alpha=0.5, cluster="spectral", criterion="score_size", exemplar_mode="topm")
    base.update(kw)
    return Cfg(base)
