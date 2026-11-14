"""[B – Bi Anh] Tier-B Candidate Verifier (GBDT).  PLAN: S9.

Huấn luyện trên FSC-147 VAL (không dùng train: CountGD++ đã học trên train), GroupKFold theo tên lớp.
"""
from __future__ import annotations


def candidate_features(cands, pos, negs, out2=None):
    raise NotImplementedError("S9 candidate_features — Bi Anh")


def train_verifier(rows, groups, seed: int = 42):
    raise NotImplementedError("S9 train_verifier — Bi Anh")
