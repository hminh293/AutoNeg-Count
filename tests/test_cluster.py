"""Spec P1.2 — autoneg/cluster.py (Minh)."""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from autoneg.cluster import cluster, cos_sim, eigengap_k
from helpers import blobs


def _norm(E):
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def test_cos_sim_properties():
    E, _, _ = blobs([5, 5])
    S = cos_sim(_norm(E))
    assert S.shape == (10, 10)
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-6)
    assert (S >= 0).all() and np.allclose(S, S.T)


def test_eigengap_four_nodes_two_pairs():
    W = np.array([[0, 1, .1, 0], [1, 0, 0, .1], [.1, 0, 0, 1], [0, .1, 1, 0]], float)
    assert eigengap_k(W) == 2


def test_eigengap_on_two_blobs():
    E, _, _ = blobs([10, 10])
    assert eigengap_k(cos_sim(_norm(E))) == 2


@pytest.mark.parametrize("method", ["spectral", "kmeans_cos"])
def test_two_blobs_recovered(method):
    E, y, _ = blobs([10, 12])
    labels = cluster(E, method=method)
    assert adjusted_rand_score(y, labels) > 0.95


def test_method_none_single_cluster():
    E, _, _ = blobs([4, 4])
    assert len(set(cluster(E, method="none"))) == 1


@pytest.mark.parametrize("n", [1, 2])
def test_tiny_inputs_single_cluster(n):
    E, _, _ = blobs([n])
    labels = cluster(E, method="spectral")
    assert len(labels) == n and len(set(labels)) == 1


def test_identical_embeddings():
    E = np.ones((6, 256))
    labels = cluster(E, method="spectral")
    assert len(set(labels)) == 1
