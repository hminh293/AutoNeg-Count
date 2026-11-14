"""Kiểm tra khung repo: import được, hợp đồng dữ liệu, config, CSV, chuyển toạ độ."""
import importlib

import numpy as np
import pytest

CPU_MODULES = ["autoneg", "autoneg.types", "autoneg.paths", "autoneg.config", "autoneg.io",
               "autoneg.candidates", "autoneg.cluster", "autoneg.cpe", "autoneg.anm", "autoneg.threshold",
               "autoneg.metrics", "autoneg.verifier", "autoneg.predict", "autoneg.datasets"]


@pytest.mark.parametrize("name", CPU_MODULES)
def test_import_without_torch(name):
    importlib.import_module(name)


def test_per_image_columns():
    from autoneg.types import PER_IMAGE_COLUMNS
    assert PER_IMAGE_COLUMNS[:4] == ["image_id", "text", "gt_count", "pred_count"]


def test_config_merge():
    from autoneg.config import load_config
    from autoneg.paths import REPO_ROOT
    cfg = load_config(REPO_ROOT / "configs" / "o3_fsc147_val.yaml")
    assert cfg.method == "o3" and cfg.cpe.enabled and cfg.anm.enabled
    assert cfg.cpe.alpha == 0.5 and cfg.thr.tau == 0.23 and cfg.seed == 42


def test_all_configs_load():
    from autoneg.config import load_config
    from autoneg.datasets import REGISTRY
    from autoneg.paths import REPO_ROOT
    for p in (REPO_ROOT / "configs").glob("*.yaml"):
        if p.name == "base.yaml":
            continue
        cfg = load_config(p)
        assert cfg.dataset in REGISTRY, p.name


def test_csv_roundtrip(tmp_path):
    from autoneg.io import make_row, read_per_image, write_per_image
    rows = [make_row("1", "apple", 10, 12, n_cand=5, time_s=0.5, tau=0.23)]
    p = write_per_image(rows, tmp_path / "a" / "per_image.csv")
    back = read_per_image(p)
    assert back[0]["signed_err"] == 2 and back[0]["abs_err"] == 2 and back[0]["gt_count"] == 10


def test_box_conversion():
    from autoneg.candidates import cxcywh_norm_to_xyxy_pixel
    b = cxcywh_norm_to_xyxy_pixel(np.array([[0.5, 0.5, 0.2, 0.4], [0.0, 0.0, 0.2, 0.2]]), (100, 200))
    np.testing.assert_allclose(b[0], [80, 30, 120, 70])
    np.testing.assert_allclose(b[1], [0, 0, 20, 10])   # đã clamp
