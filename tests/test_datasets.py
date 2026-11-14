"""Spec P1.6 — autoneg/datasets/fsc147.py (Minh), dùng JSON giả nhỏ (không cần tải dữ liệu)."""
import json

import numpy as np

from autoneg.datasets import fsc147


def _fake_json(path):
    anns, aid = [], 0
    def add(img, x, y, w=2, h=2):
        nonlocal aid
        aid += 1
        anns.append(dict(id=aid, image_id=img, bbox=[x, y, w, h], area=w * h, category_id=1, iscrowd=0))
    for i in range(5):
        add(1, 10 * i, 5)
    for _ in range(3):
        add(1, 0, 0, 20, 30)            # 3 exemplar ở cuối
    for i in range(2):
        add(2, 7, 7 * i)
    for _ in range(3):
        add(2, 0, 0, 15, 15)
    d = dict(images=[dict(id=1, file_name="1.jpg", height=384, width=576),
                     dict(id=2, file_name="2.jpg", height=384, width=512)],
             annotations=anns, categories=[dict(id=0, name="ant"), dict(id=1, name="strawberry")])
    path.write_text(json.dumps(d), encoding="utf-8")


def test_fsc147_fake(tmp_path):
    jp = tmp_path / "coco_val_exemplars.json"
    _fake_json(jp)
    samples = fsc147.load(split="val", json_path=jp, image_root=tmp_path)
    by = {s["image_id"]: s for s in samples}
    assert len(samples) == 2
    assert by["1"]["gt_count"] == 5 and by["2"]["gt_count"] == 2
    assert by["1"]["text"] == "strawberry" and by["1"]["distractor_texts"] == []
    np.testing.assert_allclose(by["1"]["gt_points"][0], [1, 6])      # (x+1, y+1)
    assert by["1"]["image_path"].endswith("1.jpg")
