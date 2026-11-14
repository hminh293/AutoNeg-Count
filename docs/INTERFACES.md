# Hợp đồng dữ liệu giữa các module

Nguồn sự thật: **`autoneg/types.py`**. Đổi hợp đồng = PR có cả A và B duyệt.

```
CountSample ──► run_pass1 / cache (Pass1Output) ──► build_candidates ──► Candidates
                                                                          │
                                         M1 cpe.select_positive (A) ◄─────┤
                                                   │ PosSelection         │
                                                   ▼                      │
                                         M2 anm.mine_negatives (B) ◄──────┘
                                                   │ list[NegClass]
                                                   ▼
                       forward_once(pos_boxes, neg_classes) ──► filter_two_stage(out2, tau)
                                                   ▲                      ▲
                                                   │        M3 threshold.pick_threshold (A)
                                                   ▼
                                               Prediction ──► io.make_row ──► per_image.csv
```

| Kiểu | Ai tạo | Ai dùng | Trường quan trọng |
|---|---|---|---|
| `CountSample` | datasets (A/B) | run_eval, predict | `text`, `gt_count`, `distractor_texts` |
| `Pass1Output` | run_pass1 / cache (AB) | build_candidates | `boxes` cxcywh chuẩn hoá, `emb` [N,256] |
| `Candidates` | build_candidates (AB) | M1, M2 | `boxes_xyxy` pixel ảnh sau resize, `scores` sắp giảm |
| `PosSelection` | M1 (A) | M2 (B), predict | `pos_idx`, `labels`, `best_cluster` |
| `NegClass` | M2 (B) | predict | `text`, `idx` |
| CSV `per_image.csv` | run_eval, run_countse_eval | make_tables | `PER_IMAGE_COLUMNS` |
