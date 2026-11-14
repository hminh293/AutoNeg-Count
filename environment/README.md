# Môi trường

| Môi trường | Dùng cho | File |
|---|---|---|
| CPU (máy cá nhân) | M1/M2/M3, metrics, datasets, test | `requirements_cpu.txt` |
| CountGD++ (GPU) | lượt 1/lượt 2, baseline B1/B2, O1–O5 | `requirements_countgdpp.txt`, `setup_colab.sh`, `setup_kaggle.sh` |
| CountSE (GPU, tiến trình riêng) | baseline B3 | `requirements_countse.txt` |

**Bẫy đã biết** (Tài liệu 1, 2, 5):
1. `transformers` mới làm hỏng `bertwarper.py` → ghim `4.39.1`.
2. Biên dịch `MultiScaleDeformableAttention` lỗi → `patches/00_ms_deform_fallback.patch` cho phép chạy bằng PyTorch thuần (chậm hơn, kết quả tương đương).
3. `engine_inference.py` import `segment_anything` ngay đầu file → đừng import module đó; copy `get_pseudo_exemplars`.
4. CountGD++ và CountSE trùng tên gói (`util`, `datasets`, `models`) → hai tiến trình riêng, trao đổi bằng CSV.
5. `requirements.txt` của CountSE lưu UTF-16.

Sau khi chạy được, ghi `pip freeze > environment/freeze_<moi_truong>_<may>.txt` và `results/env_report.md` (GPU, VRAM đỉnh, giây/ảnh).
