# patches

| File | Áp vào | Mục đích | Người |
|---|---|---|---|
| `00_ms_deform_fallback.patch` | CountGDPlusPlus | Chạy deformable attention bằng PyTorch khi không build được CUDA op | AB |
| `10_forward_hooks.patch` | CountGDPlusPlus | `return_feats`, `pos_tokens`, `neg_tokens` (mặc định tắt) — **chưa có** | B |

Quy tắc: mỗi patch có cờ tắt; khi tắt, đầu ra phải giống hệt bản gốc (tests/test_regression.py).
