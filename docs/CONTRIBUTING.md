# Cách làm việc chung

## Người phụ trách file
| File | Người giữ |
|---|---|
| `autoneg/model_io.py`, `cluster.py`, `cpe.py`, `threshold.py`, `metrics.py`, `datasets/{fsc147,carpk,shanghaitech}.py`, `scripts/run_countse_eval.py`, `scripts/make_tables.py` | **A – Minh** |
| `autoneg/forward_utils.py`, `anm.py`, `verifier.py`, `datasets/{pairtally,omnicount,bloodcell}.py`, `patches/10_forward_hooks.patch`, `scripts/demo_app.py`, `scripts/make_figures.py` | **B – Bi Anh** |
| `autoneg/types.py`, `candidates.py`, `predict.py`, `configs/`, `scripts/run_eval.py`, `scripts/cache_pass1.py`, `scripts/run_ablation.py` | **Chung** (PR cần cả hai duyệt) |

## Nhánh & Pull Request
- `main`: chỉ nhận qua PR, CI xanh, người kia duyệt.
- Nhánh làm việc: `feat/a-<việc>` (Minh), `feat/b-<việc>` (Bi Anh). Ví dụ `feat/a-cpe`, `feat/b-anm`.
- Commit nhỏ, thông điệp rõ: `cpe: thêm criterion size_only + test`.
- Mỗi PR: điền mẫu PR, cập nhật `REPORT.md`.

## Test
- `pytest -q` trên máy không GPU phải xanh (module chưa cài hiện SKIPPED).
- Test cần GPU đánh dấu `@pytest.mark.gpu` — tự bỏ qua khi không có CUDA; chạy `pytest -m gpu` trên Colab/Kaggle.
- Stub chưa cài phải `raise NotImplementedError("<bước> <hàm> — <người>")` để test tự SKIP.

## Số liệu
- Không bịa số. Chưa chạy được thì ghi "chưa chạy" trong REPORT.md.
- Seed 42. Mỗi bảng truy vết được tới CSV + YAML + commit hash.

## Kế hoạch
- `docs/plan/05_PLAN_STEPS_cho_agent.md` — kế hoạch chung (S0–S12)
- `docs/plan/07_Ke_hoach_vai_A_Minh.md` — kế hoạch vai A (P0–P7)
- Bản gốc + tài liệu giải thích nằm ở thư mục `Tai_lieu/` ngoài repo.
