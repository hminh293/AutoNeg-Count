# AutoNeg-Count — giảm đếm nhầm trong zero-shot counting

[![tests](https://github.com/hminh293/AutoNeg-Count/actions/workflows/tests.yml/badge.svg)](https://github.com/hminh293/AutoNeg-Count/actions/workflows/tests.yml)

Đồ án môn **CS331 – Thị giác máy tính nâng cao**: *hạn chế đếm nhầm trong đếm vật zero-shot*.

Ý tưởng (tên tạm **AutoNeg-Count**): chạy trên nền **CountGD++** lúc suy luận, không huấn luyện lại mô hình lớn:

1. **Lượt 1** – CountGD++ chỉ với chữ (ví dụ `"strawberry"`) → các hộp ứng viên.
2. **M1 – CPE** (*Clean Pseudo-Exemplar*): phân cụm embedding ứng viên, lấy mẫu dương từ cụm tốt nhất (điểm × kích thước).
3. **M2 – ANM** (*Auto-Negative Mining*): cụm bị loại → mẫu âm tự động.
4. **Lượt 2** – CountGD++ với mẫu dương + âm, lọc 2 tầng (`pos > τ` và `pos > neg`); **M3** chọn τ theo từng ảnh (Otsu).
5. (Tuỳ chọn) **Verifier** GBDT kiểm tra lại từng hộp.

Đối chứng: **B1** (CountGD++ chỉ chữ), **B2** (+ pseudo-exemplar top-3), **B3** (CountSE).

---

## Cấu trúc

```
AutoNeg-Count/
├── autoneg/                 # thư viện chính (import autoneg)
│   ├── types.py             # HỢP ĐỒNG dữ liệu giữa các module (CountSample, Candidates, PosSelection…)
│   ├── paths.py config.py io.py
│   ├── model_io.py          # [A] nạp CountGD++                (GPU)
│   ├── forward_utils.py     # [B] gọi model lượt 1 / lượt 2    (GPU)
│   ├── candidates.py        # [AB] cache lượt 1 → Candidates
│   ├── cluster.py cpe.py    # [A] M1
│   ├── anm.py               # [B] M2
│   ├── threshold.py         # [A] M3
│   ├── verifier.py          # [B] Tier-B
│   ├── predict.py           # [AB] pipeline đầy đủ
│   ├── metrics.py           # [A] MAE/RMSE/NAE/SRE/Bias + bootstrap
│   └── datasets/            # [A] fsc147 carpk shanghaitech · [B] pairtally omnicount bloodcell
├── scripts/                 # run_eval, run_countse_eval, cache_pass1, run_ablation, make_tables, make_figures, demo_app
├── configs/                 # YAML thí nghiệm (base.yaml + từng phương pháp)
├── tests/                   # pytest (CPU); test GPU đánh dấu @pytest.mark.gpu
├── third_party/             # submodule CountGDPlusPlus, CountSE (KHÔNG sửa trực tiếp)
├── patches/                 # mọi thay đổi vào third_party dưới dạng .patch
├── environment/             # requirements + script cài Colab/Kaggle
├── docs/                    # INTERFACES, CONTRIBUTING, kế hoạch
├── modal_app.py             # chạy trên Modal
└── results/                 # (không commit file nặng)
```

`[A]` = Minh · `[B]` = Bi Anh · `[AB]` = làm chung. Chi tiết: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## Cài đặt

### Máy cá nhân (không GPU) — viết code M1/M2/M3, chạy test
```bash
git clone --recurse-submodules <url-repo>
cd AutoNeg-Count
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r environment/requirements_cpu.txt
pip install -e .
pytest -q
```
Test của module chưa cài sẽ hiện **SKIPPED (Chưa cài: …)** thay vì lỗi — cài xong là tự chạy.

### Colab / Kaggle (GPU)
```bash
bash environment/setup_colab.sh      # hoặc setup_kaggle.sh
pytest -q -m gpu
```
Chi tiết, bẫy cài đặt và fallback deformable attention: [environment/README.md](environment/README.md).

### Biến môi trường
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `COUNTGD_ROOT` | `third_party/CountGDPlusPlus` | mã CountGD++ |
| `COUNTSE_ROOT` | `third_party/CountSE` | mã CountSE |
| `DATA_ROOT` | `data/` | thư mục dữ liệu |
| `FSC147_ROOT` | `$DATA_ROOT/FSC147_384_V2` | ảnh FSC-147 |
| `CKPT_DIR` | `checkpoints/` | trọng số |
| `RESULTS_DIR` | `results/` | đầu ra |

---

## Chạy

```bash
python scripts/run_eval.py --config configs/b2_fsc147_val.yaml          # một thí nghiệm → results/<dataset>/<method>/per_image.csv
python scripts/make_tables.py --results results/ --out results/tables/  # gộp bảng + CI
```

## Quy tắc (tóm tắt — đầy đủ ở docs/CONTRIBUTING.md)
1. Không sửa `third_party/` trực tiếp; thay đổi lưu thành `patches/*.patch`.
2. CountGD++ và CountSE trùng tên gói → **không import chung một tiến trình**.
3. Không bịa số liệu. Seed 42. Mỗi kết quả truy vết được tới CSV + config + commit.
4. Mặc định (mọi module tắt) phải cho kết quả **giống hệt** CountGD++ gốc (test hồi quy).
5. `main` chỉ nhận code qua Pull Request, CI (pytest CPU) phải xanh.

## Nguồn
- CountGD++ — https://github.com/niki-amini-naieni/CountGDPlusPlus
- CountSE — https://github.com/pppppz22/CountSE
- Giấy phép: mã trong `third_party/` thuộc tác giả gốc, theo giấy phép của từng repo.
