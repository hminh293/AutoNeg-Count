# TÀI LIỆU 7 — KẾ HOẠCH VAI A (Minh): “Nhánh dương & zero-shot chuẩn”

> **Người làm:** Minh (vai A) · **Làm cặp với:** Bi Anh (vai B)
> **Phần của Minh:** chạy lại baseline B1/B2 (CountGD++) và B3 (CountSE) · dữ liệu FSC-147, CARPK, ShanghaiTech · **M1 – CPE** (chọn mẫu sạch) · **M3** (ngưỡng thích nghi) · ablation phía dương · thuyết trình CountSE + M1/M3.
> **Đọc kèm:** `05_PLAN_STEPS_cho_agent.md` (kế hoạch chung, bản cho agent) · `04_Ke_hoach_do_an_de_doc.md` (bản đọc hiểu) · `02_CountSE_Giai_thich_code.md` · `06_Tu_dien_thuat_ngu.md` (tra thuật ngữ).
> **Cách đọc mỗi bước:** *Làm gì → File → Lệnh → Kiểm tra (xong khi nào) → Công nghệ → 🔗 Nối vào kế hoạch chung*.

---

## 0. Bức tranh một trang

| Khi nào | Giai đoạn | Việc chính | Máy |
|---|---|---|---|
| **Bây giờ** | P0 + P1 | Chốt giao diện với Bi Anh · code `cluster`, `cpe`, `threshold`, `metrics`, adapter FSC-147 + test | Windows, không GPU |
| Tuần 1 | P2 | Môi trường + `model_io` + demo 1 ảnh | Colab T4 |
| Tuần 1–2 | P3 | Test hồi quy · baseline B1, B2, B3 | Colab / Kaggle |
| Tuần 2–3 | P4 | Cache lượt 1 · adapter CARPK, ShanghaiTech | Kaggle |
| Tuần 3 | P5 | M1 trên ảnh thật (O1) · ghép `predict.py` với Bi Anh | Kaggle (lượt 2) |
| Tuần 4 | P6 | M3 (O4) · ablation phía A · đo tốc độ | CPU + ít GPU |
| Tuần 5–6 | P7 | Bảng, phân tích lỗi, báo cáo, slide | Không cần GPU |

**Nguyên tắc ưu tiên:** làm trước những gì **không cần GPU** (thuật toán + test bằng dữ liệu giả) → khi lên GPU chỉ còn việc “cắm dây” và chạy. M1 và M3 bản chất là code numpy/scikit-learn, nên **có thể viết và test xong ngay tuần này**.

> Lịch giả định bắt đầu **thứ Hai 16/11/2026**, 6 tuần. Đổi lại theo hạn nộp thật.

---

## P0. Chốt “hợp đồng” với Bi Anh (hôm nay, ~1 giờ họp)

**Làm gì**
1. Thống nhất **cấu trúc dữ liệu truyền giữa M1 và M2** — đây là chỗ duy nhất hai người phụ thuộc trực tiếp nhau:
   ```python
   # đầu vào chung (A tạo từ lượt 1 / cache)
   cands = dict(
       boxes_xyxy = np.ndarray[N, 4],   # pixel, theo ảnh SAU resize
       scores     = np.ndarray[N],      # sắp giảm dần, đã lọc > tau_low
       emb        = np.ndarray[N, 256], # roi_align embedding
   )
   # đầu ra của M1 (A) — M2 (B) đọc
   pos = dict(
       pos_idx      = list[int],        # ≤ m chỉ số mẫu dương
       labels       = np.ndarray[N],    # nhãn cụm của từng ứng viên
       best_cluster = int,
       quality      = dict[int, float], # điểm chất lượng từng cụm
   )
   ```
2. Thống nhất `CountSample` (giao diện dataset) và cột CSV kết quả: `image_id,text,gt_count,pred_count,abs_err,signed_err,n_cand,time_s,extra_json`.
3. Tạo repo Git `AutoNegCount/` (GitHub private), mời Bi Anh. Nhánh: `main` (chỉ gộp qua PR), `feat/a-core` (Minh), `feat/b-core` (Bi Anh).
4. Chia **file chung** — mỗi file chỉ một người giữ, người kia sửa qua PR:
   | File | Người giữ |
   |---|---|
   | `autoneg/model_io.py`, `autoneg/cluster.py`, `autoneg/cpe.py`, `autoneg/threshold.py`, `autoneg/metrics.py`*, `autoneg/datasets/{fsc147,carpk,shanghaitech}.py`, `scripts/run_countse_eval.py`, `scripts/make_tables.py`* | **Minh** |
   | `autoneg/forward_utils.py`, hook trong `groundingdino_app.py`, `autoneg/anm.py`, `autoneg/verifier.py`, `datasets/{pairtally,omnicount,bloodcell}.py`, `scripts/demo_app.py`, `scripts/make_figures.py`* | Bi Anh |
   | `autoneg/predict.py`, `configs/*.yaml`, `scripts/run_eval.py`, `scripts/cache_pass1.py` | làm chung (ngồi cùng) |

   \* *Đề xuất thêm của tài liệu này* (kế hoạch chung chưa gán `metrics.py`, `make_tables.py`, `make_figures.py`) — **chốt lại với Bi Anh**.

**Xong khi:** có repo, 2 nhánh, file `docs/INTERFACES.md` ghi 2 khối code trên, cả hai đồng ý.

🔗 **Nối vào kế hoạch chung:** `05` mục **1** (cấu trúc thư mục), **S1** (giao diện `CountSample`), **S3** (cột CSV), **S7** (`predict` dùng `cands`/`pos`), mục **3** (phân công). `04` mục **4**.

---

## P1. Code chạy được trên CPU (tuần này, máy Windows) ⭐ *bắt đầu ngay*

Có thể giao nguyên phần này cho Antigravity (prompt mẫu ở **Phụ lục A**). Tất cả chạy bằng `pytest` trên máy không GPU.

### P1.1 Khung repo + test
- **Làm gì:** tạo cây thư mục theo `05` mục 1 (chỉ thư mục + file rỗng có docstring), `pyproject.toml` hoặc `requirements_cpu.txt` (`numpy==1.26.4 scipy scikit-learn pandas pyyaml pytest`), `pytest.ini` khai báo marker `gpu`.
- **Kiểm tra:** `pytest -q` chạy (0 test cũng được) không lỗi import.
- **Công nghệ:** Python 3.10, venv, pytest, Git.
- 🔗 `05` mục **0** (quy tắc 1, 4, 6, 7), mục **1**.

### P1.2 `autoneg/cluster.py` — phân cụm
- **Làm gì:** `cos_sim`, `eigengap_k` (dùng `np.linalg.eigvalsh` — đã sắp tăng dần, sửa lỗi `LA.eig` của CountSE), `cluster(E, method ∈ {none, spectral, kmeans_cos}, k=None)`. Xử lý ca biên: N < 3, mọi embedding trùng nhau, ma trận có hàng toàn 0.
- **Kiểm tra (`tests/test_cluster.py`):**
  1. Ma trận ví dụ 4 nút trong Tài liệu 3 → `eigengap_k = 2`.
  2. Hai cụm Gauss tách rõ trên mặt cầu 256-d (seed 42) → spectral và kmeans_cos đều cho 2 cụm, độ khớp nhãn (ARI) > 0.95.
  3. N = 1, N = 2 không lỗi, trả 1 cụm.
- **Công nghệ:** numpy, scipy (`csgraph.laplacian`), scikit-learn (`SpectralClustering`, `KMeans`, `adjusted_rand_score`).
- 🔗 `05` **S5** (khung `cluster.py`). Kiến thức: `03` mục eigengap/spectral; `06` mục 18–24.

### P1.3 `autoneg/cpe.py` — M1 chọn mẫu dương sạch
- **Làm gì:** `select_positive(cands, cfg)` đúng chữ ký ở P0. Tiêu chí chất lượng cụm `q(c) = mean_score(c) × size(c)^alpha`. Ba cách lấy exemplar trong cụm tốt: `topm` (top-m theo điểm), `medoid`, `mean_token` (trả thêm `pos_tokens` dạng `[1,256]` cho biến thể soft-token).
- **Thêm so với kế hoạch chung (quan trọng cho tính mới):** thêm tiêu chí `criterion: size_only` = **chọn cụm đông nhất** — đúng cách của CountSE (SEG) và CountZES. Đây là đối chứng trực tiếp để chứng minh “điểm × kích thước” của M1 tốt hơn quy tắc “cụm đông nhất” của các paper trước.
- **Kiểm tra (`tests/test_cpe.py`):**
  1. Cụm đúng 8 điểm × 0.5, cụm nhiễu 4 điểm × 0.8: `alpha=1` chọn cụm đúng (4.0 > 3.2); `alpha=0` chọn cụm nhiễu (giống baseline top-điểm).
  2. Cụm nhiễu **đông hơn** (12 điểm × 0.3) và cụm đúng 6 điểm × 0.7: `size_only` chọn sai, `alpha=0.5` chọn đúng → đây chính là ca minh hoạ cho slide.
  3. `cluster="none"` hoặc N ≤ m → trả top-m theo điểm (bằng B2).
- **Công nghệ:** numpy.
- 🔗 `05` **S5** (khung `cpe.py`, tham số mặc định `N=30, tau_low=0.15, m=3, alpha=0.5`). `04` mục **2** (M1) và mục “Trung thực về tính mới”.

### P1.4 `autoneg/threshold.py` — M3 ngưỡng thích nghi
- **Làm gì:** `fixed(tau=0.23)`; `otsu(scores, bins=64, lo=0.05, clip=(0.15, 0.5))`; `valley(scores)` — điểm gãy lớn nhất của dãy điểm sắp giảm. Hàm chung `pick_threshold(probs_text_max, cfg)`.
- **Kiểm tra (`tests/test_threshold.py`):** hai mode 0.1±0.03 và 0.8±0.05 → `otsu` ∈ (0.2, 0.7); dãy điểm phẳng (1 mode) → trả về giá trị trong khoảng clip, không lỗi; mảng rỗng → trả `fixed`.
- **Công nghệ:** numpy (tự viết Otsu 1-D, ~15 dòng; có thể đối chiếu `skimage.filters.threshold_otsu` trong test).
- 🔗 `05` **S8**. Kiến thức: `03` mục Otsu; `06` mục 26, 93.

### P1.5 `autoneg/metrics.py` — chỉ số đánh giá *(đề xuất nhận thêm)*
- **Làm gì:** `mae, rmse, nae, sre, bias, overcount_rate`, `bootstrap_ci(err, B=10000, seed=42)`, `paired_bootstrap(pred_a, pred_b, gt)` → CI 95% của hiệu MAE.
- **Kiểm tra:** ví dụ tay trong Tài liệu 3 (MAE 2.5 / RMSE 3.317 / NAE 0.125 / SRE 0.671 / Bias +1.5) ra đúng tới 3 chữ số.
- 🔗 `05` **S10** (“Thống kê”). `04` mục **3** (metric).

### P1.6 `autoneg/datasets/fsc147.py` — adapter FSC-147 (phần GT chạy được ngay)
- **Làm gì:** đọc `CountSE/data/fsc147_gdino/fsc147_coco/coco_{val,test}_exemplars.json`; `gt_count = số annotation có area==4` (kiểm tra bằng `len(anns) - 3`); điểm GT = `(x+1, y+1)`; tên lớp từ `category_id` → `val_label_list` trong config CountSE. Đường dẫn ảnh đọc từ biến môi trường `FSC147_ROOT`.
- **Kiểm tra:** `python -m autoneg.datasets.fsc147 --check` in **1286 ảnh val / 1190 ảnh test**, 3 mẫu ngẫu nhiên, phân bố `gt_count`; test với JSON giả 2 ảnh.
- **Song song:** tải ảnh `FSC147_384_V2` (từ repo LearningToCountEverything) lên **Google Drive** để tuần sau Colab dùng.
- 🔗 `05` **S1** (mục FSC-147, đã có số liệu kiểm tra). `04` mục **3** (bộ dữ liệu).

**Xong P1 khi:** `pytest -q` pass toàn bộ trên máy Windows; commit lên `feat/a-core`; mở PR đầu tiên cho Bi Anh xem `INTERFACES.md` + `cpe.py`.

---

## P2. Môi trường GPU + nạp model (tuần 1, Colab T4)

| Bước | Làm gì | Kiểm tra (xong khi) |
|---|---|---|
| P2.1 | Notebook `notebooks/A_setup_colab.ipynb`: mount Drive, clone repo, cài `torch` (bản Colab sẵn có, < 2.6), `transformers==4.39.1`, `numpy==1.26.4`; tải `countgd_plusplus.pth` (gdown) + `bert-base-uncased` | `torch.cuda.is_available()` = True |
| P2.2 | Thử build CUDA op `MultiScaleDeformableAttention`; quá 30 phút thì dùng **fallback PyTorch** (`patches/00_ms_deform_fallback.patch`) | import model không cảnh báo, hoặc ghi rõ đang dùng fallback |
| P2.3 | `autoneg/model_io.py::load_countgdpp` (khung ở `05` S2) | chạy `app.py`/hàm của ta trên `img/336.jpg` + "strawberry" ra một con số |
| P2.4 | Ghi `results/env_report.md`: GPU, VRAM đỉnh (`torch.cuda.max_memory_allocated`), giây/ảnh, thời gian nạp model | file có đủ 4 số |

- **Phối hợp:** S0 là việc **làm chung** — một người đi trước, người kia chạy lại theo notebook để chắc chắn tái lập được.
- **Công nghệ:** Colab, gdown, PyTorch, CUDA.
- 🔗 `05` **S0** (toàn bộ), **S2** (`model_io` là phần của A), mục **6** rủi ro 1, 2, 4. `04` mục **6** (phần cứng).

---

## P3. Test hồi quy + Baseline (tuần 1–2)

### P3.1 Test hồi quy `tests/test_regression.py` `[gpu]`
- **Làm gì:** 20 ảnh ShanghaiTech (hoặc ảnh trong `img/`): pipeline 2 lượt của ta với M1/M2/M3 **tắt** phải ra **số đếm giống hệt** `test_shanghai_tech.py` (ngưỡng 0.23, `get_pseudo_exemplars` copy ra — **không** import `engine_inference.py`).
- **Phụ thuộc:** cần `forward_utils.forward_once` của Bi Anh. Nếu chưa có → Minh viết tạm bản tối giản, sau thay bằng bản của Bi Anh.
- 🔗 `05` **S2** (“Test hồi quy” là phần của A), quy tắc **0.5**.

### P3.2 Baseline trên FSC-147 val/test
| ID | Làm gì | Môi trường |
|---|---|---|
| **B1** | CountGD++ chỉ chữ, 1 lượt | môi trường CountGD++ |
| **B2** | B1 + pseudo-exemplar top-3, 2 lượt | môi trường CountGD++ |
| **B3** | CountSE chỉ chữ — `scripts/run_countse_eval.py`, **tiến trình/môi trường riêng** (trùng tên gói), bỏ `--crop`; bỏ `np.save("logits.npy")` | môi trường CountSE (Kaggle/Colab khác) |

- **Lưu ý code CountSE** (đã phát hiện ở Tài liệu 2): nhánh `--eval` của `main.py` gọi `evaluate` sai tham số → dùng `main_inference.py`; `requirements.txt` lưu UTF-16 → chuyển UTF-8 trước khi `pip install -r`.
- **Đầu ra:** `results/fsc147_{val,test}/{b1,b2,b3}/per_image.csv`.
- **Xong khi:** `make_tables.py` in bảng MAE/RMSE/Bias cho B1–B3, kèm **chênh lệch so với paper và nguyên nhân giả định** (không cần khớp tuyệt đối).
- **Ước lượng GPU:** ~2 476 ảnh × 2 lượt; trên T4 vài giờ → chạy trên **Kaggle** (miễn phí), không tiêu Modal.
- 🔗 `05` **S3** (B1, B2, B3 là của A), quy tắc **0.2**, **0.3**. `04` mục **3** (B1–B3). `02` (điểm yếu CountSE).

---

## P4. Cache lượt 1 + dữ liệu cross-dataset (tuần 2–3)

- **P4.1** `scripts/cache_pass1.py` *(làm chung với Bi Anh, mỗi người chạy cache cho bộ dữ liệu của mình)*: lưu `.npz` gồm `boxes[900,4]`, `probs_text`, `image_hw`, `topN_idx` (N=50, điểm ≥ 0.10), `emb[N,256]`. **Minh chạy cho:** FSC-147 val + test, CARPK, ShanghaiTech A/B. Test `tests/test_cache.py`: nạp lại khớp chạy trực tiếp ≤ 1e-4.
- **P4.2** Adapter `carpk.py` (prompt `"car"`, có thể dùng `CountSE/cross_dataset/prepare_cross_dataset.py`) và `shanghaitech.py` (`.mat`, prompt `"people"`), kèm `--check`.
- **Vì sao quan trọng:** sau khi có cache, **toàn bộ M1 và M3 chạy trên CPU** → ablation phía A gần như không tốn GPU.
- 🔗 `05` **S4**, **S1** (CARPK/SHA). `04` mục **3** (cross-dataset).

---

## P5. M1 trên dữ liệu thật (tuần 3)

- **Làm gì:** `build_candidates(cache, cfg)` (lọc `tau_low`, top-N, cxcywh → xyxy pixel có `clamp`) → `select_positive` → lượt 2 trên GPU với mẫu dương mới = **O1**.
- **Chạy:** O1 trên 100 ảnh FSC-147 val trước; sau đó toàn bộ val/test.
- **Hình:** `results/figs/cpe_*.png` — hộp ứng viên tô theo màu cụm, exemplar được chọn viền vàng; chọn 3–4 ảnh đẹp nhất cho slide.
- **Ghép với Bi Anh:** ngồi cùng viết `predict.py` (S7) — M2 lấy `pos["labels"]`, `pos["best_cluster"]` từ M1.
- **Xong khi:** bảng B2 vs O1 trên FSC-147 val (MAE, Bias, CI); 10 hình trước/sau.
- 🔗 `05` **S5** (DoD), **S7** (`predict`), **S10** (ma trận O1). `04` mục **2**, giả thuyết **H1**.

---

## P6. M3 + ablation phía A + đo tốc độ (tuần 4)

- **M3:** so `fixed / otsu / valley` trên cấu hình O3 (khi M2 của Bi Anh đã ghép) → **O4**.
- **Ablation của Minh** (mỗi lần đổi một biến quanh O3, chạy trên cache):
  | Biến | Giá trị |
  |---|---|
  | N / tau_low | {10, 20, 30} / {0.10, 0.15, 0.20} |
  | cluster | none / spectral / kmeans_cos |
  | alpha | {0, 0.5, 1} |
  | **criterion** *(thêm mới)* | score×size^alpha / **size_only (kiểu CountSE, CountZES)** |
  | m (số exemplar) | {1, 3, 5} |
  | thr mode | fixed / otsu / valley |
- **Đo tốc độ:** SEG của CountSE vs M1 của ta (giây/ảnh, CPU vs GPU) — điểm cộng khi thuyết trình.
- **Cross-dataset:** O1/O3/O4 trên CARPK, ShanghaiTech.
- 🔗 `05` **S8**, **S10** (bảng ablation, cột “Người = A”). `04` mục **3** (ablation), giả thuyết **H3**.

---

## P7. Bảng, phân tích lỗi, báo cáo, slide (tuần 5–6)

- **Bảng:** phần FSC-147 / CARPK / SHA của `results/tables/main.md` + `ablation_A_*.md`, mọi số có CI và truy vết được tới CSV.
- **Phân tích lỗi phía A:** 15 ảnh FSC-147 cải thiện nhiều nhất + 15 ảnh xấu đi nhiều nhất; chia nhóm theo mật độ (≤20 / 21–100 / >100 vật) và kích thước vật.
- **Báo cáo (phần Minh viết):** Chương 2 (Related work: CountSE, **CountZES**, VA-Count), Chương 4.1 (M1, M3), Chương 5.1 (kết quả FSC-147/CARPK/SHA).
- **Slide (~50%):** bối cảnh bài toán · CountSE (kiến trúc, SEG) · thiết kế + kết quả M1/M3 · kết quả FSC-147/CARPK/SHA. Dùng chế độ “Luồng chạy từng bước” trong `06b_So_do_khoi_keyword.html` để luyện nói.
- **Review chéo:** đọc PR của Bi Anh (M2, verifier) đủ để trả lời câu hỏi hội đồng về phần đó.
- 🔗 `05` **S10** (phân tích lỗi), mục **4** (tiêu chí hoàn thành), mục **5** (REPORT.md). `04` mục **4** (phần thuyết trình/báo cáo), mục **5** (tuần 5–6).

---

## Bảng tổng hợp: phần của Minh nối vào kế hoạch chung ở đâu

| Phần của Minh | `05_PLAN_STEPS` | `04_Ke_hoach` | Tuần | Cần GPU? | Phụ thuộc Bi Anh |
|---|---|---|---|---|---|
| P0 Hợp đồng giao diện | mục 1, S1, S3, S7, mục 3 | mục 4 | 0 | Không | **Có — chốt chung** |
| P1.2 `cluster.py` | S5 | mục 2 | 0 | Không | — |
| P1.3 `cpe.py` (M1) | S5 | mục 2 (M1) | 0 | Không | B đọc đầu ra |
| P1.4 `threshold.py` (M3) | S8 | mục 2 (M3) | 0 | Không | — |
| P1.5 `metrics.py` | S10 (Thống kê) | mục 3 | 0 | Không | B dùng chung |
| P1.6 Adapter FSC-147 | S1 | mục 3 | 0–1 | Không | — |
| P2 Môi trường + `model_io` | S0, S2 | mục 6 | 1 | Có (T4) | **Làm chung S0** |
| P3.1 Test hồi quy | S2 | — | 1–2 | Có | cần `forward_utils` (B) |
| P3.2 B1, B2, B3 | S3 | mục 3 | 2 | Có (Kaggle) | — |
| P4 Cache + CARPK/SHA | S4, S1 | mục 3 | 2–3 | Có | **Làm chung script** |
| P5 M1 thật (O1) + `predict` | S5, S7, S10 | mục 2, H1 | 3 | Lượt 2 | **Ghép `predict` chung** |
| P6 M3 + ablation A + tốc độ | S8, S10 | mục 3, H3 | 4 | Ít | cần M2 để có O3/O4 |
| P7 Bảng, lỗi, báo cáo, slide | S10, mục 4–5 | mục 4–5 | 5–6 | Không | review chéo |

**4 điểm hẹn bắt buộc với Bi Anh:** (1) P0 — chốt giao diện; (2) cuối tuần 1 — chạy lại môi trường của nhau; (3) đầu tuần 3 — ghép `predict.py`; (4) tuần 5 — review chéo + ghép bảng chính.

---

## Việc cần làm NGAY (checklist hôm nay → cuối tuần)

- [ ] Hẹn Bi Anh 1 giờ: chốt `cands`/`pos`, `CountSample`, cột CSV, chia file (P0).
- [ ] Tạo repo GitHub `AutoNegCount` + nhánh `feat/a-core`.
- [ ] Giao **Phụ lục A** cho Antigravity → nhận về `cluster.py`, `cpe.py`, `threshold.py`, `metrics.py`, `datasets/fsc147.py` + test.
- [ ] Tự chạy `pytest -q`, đọc từng file, so với mục P1 ở trên; mang về cho Claude review.
- [ ] Bắt đầu tải ảnh FSC-147 lên Google Drive.
- [ ] Đọc lại `02_CountSE_Giai_thich_code` (phần SEG) — phần Minh sẽ thuyết trình.

---

## Phụ lục A — Prompt giao cho Antigravity (P1, chỉ CPU)

```
Bạn là agent lập trình. Đọc kỹ các file: 05_PLAN_STEPS_cho_agent.md (mục 0, 1, S1, S5, S8, S10)
và 07_Ke_hoach_vai_A_Minh.md (mục P0, P1). Chỉ làm phần P1 (không cần GPU).

Yêu cầu:
1. Tạo khung thư mục AutoNegCount/ theo mục 1 của 05 (file rỗng có docstring là đủ cho phần ngoài P1).
2. Viết autoneg/cluster.py, autoneg/cpe.py, autoneg/threshold.py, autoneg/metrics.py,
   autoneg/datasets/fsc147.py đúng chữ ký và yêu cầu ở P1.2–P1.6 (kể cả criterion="size_only").
3. Viết test tương ứng trong tests/ đúng các ca kiểm tra đã liệt kê; dùng seed 42.
4. Không sửa thư mục CountGDPlusPlus/ và CountSE/ gốc. Đường dẫn đọc từ biến môi trường.
5. Chạy `pytest -q` và dán kết quả. Không bịa số; ca nào không chạy được thì ghi rõ lý do.
6. Tạo REPORT.md theo mục 5 của 05: việc đã làm, lệnh đã chạy, kết quả test, vấn đề, câu hỏi mở.
Công nghệ: Python 3.10, numpy 1.26.4, scipy, scikit-learn, pandas, pyyaml, pytest.
```

## Phụ lục B — Kiến thức cần nắm cho phần của Minh (tra trong `06_Tu_dien_thuat_ngu`)

Cosine similarity (#17) · Similarity graph (#18) · Eigenvalue (#19) · Laplacian (#20) · Eigengap (#21) · Clustering / k-means / spectral (#22–24) · Centroid/Medoid (#25) · Otsu (#26) · RoI Align (#74) · Pseudo-exemplar (#8) · CountSE / Soft exemplar / SEG (#86–88) · M1 CPE (#90) · M3 (#93) · MAE/RMSE/Bias (#108–111) · Bootstrap (#114) · FSC-147 (#117) · CountZES (#104).
