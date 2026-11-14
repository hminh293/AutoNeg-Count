# PLAN_STEPS — AutoNeg-Count (dành cho agent Antigravity)

> **Mục tiêu:** hiện thực hoá hướng đồ án "Hạn chế đếm nhầm trong zero-shot counting" trên nền **CountGD++** (có đối chứng **CountSE**). Đọc kèm: `01_CountGDPlusPlus_Giai_thich_code.md`, `02_CountSE_Giai_thich_code.md`, `04_Ke_hoach_do_an_de_doc.md`.
> **Nhãn người phụ trách:** `[A]` Người A · `[B]` Người B · `[AB]` làm chung. Mỗi bước có *Việc cần làm → Tệp → Lệnh → Tiêu chí nghiệm thu (DoD)*.
> **Người dùng sẽ mang kết quả/mã quay lại cho reviewer kiểm tra và tinh chỉnh** → mỗi bước phải để lại **log + file kết quả + commit** (xem mục "Báo cáo bàn giao").

---

## 0. Quy tắc bắt buộc cho agent

1. **Không sửa trực tiếp** hai repo gốc (`Đồ_án/CountGDPlusPlus`, `Đồ_án/CountSE`). Sao chép CountGD++ vào `AutoNegCount/third_party/CountGDPlusPlus/` (bỏ `.git`, `data/`), mọi chỉnh sửa lưu thành patch (`git diff > patches/*.patch`).
2. **Hai repo có tên gói trùng nhau** (`util`, `datasets`, `models`, `groundingdino`) → **không import chung trong 1 tiến trình Python**. Chạy CountSE và CountGD++ ở **hai tiến trình / hai môi trường riêng**, trao đổi qua file (CSV/JSON).
3. **Không bịa số liệu.** Nếu không có GPU/dữ liệu để chạy, ghi rõ "chưa chạy" và dừng ở bước đó; không điền số giả.
4. Seed cố định (`42`; CountSE dùng `215` khi cross-dataset). Mọi thí nghiệm ghi: config YAML, commit hash, thời gian, phiên bản torch/transformers.
5. **Mặc định hành vi gốc không đổi:** các patch vào model phải có cờ tắt; với cờ tắt, đầu ra phải **giống hệt** script gốc (kiểm tra bằng test hồi quy ở S2).
6. Đường dẫn đọc từ biến môi trường / config, **không hard-code** (`/path/to/...`).
7. Commit nhỏ, thông điệp rõ. Mỗi module có test đơn vị trong `tests/`.

---

## 1. Cấu trúc thư mục đích

```
AutoNegCount/
├── README.md                     # cách cài, cách chạy lại từng bảng kết quả
├── environment/                  # requirements_countgdpp.txt, requirements_countse.txt, setup_colab.sh, setup_kaggle.sh
├── third_party/CountGDPlusPlus/  # bản sao + patch (xem S2)
├── patches/                      # *.patch
├── configs/                      # YAML cho từng thí nghiệm (xem mục 4)
├── autoneg/
│   ├── model_io.py               # nạp CountGD++ + transform
│   ├── forward_utils.py          # gọi model: pass-1, pass-2 (pos/neg), lấy feature
│   ├── candidates.py             # trích ứng viên + embedding
│   ├── cluster.py                # eigengap / spectral / kmeans_cos
│   ├── cpe.py                    # M1 Clean Pseudo-Exemplar
│   ├── anm.py                    # M2 Auto-Negative Mining
│   ├── threshold.py              # M3 fixed / otsu / valley
│   ├── verifier.py               # Tier-B
│   ├── predict.py                # pipeline đầy đủ
│   ├── metrics.py                # MAE/RMSE/NAE/SRE/Bias/PRF/bootstrap
│   └── datasets/                 # fsc147.py carpk.py shanghaitech.py pairtally.py omnicount.py bloodcell.py
├── scripts/
│   ├── run_eval.py               # chạy 1 (dataset, method, config) → results/*.csv
│   ├── run_countse_eval.py       # (môi trường CountSE) → results/countse_*.csv
│   ├── cache_pass1.py            # cache đầu ra lượt 1 để ablation nhanh
│   ├── run_ablation.py
│   ├── make_tables.py            # gộp CSV → bảng markdown/LaTeX + CI
│   ├── make_figures.py           # lưới trước/sau
│   └── demo_app.py               # Gradio
├── modal_app.py                  # chạy trên Modal
├── tests/
└── results/                      # csv, json, hình (không commit file nặng)
```

---

## S0. Môi trường và phần cứng  `[AB]`  (tuần 1)

**Việc cần làm**
1. Dev trên **Linux GPU** (Colab/Kaggle/Modal). Máy Windows của người dùng **không có GPU/`nvidia-smi`** → chỉ soạn code, chạy test CPU.
2. Môi trường CountGD++ (khuyến nghị thử theo thứ tự):
   - Python 3.10; `torch==2.4.1+cu121` (hoặc bản `<2.6` có sẵn trên Colab/Kaggle nếu tương thích), `torchvision` tương ứng.
   - **`transformers==4.39.1`, `numpy==1.26.4`** (là phiên bản CountSE đã ghim; `BertModelWarper` của họ GroundingDINO dễ hỏng với transformers mới). Các gói khác theo `CountGDPlusPlus/requirements.txt`. **Không cần `detectron2`** (chỉ nằm trong docstring).
   - `segment_anything` chỉ cần nếu bật TT-Norm SAM (bỏ qua ở bản chính; nhưng `engine_inference.py` import `segment_anything` ở đầu file → chỉ import `get_pseudo_exemplars` bằng cách copy hàm, **không** import cả module).
3. **Biên dịch CUDA ops** (`models/GroundingDINO/ops`: `python setup.py build install`). Nếu thất bại trong >30 phút debug → dùng **fallback PyTorch**: trong `third_party/.../models/GroundingDINO/ms_deform_attn.py` dòng ~331 đổi
   ```python
   if torch.cuda.is_available() and value.is_cuda:
   ```
   thành
   ```python
   if torch.cuda.is_available() and value.is_cuda and '_C' in globals():
   ```
   (khi không có `_C`, rơi về `multi_scale_deformable_attn_pytorch`; chậm hơn nhưng chạy suy luận được). Lưu thành `patches/00_ms_deform_fallback.patch` và ghi nhận tốc độ cả hai đường.
4. Tải trọng số (bằng script, kiểm tra checksum/kích thước):
   - `checkpoints/countgd_plusplus.pth` (1.25 GB, link Drive trong README CountGD++; dùng `gdown`)
   - `python download_bert.py` → `checkpoints/bert-base-uncased`
   - CountSE: `pretrained_ckpts/checkpoint_best_regular.pth` (Drive trong README CountSE) + `groundingdino_swinb_cogcoor.pth` nếu cần train
5. Ghi `environment/*.txt` đã **pip freeze** sau khi chạy được.

**DoD:** (a) `python -c "import torch;print(torch.cuda.is_available())"` True; (b) `python app.py` của CountGD++ chạy demo trên 1 ảnh mẫu `img/336.jpg` với "strawberry" và trả số đếm; (c) CountSE `tools/inference_on_a_image.py` (hoặc `main_inference.py` với `--debug`) chạy 1 ảnh; (d) file `results/env_report.md` ghi: GPU, VRAM, thời gian nạp model, giây/ảnh, VRAM đỉnh (đo bằng `torch.cuda.max_memory_allocated`).

---

## S1. Dữ liệu & adapter  `[A]` FSC-147/CARPK/ShanghaiTech · `[B]` PairTally/OmniCount/BloodCell  (tuần 1–2)

Giao diện thống nhất mà mọi adapter phải cài:
```python
class CountSample(TypedDict):
    image_path: str
    image_id: str
    text: str                    # tên lớp cần đếm (positive)
    gt_count: int
    gt_points: Optional[np.ndarray]   # [N,2] (x,y) pixel, nếu có
    gt_boxes: Optional[np.ndarray]    # [N,4] xywh pixel, nếu có
    distractor_texts: List[str]       # lớp gây nhầm có GT trong cùng ảnh (PairTally/OmniCount); [] nếu không
    meta: dict
class CountDataset(Protocol):
    def __iter__(self) -> Iterator[CountSample]: ...
    def __len__(self) -> int: ...
```
- **FSC-147 `[A]`:** GT lấy từ `CountSE/data/fsc147_gdino/fsc147_coco/coco_{val,test}_exemplars.json` (đã có trong thư mục). **Mỗi ảnh có 3 hộp exemplar nối thêm ở cuối danh sách annotation** (`coco.py:376` lấy `boxes[-3:]`; README CountSE: `annotation_mode=fsc147` loại 3 hộp cuối) ⇒ `gt_count = len(annotations) - 3`. Tên lớp: theo `category_id` → `label_list`/`val_label_list` trong `config/cfg_fsc147_*.py`. Ảnh: `FSC147_384_V2/images_384_VarV2` (tải từ repo LearningToCountEverything). Ngoại lệ: `image_id==6003` (exemplar lỗi) → để cờ `remove_bad_exemplar` (chỉ liên quan khi dùng exemplar thật; zero-shot không ảnh hưởng).
  - **Đã kiểm tra trực tiếp trên file:** `coco_val_exemplars.json` = 1286 ảnh / 85 900 annotation / 30 lớp; `coco_test_exemplars.json` = 1190 ảnh / 82 406 annotation. Mỗi điểm GT là hộp **2×2 (`area==4`)** `[x,y,2,2]`; **3 annotation cuối của mỗi ảnh có `area>4`** (hộp exemplar thật). Điểm GT = `(x+1, y+1)` (tâm hộp 2×2). Có thể phân biệt thêm bằng `area==4`.
  - **Test xác minh:** với 20 ảnh ngẫu nhiên, `gt_count` khớp `len(points)` trong `annotation_FSC147_384.json` chính thức.
- **CARPK `[A]`:** dùng `CountSE/cross_dataset/prepare_cross_dataset.py carpk` (đổi sang COCO) hoặc tự đọc `Annotations/*.txt`; prompt `"car"`. **ShanghaiTech A/B `[A]`:** `.mat` (`image_info[0,0]['number']`) như `test_shanghai_tech.py`; prompt `"human"`/`"people"`.
- **PairTally `[B]`:** **đọc trước** repo `github.com/bbvisual/PairTally_Benchmark` và PR số 3 của CountGD++ trong repo đó (README CountGD++ nhắc) để lấy đúng giao thức (cách chọn prompt, GT theo từng lớp trong ảnh). Mỗi ảnh có **2 lớp** → sinh **2 CountSample** (mỗi lớp 1 mẫu), `distractor_texts=[lớp còn lại]`. Ghi lại định dạng thực tế vào `datasets/pairtally.py` (docstring).
- **OmniCount (fruits) & Blood cell `[B]`:** annotation COCO đã có trong `CountGDPlusPlus/data/*/_annotations.coco.json` (+ `external_exemplars.json`); ảnh phải tải theo README. Mỗi (ảnh, lớp) = 1 mẫu; `distractor_texts` = các lớp còn lại có trong ảnh.

**DoD (mỗi adapter):** `python -m autoneg.datasets.<ten> --check` in số mẫu, 3 mẫu ngẫu nhiên, và phân bố `gt_count`; test `tests/test_datasets.py` pass với dữ liệu giả nhỏ (không cần tải dataset).

---

## S2. Nạp model + test hồi quy + hook  `[A]` làm `model_io`/test · `[B]` làm hook forward  (tuần 1–2)

**`autoneg/model_io.py` `[A]`** — mô phỏng `build_model_and_transforms` trong `test_shanghai_tech.py`:
```python
def load_countgdpp(ckpt, cgd_root, device="cuda"):
    import sys, argparse, torch
    sys.path.insert(0, cgd_root)
    from util.slconfig import SLConfig
    import datasets.transforms_app as T
    cfg = SLConfig.fromfile(f"{cgd_root}/cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": f"{cgd_root}/checkpoints/bert-base-uncased"})
    args = argparse.Namespace(device=device)
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    from models.GroundingDINO import groundingdino_app
    model, _, _ = groundingdino_app.build_groundingdino(args)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
    return model.eval().to(device), transform
```
Tiền xử lý một ảnh: `img_t, _ = transform(PIL_image, {"exemplars": torch.tensor([])})`.

**Hook trong `groundingdino_app.py::GroundingDINO.forward` `[B]`** (patch `patches/10_forward_hooks.patch`, mặc định tắt):
1. `kw.get("return_feats", False)`: nếu True, thêm vào `out`: `out["combined_pos_features"] = combined_pos_features` (tensor `[1,256,h,w]`, stride 8).
2. `kw.get("pos_tokens")`: `Tensor[B,k,256]` — nếu có, **bỏ qua** `roi_align` của exemplar dương và dùng thẳng làm `pos_exemplar_tokens` (cho biến thể *soft-token*).
3. `kw.get("neg_tokens")`: `List[Tensor[1,k,256]]` tương tự cho từng lớp âm.
Không đổi hành vi khi không truyền các khoá này.

**Hàm tiện ích `autoneg/forward_utils.py` `[B]`**
```python
@torch.no_grad()
def forward_once(model, img_t, text, pos_boxes=None, neg_classes=None, device="cuda", return_feats=False):
    """
    img_t: [3,H,W] đã transform. text: str (có thể ""). pos_boxes: Tensor[k,4] xyxy pixel (theo ảnh đã resize) hoặc None.
    neg_classes: list of dict(text:str, boxes:Tensor[m,4] xyxy pixel)  (box cùng ảnh)
    Trả về out (dict của model), caption (str).
    """
    from util.misc import nested_tensor_from_tensor_list as nt
    x = nt(img_t.unsqueeze(0).to(device))
    pos = [pos_boxes.to(device) if pos_boxes is not None and len(pos_boxes) else torch.zeros((0,4),device=device)]
    neg_imgs, neg_boxes, caption = [], [], f"{text} . "
    for c in (neg_classes or []):
        caption += f"{c['text']} . "
        neg_imgs.append(x); neg_boxes.append([c["boxes"].to(device)])
    out = model(x, x, pos, neg_imgs, neg_boxes, captions=[caption], return_feats=return_feats)
    return out, caption
```
Lưu ý quan trọng đã xác minh trong code: exemplar âm thứ *i* được chèn vào **cụm thứ i+1** của caption → **phải có một cụm text cho mỗi lớp âm**. Với `text=""` caption vẫn có cụm rỗng (`" . "`), và logic đếm cụm (`label_count` tăng khi token trước là token đặc biệt) vẫn tìm được vị trí — **cần viết test để xác nhận** (xem dưới). Nếu lỗi, dùng placeholder `"object"`.

**Test hồi quy `tests/test_regression.py` `[A]`:** với 20 ảnh ShanghaiTech (hoặc ảnh trong `img/`), pipeline zero-shot 2 lượt của ta (cờ M1/M2/M3 tắt) phải cho **số đếm giống hệt** `test_shanghai_tech.py` (cùng `confidence_thresh=0.23`, `get_pseudo_exemplars`, `num_exemplars=3`). Nếu khác → debug trước khi làm tiếp.

**Test hook `[B]`:** (a) cờ tắt ⇒ `pred_logits` bằng nhau (`torch.allclose`) trước/sau patch; (b) `return_feats=True` ⇒ có `combined_pos_features` kích thước `[1,256,ceil(H/8),ceil(W/8)]`; (c) caption có cụm âm rỗng không báo lỗi.

**DoD:** các test trên pass trên GPU (đánh dấu `@pytest.mark.gpu`), kèm log.

---

## S3. Baseline  (tuần 2)

| ID | Việc | Người |
|---|---|---|
| B1 | CountGD++ **text-only 1 lượt** (đếm = số truy vấn có max-logit > 0.23 trên cột text) | A |
| B2 | CountGD++ text-only **+ pseudo-exemplar top-3** (đúng `test_shanghai_tech.py`) | A |
| B3 | **CountSE text-only** (checkpoint của tác giả; `main_inference.py --eval --annotation_mode fsc147 ...`, **bỏ `--crop`** để so sánh công bằng, ghi chú riêng bản có crop) | A |
| B4 | CountGD++ có **negative thật** (oracle UB) trên Blood/OmniCount/PairTally bằng exemplar từ GT (đúng `test_dataset.py`) | B |
| B5 | Chạy lại demo + `evaluate_coco_metrics.py` cho Blood cell/OmniCount, đối chiếu với README | B |

Đầu ra mỗi lần chạy: `results/<dataset>/<method>/per_image.csv` với cột
`image_id,text,gt_count,pred_count,abs_err,signed_err,n_cand,time_s,extra_json`.
`scripts/run_countse_eval.py` chạy trong **môi trường CountSE**, ghi cùng định dạng CSV (lấy `pred_cnt` từ `get_count_errs`; bỏ `np.save("logits.npy")`).

**DoD:** `make_tables.py` in bảng MAE/RMSE/Bias cho B1–B3 trên FSC-147 val/test; ghi chênh lệch so với số liệu trong paper và **nguyên nhân giả định** (ngưỡng, crop/TT-Norm tắt, exemplar giả…). Không cần khớp tuyệt đối.

---

## S4. Cache lượt 1 (tăng tốc ablation)  `[AB]`  (tuần 2–3)

`scripts/cache_pass1.py`: với mỗi mẫu chạy `forward_once(text only, return_feats=True)` rồi lưu `.npz`:
- `boxes` `[900,4]` (cxcywh chuẩn hoá), `probs_text` `[900,Lt]` (sigmoid logits của các cột text, `Lt` = chỉ số '.' đầu tiên), `image_hw` (sau resize),
- `topN_idx` (N=50, ngưỡng điểm ≥0.10), `emb` `[N,256]` = `roi_align` embedding của các ứng viên này (dùng `combined_pos_features`, hộp xyxy pixel, `spatial_scale=1/8`, `aligned=True`).

Nhờ đó M1/M2/M3 chỉ cần CPU để thay đổi tham số; **chỉ lượt 2 mới cần GPU**.

**DoD:** cache cho FSC-147 val+test, CARPK, ShanghaiTech, PairTally, OmniCount, Blood; kích thước < 2 GB tổng; test `tests/test_cache.py` (nạp lại cho kết quả giống chạy trực tiếp ≤1e-4).

---

## S5. M1 — Clean Pseudo-Exemplar (CPE)  `[A]`  (tuần 3)

**`autoneg/cluster.py`**
```python
import numpy as np
from scipy.sparse import csgraph
def cos_sim(E):                       # E [N,d] đã L2-normalize
    S = E @ E.T; S[S < 0] = 0; return S
def eigengap_k(S, kmax=4, gap_thr=0.1):
    L = csgraph.laplacian(S, normed=True)
    w = np.linalg.eigvalsh(L)         # đã sắp tăng dần (khác CountSE dùng eig không sắp xếp)
    gaps = np.diff(w[:kmax + 1])
    i = int(np.argmax(gaps));  return (i + 1) if gaps[i] > gap_thr else 1
def cluster(E, method="spectral", k=None, **kw):
    E = E / np.linalg.norm(E, axis=1, keepdims=True); S = cos_sim(E)
    if method == "none": return np.zeros(len(E), int)
    k = k or eigengap_k(S)
    if k == 1: return np.zeros(len(E), int)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0).fit_predict(S)
    if method == "kmeans_cos":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E)
    raise ValueError(method)
```
**`autoneg/cpe.py`**
```python
def select_positive(cands, cfg):
    """cands: dict(boxes_xyxy[N,4], scores[N], emb[N,256]) đã sắp giảm theo score, đã lọc score>tau_low, N<=cfg.N.
    Trả về dict(pos_idx: list[int] (≤cfg.m), labels: np.ndarray, best_cluster: int, quality: dict)."""
    N = len(cands["scores"])
    if N <= cfg.m or cfg.cluster == "none":                       # baseline B2 = top-m theo điểm
        return dict(pos_idx=list(range(min(N, cfg.m))), labels=np.zeros(N, int), best_cluster=0)
    labels = cluster(cands["emb"], cfg.cluster)
    q = {c: cands["scores"][labels == c].mean() * (labels == c).sum() ** cfg.alpha for c in np.unique(labels)}
    best = max(q, key=q.get)
    idx = [i for i in range(N) if labels[i] == best][: cfg.m]     # top-m theo điểm trong cụm tốt nhất
    return dict(pos_idx=idx, labels=labels, best_cluster=best, quality=q)
```
Tham số mặc định: `N=30, tau_low=0.15, m=3, alpha=0.5, cluster="spectral"`. Biến thể: (i) `medoid`: chọn phần tử có cos trung bình lớn nhất tới cụm; (ii) **soft-token**: `pos_tokens = mean(emb[cụm tốt nhất])` (1 token, hoặc `k` token = k-means con) truyền qua hook `pos_tokens`; (iii) tiêu chí cụm `alpha∈{0,0.5,1}`.

**Test `tests/test_cpe.py`:** dữ liệu tổng hợp 2 cụm (cụm đúng 8 điểm, điểm 0.5; cụm nhiễu 4 điểm, điểm 0.8 — điểm cao hơn nhưng ít hơn) ⇒ với `alpha=1` chọn cụm đúng (8×0.5=4.0 > 4×0.8=3.2); với `alpha=0` chọn cụm nhiễu (đúng như baseline top-điểm). Test eigengap trên ma trận ví dụ ở Tài liệu 3 (kết quả `k=2`).

**DoD:** chạy O1 trên 100 ảnh FSC-147 val; xuất lưới hình (hộp ứng viên theo màu cụm, exemplar được chọn viền vàng) vào `results/figs/cpe_*.png`.

---

## S6. M2 — Auto-Negative Mining (ANM)  `[B]`  (tuần 3)

**`autoneg/anm.py`**
```python
def mine_negatives(cands, pos, cfg, confusers=None):
    """Trả về list dict(text:str, idx:list[int]) — mỗi phần tử là một lớp âm (một cụm). Tối đa cfg.K_neg."""
    labels, best = pos["labels"], pos["best_cluster"]
    E = cands["emb"] / np.linalg.norm(cands["emb"], axis=1, keepdims=True)
    proto_pos = E[labels == best].mean(0); proto_pos /= np.linalg.norm(proto_pos)
    negs = []
    for c in np.unique(labels):
        if c == best: continue
        m = labels == c
        if m.sum() < cfg.min_cluster:                    # mặc định 2: tránh nhiễu đơn lẻ
            continue
        s = cands["scores"][m].mean()
        d = 1.0 - float(E[m].mean(0) @ proto_pos / (np.linalg.norm(E[m].mean(0)) + 1e-8))
        if s >= cfg.neg_score_min and d >= cfg.delta:   # đủ "gây nhầm" (điểm không quá thấp) và đủ "khác" (xa prototype dương)
            negs.append(dict(score=s, dist=d, idx=list(np.where(m)[0][: cfg.m_neg])))
    negs = sorted(negs, key=lambda z: -z["score"])[: cfg.K_neg]
    for n in negs:
        n["text"] = pick_negative_text(n, cfg, confusers)  # "" | "object" | từ điển confuser | ...
    return negs
```
Mặc định: `K_neg=2, m_neg=2, min_cluster=2, neg_score_min=0.20, delta=0.30`.

**Negative text** (ablation): `empty` (`""`), `placeholder` (`"object"`), `confuser_llm` (từ điển `configs/confusers.json`: `{ "<lớp>": ["tên vật dễ nhầm 1", ...] }`, **sinh một lần ngoài tập test**, ví dụ nhờ LLM cho từng tên lớp FSC-147/PairTally/OmniCount — *không gọi LLM lúc chạy*), `oracle_text` (tên lớp âm thật từ `distractor_texts`, chỉ dùng cho cận trên UB).

**Lọc cuối:** sao chép nguyên logic 2 tầng từ `app.py::get_boxes_from_prediction` (tách tại `.` đầu tiên của `out["input_ids"]`; tầng 1 `pos.max>τ`; tầng 2 `pos.max>neg.max`) vào `predict.py::filter_two_stage(out, tau)`; **không** viết lại theo ý riêng.

**Test `tests/test_anm.py`:** dữ liệu tổng hợp có cụm nhiễu (xa prototype, điểm 0.4, 4 phần tử) ⇒ `mine_negatives` trả đúng 1 lớp âm gồm các chỉ số cụm đó; cụm 1 phần tử bị bỏ; cụm gần prototype (d<delta) bị bỏ.

**DoD:** chạy O2/O3 trên 100 ảnh PairTally; lưới hình (đỏ=bị loại, xanh=giữ, vàng=exemplar dương, tím=negative); báo cáo đếm trước/sau, và **recall** (số hộp thật bị loại nhầm).

---

## S7. `predict.py` — pipeline đầy đủ  `[AB]`  (tuần 3)

```python
def predict(model, transform, sample, cfg, cache=None):
    img_t = preprocess(sample)                        # transform
    c1 = cache.get(sample) if cache else run_pass1(model, img_t, sample["text"])   # dict boxes,probs_text,emb,...
    cands = build_candidates(c1, cfg)                 # lọc tau_low, top-N, đổi cxcywh→xyxy pixel (theo kích thước ảnh sau resize)
    pos = select_positive(cands, cfg.cpe)             # M1 hoặc baseline top-m
    pos_boxes = cands["boxes_xyxy"][pos["pos_idx"]]
    negs = mine_negatives(cands, pos, cfg.anm) if cfg.anm.enabled else []
    neg_classes = [dict(text=n["text"], boxes=cands["boxes_xyxy"][n["idx"]]) for n in negs]
    out2, cap = forward_once(model, img_t, sample["text"], pos_boxes, neg_classes)
    tau = pick_threshold(c1, cfg.thr)                 # fixed 0.23 | otsu | valley (M3)
    boxes, scores = filter_two_stage(out2, tau)       # nếu không có neg → chỉ tầng 1
    return dict(count=len(boxes), boxes=boxes, scores=scores, pos=pos_boxes, negs=neg_classes, tau=tau, debug=...)
```
Chi tiết bắt buộc: toạ độ hộp ứng viên là **chuẩn hoá cxcywh theo ảnh sau resize** → đổi sang xyxy pixel với `(H,W)=img_t.shape[-2:]` (giống `get_pseudo_exemplars`, có `clamp`). Nếu `len(pos_boxes)==0` → quay về đếm lượt 1 (baseline B1).

---

## S8. M3 — Ngưỡng thích nghi  `[A]`  (tuần 4)

`threshold.py`: `fixed(0.23)`; `otsu(scores)` — Otsu 1-D trên điểm text-max của các truy vấn có điểm >0.05 (histogram 64 bin), rồi `clip(τ, 0.15, 0.5)`; `valley`: điểm "gãy" lớn nhất của dãy điểm sắp giảm (so với ngưỡng cứng). So sánh ba cách trên cùng cấu hình O3.

**Test:** dữ liệu tổng hợp hai mode (0.1±0.03 và 0.8±0.05) ⇒ `otsu` ∈ (0.2,0.7).

---

## S9. Tier-B — Candidate Verifier  `[B]`  (tuần 5, chỉ khi M1+M2 xong và còn thời gian)

- **Mục tiêu:** bộ phân loại nhẹ quyết định ứng viên "thật/nhầm" thay cho ngưỡng cứng, **huấn luyện trên FSC-147 val** (CountGD++ đã train trên FSC-147 train → điểm trên train quá tự tin, không dùng).
- **Đặc trưng/ứng viên:** `pos_max, neg_max, pos−neg, hạng điểm, cos tới prototype dương, kích thước cụm/N, w, h, area/median_area, IoU lớn nhất với ứng viên điểm cao hơn, n_cand, entropy điểm ảnh`.
- **Nhãn:** 1 nếu có điểm GT trong hộp và ghép 1-1 được (Hungarian theo khoảng cách tâm, ngưỡng `0.5*sqrt(w*h)`), ngược lại 0.
- **Mô hình:** `sklearn.ensemble.HistGradientBoostingClassifier` (+ so với `LogisticRegression`). **Cross-validation theo lớp** (GroupKFold, nhóm = tên lớp) để tránh rò rỉ; chọn ngưỡng xác suất trên fold; báo cáo trên **FSC-147 test** + PairTally (không huấn luyện trên chúng).
- Lưu `results/verifier.joblib`; thêm vào `predict` nhánh `cfg.verifier.enabled`.

**DoD:** bảng so O3 vs O5 (MAE, Bias, Recall); biểu đồ độ quan trọng đặc trưng.

---

## S10. Thí nghiệm chính & ablation  `[AB]`  (tuần 4–5)

Mỗi thí nghiệm = 1 YAML. Ví dụ `configs/o3_fsc147_test.yaml`:
```yaml
dataset: fsc147_test
method: o3
seed: 42
pass1: {tau_low: 0.15, N: 30}
cpe:  {enabled: true, cluster: spectral, m: 3, alpha: 0.5}
anm:  {enabled: true, K_neg: 2, m_neg: 2, min_cluster: 2, neg_score_min: 0.20, delta: 0.30, neg_text: confuser_llm}
thr:  {mode: fixed, tau: 0.23}
verifier: {enabled: false}
```
**Ma trận chính (mỗi hàng chạy trên mọi bộ dữ liệu):** B1, B2, B3, O1, O2, O3, O4, O5, UB (UB chỉ trên tập có `distractor_texts`).

**Ablation (một biến mỗi lần, quanh O3):**
| Biến | Giá trị | Người |
|---|---|---|
| N / tau_low | {10,20,30} / {0.10,0.15,0.20} | A |
| cluster | none / spectral / kmeans_cos | A |
| alpha | {0,0.5,1} | A |
| m (số exemplar dương) | {1,3,5} | A |
| thr mode | fixed/otsu/valley | A |
| K_neg | {0,1,2,3} | B |
| delta | {0.2,0.3,0.4} | B |
| neg_text | empty/placeholder/confuser_llm/oracle | B |
| verifier | off/on | B |

**Phân tích lỗi bắt buộc:** (1) 15 ảnh cải thiện nhiều nhất + 15 ảnh xấu đi nhiều nhất (hình trước/sau, kèm nguyên nhân); (2) tỉ lệ ca "negative gạt nhầm vật thật" (Recall giảm); (3) theo nhóm: mật độ (≤20/21–100/>100 vật), kích thước vật, loại cặp PairTally (inter vs intra-category).

**Thống kê:** `metrics.paired_bootstrap(pred_a, pred_b, gt, B=10000)` → CI 95% của hiệu MAE; ghi "có ý nghĩa" chỉ khi CI không chứa 0.

**DoD:** `make_tables.py` sinh `results/tables/main.md`, `ablation_*.md` (kèm CI); `make_figures.py` sinh hình; mọi số trong bảng truy vết được tới CSV.

---

## S11. Demo  `[B]`  (tuần 5)

`scripts/demo_app.py` (Gradio, tránh phụ thuộc `gradio_image_prompter`): nhập ảnh + tên vật + công tắc (CPE, ANM, ngưỡng thích nghi). Hiển thị: ảnh với hộp (xanh = đếm, đỏ = bị loại), exemplar dương (vàng), negative tự động (tím), số đếm baseline vs đề xuất, và bảng điểm. Chạy được trên Colab (`share=True`).

---

## S12. Chạy trên Modal / Kaggle  `[AB]`

**Phân bổ:** debug trên Colab T4; **cache lượt 1 + eval lớn trên Kaggle** (miễn phí, nhớ bật Internet, lưu kết quả vào `/kaggle/working`); **Modal** cho các lần chạy dài/ổn định.

**Khung `modal_app.py`** (agent phải đối chiếu với tài liệu Modal hiện hành trước khi dùng):
```python
import modal
app = modal.App("autoneg-count")
vol = modal.Volume.from_name("autoneg-data", create_if_missing=True)   # checkpoints + datasets + cache
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("transformers==4.39.1", "numpy==1.26.4", "scipy", "scikit-learn", "timm", "opencv-python-headless",
                 "pycocotools", "addict", "yapf==0.40.1", "termcolor", "colorlog", "pandas", "pyyaml", "tqdm", "gdown")
    .add_local_dir("AutoNegCount", "/root/AutoNegCount")
)
@app.function(image=image, gpu="A100-80GB", timeout=3 * 3600, volumes={"/data": vol})
def run(config_path: str):
    import subprocess, os
    os.environ["COUNTGD_ROOT"] = "/root/AutoNegCount/third_party/CountGDPlusPlus"
    subprocess.run(["python", "scripts/run_eval.py", "--config", config_path, "--data_root", "/data"],
                   cwd="/root/AutoNegCount", check=True)
    vol.commit()
@app.local_entrypoint()
def main(config: str): run.remote(config)
```
**Giá (từ ảnh người dùng gửi):** A100-80GB $2.50/h · A100-40GB $2.10 · L40S $1.95 · L4 $0.80 · T4 $0.59. Dùng **T4/L4 để debug**, A100/L40S để chạy lớn; luôn đặt `timeout`; ghi `cost_log.md` (giờ chạy × giá) mỗi lần chạy. **Dùng fallback PyTorch cho deformable attention** để khỏi biên dịch trong image (hoặc build trong `image.run_commands` nếu cần tốc độ; `TORCH_CUDA_ARCH_LIST` phù hợp GPU).

**Ngân sách mục tiêu ($60):** baseline ≤ $10 · thí nghiệm chính ≤ $20 · ablation ≤ $10 · dự phòng ≥ $15 · còn lại đệm. Dừng và báo người dùng nếu đã tiêu >$40 mà chưa có bảng chính.

---

## 3. Phân công theo người (tóm tắt, chia đều)

| | **Người A** | **Người B** |
|---|---|---|
| Hạ tầng | S0 (cả hai), `model_io`, test hồi quy | S0 (cả hai), hook forward, `forward_utils` |
| Dữ liệu | FSC-147, CARPK, ShanghaiTech | PairTally, OmniCount, Blood cell |
| Baseline | B1, B2, B3 (CountSE) | B4 (UB), B5 (demo gốc) + B1/B2 trên PairTally |
| Module chính | **M1 CPE** (S5) | **M2 ANM** (S6) |
| Module phụ | **M3** ngưỡng (S8) + đo tốc độ SEG vs ta | **Verifier** (S9) + **Demo** (S11) |
| Ablation | N, tau_low, cluster, alpha, m, thr | K_neg, delta, neg_text, verifier |
| Báo cáo/Slide | CountSE + M1/M3 + FSC-147/CARPK/SHA | CountGD++ + M2/Verifier + PairTally/OmniCount/Blood + Demo |

---

## 4. Kiểm thử tổng & tiêu chí hoàn thành

- `pytest -q` (không GPU) pass; `pytest -m gpu` pass trên Colab/Kaggle.
- `scripts/run_eval.py --config configs/o3_fsc147_test.yaml` chạy lại đúng số liệu trong bảng (sai khác ≤ 1e-6 vì seed cố định, cùng phiên bản thư viện).
- Bảng chính có CI; mọi cải thiện được nêu kèm CI; kết quả âm được báo cáo.
- README tái lập từng bảng bằng một lệnh.

---

## 5. Báo cáo bàn giao (để người dùng đưa lại cho reviewer)

Sau mỗi bước, tạo/ cập nhật `REPORT.md` gồm: (1) việc đã làm, (2) lệnh đã chạy + commit hash, (3) bảng số liệu (kèm đường dẫn CSV), (4) vấn đề gặp & cách xử lý, (5) việc chưa làm/không kiểm chứng được, (6) câu hỏi mở. Đính kèm: `git diff` các patch, `environment/*.txt`, 10 hình trước/sau, `results/env_report.md`, `cost_log.md`.

---

## 6. Rủi ro đã biết (agent cần chú ý)

1. **Biên dịch CUDA ops** → fallback ở S0.3.
2. **`transformers` quá mới làm hỏng `bertwarper.py`** → ghim 4.39.1.
3. **Trùng tên gói giữa 2 repo** → tiến trình riêng (quy tắc 0.2).
4. **`engine_inference.py` import `segment_anything` ngay đầu file** → đừng import module đó; copy `get_pseudo_exemplars`.
5. **Caption âm cần có cụm riêng**; kiểm thử cụm rỗng (S2).
6. **Thiếu `test_fscd147.py`, JSON `data/fscd147/*` rỗng** → dùng harness riêng + GT từ CountSE COCO JSON.
7. **CountGD++ đã train trên FSC-147 train** → không dùng train làm dữ liệu huấn luyện cho Verifier.
8. **ANM có thể làm giảm recall** → luôn báo Recall cạnh MAE/Bias.
