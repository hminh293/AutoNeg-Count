"""[A – Minh] Nạp CountGD++ (bản groundingdino_app) + transform.  Cần GPU.  PLAN: P2.3 / S2.

Không import module này trong test CPU. Chạy trong MÔI TRƯỜNG CountGD++ (không chung tiến trình với CountSE).
"""
from __future__ import annotations

from pathlib import Path

from .paths import countgd_root, ckpt_dir


def load_countgdpp(ckpt: str | Path | None = None, cgd_root: str | Path | None = None, device: str = "cuda"):
    """Trả (model.eval() trên device, transform). Mô phỏng build_model_and_transforms của test_shanghai_tech.py.

    Khung tham khảo (05_PLAN_STEPS, S2):
        sys.path.insert(0, cgd_root); SLConfig.fromfile(cgd_root/"cfg_app.py");
        cfg.text_encoder_type = <ckpt_dir>/bert-base-uncased;
        groundingdino_app.build_groundingdino(args); load_state_dict(torch.load(ckpt)["model"], strict=False);
        transform = Compose([RandomResize([800], max_size=1333), ToTensor, Normalize(ImageNet)])
    """
    cgd_root = Path(cgd_root or countgd_root())
    ckpt = Path(ckpt or ckpt_dir() / "countgd_plusplus.pth")
    raise NotImplementedError("P2.3 load_countgdpp — Minh")


def preprocess(transform, pil_image):
    """img_t, _ = transform(pil_image, {"exemplars": torch.tensor([])}); trả img_t [3,H,W]."""
    raise NotImplementedError("P2.3 preprocess — Minh")
