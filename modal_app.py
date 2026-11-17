"""Chạy thí nghiệm trên Modal (GPU trả theo giây).  PLAN: S12.

ĐỐI CHIẾU tài liệu Modal hiện hành trước khi dùng. Luôn đặt timeout; ghi cost_log.md (giờ × giá).
Debug: T4/L4. Chạy lớn: A100/L40S. Ngân sách tổng $60 — dừng nếu tiêu > $40 mà chưa có bảng chính.

    modal run modal_app.py --config configs/o3_fsc147_val.yaml
"""
import modal

app = modal.App("autoneg-count")
vol = modal.Volume.from_name("autoneg-data", create_if_missing=True)   # checkpoints + datasets + cache
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_requirements("environment/requirements_countgdpp.txt")
    .add_local_dir(".", "/root/repo", ignore=["data", "checkpoints", "results", ".venv", ".git"])
)


@app.function(image=image, gpu="L4", timeout=3 * 3600, volumes={"/data": vol})
def run(config_path: str):
    import os
    import subprocess
    env = dict(os.environ, DATA_ROOT="/data", CKPT_DIR="/data/checkpoints", RESULTS_DIR="/data/results")
    subprocess.run(["python", "scripts/run_eval.py", "--config", config_path], cwd="/root/repo", env=env, check=True)
    vol.commit()


@app.local_entrypoint()
def main(config: str):
    run.remote(config)
