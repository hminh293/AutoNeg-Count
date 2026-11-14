#!/usr/bin/env bash
# Cài môi trường CountGD++ trên Colab (GPU). Chạy từ thư mục gốc repo.
set -euo pipefail
git submodule update --init --recursive
pip install -q -r environment/requirements_countgdpp.txt
pip install -q -e .
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Áp patch (bản sao làm việc của third_party, không commit vào submodule)
for p in patches/*.patch; do
  [ -e "$p" ] || continue
  git -C third_party/CountGDPlusPlus apply --check "../../$p" && git -C third_party/CountGDPlusPlus apply "../../$p" || echo "bỏ qua $p (đã áp hoặc không khớp)"
done

# Biên dịch CUDA op (tuỳ chọn; lỗi thì dùng fallback PyTorch trong patch 00)
( cd third_party/CountGDPlusPlus/models/GroundingDINO/ops && python setup.py build install ) || echo "CUDA op build thất bại → dùng fallback PyTorch"

# Trọng số
mkdir -p checkpoints
[ -d checkpoints/bert-base-uncased ] || python - <<'EOF'
from transformers import BertModel, BertTokenizer
BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained("checkpoints/bert-base-uncased")
BertModel.from_pretrained("bert-base-uncased").save_pretrained("checkpoints/bert-base-uncased")
EOF
echo "Tải countgd_plusplus.pth (link Google Drive trong README CountGD++) vào checkpoints/ bằng gdown."
