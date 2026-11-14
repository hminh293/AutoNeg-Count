#!/usr/bin/env bash
# Kaggle: bật Internet + GPU trong Settings. Lưu kết quả vào /kaggle/working.
set -euo pipefail
export RESULTS_DIR=${RESULTS_DIR:-/kaggle/working/results}
bash environment/setup_colab.sh
