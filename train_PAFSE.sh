#!/usr/bin/env bash

set -Eeuo pipefail

# 论文实验参数对应的 PAFSE 训练脚本。
# 用法：
#   bash train_PAFSE.sh all
#   bash train_PAFSE.sh ICEWS14
#   bash train_PAFSE.sh ICEWS05-15
#   bash train_PAFSE.sh GDELT
#
# 指定 GPU：
#   CUDA_VISIBLE_DEVICES=1 bash train_PAFSE.sh ICEWS14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"
VALID_FREQ="${VALID_FREQ:-5}"
FREQ_BETA="1e-5"

if [[ -n "${PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${PROJECT_DIR}"
elif [[ -f "${SCRIPT_DIR}/learner.py" ]]; then
    PROJECT_DIR="${SCRIPT_DIR}"
else
    PROJECT_DIR="${SCRIPT_DIR}/PAFSE模型代码"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${PROJECT_DIR}/learner.py" ]]; then
    echo "错误：未找到 ${PROJECT_DIR}/learner.py" >&2
    echo "可通过 PROJECT_DIR=/path/to/PAFSE模型代码 指定项目目录。" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs

run_training() {
    local dataset="$1"
    local rank="$2"
    local batch_size="$3"
    local max_epochs="$4"
    local learning_rate="$5"
    local emb_reg="$6"
    local time_reg="$7"

    if [[ ! -d "data/${dataset}" ]]; then
        echo "错误：缺少预处理后的数据目录 data/${dataset}" >&2
        exit 1
    fi

    echo "============================================================"
    echo "开始训练 PAFSE：${dataset}"
    echo "GPU=${CUDA_VISIBLE_DEVICES}, rank=${rank}, batch=${batch_size}, epochs=${max_epochs}"
    echo "lr=${learning_rate}, emb_reg=${emb_reg}, time_reg=${time_reg}, freq_beta=${FREQ_BETA}"
    echo "============================================================"

    "${PYTHON_BIN}" -u learner.py \
        --model PAFSE \
        --dataset "${dataset}" \
        --rank "${rank}" \
        --batch_size "${batch_size}" \
        --max_epochs "${max_epochs}" \
        --valid_freq "${VALID_FREQ}" \
        --learning_rate "${learning_rate}" \
        --emb_reg "${emb_reg}" \
        --time_reg "${time_reg}" \
        --use_freq_loss \
        --freq_beta "${FREQ_BETA}" \
        2>&1 | tee "logs/PAFSE_${dataset}.log"
}

train_icews14() {
    run_training "ICEWS14" 6000 4000 5000 0.03 0.01 0.01
}

train_icews0515() {
    run_training "ICEWS05-15" 8000 6000 3000 0.03 0.002 0.1
}

train_gdelt() {
    run_training "GDELT" 8000 2000 400 0.2 0.001 0.001
}

case "${TARGET}" in
    all)
        train_icews14
        train_icews0515
        train_gdelt
        ;;
    ICEWS14|icews14)
        train_icews14
        ;;
    ICEWS05-15|icews05-15|ICEWS0515|icews0515)
        train_icews0515
        ;;
    GDELT|gdelt)
        train_gdelt
        ;;
    *)
        echo "用法：bash $(basename "$0") {all|ICEWS14|ICEWS05-15|GDELT}" >&2
        exit 2
        ;;
esac
