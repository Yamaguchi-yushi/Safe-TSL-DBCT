#!/bin/bash
# MARL4DRP / Safe-TSL-DBCT 用 conda 環境セットアップスクリプト
# 使い方: bash install.sh [env_name]
#   env_name を指定しない場合は "drp" になります

set -e

ENV_NAME=${1:-drp}

echo "================================================"
echo "Creating conda environment: ${ENV_NAME} (Python 3.8)"
echo "================================================"

# conda コマンドが利用可能か確認
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# 既存環境のチェック
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Remove and recreate? (y/N): " yn
    case "$yn" in
        [yY]*)
            conda env remove -n "${ENV_NAME}" -y
            ;;
        *)
            echo "Aborting."
            exit 1
            ;;
    esac
fi

# conda 環境を作成 (Python 3.8)
conda create -n "${ENV_NAME}" python=3.8 -y

# conda activate を有効化するため、shell hook を読み込む
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "================================================"
echo "Installing Python packages via pip"
echo "================================================"

# pip 自体を更新
pip install --upgrade pip setuptools wheel

# requirements.txt からまとめてインストール
# gym==0.21.0 は古い setuptools が必要なので setuptools<66 で固定
pip install "setuptools<66" "wheel<0.40"
pip install -r requirements.txt

echo "================================================"
echo "Installing local packages (drp_env / epymarl)"
echo "================================================"

# このリポジトリを editable install
pip install -e .

echo ""
echo "================================================"
echo "Done!"
echo "================================================"
echo "Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "Run cost calculation:"
echo "    python calculate_cost.py"
echo ""
echo "Run training:"
echo "    cd epymarl"
echo "    python src/main.py --config=qmix --env-config=gymma with \\"
echo "        env_args.time_limit=100 \\"
echo "        'env_args.key=drp_env:drp-5agent_map_8x5-v2' \\"
echo "        env_args.state_repre_flag=\"onehot_fov\""
echo ""
