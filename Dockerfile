# ============================================================
# MARL4DRP Docker Image
# Base: CUDA 12.8 + cuDNN + Ubuntu 22.04
# Python 3.9, PyTorch 2.7.0+cu128
# ============================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---- System packages ----
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 3.9 をデフォルトに ----
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch 2.7.0 + CUDA 12.8 ----
RUN pip install \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ---- gym 0.21.0 (setup.pyの自己参照extras_requireをパッチして対応) ----
RUN git clone --depth 1 --branch v0.21.0 https://github.com/openai/gym.git /tmp/gym && \
    python3 -c "\
content = open('/tmp/gym/setup.py').read(); \
import re; \
content = re.sub(r'extras_require\[.all.\]\s*=\s*\[.*?\]', \"extras_require['all'] = []\", content, flags=re.DOTALL); \
open('/tmp/gym/setup.py', 'w').write(content)" && \
    pip install /tmp/gym --no-deps && \
    rm -rf /tmp/gym

# ---- SMAC (特定コミット) ----
RUN pip install "SMAC @ git+https://github.com/oxwhirl/smac.git@d6aab33f76abc3849c50463a8592a84f59a5ef84"

# ---- その他の依存パッケージ ----
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ---- プロジェクトファイルをコピー ----
COPY . /workspace/

# ---- drp パッケージをローカルインストール ----
RUN pip install -e .

WORKDIR /workspace

CMD ["bash"]
