FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && python -m pip install --upgrade pip setuptools wheel

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

RUN pip install --no-cache-dir \
    open_clip_torch \
    opencv-python-headless \
    pillow \
    tqdm \
    numpy

ARG SAM3_PIP=""
RUN if [ -n "${SAM3_PIP}" ]; then pip install --no-cache-dir "${SAM3_PIP}"; fi

WORKDIR /workspace

COPY . /workspace

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache

CMD ["bash"]
