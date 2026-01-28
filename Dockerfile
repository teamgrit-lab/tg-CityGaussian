FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements/ requirements/
COPY requirements.txt requirements.txt

RUN pip install -r requirements/pyt201_cu118.txt \
  && PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt \
  && PIP_NO_BUILD_ISOLATION=1 pip install -r requirements/CityGS.txt \
  && pip install awscli

COPY . /workspace

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache

CMD ["bash"]
