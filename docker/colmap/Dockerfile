FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap \
    python3 \
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

RUN python3 -m pip install --upgrade pip \
  && pip install awscli

WORKDIR /workspace

ENV QT_QPA_PLATFORM=offscreen

CMD ["bash"]
