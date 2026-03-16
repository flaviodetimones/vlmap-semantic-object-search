FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY docker/requirements.txt /workspace/docker/requirements.txt
RUN pip install -r /workspace/docker/requirements.txt

COPY . /workspace

RUN chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["/bin/bash"]
