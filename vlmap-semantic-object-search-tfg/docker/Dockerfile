FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System dependencies (including X11/OpenGL for interactive viewer)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libglm-dev \
    libegl1-mesa-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
    ffmpeg \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge by default, no ToS issues unlike Miniconda)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Create tfg env: habitat-sim 0.3.1 requires Python >=3.9,<3.10
# withbullet: enables physics; no headless flag so OpenGL viewer works via X11
RUN conda create -n tfg habitat-sim=0.3.1 withbullet \
    -c conda-forge -c aihabitat -y \
    && conda clean -afy

# PyTorch with CUDA 12.1
RUN conda run -n tfg pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Python requirements
COPY docker/requirements.txt /tmp/requirements.txt
RUN conda run -n tfg pip install -r /tmp/requirements.txt

# Auto-activate tfg for all bash sessions (interactive and exec)
RUN conda init bash && echo "conda activate tfg" >> /root/.bashrc

# Make the tfg env the default for non-interactive shells
ENV PATH=/opt/conda/envs/tfg/bin:$PATH

WORKDIR /workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
