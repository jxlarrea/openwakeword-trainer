# syntax=docker/dockerfile:1.7
#
# OpenWakeWord Trainer
# Multi-arch (linux/amd64 + linux/arm64) image with CUDA 12.8 + cuDNN.
# Blackwell-class GPUs (DGX Spark GB10, RTX 50-series) require CUDA 12.8+.
#
# Build:
#   docker buildx build --platform linux/amd64,linux/arm64 -t oww-trainer .
#
# DGX Spark host (aarch64): docker compose up trainer
# x86_64 host:               docker compose up trainer

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Etc/UTC

# System deps. libsndfile + ffmpeg are required for audiomentations / soundfile / mp3 augmentation.
# espeak-ng is required by Piper for phoneme generation. git+curl needed for HF downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential pkg-config \
        ffmpeg libsndfile1 libsox-fmt-all sox \
        espeak-ng \
        curl ca-certificates git tini \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Use a venv so PEP 668 (externally-managed) doesn't bite us on Ubuntu 24.04
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Install PyTorch from the cu128 index first (so PyPI doesn't pull a generic CPU wheel).
# cu128 publishes both x86_64 and aarch64 wheels - works on DGX Spark.
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

# Project requirements
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Pre-bake openWakeWord feature models (melspec + embedding + silero-vad) so
# first run doesn't depend on the GitHub release endpoint being reachable.
RUN python -c "import openwakeword; openwakeword.utils.download_models()" \
 || echo "WARN: openwakeword model pre-download failed; will retry at runtime"

# App code
COPY src /app/src

# Volumes for persistent state. Mounted by compose; declared here for clarity.
VOLUME ["/data"]

EXPOSE 8000

# tini handles PID 1 / signals so SSE connections close cleanly on Ctrl-C.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
