# syntax=docker/dockerfile:1.7
#
# OpenWakeWord Trainer - DGX Spark / aarch64+CUDA build.
#
# Two-stage build:
#   Stage 1 (ort-builder): compile onnxruntime-gpu from source with sm_100
#                          (Blackwell GB10) and sm_120 kernels. This is the
#                          only way to get aarch64+CUDA onnxruntime as no
#                          public wheel exists.
#   Stage 2 (base):        slim runtime image that pip-installs the wheel
#                          built in stage 1, plus our app deps.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG ORT_VERSION=v1.20.1
# GB10 = sm_120. We do NOT build for sm_100 (datacenter B100/B200) because
# (a) we don't run on those and (b) building both doubled CUDA template
# memory pressure and OOM'd the DGX Spark during compile.
ARG CUDA_ARCHITECTURES=120
# Build parallelism. CUDA template compilation peaks at ~5-15 GB per NVCC
# instance during flash_attention compile. 8 workers x ~10 GB avg = ~80 GB peak,
# fits comfortably on DGX Spark's 128 GB with other containers running
# (observed steady state during CXX phase was only 48 GB used with parallel=4).
# --nvcc_threads stays at 1 to keep peak memory bounded per NVCC instance.
ARG BUILD_PARALLEL=8

# ============================================================================
# Stage 1: Build onnxruntime-gpu wheel
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS ort-builder

ARG ORT_VERSION
ARG CUDA_ARCHITECTURES
ARG BUILD_PARALLEL

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential cmake git ninja-build \
        ca-certificates curl \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies. Ubuntu 22.04 ships cmake 3.22, but onnxruntime v1.20+
# requires >= 3.26. Install a current cmake via pip (lands at /usr/local/bin
# which precedes /usr/bin on PATH). psutil is for nvcc thread auto-tuning.
# cmake<4 because some ORT subdeps (google_nsync) use legacy
# cmake_minimum_required syntax that 4.x rejects.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install "cmake>=3.28,<4" numpy packaging psutil \
 && cmake --version

WORKDIR /build

# Pin to a release tag for reproducibility.
RUN git clone --depth 1 --branch ${ORT_VERSION} https://github.com/microsoft/onnxruntime.git

WORKDIR /build/onnxruntime

# Patch the Eigen download URL: gitlab.com is Cloudflare-challenged inside the
# build container, so we redirect FetchContent to the GitHub mirror. We have to
# download the archive once to compute its SHA1 (the GitHub archive ZIP has
# a different layout than GitLab's so the original hash is wrong) and then
# patch deps.txt with the new URL + new hash. ExternalProject requires a
# valid URL_HASH; empty fails parsing.
RUN EIGEN_COMMIT=$(grep '^eigen;' cmake/deps.txt | grep -oE 'archive/[a-f0-9]{40}' | head -1 | cut -d/ -f2) \
 && echo "Eigen commit: $EIGEN_COMMIT" \
 && EIGEN_URL="https://github.com/eigen-mirror/eigen/archive/${EIGEN_COMMIT}.zip" \
 && echo "Downloading $EIGEN_URL to compute SHA1..." \
 && curl -sSfL -o /tmp/eigen.zip "$EIGEN_URL" \
 && EIGEN_SHA1=$(sha1sum /tmp/eigen.zip | awk '{print $1}') \
 && echo "Eigen SHA1: $EIGEN_SHA1" \
 && rm /tmp/eigen.zip \
 && sed -i -E "s|^eigen;.*$|eigen;${EIGEN_URL};${EIGEN_SHA1}|" cmake/deps.txt \
 && grep -i eigen cmake/deps.txt

# Compile. --use_cuda enables the CUDA EP; CMAKE_CUDA_ARCHITECTURES picks
# which sm_* kernels to emit. Build is single-config Release with all cores.
RUN python tools/ci_build/build.py \
        --build_dir /build/ort-build \
        --config Release \
        --parallel ${BUILD_PARALLEL} \
        --nvcc_threads 1 \
        --skip_tests \
        --allow_running_as_root \
        --compile_no_warning_as_error \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cuda \
        --cmake_extra_defines \
            "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
            "onnxruntime_BUILD_UNIT_TESTS=OFF" \
            "FETCHCONTENT_QUIET=OFF" \
        --build_wheel \
        --update --build \
    && ls -la /build/ort-build/Release/dist/

# ============================================================================
# Stage 2: Runtime image
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential pkg-config \
        ffmpeg libsndfile1 libsox-fmt-all sox \
        espeak-ng \
        zip unzip \
        curl ca-certificates git tini \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# PyTorch from cu128 (aarch64 + Blackwell native).
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

# Copy the locally-built onnxruntime-gpu wheel from stage 1 but DO NOT install yet.
COPY --from=ort-builder /build/ort-build/Release/dist/*.whl /tmp/wheels/

# Project requirements first. piper-tts + openwakeword list `onnxruntime` as a
# dep (the CPU package), and pip will install it. We accept that, then
# uninstall it, then install our GPU build LAST so its files are the
# survivors. (If we installed GPU first, pip's CPU-package uninstall would
# remove shared files like __init__.py because both wheels register them.)
COPY requirements.txt /app/requirements.txt
RUN grep -vE '^onnxruntime' /app/requirements.txt > /tmp/req-no-ort.txt \
 && python -m pip install -r /tmp/req-no-ort.txt \
 && python -m pip uninstall -y onnxruntime onnxruntime-gpu \
 && python -m pip install /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && python -m pip list | grep -i onnx \
 && python -c "import onnxruntime as ort; print('providers:', ort.get_available_providers())"

# Pre-bake openWakeWord feature models.
RUN python -c "import openwakeword; openwakeword.utils.download_models()" \
 || echo "WARN: openwakeword model pre-download failed; will retry at runtime"

COPY src /app/src

VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
