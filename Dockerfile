# syntax=docker/dockerfile:1.7
#
# OpenWakeWord Trainer - CUDA build.
#
# Two-stage build:
#   Stage 1 (ort-builder): produce an onnxruntime-gpu wheel. On linux/amd64
#                          this downloads the public PyPI wheel. On linux/arm64
#                          it compiles from source because no public aarch64
#                          CUDA wheel exists.
#   Stage 2 (base):        slim runtime image that pip-installs the wheel
#                          built in stage 1, plus our app deps.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG ORT_VERSION=v1.20.1
ARG ORT_PIP_VERSION=1.20.1
# auto: use public onnxruntime-gpu wheel on amd64, source-build on arm64.
# 1/true/yes: force source-build, useful if you need a custom CUDA arch set.
ARG ORT_BUILD_FROM_SOURCE=auto
# Used only when ORT is built from source. GB10/RTX 50-series Blackwell use
# sm_120. RTX 4090 Ada uses sm_89. Multiple values are allowed, e.g. "89;120",
# but compile time and memory use increase significantly.
ARG CUDA_ARCHITECTURES=120
# Build parallelism. CUDA template compilation peaks at ~5-15 GB per NVCC
# instance during flash_attention compile. 8 workers x ~10 GB avg = ~80 GB peak,
# fits comfortably on DGX Spark's 128 GB with other containers running
# (observed steady state during CXX phase was only 48 GB used with parallel=4).
# --nvcc_threads stays at 1 to keep peak memory bounded per NVCC instance.
ARG BUILD_PARALLEL=8

# ============================================================================
# Stage 1: Produce onnxruntime-gpu wheel
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS ort-builder

ARG ORT_VERSION
ARG ORT_PIP_VERSION
ARG ORT_BUILD_FROM_SOURCE
ARG CUDA_ARCHITECTURES
ARG BUILD_PARALLEL
ARG TARGETARCH

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

RUN set -eux; \
    mkdir -p /build/ort-wheel; \
    build_from_source="${ORT_BUILD_FROM_SOURCE}"; \
    if [ "${build_from_source}" = "auto" ]; then \
        if [ "${TARGETARCH}" = "amd64" ]; then build_from_source="0"; else build_from_source="1"; fi; \
    fi; \
    case "${build_from_source}" in \
        0|false|False|no|No) \
            echo "Using public onnxruntime-gpu==${ORT_PIP_VERSION} wheel for TARGETARCH=${TARGETARCH}"; \
            python -m pip download \
                --only-binary=:all: \
                --no-deps \
                --dest /build/ort-wheel \
                "onnxruntime-gpu==${ORT_PIP_VERSION}"; \
            ;; \
        1|true|True|yes|Yes) \
            echo "Building onnxruntime-gpu ${ORT_VERSION} from source for TARGETARCH=${TARGETARCH}, CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"; \
            git clone --depth 1 --branch "${ORT_VERSION}" https://github.com/microsoft/onnxruntime.git /build/onnxruntime; \
            cd /build/onnxruntime; \
            EIGEN_COMMIT=$(grep '^eigen;' cmake/deps.txt | grep -oE 'archive/[a-f0-9]{40}' | head -1 | cut -d/ -f2); \
            echo "Eigen commit: ${EIGEN_COMMIT}"; \
            EIGEN_URL="https://github.com/eigen-mirror/eigen/archive/${EIGEN_COMMIT}.zip"; \
            curl -sSfL -o /tmp/eigen.zip "${EIGEN_URL}"; \
            EIGEN_SHA1=$(sha1sum /tmp/eigen.zip | awk '{print $1}'); \
            rm /tmp/eigen.zip; \
            sed -i -E "s|^eigen;.*$|eigen;${EIGEN_URL};${EIGEN_SHA1}|" cmake/deps.txt; \
            python tools/ci_build/build.py \
                --build_dir /build/ort-build \
                --config Release \
                --parallel "${BUILD_PARALLEL}" \
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
                --update --build; \
            cp /build/ort-build/Release/dist/*.whl /build/ort-wheel/; \
            ;; \
        *) \
            echo "Invalid ORT_BUILD_FROM_SOURCE=${ORT_BUILD_FROM_SOURCE}; use auto, true, or false" >&2; \
            exit 2; \
            ;; \
    esac; \
    ls -la /build/ort-wheel/

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

# PyTorch from cu128.
ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

# Copy the onnxruntime-gpu wheel from stage 1 but DO NOT install yet.
COPY --from=ort-builder /build/ort-wheel/*.whl /tmp/wheels/

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
