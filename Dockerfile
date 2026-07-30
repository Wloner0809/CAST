# Match the published Linux x86-64 / Python 3.10 / CUDA 12.6 runtime without
# pre-installing the mutually exclusive vLLM rollout profile.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates git openjdk-11-jre-headless \
        python3.10 python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/cast

# Install this checkout, not a separately cloned upstream rllm revision.
COPY . .
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel ninja packaging psutil && \
    python3 -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    python3 -m pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl && \
    python3 -m pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
