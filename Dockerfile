FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git python3.10 python3.10-dev python3-pip ninja-build ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv /opt/venv --python 3.10 --seed
RUN uv pip install --no-cache torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --no-cache "huggingface_hub[cli]" hf_transfer tokenizers numpy rich typing_extensions safetensors ninja pillow pyyaml \
    marisa_trie pydantic "llguidance>=1.7.0" setuptools wheel

# Pinned upstream exllamav3 (turboderp-org, MIT) + Inkling-2 overlay
ARG EXLLAMAV3_REF=v1.6.6
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3 && git -C /opt/exllamav3 checkout ${EXLLAMAV3_REF}
COPY patches /opt/inkling2/patches
COPY overlay /opt/inkling2/overlay
RUN git -C /opt/exllamav3 apply /opt/inkling2/patches/upstream-modifications.patch \
    && cp -r /opt/inkling2/overlay/exllamav3/. /opt/exllamav3/exllamav3/

# Build the CUDA extension for Blackwell (sm_120); no GPU needed at build time
ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8
RUN cd /opt/exllamav3 && uv pip install --no-cache --no-build-isolation .

COPY bench /opt/inkling2/bench
COPY scripts /opt/inkling2/scripts
ENV MODEL_DIR=/models/Inkling-2-Small-EXL3-3.48bpw RESULTS_DIR=/models/results SPEC=on HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /opt/inkling2
ENTRYPOINT ["/opt/inkling2/scripts/entrypoint.sh"]
CMD ["sweep"]
