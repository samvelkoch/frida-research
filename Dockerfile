# FRIDA-Decisions behind an OpenAI-compatible API (/v1/chat/completions, /v1/models, /health).
# CUDA 12.8 build: covers Turing..Blackwell (RTX 20xx..50xx, A100, H100, L4, ...).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG FRIDA_VERSION=v0.2.0
# Pinned HF commit of ai-forever/FRIDA-Decisions, so the image is reproducible.
ARG MODEL_REVISION=0096b5384e821c68791cb2b3b7292c2b937dcec8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# torch comes from the base image; the package is installed without its [torch] extra
# so pip never replaces the CUDA build of torch.
RUN pip install \
        "transformers==4.56.2" \
        "fastapi==0.118.0" \
        "uvicorn[standard]==0.37.0" \
        "huggingface_hub>=0.34" \
        "https://github.com/ai-forever/FRIDA-Decisions/archive/refs/tags/${FRIDA_VERSION}.tar.gz" \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch._C._cuda_getArchFlags())"

# Weights are baked in: no HF access at start-up (~1.9 GB, ONNX build skipped).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ai-forever/FRIDA-Decisions', revision='${MODEL_REVISION}', allow_patterns=['*.json', 'model.safetensors', 'head.safetensors'], local_dir='/models/FRIDA-Decisions')" \
 && rm -rf /models/FRIDA-Decisions/.cache /root/.cache

ENV HF_HUB_OFFLINE=1 \
    MODEL_DIR=/models/FRIDA-Decisions \
    MODEL_ID=frida-decisions \
    STATE_MAX=384 \
    STATE_CACHE_MB=512 \
    PORT=8000

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# One worker: one model copy per GPU; requests are serialized inside the process.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
