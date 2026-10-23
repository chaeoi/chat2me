FROM ollama/ollama:latest AS upstream

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=upstream /usr/bin/ollama /usr/bin/ollama
COPY --from=upstream /usr/lib/ollama /usr/lib/ollama

COPY services/llm/requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY services/llm/entrypoint.sh /entrypoint.sh
COPY runtime/shared/common.py ./app/common.py
COPY services/llm/app ./app
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/cuda_jetpack6:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    VOICE_ROLE=chat2me-llm \
    OLLAMA_HOST=0.0.0.0:11434

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082", "--no-access-log", "--log-level", "warning"]
