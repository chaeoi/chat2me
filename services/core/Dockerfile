FROM python:3.11-slim

ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV PYTHONDONTWRITEBYTECODE=1 \
    VOICE_ROLE=chat2me-core \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

COPY services/core/requirements.txt .
RUN pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY runtime/entrypoints/config.sh /config.sh
COPY runtime/shared/common.py ./app/common.py
COPY services/core/app ./app
RUN chmod +x /config.sh

ENTRYPOINT ["/config.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "--log-level", "warning"]
