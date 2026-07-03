FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    VOICE_ROLE=chat2me-relay \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/relay/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY runtime/entrypoints/config.sh /config.sh
COPY runtime/shared/common.py ./app/common.py
COPY services/relay/app/main.py ./app/main.py
RUN chmod +x /config.sh

ENTRYPOINT ["/config.sh"]
CMD ["python3", "-m", "app.main"]
