FROM ubuntu:20.04

ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-speech \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       alsa-utils \
       bzip2 \
       ca-certificates \
       curl \
       libasound2-plugins \
       libasound2 \
       libgomp1 \
       libopenblas-base \
       libportaudio2 \
       pulseaudio-utils \
       libusb-1.0-0 \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/speech/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt

COPY runtime/tools/preload.sh /opt/preload.sh
RUN chmod +x /opt/preload.sh \
    && DEFAULT_MODEL_SET=kws /opt/preload.sh \
    && rm -f /opt/preload.sh

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY runtime/entrypoints/audio.sh /audio.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/speech/app/respeaker.py ./app/respeaker.py
COPY services/speech/app/main.py ./app/main.py
RUN chmod +x /audio.sh

ENTRYPOINT ["/audio.sh"]
CMD ["python3", "-m", "app.main"]
