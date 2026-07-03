FROM ubuntu:20.04

ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-asr \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       libgomp1 \
       libopenblas-base \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY services/asr/requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt

ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    ASR_HOMOPHONE_GENERATOR_PYTHON=/opt/homophone-fst/bin/python

RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-aarch64/latest -o /tmp/micromamba.tar.bz2 \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -p /opt/homophone-fst -c conda-forge python=3.12 pynini pyyaml pypinyin \
    && micromamba clean -a -y \
    && rm -f /tmp/micromamba.tar.bz2

COPY runtime/tools/preload.sh /opt/preload.sh
RUN chmod +x /opt/preload.sh \
    && DEFAULT_MODEL_SET=asr,homophone /opt/preload.sh \
    && rm -f /opt/preload.sh

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY runtime/entrypoints/audio.sh /audio.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/asr/app/main.py ./app/main.py
COPY runtime/tools/homophone.py ./runtime/tools/homophone.py
RUN chmod +x /audio.sh

ENTRYPOINT ["/audio.sh"]
CMD ["python3", "-m", "app.main"]
