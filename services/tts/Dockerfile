FROM ubuntu:22.04

ARG JETSON_L4T_PLATFORM=t234
ARG JETSON_L4T_RELEASE=r36.5
ARG JETSON_TORCH_WHEEL_URL=https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
ARG CUSPARSELT_URL=https://developer.download.nvidia.cn/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.2.3-archive.tar.xz
ARG MELOTTS_SOURCE_REF=209145371cff8fc3bd60d7be902ea69cbdb7965a
ARG CHAT2ME_DOWNLOAD_RETRIES=10
ARG CHAT2ME_PIP_RETRIES=10
ARG CHAT2ME_PIP_TIMEOUT=120

ENV DEBIAN_FRONTEND=noninteractive \
    VOICE_ROLE=chat2me-tts \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia \
    CHAT2ME_DOWNLOAD_RETRIES=${CHAT2ME_DOWNLOAD_RETRIES} \
    CHAT2ME_PIP_RETRIES=${CHAT2ME_PIP_RETRIES} \
    CHAT2ME_PIP_TIMEOUT=${CHAT2ME_PIP_TIMEOUT}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       xz-utils \
    && curl -fsSL \
       --retry "$CHAT2ME_DOWNLOAD_RETRIES" \
       --retry-connrefused \
       --connect-timeout 20 \
       https://repo.download.nvidia.cn/jetson/jetson-ota-public.asc \
       -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
    && printf '%s\n' \
       "deb https://repo.download.nvidia.cn/jetson/common ${JETSON_L4T_RELEASE} main" \
       "deb https://repo.download.nvidia.cn/jetson/${JETSON_L4T_PLATFORM} ${JETSON_L4T_RELEASE} main" \
       > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       cuda-cudart-12-6 \
       cuda-cupti-12-6 \
       cuda-nvrtc-12-6 \
       cuda-nvtx-12-6 \
       ffmpeg \
       libgomp1 \
       libopenblas-base \
       libsndfile1 \
       libcublas-12-6 \
       libcufft-12-6 \
       libcudnn9-cuda-12 \
       libcurand-12-6 \
       libnvjitlink-12-6 \
       libcusolver-12-6 \
       libcusparse-12-6 \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA's JetPack 6 PyTorch wheel also links against cuSPARSELt.
RUN curl -fsSL --retry "$CHAT2ME_DOWNLOAD_RETRIES" --retry-connrefused \
       --connect-timeout 20 "$CUSPARSELT_URL" -o /tmp/cusparselt.tar.xz \
    && mkdir /tmp/cusparselt \
    && tar -xf /tmp/cusparselt.tar.xz -C /tmp/cusparselt --strip-components=1 \
    && cp -a /tmp/cusparselt/lib/libcusparseLt.so* /usr/local/cuda-12.6/targets/aarch64-linux/lib/ \
    && rm -rf /tmp/cusparselt /tmp/cusparselt.tar.xz \
    && ldconfig

COPY services/tts/requirements.txt requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      -r requirements.txt \
    && python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      --force-reinstall \
      --no-deps \
      "$JETSON_TORCH_WHEEL_URL" \
    && python3 -m pip install --no-cache-dir \
      --retries "$CHAT2ME_PIP_RETRIES" \
      --timeout "$CHAT2ME_PIP_TIMEOUT" \
      "filelock==3.13.4" \
      "fsspec==2024.6.1" \
      "jinja2==3.1.4" \
      "networkx==3.1" \
      "sympy==1.13.1" \
    && python3 -m pip check

# Pin corpus contents instead of consulting NLTK's mutable download index.
ARG NLTK_DATA_REF=550b6625bcef1f2abff2ff770a5a0d272c9c6b2a
RUN python3 - <<'PY'
import hashlib
from http.client import HTTPException
import io
import os
from pathlib import Path
import time
import urllib.request
import zipfile

root = Path('/usr/local/share/nltk_data')
packages = {
    'corpora/cmudict': 'd07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6',
    'taggers/averaged_perceptron_tagger': 'e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9',
    'taggers/averaged_perceptron_tagger_eng': '6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b',
}
for package, expected in packages.items():
    url = f"https://api.github.com/repos/nltk/nltk_data/contents/packages/{package}.zip?ref={os.environ['NLTK_DATA_REF']}"
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'chat2me-build'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            break
        except (OSError, HTTPException):
            if attempt == 3:
                raise
            time.sleep(2)
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f'NLTK data checksum mismatch: {package}')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(root / package.split('/')[0])
PY

COPY runtime/tools/melotts.sh /opt/melotts.sh
ARG MELOTTS_SOURCE_URL
RUN chmod +x /opt/melotts.sh \
    && MELOTTS_SOURCE_REF="$MELOTTS_SOURCE_REF" \
       /opt/melotts.sh \
    && rm -f /opt/melotts.sh \
    && python3 -m compileall -q /opt/MeloTTS

COPY config /defaults/config
COPY runtime/entrypoints/lib.sh /entrypoint-lib.sh
COPY runtime/entrypoints/audio.sh /audio.sh
COPY runtime/shared/common.py ./app/common.py
COPY runtime/shared/voice.py ./app/voice.py
COPY services/tts/app/main.py ./app/main.py
RUN chmod +x /audio.sh

ENV PYTHONPATH=/opt/MeloTTS

ENTRYPOINT ["/audio.sh"]
CMD ["python3", "-m", "app.main"]
