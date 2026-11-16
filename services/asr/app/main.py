from __future__ import annotations

import io
import os
import time
import threading
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import voice
from app.common import log


ASR_HOST = os.getenv("ASR_HOST", "0.0.0.0")
ASR_PORT = int(os.getenv("ASR_PORT", "8092"))
CONFIGURED_ASR_ENGINE = voice.VOICE_ASR_ENGINE
CONFIGURED_ASR_MODEL = voice.VOICE_ASR_MODEL


class HealthResponse(BaseModel):
    ok: bool
    configured_engine: str
    local_engine: str
    local_model: str


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


class TranscribeResponse(BaseModel):
    text: str
    route: str
    engine: str
    model: str
    latency_ms: int


LOCAL_RECOGNIZER: voice.StreamingRecognizer | None = None
ASR_INFERENCE_LOCK = threading.Lock()


def local_engine() -> str:
    return CONFIGURED_ASR_ENGINE


def local_model() -> str:
    return CONFIGURED_ASR_MODEL


def with_asr_env(engine: str, model: str) -> None:
    voice.VOICE_ASR_ENGINE = engine
    voice.VOICE_ASR_MODEL = model
    voice.ASR_MODEL_DIR = voice.MODELS_DIR / engine / model


def create_local_recognizer() -> voice.StreamingRecognizer:
    engine = local_engine()
    model = local_model()
    with_asr_env(engine, model)
    recognizer = voice.create_asr()
    log(f"local ASR ready: engine={engine} model={model}")
    return recognizer


def read_wav_upload(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise HTTPException(status_code=400, detail="only 16-bit PCM wav is supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


def transcribe_with_recognizer(recognizer: voice.StreamingRecognizer, samples: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    try:
        chunk = max(1, int(sample_rate * voice.CHUNK_SECONDS))
        for offset in range(0, len(samples), chunk):
            recognizer.accept_waveform(stream, sample_rate, samples[offset : offset + chunk])
        recognizer.input_finished(stream)
        return recognizer.decode_ready(stream).strip()
    finally:
        del stream


async def startup() -> None:
    global LOCAL_RECOGNIZER
    LOCAL_RECOGNIZER = create_local_recognizer()


async def shutdown() -> None:
    return


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="Chat2Me ASR", version=os.getenv("APP_VERSION", "dev"), lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ok=LOCAL_RECOGNIZER is not None,
        configured_engine=CONFIGURED_ASR_ENGINE,
        local_engine=local_engine(),
        local_model=local_model(),
    )


@app.get("/asr/reachability", response_model=ReachabilityResponse)
async def reachability() -> ReachabilityResponse:
    return ReachabilityResponse(
        online=False,
        provider="local",
        model=None,
        status="online_asr_removed",
        checked_at=time.time(),
    )


@app.post("/asr/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
) -> TranscribeResponse:
    started_at = time.perf_counter()
    payload = await file.read()
    samples, sample_rate = read_wav_upload(payload)

    with ASR_INFERENCE_LOCK:
        if LOCAL_RECOGNIZER is None:
            raise HTTPException(status_code=503, detail="local ASR is not ready")
        with_asr_env(local_engine(), local_model())
        text = transcribe_with_recognizer(LOCAL_RECOGNIZER, samples, sample_rate)
    return TranscribeResponse(
        text=text,
        route="local",
        engine=local_engine(),
        model=local_model(),
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ASR_HOST, port=ASR_PORT, access_log=False, log_level="warning")
