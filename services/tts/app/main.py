from __future__ import annotations

import asyncio
import math
import os
import threading
import time
import wave
import numpy as np
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import voice
from app.common import env_bool, env_float, log


TTS_HOST = os.getenv("TTS_HOST", "0.0.0.0")
TTS_PORT = int(os.getenv("TTS_PORT", "8093"))
LOCAL_TTS_ENGINE = "melotts"
LOCAL_TTS_MODEL = "MeloTTS-Chinese"
CONFIGURED_TTS_ENGINE = voice.VOICE_TTS_ENGINE
CONFIGURED_TTS_MODEL = voice.VOICE_TTS_MODEL

TTS_REACHABILITY_INTERVAL_SECONDS = env_float("TTS_REACHABILITY_INTERVAL_SECONDS", 5.0)
TTS_REACHABILITY_TIMEOUT_SECONDS = env_float("TTS_REACHABILITY_TIMEOUT_SECONDS", 3.0)
TTS_ONLINE_SYNTHESIS_TIMEOUT_SECONDS = env_float("TTS_ONLINE_SYNTHESIS_TIMEOUT_SECONDS", 8.0)
TTS_ONLINE_FAILURE_COOLDOWN_SECONDS = env_float("TTS_ONLINE_FAILURE_COOLDOWN_SECONDS", 30.0)
TTS_ONLINE_RECENT_SUCCESS_GRACE_SECONDS = env_float("TTS_ONLINE_RECENT_SUCCESS_GRACE_SECONDS", 30.0)
TTS_LOCAL_SYNTHESIS_TIMEOUT_SECONDS = env_float("TTS_LOCAL_SYNTHESIS_TIMEOUT_SECONDS", 20.0)
TTS_LOCAL_LOCK_TIMEOUT_SECONDS = env_float("TTS_LOCAL_LOCK_TIMEOUT_SECONDS", 5.0)
TTS_NORMALIZE_ENABLED = env_bool("TTS_NORMALIZE_ENABLED", True)
TTS_TARGET_RMS_DBFS = env_float("TTS_TARGET_RMS_DBFS", -22.0)
TTS_PEAK_LIMIT_DBFS = env_float("TTS_PEAK_LIMIT_DBFS", -2.0)
TTS_NORMALIZE_ACTIVE_FLOOR_DBFS = env_float("TTS_NORMALIZE_ACTIVE_FLOOR_DBFS", -45.0)
TTS_NORMALIZE_MIN_GAIN = env_float("TTS_NORMALIZE_MIN_GAIN", 0.25)
TTS_NORMALIZE_MAX_GAIN = env_float("TTS_NORMALIZE_MAX_GAIN", 6.0)
if TTS_NORMALIZE_MIN_GAIN <= 0 or TTS_NORMALIZE_MAX_GAIN <= 0:
    raise RuntimeError("TTS_NORMALIZE_MIN_GAIN and TTS_NORMALIZE_MAX_GAIN must be greater than zero")
if TTS_NORMALIZE_MIN_GAIN > TTS_NORMALIZE_MAX_GAIN:
    raise RuntimeError("TTS_NORMALIZE_MIN_GAIN must be less than or equal to TTS_NORMALIZE_MAX_GAIN")


@dataclass(frozen=True)
class Reachability:
    online: bool
    status: str
    checked_at: float | None = None


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    ok: bool
    configured_engine: str
    active_online: bool
    online_status: str
    online_cooldown_remaining_seconds: float = 0.0
    local_engine: str
    local_model: str
    sample_rate: int


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None
    cooldown_remaining_seconds: float = 0.0


ONLINE_REACHABILITY = Reachability(online=False, status="not_checked")
ONLINE_DISABLED_UNTIL = 0.0
ONLINE_LAST_SUCCESS_AT = 0.0
ONLINE_REACHABILITY_TASK: asyncio.Task[None] | None = None
LOCAL_VOICE: voice.TextToSpeech | None = None
ONLINE_VOICE: voice.TextToSpeech | None = None
ONLINE_STATE_LOCK = threading.Lock()
ONLINE_TTS_LOCK = threading.Lock()
LOCAL_TTS_LOCK = threading.Lock()


def online_enabled() -> bool:
    return CONFIGURED_TTS_ENGINE == "online"


def local_engine() -> str:
    return LOCAL_TTS_ENGINE if online_enabled() else CONFIGURED_TTS_ENGINE


def local_model() -> str:
    return LOCAL_TTS_MODEL if online_enabled() else CONFIGURED_TTS_MODEL


def validate_tts_selection() -> None:
    if CONFIGURED_TTS_ENGINE not in {"melotts", "online"}:
        raise RuntimeError("VOICE_TTS_ENGINE must be melotts or online")
    if online_enabled() and CONFIGURED_TTS_MODEL != "edge-tts":
        raise RuntimeError("online TTS only supports VOICE_TTS_MODEL=edge-tts")
    if local_model() != "MeloTTS-Chinese":
        raise RuntimeError("local TTS only supports MeloTTS-Chinese")


def online_state_snapshot() -> tuple[Reachability, float]:
    with ONLINE_STATE_LOCK:
        cooldown_remaining = max(0.0, ONLINE_DISABLED_UNTIL - time.monotonic())
        return ONLINE_REACHABILITY, cooldown_remaining


def publish_online_reachability(result: Reachability) -> None:
    global ONLINE_REACHABILITY, ONLINE_DISABLED_UNTIL, ONLINE_LAST_SUCCESS_AT
    with ONLINE_STATE_LOCK:
        now = time.monotonic()
        if result.online and ONLINE_DISABLED_UNTIL > time.monotonic():
            return
        if not result.online and now - ONLINE_LAST_SUCCESS_AT < max(0.0, TTS_ONLINE_RECENT_SUCCESS_GRACE_SECONDS):
            return
        ONLINE_REACHABILITY = result
        if result.online:
            ONLINE_DISABLED_UNTIL = 0.0
            ONLINE_LAST_SUCCESS_AT = now


def mark_online_success() -> None:
    publish_online_reachability(Reachability(online=True, status="ok", checked_at=time.time()))


def disable_online_temporarily(status: str, exc: Exception | None = None) -> None:
    global ONLINE_REACHABILITY, ONLINE_DISABLED_UNTIL
    cooldown = max(0.0, TTS_ONLINE_FAILURE_COOLDOWN_SECONDS)
    with ONLINE_STATE_LOCK:
        ONLINE_DISABLED_UNTIL = time.monotonic() + cooldown
        ONLINE_REACHABILITY = Reachability(online=False, status=status, checked_at=time.time())
    detail = f": {exc}" if exc is not None else ""
    log(f"online TTS disabled for {cooldown:.1f}s after {status}{detail}", level="warning")


def online_available() -> bool:
    if not online_enabled() or ONLINE_VOICE is None:
        return False
    reachability, cooldown_remaining = online_state_snapshot()
    return reachability.online and cooldown_remaining <= 0.0


def online_skip_reason() -> str:
    if not online_enabled():
        return "online_engine_disabled"
    if ONLINE_VOICE is None:
        return "online_voice_not_ready"
    reachability, cooldown_remaining = online_state_snapshot()
    if cooldown_remaining > 0.0:
        return "online_cooling_down"
    if not reachability.online:
        return reachability.status
    return "online_not_selected"


def with_tts_env(engine: str, model: str) -> None:
    voice.VOICE_TTS_ENGINE = engine
    voice.VOICE_TTS_MODEL = model
    voice.TTS_MODEL_DIR = voice.MODELS_DIR / engine / model
    voice.MELOTTS_CONFIG_FILE = Path(os.getenv("MELOTTS_CONFIG_FILE", str(voice.TTS_MODEL_DIR / "config.json")))
    voice.MELOTTS_CKPT_FILE = Path(os.getenv("MELOTTS_CKPT_FILE", str(voice.TTS_MODEL_DIR / "checkpoint.pth")))


def create_local_voice() -> voice.TextToSpeech:
    engine = local_engine()
    model = local_model()
    with_tts_env(engine, model)
    tts_voice = voice.create_tts()
    voice.preload_tts_cache(tts_voice, voice.WAKE_RESPONSE, voice.SESSION_END_RESPONSE, voice.SESSION_IDLE_RESPONSE)
    voice.warmup_tts(tts_voice)
    log(f"local TTS ready: engine={engine} model={model} sample_rate={tts_voice.config.sample_rate}")
    return tts_voice


def create_online_voice() -> voice.TextToSpeech:
    if not online_enabled():
        raise RuntimeError("online TTS is not configured")
    with_tts_env("online", CONFIGURED_TTS_MODEL)
    tts_voice = voice.create_tts()
    log(f"online TTS ready: model={CONFIGURED_TTS_MODEL} sample_rate={tts_voice.config.sample_rate}")
    return tts_voice


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def dbfs_to_amplitude(dbfs: float) -> float:
    return 10 ** (dbfs / 20.0)


def normalize_pcm_loudness(pcm: bytes) -> bytes:
    if not TTS_NORMALIZE_ENABLED or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    normalized = samples / 32768.0
    floor = dbfs_to_amplitude(TTS_NORMALIZE_ACTIVE_FLOOR_DBFS)
    active = normalized[np.abs(normalized) >= floor]
    if active.size == 0:
        active = normalized
    rms = math.sqrt(float(np.mean(active * active))) if active.size else 0.0
    if rms <= 0:
        return pcm

    target_rms = dbfs_to_amplitude(TTS_TARGET_RMS_DBFS)
    gain = target_rms / rms
    gain = min(TTS_NORMALIZE_MAX_GAIN, max(TTS_NORMALIZE_MIN_GAIN, gain))

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    peak_limit = dbfs_to_amplitude(TTS_PEAK_LIMIT_DBFS)
    if peak > 0 and peak * gain > peak_limit:
        gain = peak_limit / peak
    if abs(gain - 1.0) < 0.001:
        return pcm

    samples *= gain
    np.clip(samples, -32768, 32767, out=samples)
    return samples.astype(np.int16).tobytes()


def synthesize_wav(tts_voice: voice.TextToSpeech, text: str) -> bytes:
    pcm = b"".join(chunk for chunk in tts_voice.synthesize_pcm(text) if chunk)
    pcm = normalize_pcm_loudness(pcm)
    return pcm_to_wav(pcm, int(tts_voice.config.sample_rate))


def synthesize_wav_locked(
    tts_voice: voice.TextToSpeech,
    text: str,
    lock: threading.Lock,
    name: str,
    lock_timeout_seconds: float,
) -> bytes:
    if lock_timeout_seconds <= 0:
        acquired = lock.acquire(blocking=False)
    else:
        acquired = lock.acquire(timeout=lock_timeout_seconds)
    if not acquired:
        raise RuntimeError(f"{name} TTS is busy")
    try:
        return synthesize_wav(tts_voice, text)
    finally:
        lock.release()


async def run_blocking_synthesis(
    tts_voice: voice.TextToSpeech,
    text: str,
    lock: threading.Lock,
    name: str,
    lock_timeout_seconds: float,
    timeout_seconds: float,
) -> bytes:
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None,
        synthesize_wav_locked,
        tts_voice,
        text,
        lock,
        name,
        lock_timeout_seconds,
    )
    return await asyncio.wait_for(future, timeout=max(0.1, timeout_seconds))


async def probe_online() -> Reachability:
    if not online_enabled():
        return Reachability(online=False, status="online_engine_disabled", checked_at=time.time())
    try:
        import edge_tts

        await asyncio.wait_for(
            edge_tts.list_voices(proxy=voice.EDGE_TTS_PROXY),
            timeout=TTS_REACHABILITY_TIMEOUT_SECONDS,
        )
        return Reachability(online=True, status="ok", checked_at=time.time())
    except asyncio.TimeoutError:
        return Reachability(online=False, status="timeout", checked_at=time.time())
    except Exception:
        return Reachability(online=False, status="unreachable", checked_at=time.time())


async def reachability_loop() -> None:
    interval = max(0.5, TTS_REACHABILITY_INTERVAL_SECONDS)
    while True:
        _, cooldown_remaining = online_state_snapshot()
        if cooldown_remaining > 0:
            await asyncio.sleep(min(interval, cooldown_remaining))
            continue
        publish_online_reachability(await probe_online())
        await asyncio.sleep(interval)


async def startup() -> None:
    global LOCAL_VOICE, ONLINE_VOICE, ONLINE_REACHABILITY_TASK
    validate_tts_selection()
    LOCAL_VOICE = create_local_voice()
    if online_enabled():
        ONLINE_VOICE = create_online_voice()
        publish_online_reachability(await probe_online())
        ONLINE_REACHABILITY_TASK = asyncio.create_task(reachability_loop())


async def shutdown() -> None:
    if ONLINE_REACHABILITY_TASK is None:
        return
    ONLINE_REACHABILITY_TASK.cancel()
    try:
        await ONLINE_REACHABILITY_TASK
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="Chat2Me TTS", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    reachability, cooldown_remaining = online_state_snapshot()
    active_online = online_enabled() and reachability.online and cooldown_remaining <= 0.0
    active_voice = ONLINE_VOICE if active_online and ONLINE_VOICE is not None else LOCAL_VOICE
    return HealthResponse(
        ok=LOCAL_VOICE is not None,
        configured_engine=CONFIGURED_TTS_ENGINE,
        active_online=active_online,
        online_status=reachability.status,
        online_cooldown_remaining_seconds=round(cooldown_remaining, 1),
        local_engine=local_engine(),
        local_model=local_model(),
        sample_rate=int(getattr(getattr(active_voice, "config", None), "sample_rate", 0) or 0),
    )


@app.get("/tts/reachability", response_model=ReachabilityResponse)
async def reachability() -> ReachabilityResponse:
    reachability, cooldown_remaining = online_state_snapshot()
    return ReachabilityResponse(
        online=reachability.online and cooldown_remaining <= 0.0,
        provider="online-tts" if online_enabled() else "local",
        model=CONFIGURED_TTS_MODEL if online_enabled() else None,
        status=reachability.status if online_enabled() else "online_engine_disabled",
        checked_at=reachability.checked_at,
        cooldown_remaining_seconds=round(cooldown_remaining, 1),
    )


@app.post("/tts/speak")
async def speak(request: SpeakRequest) -> Response:
    fallback_reason = ""
    if online_available() and ONLINE_VOICE is not None:
        try:
            wav = await run_blocking_synthesis(
                ONLINE_VOICE,
                request.text,
                ONLINE_TTS_LOCK,
                "online",
                0.0,
                TTS_ONLINE_SYNTHESIS_TIMEOUT_SECONDS,
            )
            mark_online_success()
            reachability, _ = online_state_snapshot()
            return Response(
                content=wav,
                media_type="audio/wav",
                headers={
                    "X-Chat2Me-TTS-Route": "online",
                    "X-Chat2Me-TTS-Engine": "online",
                    "X-Chat2Me-TTS-Model": CONFIGURED_TTS_MODEL,
                    "X-Chat2Me-Online-Status": reachability.status,
                },
            )
        except asyncio.TimeoutError as exc:
            fallback_reason = "online_timeout"
            disable_online_temporarily(fallback_reason, exc)
        except Exception as exc:
            fallback_reason = f"online_failed:{exc.__class__.__name__}"
            disable_online_temporarily(fallback_reason, exc)
    else:
        fallback_reason = online_skip_reason()

    if LOCAL_VOICE is None:
        raise HTTPException(status_code=503, detail="local TTS is not ready")
    try:
        wav = await run_blocking_synthesis(
            LOCAL_VOICE,
            request.text,
            LOCAL_TTS_LOCK,
            "local",
            TTS_LOCAL_LOCK_TIMEOUT_SECONDS,
            TTS_LOCAL_SYNTHESIS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="local TTS synthesis timed out") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"local TTS synthesis failed: {exc}") from exc

    reachability, _ = online_state_snapshot()
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Chat2Me-TTS-Route": "local",
            "X-Chat2Me-TTS-Engine": local_engine(),
            "X-Chat2Me-TTS-Model": local_model(),
            "X-Chat2Me-TTS-Fallback": "1" if online_enabled() else "0",
            "X-Chat2Me-TTS-Fallback-Reason": fallback_reason,
            "X-Chat2Me-Online-Status": reachability.status,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=TTS_HOST, port=TTS_PORT, access_log=False, log_level="warning")
