from __future__ import annotations

import gc
import io
import logging
import os
import re
import subprocess
import threading
import time
import wave
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Protocol, TypedDict

from app.common import DisplayClient, env_bool, env_csv, env_float, env_int, env_value, log
import numpy as np


def is_default_audio_selector(value: str) -> bool:
    return value.strip().lower() in {"", "auto", "default"}


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
VOICE_KWS_MODEL = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
VOICE_ASR_ENGINE = env_value("VOICE_ASR_ENGINE")
VOICE_ASR_MODEL = env_value("VOICE_ASR_MODEL")
VOICE_TTS_MODEL = env_value("VOICE_TTS_MODEL")
VOICE_TTS_ENGINE = env_value("VOICE_TTS_ENGINE").strip().lower()
KWS_MODEL_DIR = MODELS_DIR / VOICE_KWS_MODEL
ASR_MODEL_DIR = MODELS_DIR / VOICE_ASR_ENGINE / VOICE_ASR_MODEL
GENERATED_KEYWORDS_FILE = MODELS_DIR / "wake_words.txt"
GENERATED_KEYWORDS_RAW = MODELS_DIR / "wake_words_raw.txt"
WAKE_WORDS = env_csv("WAKE_WORDS")
if not WAKE_WORDS:
    raise RuntimeError("WAKE_WORDS must contain at least one wake word in runtime.env")

CORE_URL = "http://chat2me-core:8080/chat"
ASR_SERVICE_URL = "http://chat2me-asr:8092/asr/transcribe"
TTS_SERVICE_URL = "http://chat2me-tts:8093/tts/speak"
DISPLAY_SERIAL_BAUD = env_int("DISPLAY_SERIAL_BAUD")
INPUT_DEVICE = env_value("AUDIO_INPUT_DEVICE", allow_empty=True)
INPUT_DEVICE_REQUIRED = bool(
    INPUT_DEVICE
    and not INPUT_DEVICE.isdigit()
    and not is_default_audio_selector(INPUT_DEVICE)
)
OUTPUT_DEVICE = env_value("AUDIO_OUTPUT_DEVICE", allow_empty=True)
SAMPLE_RATE = env_int("AUDIO_SAMPLE_RATE")
CHUNK_SECONDS = env_float("AUDIO_CHUNK_SECONDS")
INPUT_CHANNELS = env_int("AUDIO_INPUT_CHANNELS")
if INPUT_CHANNELS != 6:
    raise RuntimeError("AUDIO_INPUT_CHANNELS must be 6 for ReSpeaker Mic Array v3.0 factory firmware")
INPUT_CHANNEL_INDEX = env_int("AUDIO_INPUT_CHANNEL_INDEX")
if INPUT_CHANNEL_INDEX != 0:
    raise RuntimeError("AUDIO_INPUT_CHANNEL_INDEX must be 0 for ReSpeaker Mic Array v3.0 ASR channel")
KWS_THREADS = env_int("KWS_THREADS")
ASR_THREADS = env_int("ASR_THREADS")
SENSEVOICE_LANGUAGE = os.getenv("SENSEVOICE_LANGUAGE", "auto").strip().lower() or "auto"
if SENSEVOICE_LANGUAGE not in {"auto", "zh", "en", "ja", "ko", "yue"}:
    raise RuntimeError("SENSEVOICE_LANGUAGE must be auto, zh, en, ja, ko, or yue in runtime.env")
SENSEVOICE_USE_ITN = env_bool("SENSEVOICE_USE_ITN", True)
ASR_HOMOPHONE_REPLACER_ENABLED = env_bool("ASR_HOMOPHONE_REPLACER_ENABLED", False)
ASR_HOMOPHONE_LEXICON_RAW = os.getenv("ASR_HOMOPHONE_LEXICON", str(MODELS_DIR / "homophone" / "lexicon.txt")).strip()
ASR_HOMOPHONE_RULE_FSTS = os.getenv("ASR_HOMOPHONE_RULE_FSTS", str(MODELS_DIR / "homophone" / "replace.fst")).strip()
# Legacy deployments may still set GATEWAY_* keys in runtime.env.
CORE_REQUEST_TIMEOUT_SECONDS = env_float("CORE_REQUEST_TIMEOUT_SECONDS", env_float("GATEWAY_REQUEST_TIMEOUT_SECONDS", 30))
CORE_UNAVAILABLE_RESPONSE = os.getenv("CORE_UNAVAILABLE_RESPONSE") or env_value("GATEWAY_UNAVAILABLE_RESPONSE")
COMMAND_TIMEOUT_SECONDS = env_float("COMMAND_TIMEOUT_SECONDS")
COMMAND_MIN_SECONDS = env_float("COMMAND_MIN_SECONDS")
COMMAND_LEADING_SILENCE_SECONDS = env_float("COMMAND_LEADING_SILENCE_SECONDS")
COMMAND_INITIAL_GRACE_SECONDS = env_float("COMMAND_INITIAL_GRACE_SECONDS")
PRE_BEEP_DRAIN_SECONDS = env_float("PRE_BEEP_DRAIN_SECONDS")
POST_BEEP_DRAIN_SECONDS = env_float("POST_BEEP_DRAIN_SECONDS")
POST_RESPONSE_DRAIN_SECONDS = env_float("POST_RESPONSE_DRAIN_SECONDS")
SPEECH_RMS_THRESHOLD = env_float("SPEECH_RMS_THRESHOLD")
ASR_NOISE_GATE_ENABLED = env_bool("ASR_NOISE_GATE_ENABLED")
ASR_NOISE_CALIBRATION_SECONDS = env_float("ASR_NOISE_CALIBRATION_SECONDS")
ASR_NOISE_GATE_PERCENTILE = env_float("ASR_NOISE_GATE_PERCENTILE")
ASR_NOISE_GATE_RATIO = env_float("ASR_NOISE_GATE_RATIO")
ASR_NOISE_GATE_OFFSET = env_float("ASR_NOISE_GATE_OFFSET")
ASR_PREROLL_SECONDS = env_float("ASR_PREROLL_SECONDS")
RESPEAKER_VAD_GATE_ENABLED = env_bool("RESPEAKER_VAD_GATE_ENABLED")
MELOTTS_LANGUAGE = "ZH_MIX_EN"
MELOTTS_SPEAKER = "ZH"
MELOTTS_SPEED = env_float("MELOTTS_SPEED")
MELOTTS_SDP_RATIO = env_float("MELOTTS_SDP_RATIO")
MELOTTS_NOISE_SCALE = env_float("MELOTTS_NOISE_SCALE")
MELOTTS_NOISE_SCALE_W = env_float("MELOTTS_NOISE_SCALE_W")
MELOTTS_DISABLE_BERT = env_bool("MELOTTS_DISABLE_BERT")
TTS_PLAYER_TIMEOUT_SECONDS = env_float("TTS_PLAYER_TIMEOUT_SECONDS")
SPEECH_TTS_MAX_CHARS = env_int("SPEECH_TTS_MAX_CHARS")
TTS_CACHE_ENABLED = env_bool("TTS_CACHE_ENABLED")
TTS_CACHE_MAX_ITEMS = env_int("TTS_CACHE_MAX_ITEMS")
TTS_CACHE_MAX_BYTES = env_int("TTS_CACHE_MAX_BYTES")
TTS_PLAYBACK_MODE = env_value("TTS_PLAYBACK_MODE", default="buffered").lower()
TTS_PREBUFFER_SECONDS = env_float("TTS_PREBUFFER_SECONDS", 2.4)
TTS_PLAYBACK_RETRY_SECONDS = env_float("TTS_PLAYBACK_RETRY_SECONDS", 3)
TTS_WARMUP_TEXTS = tuple(
    text.strip()
    for text in os.getenv("TTS_WARMUP_TEXTS", "").split("|")
    if text.strip()
)
TTS_SERVICE_CONNECT_TIMEOUT_SECONDS = env_float("TTS_SERVICE_CONNECT_TIMEOUT_SECONDS", 3.0)
TTS_SERVICE_READ_TIMEOUT_SECONDS = env_float("TTS_SERVICE_READ_TIMEOUT_SECONDS", 35.0)
TTS_MODEL_DIR = MODELS_DIR / VOICE_TTS_ENGINE / VOICE_TTS_MODEL
VOICE_TTS_DEVICE = os.getenv("VOICE_TTS_DEVICE", "auto").strip().lower()
MELOTTS_CONFIG_FILE = Path(os.getenv("MELOTTS_CONFIG_FILE", str(TTS_MODEL_DIR / "config.json")))
MELOTTS_CKPT_FILE = Path(os.getenv("MELOTTS_CKPT_FILE", str(TTS_MODEL_DIR / "checkpoint.pth")))
EDGE_TTS_VOICE = env_value("EDGE_TTS_VOICE")
EDGE_TTS_RATE = env_value("EDGE_TTS_RATE")
EDGE_TTS_VOLUME = env_value("EDGE_TTS_VOLUME")
EDGE_TTS_PITCH = env_value("EDGE_TTS_PITCH")
EDGE_TTS_PROXY = env_value("EDGE_TTS_PROXY", allow_empty=True) or None
EDGE_TTS_SAMPLE_RATE = env_int("EDGE_TTS_SAMPLE_RATE")
EDGE_TTS_CONNECT_TIMEOUT_SECONDS = env_int("EDGE_TTS_CONNECT_TIMEOUT_SECONDS")
EDGE_TTS_RECEIVE_TIMEOUT_SECONDS = env_int("EDGE_TTS_RECEIVE_TIMEOUT_SECONDS")
DISPLAY_TEXT_MAX_CHARS = env_int("DISPLAY_TEXT_MAX_CHARS")
DISPLAY_SERIAL_RETRY_SECONDS = env_float("DISPLAY_SERIAL_RETRY_SECONDS")
WAKE_RESPONSE = env_value("WAKE_RESPONSE")
SESSION_IDLE_RESPONSE = env_value("SESSION_IDLE_RESPONSE", allow_empty=True)
SESSION_END_RESPONSE = env_value("SESSION_END_RESPONSE")
MAX_SESSION_TURNS = env_int("MAX_SESSION_TURNS")
SESSION_END_PHRASES = env_csv("SESSION_END_PHRASES")
if not SESSION_END_PHRASES:
    raise RuntimeError("SESSION_END_PHRASES must contain at least one phrase in runtime.env")
KWS_KEYWORDS_SCORE = env_float("KWS_KEYWORDS_SCORE")
KWS_KEYWORDS_THRESHOLD = env_float("KWS_KEYWORDS_THRESHOLD")


class VoiceActivityState(TypedDict):
    voice_activity: bool | None
    speech_detected: bool | None


class StreamingRecognizer(Protocol):
    def create_stream(self) -> Any:
        ...

    def accept_waveform(self, stream: Any, sample_rate: int, samples: np.ndarray) -> None:
        ...

    def input_finished(self, stream: Any) -> None:
        ...

    def decode_ready(self, stream: Any) -> str:
        ...

    def is_endpoint(self, stream: Any) -> bool:
        ...


class TextToSpeech(Protocol):
    config: Any

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        ...


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def wake_words_display() -> str:
    return " / ".join(WAKE_WORDS)


def select_input_device(selector: str) -> int | str | None:
    if not selector or is_default_audio_selector(selector):
        return None
    if selector.isdigit():
        return int(selector)

    import sounddevice as sd

    devices = sd.query_devices()
    selector_lower = selector.lower()
    for index, device in enumerate(devices):
        if selector_lower not in str(device.get("name", "")).lower():
            continue
        if device.get("max_input_channels", 0) > 0:
            return index

    return None


def ensure_keywords_file() -> Path:
    require_file(KWS_MODEL_DIR / "tokens.txt")
    require_file(KWS_MODEL_DIR / "en.phone")

    raw_file = GENERATED_KEYWORDS_RAW
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("".join(f"{word} @{word}\n" for word in WAKE_WORDS), encoding="utf-8")

    keywords_file = GENERATED_KEYWORDS_FILE
    keywords_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sherpa-onnx-cli",
        "text2token",
        "--tokens",
        str(KWS_MODEL_DIR / "tokens.txt"),
        "--tokens-type",
        "phone+ppinyin",
        "--lexicon",
        str(KWS_MODEL_DIR / "en.phone"),
        str(raw_file),
        str(keywords_file),
    ]
    log(f"generating wake keyword tokens: {wake_words_display()}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"wake keyword token generation failed with exit code {result.returncode}")
    return keywords_file


def create_kws() -> Any:
    import sherpa_onnx

    keywords_file = ensure_keywords_file()
    require_file(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    require_file(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
    require_file(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")

    spotter = sherpa_onnx.KeywordSpotter(
        tokens=str(KWS_MODEL_DIR / "tokens.txt"),
        encoder=str(KWS_MODEL_DIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(KWS_MODEL_DIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(KWS_MODEL_DIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        num_threads=KWS_THREADS,
        keywords_file=str(keywords_file),
        keywords_score=KWS_KEYWORDS_SCORE,
        keywords_threshold=KWS_KEYWORDS_THRESHOLD,
        provider="cpu",
    )
    log(f"KWS config: model={KWS_MODEL_DIR} provider=cpu threads={KWS_THREADS}")
    return spotter


def sensevoice_model_file() -> Path:
    model = ASR_MODEL_DIR / "model.int8.onnx"
    if model.is_file():
        return model
    return ASR_MODEL_DIR / "model.onnx"


def offline_result_text(result: Any) -> str:
    text = getattr(result, "text", result)
    return str(text or "").strip()


class SenseVoiceRecognizer:
    def __init__(self, recognizer: Any) -> None:
        self.recognizer = recognizer

    def create_stream(self) -> dict[str, Any]:
        return {
            "chunks": [],
            "sample_rate": None,
            "finalized": False,
            "text": "",
        }

    def accept_waveform(self, stream: dict[str, Any], sample_rate: int, samples: np.ndarray) -> None:
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"SenseVoice ASR input must be {SAMPLE_RATE} Hz")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        previous_sample_rate = stream.get("sample_rate")
        if previous_sample_rate is None:
            stream["sample_rate"] = int(sample_rate)
        elif int(previous_sample_rate) != int(sample_rate):
            raise RuntimeError("SenseVoice ASR stream sample rate changed")
        stream["chunks"].append(audio.copy())

    def input_finished(self, stream: dict[str, Any]) -> None:
        if stream.get("finalized"):
            return
        stream["finalized"] = True
        chunks = stream.get("chunks") or []
        if not chunks:
            stream["text"] = ""
            return

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        offline_stream = self.recognizer.create_stream()
        try:
            offline_stream.accept_waveform(int(stream.get("sample_rate") or SAMPLE_RATE), audio)
            self.recognizer.decode_stream(offline_stream)
            stream["text"] = offline_result_text(getattr(offline_stream, "result", ""))
        finally:
            del offline_stream

    def decode_ready(self, stream: dict[str, Any]) -> str:
        return str(stream.get("text") or "").strip()

    def is_endpoint(self, stream: dict[str, Any]) -> bool:
        return False


class RemoteBatchRecognizer:
    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE

    def create_stream(self) -> dict[str, Any]:
        return {
            "chunks": [],
            "seconds": 0.0,
            "active_seconds": 0.0,
            "speech_started": False,
            "endpoint": False,
            "finalized": False,
            "text": "",
        }

    def accept_waveform(self, stream: dict[str, Any], sample_rate: int, samples: np.ndarray) -> None:
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"remote ASR input must be {self.sample_rate} Hz")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        stream["chunks"].append(audio.copy())
        seconds = len(audio) / max(1, sample_rate)
        stream["seconds"] = float(stream["seconds"]) + seconds
        rms = audio_rms(audio)
        if rms >= SPEECH_RMS_THRESHOLD:
            stream["speech_started"] = True
            stream["active_seconds"] = float(stream["seconds"])
        elif stream["speech_started"]:
            trailing = float(stream["seconds"]) - float(stream["active_seconds"])
            if float(stream["seconds"]) >= COMMAND_MIN_SECONDS and trailing >= COMMAND_INITIAL_GRACE_SECONDS:
                stream["endpoint"] = True

    def input_finished(self, stream: dict[str, Any]) -> None:
        if stream.get("finalized"):
            return
        stream["finalized"] = True
        chunks = stream.get("chunks") or []
        if not chunks:
            stream["text"] = ""
            return
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        stream["text"] = transcribe_remote_audio(audio, self.sample_rate)

    def decode_ready(self, stream: dict[str, Any]) -> str:
        return str(stream.get("text") or "").strip()

    def is_endpoint(self, stream: dict[str, Any]) -> bool:
        return bool(stream.get("endpoint", False))


# 语音链路对 ASR/TTS/Core 的调用是串行低频的，共用一个连接池避免每轮握手。
# httpx 延迟导入：并非所有引用本模块的镜像都装了 httpx。
_SERVICE_CLIENT: Any | None = None
_SERVICE_CLIENT_LOCK = threading.Lock()


def service_http_client() -> Any:
    global _SERVICE_CLIENT
    if _SERVICE_CLIENT is None:
        import httpx

        with _SERVICE_CLIENT_LOCK:
            if _SERVICE_CLIENT is None:
                _SERVICE_CLIENT = httpx.Client()
    return _SERVICE_CLIENT


def float_audio_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def transcribe_remote_audio(samples: np.ndarray, sample_rate: int) -> str:
    import httpx

    wav_bytes = float_audio_to_wav_bytes(samples, sample_rate)
    files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
    timeout = httpx.Timeout(connect=5.0, read=COMMAND_TIMEOUT_SECONDS + 10, write=15.0, pool=5.0)
    response = service_http_client().post(ASR_SERVICE_URL, files=files, timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    route = payload.get("route")
    engine = payload.get("engine")
    model = payload.get("model")
    fallback = payload.get("fallback")
    log(f"remote asr result: route={route} engine={engine} model={model} fallback={fallback}", level="debug")
    return str(payload.get("text") or "").strip()


def create_sensevoice_asr() -> StreamingRecognizer:
    import sherpa_onnx

    require_file(ASR_MODEL_DIR / "tokens.txt")
    model = sensevoice_model_file()
    require_file(model)
    hr_kwargs = homophone_replacer_kwargs()
    log(
        "SenseVoice ASR config: "
        f"model={ASR_MODEL_DIR} language={SENSEVOICE_LANGUAGE} "
        f"use_itn={SENSEVOICE_USE_ITN} threads={ASR_THREADS} provider=cpu "
        f"homophone_replacer={'enabled' if hr_kwargs else 'disabled'}"
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        tokens=str(ASR_MODEL_DIR / "tokens.txt"),
        model=str(model),
        num_threads=ASR_THREADS,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        language=SENSEVOICE_LANGUAGE,
        use_itn=SENSEVOICE_USE_ITN,
        provider="cpu",
        **hr_kwargs,
    )
    log("SenseVoice ASR loaded: provider=cpu")
    return SenseVoiceRecognizer(recognizer)


def homophone_replacer_kwargs() -> dict[str, str]:
    if not ASR_HOMOPHONE_REPLACER_ENABLED:
        return {}
    if not ASR_HOMOPHONE_LEXICON_RAW:
        raise RuntimeError("homophone replacer enabled but ASR_HOMOPHONE_LEXICON is empty")

    lexicon = Path(ASR_HOMOPHONE_LEXICON_RAW)
    if not lexicon.is_file():
        raise FileNotFoundError(f"homophone replacer lexicon is missing: {lexicon}")

    rule_files = tuple(Path(item.strip()) for item in ASR_HOMOPHONE_RULE_FSTS.split(",") if item.strip())
    if not rule_files:
        raise RuntimeError("homophone replacer enabled but ASR_HOMOPHONE_RULE_FSTS is empty")

    missing = [str(path) for path in rule_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"homophone replacer rule FST is missing: {', '.join(missing)}")

    rule_fsts = ",".join(str(path) for path in rule_files)
    log(f"homophone replacer files: lexicon={lexicon} rule_fsts={rule_fsts}")
    return {"hr_lexicon": str(lexicon), "hr_rule_fsts": rule_fsts}


def create_asr() -> StreamingRecognizer:
    if VOICE_ASR_ENGINE == "sensevoice":
        return create_sensevoice_asr()
    raise RuntimeError("only VOICE_ASR_ENGINE=sensevoice is supported")


def create_remote_asr() -> StreamingRecognizer:
    log(f"remote ASR service: {ASR_SERVICE_URL}")
    return RemoteBatchRecognizer()


def write_beep(path: Path) -> None:
    sample_rate = 16000
    duration = 0.13
    freq = 880
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * freq * t)
    samples = np.int16(tone * 32767)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())


def play_wav(path: Path) -> None:
    try:
        subprocess.run(["aplay", "-q", "-D", OUTPUT_DEVICE, str(path)], check=False)
    except FileNotFoundError as exc:
        log(f"aplay is unavailable: {exc}", level="warning")


class MeloTextToSpeech:
    def __init__(self, model: Any, speaker_id: int, sample_rate: int) -> None:
        self.model = model
        self.speaker_id = speaker_id
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        text = text.strip()
        if not text:
            return
        audio = self.model.tts_to_file(
            text,
            self.speaker_id,
            output_path=None,
            sdp_ratio=MELOTTS_SDP_RATIO,
            noise_scale=MELOTTS_NOISE_SCALE,
            noise_scale_w=MELOTTS_NOISE_SCALE_W,
            speed=MELOTTS_SPEED,
            quiet=True,
        )
        yield tensor_audio_bytes(audio)


class EdgeTextToSpeech:
    def __init__(self) -> None:
        self.config = SimpleNamespace(sample_rate=EDGE_TTS_SAMPLE_RATE)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        audio = synthesize_edge_tts_audio(text)
        yield decode_audio_to_pcm(audio, EDGE_TTS_SAMPLE_RATE)


class RemoteTextToSpeech:
    def __init__(self) -> None:
        self.config = SimpleNamespace(sample_rate=SAMPLE_RATE)

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        wav_bytes = synthesize_remote_wav(text)
        pcm, sample_rate = wav_bytes_to_pcm(wav_bytes)
        self.config.sample_rate = sample_rate
        yield pcm


class CachedTextToSpeech:
    def __init__(self, voice: TextToSpeech, max_items: int, max_bytes: int) -> None:
        self.voice = voice
        self.config = voice.config
        self.max_items = max(0, max_items)
        self.max_bytes = max(0, max_bytes)
        self.cache: dict[str, tuple[bytes, ...]] = {}
        self.order: deque[str] = deque()

    def synthesize_pcm(self, text: str) -> Iterable[bytes]:
        cached = self.cache.get(text)
        if cached is not None:
            for chunk in cached:
                yield chunk
            return

        chunks: list[bytes] = []
        total_bytes = 0
        cacheable = bool(text) and self.max_items > 0 and self.max_bytes > 0
        for chunk in self.voice.synthesize_pcm(text):
            data = bytes(chunk)
            if cacheable:
                total_bytes += len(data)
                if total_bytes <= self.max_bytes:
                    chunks.append(data)
                else:
                    chunks.clear()
                    cacheable = False
            yield data

        if cacheable and chunks:
            self.cache[text] = tuple(chunks)
            self.order.append(text)
            while len(self.order) > self.max_items:
                old_text = self.order.popleft()
                self.cache.pop(old_text, None)

    def preload(self, text: str) -> None:
        if text and text not in self.cache:
            for _ in self.synthesize_pcm(text):
                pass


def resolve_torch_device(requested: str) -> str:
    import torch

    value = requested.strip().lower() or "auto"
    if value in {"auto", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("VOICE_TTS_DEVICE=cuda was requested, but torch CUDA is unavailable")
        return "cuda"
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"VOICE_TTS_DEVICE={requested} was requested, but torch CUDA is unavailable")
        device_index = value.split(":", 1)[1]
        if not device_index.isdigit():
            raise RuntimeError("VOICE_TTS_DEVICE must be auto, cpu, cuda, gpu, or cuda:<index>")
        if int(device_index) >= torch.cuda.device_count():
            raise RuntimeError(
                f"VOICE_TTS_DEVICE={requested} is not available; torch sees {torch.cuda.device_count()} CUDA device(s)"
            )
        torch.cuda.set_device(int(device_index))
        return value
    raise RuntimeError("VOICE_TTS_DEVICE must be auto, cpu, cuda, gpu, or cuda:<index>")


def resolve_melotts_speaker(model: Any) -> int:
    speaker = MELOTTS_SPEAKER.strip()
    if speaker.isdigit():
        return int(speaker)
    hps = getattr(model, "hps", None)
    data = getattr(hps, "data", None)
    speaker_ids = getattr(data, "spk2id", {}) or {}
    if speaker in speaker_ids:
        return int(speaker_ids[speaker])
    available = ", ".join(str(name) for name in speaker_ids.keys())
    raise RuntimeError(f"MeloTTS speaker '{speaker}' is not available. Available speakers: {available}")


def create_melotts_tts() -> TextToSpeech:
    from melo.api import TTS

    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
    except Exception:
        logging.getLogger("jieba").setLevel(logging.WARNING)

    require_file(MELOTTS_CONFIG_FILE)
    require_file(MELOTTS_CKPT_FILE)
    device = resolve_torch_device(VOICE_TTS_DEVICE)
    log(
        "MeloTTS config: "
        f"model={TTS_MODEL_DIR} language={MELOTTS_LANGUAGE} device={device} "
        f"speaker={MELOTTS_SPEAKER} speed={MELOTTS_SPEED} "
        f"sdp_ratio={MELOTTS_SDP_RATIO} noise_scale={MELOTTS_NOISE_SCALE} "
        f"noise_scale_w={MELOTTS_NOISE_SCALE_W}"
    )
    started = time.monotonic()
    init_output = io.StringIO()
    try:
        with redirect_stdout(init_output), redirect_stderr(init_output):
            model = TTS(
                language=MELOTTS_LANGUAGE,
                device=device,
                use_hf=False,
                config_path=str(MELOTTS_CONFIG_FILE),
                ckpt_path=str(MELOTTS_CKPT_FILE),
            )
    except Exception:
        detail = init_output.getvalue().strip()
        if detail:
            log(f"MeloTTS init output before failure: {detail[-2000:]}", level="error")
        raise
    if MELOTTS_DISABLE_BERT:
        setattr(model.hps.data, "disable_bert", True)
        log("MeloTTS BERT features disabled")
    speaker_id = resolve_melotts_speaker(model)
    sample_rate = int(getattr(model.hps.data, "sampling_rate", 44100) or 44100)
    log(
        "MeloTTS loaded: "
        f"speaker={MELOTTS_SPEAKER}:{speaker_id} sample_rate={sample_rate} "
        f"elapsed={time.monotonic() - started:.2f}s"
    )
    return MeloTextToSpeech(model, speaker_id, sample_rate)


def create_tts() -> TextToSpeech:
    if VOICE_TTS_ENGINE == "melotts":
        return wrap_tts(create_melotts_tts())
    if VOICE_TTS_ENGINE == "online":
        return wrap_tts(create_edge_tts())
    raise RuntimeError(f"VOICE_TTS_ENGINE '{VOICE_TTS_ENGINE}' is not supported")


def create_remote_tts() -> TextToSpeech:
    log(f"remote TTS service: {TTS_SERVICE_URL}")
    return RemoteTextToSpeech()


def wrap_tts(voice: TextToSpeech) -> TextToSpeech:
    if not TTS_CACHE_ENABLED:
        return voice
    return CachedTextToSpeech(voice, TTS_CACHE_MAX_ITEMS, TTS_CACHE_MAX_BYTES)


def preload_tts_cache(voice: TextToSpeech, *texts: str) -> None:
    if isinstance(voice, CachedTextToSpeech):
        for text in texts:
            voice.preload(text)


def warmup_tts(voice: TextToSpeech) -> None:
    for text in TTS_WARMUP_TEXTS:
        started = time.monotonic()
        chunks = 0
        bytes_total = 0
        for chunk in voice.synthesize_pcm(text):
            if chunk:
                chunks += 1
                bytes_total += len(chunk)
        log(
            "tts warmup: "
            f"text_chars={len(text)} chunks={chunks} bytes={bytes_total} "
            f"elapsed={time.monotonic() - started:.2f}s"
        )


def synthesize_edge_tts_audio(text: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        EDGE_TTS_VOICE,
        rate=EDGE_TTS_RATE,
        volume=EDGE_TTS_VOLUME,
        pitch=EDGE_TTS_PITCH,
        proxy=EDGE_TTS_PROXY,
        connect_timeout=EDGE_TTS_CONNECT_TIMEOUT_SECONDS,
        receive_timeout=EDGE_TTS_RECEIVE_TIMEOUT_SECONDS,
    )
    chunks: list[bytes] = []
    for message in communicate.stream_sync():
        if message.get("type") == "audio":
            chunks.append(bytes(message.get("data") or b""))
    if not chunks:
        raise RuntimeError("EdgeTTS returned no audio")
    return b"".join(chunks)


def synthesize_remote_wav(text: str) -> bytes:
    import httpx

    timeout = httpx.Timeout(
        connect=max(0.1, TTS_SERVICE_CONNECT_TIMEOUT_SECONDS),
        read=max(1.0, TTS_SERVICE_READ_TIMEOUT_SECONDS),
        write=15.0,
        pool=5.0,
    )
    response = service_http_client().post(TTS_SERVICE_URL, json={"text": text}, timeout=timeout)
    response.raise_for_status()
    route = response.headers.get("X-Chat2Me-TTS-Route", "")
    engine = response.headers.get("X-Chat2Me-TTS-Engine", "")
    model = response.headers.get("X-Chat2Me-TTS-Model", "")
    fallback = response.headers.get("X-Chat2Me-TTS-Fallback", "")
    log(f"remote tts result: route={route} engine={engine} model={model} fallback={fallback}", level="debug")
    return response.content


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise RuntimeError("remote TTS must return 16-bit PCM wav")
    if channels == 1:
        return frames, int(sample_rate)
    samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono_samples = samples.mean(axis=1).astype(np.int16)
    return mono_samples.tobytes(), int(sample_rate)


def decode_audio_to_pcm(audio: bytes, sample_rate: int) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"TTS audio decode failed: {detail}")
    return result.stdout


def create_edge_tts() -> TextToSpeech:
    if VOICE_TTS_MODEL != "edge-tts":
        raise RuntimeError("online TTS only supports VOICE_TTS_MODEL=edge-tts")
    log(
        "EdgeTTS config: "
        f"model={VOICE_TTS_MODEL} voice={EDGE_TTS_VOICE} rate={EDGE_TTS_RATE} "
        f"volume={EDGE_TTS_VOLUME} pitch={EDGE_TTS_PITCH} sample_rate={EDGE_TTS_SAMPLE_RATE}"
    )
    return EdgeTextToSpeech()


def tensor_audio_bytes(audio: Any) -> bytes:
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
    except Exception:
        pass
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def tts_playback_mode() -> str:
    if TTS_PLAYBACK_MODE in {"buffered", "hybrid", "stream"}:
        return TTS_PLAYBACK_MODE
    raise RuntimeError("TTS_PLAYBACK_MODE must be buffered, hybrid, or stream")


def write_player_stdin(player: subprocess.Popen[bytes], chunks: Iterable[bytes]) -> None:
    assert player.stdin is not None
    try:
        for chunk in chunks:
            if chunk:
                player.stdin.write(chunk)
    finally:
        player.stdin.close()


def run_aplay(command: list[str], input_data: bytes) -> None:
    deadline = time.monotonic() + min(max(0.0, TTS_PLAYBACK_RETRY_SECONDS), TTS_PLAYER_TIMEOUT_SECONDS)
    last_return_code: int | None = None
    while True:
        result = subprocess.run(command, input=input_data, check=False, timeout=TTS_PLAYER_TIMEOUT_SECONDS)
        if result.returncode == 0:
            return
        last_return_code = result.returncode
        if time.monotonic() >= deadline:
            raise RuntimeError(f"aplay exited with status {last_return_code}")
        time.sleep(0.25)


def play_pcm_chunks(command: list[str], chunks: Iterable[bytes]) -> None:
    with subprocess.Popen(command, stdin=subprocess.PIPE) as player:
        write_player_stdin(player, chunks)
        return_code = player.wait(timeout=TTS_PLAYER_TIMEOUT_SECONDS)
    if return_code != 0:
        raise RuntimeError(f"aplay exited with status {return_code}")


def speak(text: str, voice: TextToSpeech) -> None:
    if not text:
        return
    log(f"{VOICE_TTS_ENGINE} tts: text={text}")

    def playback_command() -> list[str]:
        return [
            "aplay",
            "-q",
            "-D",
            OUTPUT_DEVICE,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(voice.config.sample_rate),
        ]

    mode = tts_playback_mode()
    if mode == "buffered":
        started = time.monotonic()
        pcm = b"".join(chunk for chunk in voice.synthesize_pcm(text) if chunk)
        duration = len(pcm) / max(1, voice.config.sample_rate * 2)
        log(
            f"tts buffered pcm: bytes={len(pcm)} duration={duration:.2f}s synth={time.monotonic() - started:.2f}s",
            level="debug",
        )
        run_aplay(playback_command(), pcm)
        return

    if mode == "hybrid":
        started = time.monotonic()
        min_bytes = int(max(0.0, TTS_PREBUFFER_SECONDS) * voice.config.sample_rate * 2)
        iterator = iter(voice.synthesize_pcm(text))
        buffered: list[bytes] = []
        buffered_bytes = 0
        for chunk in iterator:
            if not chunk:
                continue
            buffered.append(chunk)
            buffered_bytes += len(chunk)
            if buffered_bytes >= min_bytes:
                break
        duration = buffered_bytes / max(1, voice.config.sample_rate * 2)
        log(
            f"tts hybrid prebuffer: bytes={buffered_bytes} duration={duration:.2f}s wait={time.monotonic() - started:.2f}s",
            level="debug",
        )
        play_pcm_chunks(playback_command(), chain(buffered, iterator))
        return

    play_pcm_chunks(playback_command(), voice.synthesize_pcm(text))


def spoken_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if SPEECH_TTS_MAX_CHARS <= 0 or len(cleaned) <= SPEECH_TTS_MAX_CHARS:
        return cleaned

    parts = re.findall(r"[^。！？!?]+[。！？!?]?", cleaned)
    output = ""
    for part in parts:
        if not part:
            continue
        if len(output) + len(part) > SPEECH_TTS_MAX_CHARS:
            break
        output += part

    if not output:
        output = cleaned[:SPEECH_TTS_MAX_CHARS].rstrip("，,、；;：: ")
    output = output.rstrip()
    if output and output[-1] not in "。！？!?":
        output += "。"
    return output


def speak_pausing_input(
    audio: Any,
    text: str,
    voice: TextToSpeech,
    display: DisplayClient,
    expression: str | None = None,
) -> None:
    if not text:
        return
    display.set_state("speaking", text, expression=expression)
    audio.stop()
    try:
        speak(text, voice)
    finally:
        audio.start()
        display.set_state("listening")


def drain_audio(audio: Any, seconds: float) -> None:
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        audio.read(chunk)


def mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        raise RuntimeError("expected 6-channel ReSpeaker v3.0 audio, got mono input")
    if samples.shape[1] != INPUT_CHANNELS:
        raise RuntimeError(f"expected {INPUT_CHANNELS}-channel ReSpeaker v3.0 audio, got {samples.shape[1]} channels")
    return samples[:, INPUT_CHANNEL_INDEX]


def read_mono(audio: Any, frames: int) -> np.ndarray:
    samples, overflowed = audio.read(frames)
    if overflowed:
        log("audio input overflowed; command audio may be clipped", level="warning")
    return mono(samples).reshape(-1)


def audio_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def feed_asr(
    recognizer: StreamingRecognizer,
    stream: Any,
    samples: np.ndarray,
    last_text: str,
) -> str:
    recognizer.accept_waveform(stream, SAMPLE_RATE, samples)
    result = recognizer.decode_ready(stream)
    if result and result != last_text:
        log(f"asr partial: {result}", level="debug")
        return result
    return last_text


def empty_voice_activity_state() -> VoiceActivityState:
    return {"voice_activity": None, "speech_detected": None}


def read_voice_activity_state(
    voice_activity_probe: Callable[[], VoiceActivityState | bool | None] | None,
) -> VoiceActivityState:
    if not RESPEAKER_VAD_GATE_ENABLED or voice_activity_probe is None:
        return empty_voice_activity_state()
    try:
        activity = voice_activity_probe()
    except Exception as exc:
        log(f"respeaker voice activity read failed: {exc}", level="debug")
        return empty_voice_activity_state()

    if isinstance(activity, dict):
        return {
            "voice_activity": activity.get("voice_activity"),
            "speech_detected": activity.get("speech_detected"),
        }
    return {
        "voice_activity": activity,
        "speech_detected": None,
    }


def calibrate_asr_noise(audio: Any, frames: int) -> tuple[float, list[tuple[np.ndarray, float]]]:
    chunks: list[tuple[np.ndarray, float]] = []
    if not ASR_NOISE_GATE_ENABLED or ASR_NOISE_CALIBRATION_SECONDS <= 0:
        return SPEECH_RMS_THRESHOLD, chunks

    deadline = time.monotonic() + ASR_NOISE_CALIBRATION_SECONDS
    while time.monotonic() < deadline:
        samples = read_mono(audio, frames)
        chunks.append((samples.copy(), audio_rms(samples)))

    if not chunks:
        return SPEECH_RMS_THRESHOLD, chunks

    percentile = min(max(ASR_NOISE_GATE_PERCENTILE, 0.0), 100.0)
    noise_floor = float(np.percentile([rms for _, rms in chunks], percentile))
    threshold = max(SPEECH_RMS_THRESHOLD, noise_floor * (1.0 + ASR_NOISE_GATE_RATIO) + ASR_NOISE_GATE_OFFSET)
    log(f"asr noise gate: floor={noise_floor:.4f} threshold={threshold:.4f}")
    return threshold, chunks


def listen_command(
    audio: Any,
    ready_beep_path: Path | None,
    recognizer: StreamingRecognizer,
    play_ready_beep: bool = True,
    voice_activity_probe: Callable[[], VoiceActivityState | bool | None] | None = None,
) -> str:
    stream = recognizer.create_stream()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    last_text = ""
    speech_started = False
    max_rms = 0.0
    rms_active_chunks = 0
    vad_active_chunks = 0
    speech_detected_chunks = 0
    last_active_elapsed: float | None = None
    preroll_chunks = max(1, int(max(ASR_PREROLL_SECONDS, CHUNK_SECONDS) / CHUNK_SECONDS))
    pre_roll: deque[tuple[np.ndarray, float]] = deque(maxlen=preroll_chunks)

    if play_ready_beep and ready_beep_path is not None:
        drain_audio(audio, PRE_BEEP_DRAIN_SECONDS)
        play_wav(ready_beep_path)
        drain_audio(audio, POST_BEEP_DRAIN_SECONDS)
    log("listening for command")
    started = time.monotonic()
    gate_threshold, calibration_chunks = calibrate_asr_noise(audio, chunk)
    for samples, rms in calibration_chunks:
        max_rms = max(max_rms, rms)
        pre_roll.append((samples, rms))
    if ASR_NOISE_GATE_ENABLED and pre_roll and max(rms for _, rms in pre_roll) >= gate_threshold:
        speech_started = True
        for samples, _ in pre_roll:
            last_text = feed_asr(recognizer, stream, samples, last_text)
        pre_roll.clear()

    while time.monotonic() - started < COMMAND_TIMEOUT_SECONDS:
        samples = read_mono(audio, chunk)
        rms = audio_rms(samples)
        max_rms = max(max_rms, rms)
        activity_state = read_voice_activity_state(voice_activity_probe)
        vad_active = activity_state["voice_activity"]
        speech_detected = activity_state["speech_detected"]
        rms_active = rms >= gate_threshold
        if rms_active:
            rms_active_chunks += 1
        if vad_active:
            vad_active_chunks += 1
        if speech_detected:
            speech_detected_chunks += 1
        active = rms_active or bool(vad_active) or bool(speech_detected)

        if ASR_NOISE_GATE_ENABLED:
            if not speech_started:
                pre_roll.append((samples.copy(), rms))
                if active:
                    speech_started = True
                    for buffered_samples, _ in pre_roll:
                        last_text = feed_asr(recognizer, stream, buffered_samples, last_text)
                    pre_roll.clear()
            else:
                gated_samples = samples if active else np.zeros_like(samples)
                last_text = feed_asr(recognizer, stream, gated_samples, last_text)
        else:
            if rms >= SPEECH_RMS_THRESHOLD:
                speech_started = True
            last_text = feed_asr(recognizer, stream, samples, last_text)

        elapsed = time.monotonic() - started
        if active:
            last_active_elapsed = elapsed

        if not speech_started and not last_text and elapsed >= COMMAND_LEADING_SILENCE_SECONDS:
            break

        if elapsed < max(COMMAND_MIN_SECONDS, COMMAND_INITIAL_GRACE_SECONDS):
            continue

        if speech_started and last_active_elapsed is not None:
            trailing = elapsed - last_active_elapsed
            if trailing >= COMMAND_INITIAL_GRACE_SECONDS:
                break

        if recognizer.is_endpoint(stream) and (speech_started or last_text):
            break

    recognizer.input_finished(stream)
    final_text = (recognizer.decode_ready(stream) or last_text).strip()
    if final_text and voice_activity_probe is not None and speech_detected_chunks == 0:
        log(
            "asr rejected without speech detection: "
            f"text='{final_text}' rms_active_chunks={rms_active_chunks} "
            f"vad_active_chunks={vad_active_chunks} speech_detected_chunks={speech_detected_chunks}",
            level="info",
        )
        final_text = ""
    del stream
    gc.collect()
    log(
        "asr finished: "
        f"text='{final_text}' speech_started={speech_started} max_rms={max_rms:.4f} "
        f"rms_active_chunks={rms_active_chunks} vad_active_chunks={vad_active_chunks} "
        f"speech_detected_chunks={speech_detected_chunks} elapsed={time.monotonic() - started:.1f}s"
    )
    return final_text


def ask_core(text: str) -> dict[str, Any]:
    response = service_http_client().post(
        CORE_URL, json={"message": text}, timeout=CORE_REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def is_session_end(command: str) -> bool:
    # 退出词必须占据输入的主体：整句以外最多允许 2 个字的语气/称呼修饰
    # （"退下吧"、"那再见"），避免"怎么跟朋友说再见"这类长句被误判为退出。
    normalized = re.sub(r"[\s，。！？,.!?]", "", command)
    return any(
        phrase in normalized and len(normalized) <= len(phrase) + 2
        for phrase in SESSION_END_PHRASES
    )


def handle_conversation_turn(
    audio: Any,
    command: str,
    voice: TextToSpeech,
    display: DisplayClient,
) -> bool:
    if is_session_end(command):
        log(f"session end command: {command}")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, display)
        display.set_state("idle")
        return False

    log(f"recognized command: {command}")
    display.set_state("thinking", command)
    try:
        core_result = ask_core(command)
    except Exception as exc:
        log(f"core request failed: {exc}", level="warning")
        display.set_state("error", "core unavailable")
        speak_pausing_input(audio, CORE_UNAVAILABLE_RESPONSE, voice, display)
        return True

    answer = str(core_result.get("answer", "")).strip()
    route = str(core_result.get("route", "")).strip()
    log(f"core result: route={route} model={core_result.get('model')} latency_ms={core_result.get('latency_ms')}")
    expression = str(core_result.get("expression") or "").strip() or None
    if route == "invalid_input":
        log(f"invalid input discarded: {command}")
        display.set_state("idle")
        return False

    log(f"answer: {answer}")
    speech_answer = spoken_text(answer)
    if speech_answer != answer:
        log(f"answer shortened for speech: {speech_answer}")
    speak_pausing_input(audio, speech_answer, voice, display, expression=expression)
    if route == "session_end":
        log("session ended by core intent route")
        display.set_state("idle")
        return False
    return True
