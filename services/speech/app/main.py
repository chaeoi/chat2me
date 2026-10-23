from __future__ import annotations

import base64
import binascii
import io
import json
import os
import sys
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import httpx
import sounddevice as sd

from app.voice import (
    ASR_SERVICE_URL,
    CHUNK_SECONDS,
    CORE_URL,
    DISPLAY_SERIAL_BAUD,
    DISPLAY_TEXT_MAX_CHARS,
    INPUT_CHANNELS,
    INPUT_DEVICE,
    INPUT_DEVICE_REQUIRED,
    KWS_MODEL_DIR,
    MAX_SESSION_TURNS,
    POST_RESPONSE_DRAIN_SECONDS,
    SAMPLE_RATE,
    SESSION_END_RESPONSE,
    SESSION_IDLE_RESPONSE,
    TTS_SERVICE_URL,
    WAKE_RESPONSE,
    DisplayClient,
    create_kws,
    create_remote_asr,
    create_remote_tts,
    drain_audio,
    env_float,
    handle_conversation_turn,
    listen_command,
    log,
    read_mono,
    select_input_device,
    spoken_text,
    speak_pausing_input,
    wake_words_display,
    write_beep,
)
from app.respeaker import direction_answer_from_snapshot, direction_label, open_respeaker


SPEECH_HOST = os.getenv("SPEECH_HOST", "0.0.0.0")
SPEECH_PORT = int(os.getenv("SPEECH_PORT", "8090"))
DIRECTION_CHANGE_POLL_SECONDS = float(os.getenv("DIRECTION_CHANGE_POLL_SECONDS", "0.5").strip() or "0.5")
SPEECH_WAIT_LOG_SECONDS = env_float("SPEECH_WAIT_LOG_SECONDS")
SPEECH_WAIT_POLL_SECONDS = env_float("SPEECH_WAIT_POLL_SECONDS")
SPEECH_DIAGNOSTIC_MAX_BODY_BYTES = int(os.getenv("SPEECH_DIAGNOSTIC_MAX_BODY_BYTES", "25165824").strip() or "25165824")
SPEECH_DIAGNOSTIC_MAX_AUDIO_BYTES = int(os.getenv("SPEECH_DIAGNOSTIC_MAX_AUDIO_BYTES", "16777216").strip() or "16777216")
SPEECH_DIAGNOSTIC_TIMEOUT_SECONDS = float(os.getenv("SPEECH_DIAGNOSTIC_TIMEOUT_SECONDS", "180").strip() or "180")


STATE_LOCK = threading.Lock()
_UNSET = object()
CURRENT_STATE: dict[str, object] = {
    "state": "idle",
    "text": "",
    "expression": None,
    "direction": {
        "ok": False,
        "angle_degrees": None,
        "error": "not_reported",
        "updated_at": time.time(),
    },
    "seq": 0,
    "changed_at": time.time(),
}


class SpeechState(DisplayClient):
    def __init__(self) -> None:
        super().__init__("", DISPLAY_SERIAL_BAUD)
        self.audio_source = None

    @property
    def enabled(self) -> bool:
        return True

    def set_state(self, state: str, text: str = "", expression: str | None = None) -> None:
        text = text[:DISPLAY_TEXT_MAX_CHARS]
        update_speech_state(
            state=state,
            text=text,
            expression=expression,
            direction=direction_status(self.audio_source),
        )

class WakeHandler(BaseHTTPRequestHandler):
    recognizer = None
    voice = None
    display = None
    beep_path = None
    audio_source = None
    busy_lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"ok": True, "busy": WakeHandler.busy_lock.locked()})
            return
        if path == "/state":
            self._send_json(update_speech_state(direction=direction_status(WakeHandler.audio_source)))
            return
        self.send_response(404)
        self.end_headers()

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/diagnostics/turn":
            self._handle_diagnostic_turn()
            return

        if path != "/wake":
            self.send_response(404)
            self.end_headers()
            return

        if not WakeHandler.busy_lock.acquire(blocking=False):
            self.send_response(409)
            self.end_headers()
            return
        try:
            threading.Thread(
                target=run_session_thread,
                args=(
                    WakeHandler.recognizer,
                    WakeHandler.voice,
                    WakeHandler.display,
                    WakeHandler.beep_path,
                    WakeHandler.audio_source,
                ),
                daemon=True,
            ).start()
        except Exception:
            # run_session_thread 只有在成功启动后才会释放 busy_lock。
            WakeHandler.busy_lock.release()
            raise
        self.send_response(202)
        self.end_headers()

    def _read_json_body(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return {}
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > SPEECH_DIAGNOSTIC_MAX_BODY_BYTES:
            raise ValueError("diagnostic request body is too large")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise ValueError("diagnostic request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("diagnostic request body must be a JSON object")
        return payload

    def _handle_diagnostic_turn(self) -> None:
        try:
            payload = self._read_json_body()
            result = run_diagnostic_turn(payload)
            self._send_json(result)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except httpx.HTTPStatusError as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": "upstream_http_error",
                    "status_code": exc.response.status_code,
                    "body": exc.response.text[:800],
                },
                HTTPStatus.BAD_GATEWAY,
            )
        except Exception as exc:
            log(f"diagnostic turn failed: {exc}", level="warning")
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def decode_audio_wav_base64(value: object) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio_wav_base64 must be a non-empty base64 string")
    encoded = value.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        audio = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("audio_wav_base64 is not valid base64") from exc
    if len(audio) > SPEECH_DIAGNOSTIC_MAX_AUDIO_BYTES:
        raise ValueError("diagnostic audio is too large")
    return audio


def wav_metadata(wav_bytes: bytes) -> dict[str, object]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except wave.Error as exc:
        raise ValueError("audio_wav_base64 must contain a valid WAV file") from exc
    duration = frames / sample_rate if sample_rate else 0.0
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_seconds": round(duration, 3),
    }


def timed_post_asr(client: httpx.Client, wav_bytes: bytes) -> dict[str, object]:
    started = time.perf_counter()
    response = client.post(
        ASR_SERVICE_URL,
        files={"file": ("diagnostic.wav", wav_bytes, "audio/wav")},
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    body = response.json()
    return {
        "ok": True,
        "wall_ms": wall_ms,
        "body": body,
        "text": str(body.get("text") or "").strip() if isinstance(body, dict) else "",
    }


def timed_post_core(client: httpx.Client, text: str) -> dict[str, object]:
    payload: dict[str, object] = {"message": text}
    started = time.perf_counter()
    response = client.post(CORE_URL, json=payload)
    wall_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    body = response.json()
    return {
        "ok": True,
        "wall_ms": wall_ms,
        "body": body,
        "answer": str(body.get("answer") or "").strip() if isinstance(body, dict) else "",
    }


def timed_post_tts(client: httpx.Client, text: str, return_audio_base64: bool) -> dict[str, object]:
    started = time.perf_counter()
    first_audio_chunk_ms: int | None = None
    audio = bytearray()
    with client.stream(
        "POST",
        TTS_SERVICE_URL,
        json={"text": text},
    ) as response:
        headers = {
            "route": response.headers.get("x-chat2me-tts-route"),
            "engine": response.headers.get("x-chat2me-tts-engine"),
            "model": response.headers.get("x-chat2me-tts-model"),
            "content_type": response.headers.get("content-type"),
        }
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if chunk and first_audio_chunk_ms is None:
                first_audio_chunk_ms = int((time.perf_counter() - started) * 1000)
            audio.extend(chunk)
    payload: dict[str, object] = {
        "ok": True,
        "wall_ms": int((time.perf_counter() - started) * 1000),
        "time_to_first_audio_chunk_ms": first_audio_chunk_ms,
        "bytes": len(audio),
        "headers": headers,
    }
    if return_audio_base64:
        payload["audio_wav_base64"] = base64.b64encode(bytes(audio)).decode("ascii")
    return payload


def run_diagnostic_turn(payload: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    input_text = str(payload.get("text") or "").strip()
    audio_base64 = payload.get("audio_wav_base64")
    return_audio_base64 = bool(payload.get("return_audio_base64"))
    result: dict[str, object] = {
        "ok": True,
        "input": {},
        "asr": {"skipped": True},
    }

    timeout = httpx.Timeout(connect=5.0, read=SPEECH_DIAGNOSTIC_TIMEOUT_SECONDS, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        if audio_base64 is not None:
            wav_bytes = decode_audio_wav_base64(audio_base64)
            result["input"] = {"audio": wav_metadata(wav_bytes)}
            asr_result = timed_post_asr(client, wav_bytes)
            result["asr"] = asr_result
            input_text = asr_result["text"] or input_text
        else:
            result["input"] = {"text_chars": len(input_text)}

        if not input_text:
            raise ValueError("diagnostic turn requires text or audio that transcribes to text")

        core_result = timed_post_core(client, input_text)
        result["core"] = core_result
        answer = core_result["answer"]
        tts_input = spoken_text(answer) or answer
        if not tts_input:
            raise ValueError("core returned an empty answer")
        result["tts_input"] = {
            "text": tts_input,
            "text_chars": len(tts_input),
            "shortened": tts_input != answer,
        }
        result["tts"] = timed_post_tts(client, tts_input, return_audio_base64)

    result["total_latency_ms"] = int((time.perf_counter() - started) * 1000)
    log(
        "diagnostic turn completed: "
        f"asr_ms={dict(result.get('asr') or {}).get('wall_ms')} "
        f"core_ms={dict(result.get('core') or {}).get('wall_ms')} "
        f"tts_ms={dict(result.get('tts') or {}).get('wall_ms')} "
        f"total_ms={result['total_latency_ms']}"
    )
    return result


def direction_snapshot(audio_source) -> dict[str, object]:
    if audio_source is None:
        return {"ok": False, "source": "respeaker", "error": "unavailable", "updated_at": time.time()}
    return audio_source.snapshot()


def direction_status(audio_source) -> dict[str, object]:
    snapshot = direction_snapshot(audio_source)
    if not snapshot.get("ok"):
        return {
            "ok": False,
            "angle_degrees": None,
            "updated_at": snapshot.get("updated_at", time.time()),
            "error": snapshot.get("error", "unavailable"),
        }
    return {
        "ok": True,
        "angle_degrees": snapshot.get("angle_degrees"),
        "voice_activity": snapshot.get("voice_activity"),
        "speech_detected": snapshot.get("speech_detected"),
        "updated_at": snapshot.get("updated_at", time.time()),
    }


def normalize_direction(direction: dict[str, object]) -> dict[str, object]:
    ok = bool(direction.get("ok"))
    angle = direction.get("angle_degrees")
    if ok and isinstance(angle, (int, float)):
        return {
            "ok": True,
            "angle_degrees": int(round(float(angle))) % 360,
            "voice_activity": direction.get("voice_activity"),
            "speech_detected": direction.get("speech_detected"),
            "updated_at": direction.get("updated_at") or time.time(),
        }
    return {
        "ok": False,
        "angle_degrees": None,
        "error": str(direction.get("error") or "unavailable"),
        "updated_at": direction.get("updated_at") or time.time(),
    }


def current_state() -> dict[str, object]:
    with STATE_LOCK:
        payload = dict(CURRENT_STATE)
        payload["direction"] = dict(CURRENT_STATE.get("direction") or {})
        return payload


def direction_bucket(direction: dict[str, object]) -> str | None:
    if not direction.get("ok"):
        return None
    angle = direction.get("angle_degrees")
    if not isinstance(angle, (int, float)):
        return None
    return direction_label(float(angle))


def update_speech_state(
    *,
    state: str | None = None,
    text: str | None = None,
    expression: str | None | object = _UNSET,
    direction: dict[str, object] | None = None,
    force_seq: bool = False,
) -> dict[str, object]:
    with STATE_LOCK:
        changed = force_seq
        if state is not None and CURRENT_STATE["state"] != state:
            CURRENT_STATE["state"] = state
            changed = True
        if text is not None and CURRENT_STATE["text"] != text:
            CURRENT_STATE["text"] = text
            changed = True
        if expression is not _UNSET and CURRENT_STATE["expression"] != expression:
            CURRENT_STATE["expression"] = expression
            changed = True
        if direction is not None:
            normalized = normalize_direction(direction)
            old_direction = CURRENT_STATE.get("direction") or {}
            if direction_bucket(old_direction) != direction_bucket(normalized) or old_direction.get("ok") != normalized.get("ok"):
                changed = True
            CURRENT_STATE["direction"] = normalized
        if changed:
            CURRENT_STATE["seq"] = int(CURRENT_STATE["seq"]) + 1
            CURRENT_STATE["changed_at"] = time.time()
        payload = dict(CURRENT_STATE)
        payload["direction"] = dict(CURRENT_STATE.get("direction") or {})
        return payload


def watch_direction_changes(state_store: SpeechState) -> None:
    interval = max(0.2, DIRECTION_CHANGE_POLL_SECONDS)
    last_bucket: str | None = None
    while True:
        payload = update_speech_state(direction=direction_status(state_store.audio_source))
        bucket = direction_bucket(dict(payload.get("direction") or {}))
        if bucket != last_bucket:
            last_bucket = bucket
        time.sleep(interval)


def run_session_thread(recognizer, voice, display: SpeechState, beep_path: Path, audio_source) -> None:
    try:
        log("wake signal received")
        run_session(recognizer, voice, display, beep_path, audio_source)
    except Exception as exc:
        log(f"session failed: {exc}", level="warning")
        if display is not None:
            display.set_state("error", str(exc))
    finally:
        if display is not None:
            display.set_state("idle")
        WakeHandler.busy_lock.release()


def listen_for_wake(kws, chunk: int, display: SpeechState) -> str:
    stream = kws.create_stream()
    matched = ""
    last_error_log = 0.0
    logged_input_device: object = object()
    log(f"wake listener active: {wake_words_display()}")

    while not matched:
        input_device = wait_for_input_device(display)
        if input_device != logged_input_device:
            log(f"input device: {input_device if input_device is not None else 'default'}")
            logged_input_device = input_device
        try:
            with sd.InputStream(
                channels=INPUT_CHANNELS,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                device=input_device,
                blocksize=chunk,
            ) as audio:
                while not matched:
                    samples = read_mono(audio, chunk)
                    stream.accept_waveform(SAMPLE_RATE, samples)
                    while kws.is_ready(stream):
                        kws.decode_stream(stream)
                        result = kws.get_result(stream)
                        if not result:
                            continue
                        matched = result
                        log(f"wake keyword matched: {matched}")
                        break
        except Exception as exc:
            now = time.monotonic()
            if now - last_error_log >= SPEECH_WAIT_LOG_SECONDS:
                log(f"wake input stream unavailable; retrying: {exc}")
                last_error_log = now
            display.set_state("error", "audio input unavailable")
            time.sleep(SPEECH_WAIT_POLL_SECONDS)
            stream = kws.create_stream()

    return matched


def wait_for_input_device(display: SpeechState) -> int | str | None:
    logged_waiting = False
    while True:
        try:
            input_device = select_input_device(INPUT_DEVICE)
        except Exception as exc:
            input_device = None
            if not logged_waiting:
                log(f"audio input device lookup failed; retrying: {exc}")
                logged_waiting = True
        if input_device is not None or not INPUT_DEVICE_REQUIRED:
            return input_device

        if not logged_waiting:
            log(f"waiting for configured input device: {INPUT_DEVICE}")
            logged_waiting = True
        display.set_state("error", "audio input unavailable")
        time.sleep(SPEECH_WAIT_POLL_SECONDS)


def run_embedded_wake_loop(recognizer, voice, display: SpeechState, beep_path: Path, audio_source) -> None:
    log(f"loading wake-word model: {KWS_MODEL_DIR}")
    kws = create_kws()
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    display.set_state("idle")

    while True:
        listen_for_wake(kws, chunk, display)
        if not WakeHandler.busy_lock.acquire(blocking=False):
            log("wake ignored because a session is already running")
            continue
        run_session_thread(recognizer, voice, display, beep_path, audio_source)


def open_session_input(display: SpeechState, chunk: int) -> sd.InputStream:
    deadline = time.monotonic() + 5.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        audio: sd.InputStream | None = None
        input_device = wait_for_input_device(display)
        try:
            audio = sd.InputStream(
                channels=INPUT_CHANNELS,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                device=input_device,
                blocksize=chunk,
            )
            audio.start()
            return audio
        except Exception as exc:
            last_error = exc
            if audio is not None:
                try:
                    audio.close()
                except Exception:
                    pass
            log(f"speech input stream unavailable; retrying: {exc}", level="debug")
            time.sleep(0.25)

    if last_error is not None:
        raise last_error
    raise RuntimeError("speech input stream unavailable")


def run_session(recognizer, voice, display: SpeechState, beep_path: Path, audio_source) -> None:
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    audio = open_session_input(display, chunk)
    voice_activity_probe = getattr(audio_source, "activity_state", None) or getattr(audio_source, "is_voice_active", None)
    try:
        display.set_state("listening", "wake")
        speak_pausing_input(audio, WAKE_RESPONSE, voice, display)
        drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        for turn in range(1, MAX_SESSION_TURNS + 1):
            log(f"conversation turn {turn}/{MAX_SESSION_TURNS}")
            command = listen_command(
                audio,
                beep_path,
                recognizer,
                play_ready_beep=False,
                voice_activity_probe=voice_activity_probe,
            )
            if not command:
                log("conversation idle timeout")
                if SESSION_IDLE_RESPONSE:
                    speak_pausing_input(audio, SESSION_IDLE_RESPONSE, voice, display)
                display.set_state("idle")
                return
            direction_payload = update_speech_state(direction=direction_status(audio_source))
            direction_reply = direction_answer_from_snapshot(command, dict(direction_payload.get("direction") or {}))
            if direction_reply is not None:
                log(f"direction answer: {direction_reply}")
                speak_pausing_input(audio, direction_reply, voice, display)
                drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)
                continue
            if not handle_conversation_turn(audio, command, voice, display):
                return
            drain_audio(audio, POST_RESPONSE_DRAIN_SECONDS)

        log("conversation reached max turns")
        speak_pausing_input(audio, SESSION_END_RESPONSE, voice, display)
        display.set_state("idle")
    finally:
        audio.close()


def main() -> None:
    log(f"core url: {CORE_URL}")
    log("using remote ASR service")
    recognizer = create_remote_asr()
    log("using remote TTS service")
    voice = create_remote_tts()
    audio_source = open_respeaker()
    beep_path = Path("/tmp/chat2me_wake.wav")
    write_beep(beep_path)
    display = SpeechState()
    display.audio_source = audio_source
    threading.Thread(target=watch_direction_changes, args=(display,), daemon=True).start()

    WakeHandler.recognizer = recognizer
    WakeHandler.voice = voice
    WakeHandler.display = display
    WakeHandler.beep_path = beep_path
    WakeHandler.audio_source = audio_source

    display.set_state("idle")
    server = ThreadingHTTPServer((SPEECH_HOST, SPEECH_PORT), WakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"speech service listening on {SPEECH_HOST}:{SPEECH_PORT}")
    run_embedded_wake_loop(recognizer, voice, display, beep_path, audio_source)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}", level="error")
        sys.exit(1)
