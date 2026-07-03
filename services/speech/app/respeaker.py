from __future__ import annotations

import os
import re
import struct
import threading
import time
from typing import Any

from app.common import direction_label, direction_sector, env_bool, env_float, env_value, log

try:
    import usb.core
    import usb.util
except ImportError:  # pragma: no cover - handled at runtime in slim environments
    usb = None  # type: ignore[assignment]


PARAMETERS = {
    "AECFREEZEONOFF": (18, 7, "int", "rw"),
    "AECNORM": (18, 19, "float", "rw"),
    "HPFONOFF": (18, 27, "int", "rw"),
    "RT60ONOFF": (18, 28, "int", "rw"),
    "AECSILENCELEVEL": (18, 30, "float", "rw"),
    "AGCONOFF": (19, 0, "int", "rw"),
    "AGCMAXGAIN": (19, 1, "float", "rw"),
    "AGCDESIREDLEVEL": (19, 2, "float", "rw"),
    "AGCGAIN": (19, 3, "float", "rw"),
    "AGCTIME": (19, 4, "float", "rw"),
    "CNIONOFF": (19, 5, "int", "rw"),
    "FREEZEONOFF": (19, 6, "int", "rw"),
    "STATNOISEONOFF": (19, 8, "int", "rw"),
    "GAMMA_NS": (19, 9, "float", "rw"),
    "MIN_NS": (19, 10, "float", "rw"),
    "NONSTATNOISEONOFF": (19, 11, "int", "rw"),
    "GAMMA_NN": (19, 12, "float", "rw"),
    "MIN_NN": (19, 13, "float", "rw"),
    "ECHOONOFF": (19, 14, "int", "rw"),
    "GAMMA_E": (19, 15, "float", "rw"),
    "GAMMA_ETAIL": (19, 16, "float", "rw"),
    "GAMMA_ENL": (19, 17, "float", "rw"),
    "NLATTENONOFF": (19, 18, "int", "rw"),
    "NLAEC_MODE": (19, 20, "int", "rw"),
    "TRANSIENTONOFF": (19, 29, "int", "rw"),
    "VOICEACTIVITY": (19, 32, "int", "ro"),
    "STATNOISEONOFF_SR": (19, 33, "int", "rw"),
    "NONSTATNOISEONOFF_SR": (19, 34, "int", "rw"),
    "GAMMA_NS_SR": (19, 35, "float", "rw"),
    "GAMMA_NN_SR": (19, 36, "float", "rw"),
    "MIN_NS_SR": (19, 37, "float", "rw"),
    "MIN_NN_SR": (19, 38, "float", "rw"),
    "GAMMAVAD_SR": (19, 39, "float", "rw"),
    "SPEECHDETECTED": (19, 22, "int", "ro"),
    "DOAANGLE": (21, 0, "int", "ro"),
}

DIRECTION_QUERY_RE = re.compile(
    r"(我在你(的)?(哪边|哪一边|哪儿|哪里|什么方向|哪个方向|左边|右边|前面|后面)|"
    r"你(能|可以)?(听出|判断|知道)?我在(哪边|哪一边|哪儿|哪里|什么方向|哪个方向))"
)


def env_int_base(key: str) -> int:
    try:
        return int(env_value(key), 0)
    except ValueError:
        raise RuntimeError(f"{key} must be an integer in runtime.env") from None


def optional_env_bool(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(env_bool(name))


def optional_env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        raise RuntimeError(f"{name} must be a number in runtime.env") from None


def is_direction_query(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？、,.!?]", "", text).replace("您", "你")
    normalized = re.sub(r"(哪|那){2,}", r"\1", normalized)
    return bool(DIRECTION_QUERY_RE.search(normalized))


class ReSpeakerAudioSource:
    TIMEOUT_MS = 2000

    def __init__(self, dev: Any) -> None:
        self.dev = dev
        self.front_offset_degrees = env_float("RESPEAKER_DOA_FRONT_OFFSET_DEGREES")
        self.clockwise = env_bool("RESPEAKER_DOA_CLOCKWISE")
        self._lock = threading.Lock()

    def close(self) -> None:
        if usb is not None:
            with self._lock:
                usb.util.dispose_resources(self.dev)

    def read(self, name: str) -> int | float:
        parameter = PARAMETERS[name]
        parameter_id, offset, value_type, _ = parameter
        command = 0x80 | int(offset)
        if value_type == "int":
            command |= 0x40
        with self._lock:
            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                command,
                int(parameter_id),
                8,
                self.TIMEOUT_MS,
            )
        raw = bytes(response)
        integer, exponent = struct.unpack("ii", raw)
        if value_type == "int":
            return integer
        return integer * (2.0**exponent)

    def write(self, name: str, value: int | float) -> None:
        parameter = PARAMETERS[name]
        parameter_id, offset, value_type, mode = parameter
        if mode != "rw":
            raise ValueError(f"{name} is read-only")
        if value_type == "int":
            payload = struct.pack("iii", int(offset), int(value), 1)
        else:
            payload = struct.pack("ifi", int(offset), float(value), 0)
        with self._lock:
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                0,
                int(parameter_id),
                payload,
                self.TIMEOUT_MS,
            )

    def apply_tuning(self) -> None:
        if not env_bool("RESPEAKER_TUNING_ENABLED"):
            return

        writes: list[tuple[str, int | float]] = [
            ("HPFONOFF", env_int_base("RESPEAKER_HPF_MODE")),
            ("AGCONOFF", int(env_bool("RESPEAKER_AGC_ENABLED"))),
            ("AGCMAXGAIN", env_float("RESPEAKER_AGC_MAX_GAIN")),
            ("AGCDESIREDLEVEL", env_float("RESPEAKER_AGC_DESIRED_LEVEL")),
            ("AGCTIME", env_float("RESPEAKER_AGC_TIME_SECONDS")),
            ("STATNOISEONOFF", int(env_bool("RESPEAKER_STATIONARY_NOISE_SUPPRESSION"))),
            ("NONSTATNOISEONOFF", int(env_bool("RESPEAKER_NONSTATIONARY_NOISE_SUPPRESSION"))),
            ("GAMMA_NS", env_float("RESPEAKER_STATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("GAMMA_NN", env_float("RESPEAKER_NONSTATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("MIN_NS", env_float("RESPEAKER_STATIONARY_NOISE_SUPPRESSION_FLOOR")),
            ("MIN_NN", env_float("RESPEAKER_NONSTATIONARY_NOISE_SUPPRESSION_FLOOR")),
            ("STATNOISEONOFF_SR", int(env_bool("RESPEAKER_ASR_STATIONARY_NOISE_SUPPRESSION"))),
            ("NONSTATNOISEONOFF_SR", int(env_bool("RESPEAKER_ASR_NONSTATIONARY_NOISE_SUPPRESSION"))),
            ("GAMMA_NS_SR", env_float("RESPEAKER_ASR_STATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("GAMMA_NN_SR", env_float("RESPEAKER_ASR_NONSTATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("MIN_NS_SR", env_float("RESPEAKER_ASR_STATIONARY_NOISE_SUPPRESSION_FLOOR")),
            ("MIN_NN_SR", env_float("RESPEAKER_ASR_NONSTATIONARY_NOISE_SUPPRESSION_FLOOR")),
            ("GAMMAVAD_SR", env_float("RESPEAKER_VAD_THRESHOLD_DB")),
            ("ECHOONOFF", int(env_bool("RESPEAKER_ECHO_SUPPRESSION_ENABLED"))),
            ("GAMMA_E", env_float("RESPEAKER_ECHO_SUPPRESSION_LEVEL")),
            ("GAMMA_ETAIL", env_float("RESPEAKER_ECHO_TAIL_SUPPRESSION_LEVEL")),
            ("GAMMA_ENL", env_float("RESPEAKER_NONLINEAR_ECHO_SUPPRESSION_LEVEL")),
            ("NLATTENONOFF", int(env_bool("RESPEAKER_NONLINEAR_AEC_ENABLED"))),
            ("NLAEC_MODE", env_int_base("RESPEAKER_NONLINEAR_AEC_MODE")),
            ("TRANSIENTONOFF", int(env_bool("RESPEAKER_TRANSIENT_ECHO_SUPPRESSION_ENABLED"))),
        ]
        optional_writes: tuple[tuple[str, int | float | None], ...] = (
            ("AECFREEZEONOFF", optional_env_bool("RESPEAKER_AEC_FREEZE_ENABLED")),
            ("AECNORM", optional_env_float("RESPEAKER_AEC_NORM")),
            ("RT60ONOFF", optional_env_bool("RESPEAKER_RT60_ENABLED")),
            ("AECSILENCELEVEL", optional_env_float("RESPEAKER_AEC_SILENCE_LEVEL")),
            ("AGCGAIN", optional_env_float("RESPEAKER_AGC_FIXED_GAIN")),
            ("CNIONOFF", optional_env_bool("RESPEAKER_COMFORT_NOISE_ENABLED")),
            ("FREEZEONOFF", optional_env_bool("RESPEAKER_BEAMFORMER_FREEZE_ENABLED")),
        )
        writes.extend((name, value) for name, value in optional_writes if value is not None)

        applied: dict[str, int | float] = {}
        for name, value in writes:
            try:
                self.write(name, value)
                applied[name] = value
            except Exception as exc:
                log(f"respeaker tuning write failed: {name}={value}: {exc}", level="warning")
        log(f"respeaker tuning applied: {applied}")

    def normalize_angle(self, raw_angle: int) -> int:
        angle = (raw_angle - self.front_offset_degrees) % 360
        if not self.clockwise:
            angle = (-angle) % 360
        return int(round(angle)) % 360

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        try:
            raw_angle = int(self.read("DOAANGLE"))
            angle = self.normalize_angle(raw_angle)
        except Exception as exc:
            return {
                "ok": False,
                "source": "respeaker",
                "error": str(exc),
                "updated_at": now,
            }

        activity = self.activity_state()
        voice_activity = activity["voice_activity"]
        speech_detected = activity["speech_detected"]

        sector = direction_sector(angle)
        return {
            "ok": True,
            "source": "respeaker",
            "raw_angle_degrees": raw_angle,
            "angle_degrees": angle,
            "sector": sector["code"],
            "label": sector["label"],
            "voice_activity": voice_activity,
            "speech_detected": speech_detected,
            "coordinate": {
                "zero": "front",
                "positive": "clockwise",
                "unit": "degrees",
                "front_offset_degrees": self.front_offset_degrees,
                "device_clockwise": self.clockwise,
            },
            "updated_at": now,
        }

    def activity_state(self) -> dict[str, bool | None]:
        try:
            voice_activity: bool | None = bool(self.read("VOICEACTIVITY"))
        except Exception as exc:
            log(f"respeaker voice activity read failed: {exc}", level="debug")
            voice_activity = None

        try:
            speech_detected: bool | None = bool(self.read("SPEECHDETECTED"))
        except Exception as exc:
            log(f"respeaker speech detected read failed: {exc}", level="debug")
            speech_detected = None

        return {
            "voice_activity": voice_activity,
            "speech_detected": speech_detected,
        }

    def is_voice_active(self) -> bool | None:
        return self.activity_state()["voice_activity"]

    def answer_direction(self) -> str:
        snapshot = self.snapshot()
        if not snapshot.get("ok"):
            return "我现在读不到麦克风方向信息。"
        return f"您在我的{snapshot['label']}。"


def _open_respeaker_once(*, log_missing: bool = True) -> ReSpeakerAudioSource | None:
    vendor_id = env_int_base("RESPEAKER_USB_VENDOR_ID")
    product_id = env_int_base("RESPEAKER_USB_PRODUCT_ID")
    try:
        dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    except Exception as exc:
        if log_missing:
            log(f"respeaker control unavailable: {exc}", level="warning")
        return None
    if dev is None:
        if log_missing:
            log(f"respeaker control device not found: vid=0x{vendor_id:04x} pid=0x{product_id:04x}")
        return None

    source = ReSpeakerAudioSource(dev)
    source.apply_tuning()
    snapshot = source.snapshot()
    log(f"respeaker control ready: {snapshot}")
    return source


class ReSpeakerAudioSourceManager:
    def __init__(self, retry_seconds: float = 5.0) -> None:
        self.retry_seconds = max(1.0, retry_seconds)
        self._source: ReSpeakerAudioSource | None = None
        self._lock = threading.Lock()
        self._next_open_at = 0.0
        self._last_missing_log_at = -30.0

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def _open_locked(self) -> ReSpeakerAudioSource | None:
        if self._source is not None:
            return self._source

        now = time.monotonic()
        if now < self._next_open_at:
            return None

        log_missing = now - self._last_missing_log_at >= 30.0
        source = _open_respeaker_once(log_missing=log_missing)
        if source is None:
            if log_missing:
                self._last_missing_log_at = now
            self._next_open_at = now + self.retry_seconds
            return None

        self._source = source
        self._next_open_at = 0.0
        self._last_missing_log_at = 0.0
        return source

    def _drop_source(self, source: ReSpeakerAudioSource, reason: str) -> None:
        with self._lock:
            if self._source is source:
                log(f"respeaker control lost; will retry: {reason}", level="warning")
                self._close_locked()
                self._next_open_at = time.monotonic() + self.retry_seconds

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            source = self._open_locked()
        if source is None:
            return {
                "ok": False,
                "source": "respeaker",
                "error": "unavailable",
                "updated_at": time.time(),
            }

        snapshot = source.snapshot()
        if not snapshot.get("ok"):
            self._drop_source(source, str(snapshot.get("error") or "read failed"))
        return snapshot

    def answer_direction(self) -> str:
        snapshot = self.snapshot()
        if not snapshot.get("ok"):
            return "我现在读不到麦克风方向信息。"
        return f"您在我的{snapshot['label']}。"

    def is_voice_active(self) -> bool | None:
        with self._lock:
            source = self._open_locked()
        if source is None:
            return None
        active = source.is_voice_active()
        if active is None:
            self._drop_source(source, "voice activity read failed")
        return active

    def activity_state(self) -> dict[str, bool | None]:
        with self._lock:
            source = self._open_locked()
        if source is None:
            return {"voice_activity": None, "speech_detected": None}
        activity = source.activity_state()
        if activity["voice_activity"] is None and activity["speech_detected"] is None:
            self._drop_source(source, "activity state read failed")
        return activity


def open_respeaker() -> ReSpeakerAudioSourceManager | None:
    if not env_bool("RESPEAKER_ENABLED"):
        log("respeaker control disabled")
        return None
    if usb is None:
        log("respeaker control unavailable: pyusb is not installed", level="warning")
        return None
    return ReSpeakerAudioSourceManager(env_float("RESPEAKER_RECONNECT_SECONDS", 5))


def direction_answer(text: str, source: ReSpeakerAudioSource | ReSpeakerAudioSourceManager | None) -> str | None:
    if not is_direction_query(text):
        return None
    if source is None:
        return "我现在读不到麦克风方向信息。"
    return source.answer_direction()


def direction_answer_from_snapshot(text: str, snapshot: dict[str, Any]) -> str | None:
    if not is_direction_query(text):
        return None
    if not snapshot.get("ok"):
        return "我现在读不到麦克风方向信息。"
    angle = snapshot.get("angle_degrees")
    if angle is None:
        return "我现在读不到麦克风方向信息。"
    try:
        label = direction_label(float(angle))
    except (TypeError, ValueError):
        return "我现在读不到麦克风方向信息。"
    return f"您在我的{label}。"
