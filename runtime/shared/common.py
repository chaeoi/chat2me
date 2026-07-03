from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


def load_runtime_env() -> None:
    path = Path(os.getenv("RUNTIME_CONFIG_PATH", "/app/config/runtime.env"))
    if not path.is_file():
        return
    protected_keys = set(os.environ)
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z0-9_]+", key):
            continue
        values[key] = value.strip()
    for key, value in values.items():
        if key not in protected_keys:
            os.environ[key] = value


load_runtime_env()


def quiet_http_client_logging() -> None:
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


quiet_http_client_logging()


def env_value(key: str, *, allow_empty: bool = False, default: str | None = None) -> str:
    # 未设置 -> default（无 default 则报错）；显式空值在 allow_empty 时原样返回，
    # 否则同样回退到 default（无 default 则报错）。
    value = os.getenv(key)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"{key} must be set in runtime.env")
    value = value.strip()
    if not value and not allow_empty:
        if default is not None:
            return default
        raise RuntimeError(f"{key} must not be empty in runtime.env")
    return value


def env_int(key: str, default: int | None = None) -> int:
    value = env_value(key, default=None if default is None else str(default))
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"{key} must be an integer in runtime.env") from None


def env_float(key: str, default: float | None = None) -> float:
    value = env_value(key, default=None if default is None else str(default))
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{key} must be a number in runtime.env") from None


def env_bool(key: str, default: bool | None = None) -> bool:
    value = env_value(key, default=None if default is None else ("1" if default else "0")).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{key} must be a boolean in runtime.env")


def env_csv(key: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in env_value(key).split(",") if item.strip())


DIRECTION_SECTORS = (
    ("front", "正前方"),
    ("front_right", "右前方"),
    ("right", "右侧"),
    ("back_right", "右后方"),
    ("back", "正后方"),
    ("back_left", "左后方"),
    ("left", "左侧"),
    ("front_left", "左前方"),
)


def direction_sector(angle: int | float) -> dict[str, str]:
    code, label = DIRECTION_SECTORS[int(((float(angle) + 22.5) % 360) // 45)]
    return {"code": code, "label": label}


def direction_label(angle: int | float) -> str:
    return direction_sector(angle)["label"]


LOG_LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}
LOG_LEVEL_ALIASES = {
    "warn": "warning",
    "err": "error",
}
LOG_DIR = Path("/app/log")
LOG_LOCK = threading.Lock()
LOG_FILE_ERROR_REPORTED = False


def normalize_log_level(value: str | None, default: str) -> str:
    normalized = (value or default).strip().lower()
    normalized = LOG_LEVEL_ALIASES.get(normalized, normalized)
    if normalized in LOG_LEVELS:
        return normalized
    return default


def log_threshold(env_key: str, default: str) -> int:
    return LOG_LEVELS[normalize_log_level(os.getenv(env_key), default)]


def log_file_path(role: str) -> Path:
    return LOG_DIR / f"{role}.log"


def log(message: str, *, level: str = "info") -> None:
    global LOG_FILE_ERROR_REPORTED

    level_name = normalize_log_level(level, "info")
    level_value = LOG_LEVELS[level_name]
    role = os.getenv("VOICE_ROLE", "chat2me")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    file_line = f"{timestamp} [{level_name}] [{role}] {message}"
    console_line = f"[{role}] {level_name}: {message}"

    if level_value >= log_threshold("CHAT2ME_LOG_LEVEL", "info"):
        path = log_file_path(role)
        try:
            with LOG_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as file:
                    file.write(file_line + "\n")
        except OSError as exc:
            if not LOG_FILE_ERROR_REPORTED:
                LOG_FILE_ERROR_REPORTED = True
                print(f"[{role}] warning: log file write failed: {exc}", file=sys.stderr, flush=True)

    if level_value >= log_threshold("CHAT2ME_CONSOLE_LOG_LEVEL", "warning"):
        stream = sys.stderr if level_value >= LOG_LEVELS["warning"] else sys.stdout
        print(console_line, file=stream, flush=True)


class DisplayClient:
    def __init__(
        self,
        port: str,
        baud: int,
        text_max_chars: int | None = None,
        retry_seconds: float | None = None,
        port_resolver: Callable[[], str] | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.text_max_chars = env_int("DISPLAY_TEXT_MAX_CHARS") if text_max_chars is None else text_max_chars
        self.retry_seconds = env_float("DISPLAY_SERIAL_RETRY_SECONDS") if retry_seconds is None else retry_seconds
        self._port_resolver = port_resolver
        self._serial: Any | None = None
        self._lock = threading.Lock()
        self._disabled_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.port or self._port_resolver is not None)

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _resolve_port(self) -> str:
        if self._port_resolver is None:
            return self.port
        port = self._port_resolver()
        if port and port != self.port:
            self.port = port
            log(f"display serial resolved: {port}")
        return port

    def set_state(self, state: str, text: str = "") -> None:
        if not self.enabled:
            return
        if time.monotonic() < self._disabled_until:
            return
        payload = {
            "state": state,
            "text": text[: self.text_max_chars],
            "ts": int(time.time()),
        }
        line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            import serial
        except ImportError as exc:
            log(f"display serial dependency unavailable: {exc}", level="warning")
            self._disabled_until = time.monotonic() + self.retry_seconds
            return

        with self._lock:
            try:
                if self._serial is None or not self._serial.is_open:
                    port = self._resolve_port()
                    if not port:
                        raise serial.SerialException("display serial port unavailable")
                    serial_conn = serial.Serial()
                    serial_conn.port = port
                    serial_conn.baudrate = self.baud
                    serial_conn.timeout = 0
                    serial_conn.write_timeout = 1
                    serial_conn.rtscts = False
                    serial_conn.dsrdtr = False
                    serial_conn.dtr = False
                    serial_conn.rts = False
                    try:
                        serial_conn.open()
                        serial_conn.dtr = False
                        serial_conn.rts = False
                        serial_conn.reset_input_buffer()
                        serial_conn.reset_output_buffer()
                    except serial.SerialException:
                        try:
                            serial_conn.close()
                        except Exception:
                            pass
                        raise
                    self._serial = serial_conn
                    log(f"display serial opened: {port}")
                    time.sleep(0.25)
                written = self._serial.write(line)
                self._serial.flush()
                if written != len(line):
                    raise serial.SerialTimeoutException(
                        f"display serial partial write: {written}/{len(line)} bytes"
                    )
            except (serial.SerialException, OSError) as exc:
                log(f"display serial write failed: {exc}", level="warning")
                self._close_locked()
                self._disabled_until = time.monotonic() + self.retry_seconds
