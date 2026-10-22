from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

from app.common import DisplayClient, env_float, env_int, env_value, log


DISPLAY_SERIAL_PORT = env_value("DISPLAY_SERIAL_PORT", allow_empty=True)
DISPLAY_SERIAL_CANDIDATES_ENV = os.getenv("DISPLAY_SERIAL_CANDIDATES", "")
DISPLAY_SERIAL_CANDIDATES = tuple(candidate.strip() for candidate in DISPLAY_SERIAL_CANDIDATES_ENV.split(",") if candidate.strip())
DISPLAY_SERIAL_BAUD = env_int("DISPLAY_SERIAL_BAUD")
DISPLAY_SYNC_SECONDS = env_float("DISPLAY_SYNC_SECONDS")
DISPLAY_TEXT_MAX_CHARS = env_int("DISPLAY_TEXT_MAX_CHARS")
DISPLAY_SERIAL_RETRY_SECONDS = env_float("DISPLAY_SERIAL_RETRY_SECONDS")
RELAY_HOST = os.getenv("RELAY_HOST", "0.0.0.0")
RELAY_PORT = int(os.getenv("RELAY_PORT", "8091"))
RELAY_SOURCE_URL = "http://chat2me-speech:8090/state"
RELAY_POLL_SECONDS = float(os.getenv("RELAY_POLL_SECONDS", "0.2").strip() or "0.2")
RELAY_SOURCE_TIMEOUT_SECONDS = float(os.getenv("RELAY_SOURCE_TIMEOUT_SECONDS", "1.0").strip() or "1.0")


def resolve_display_port(port: str) -> str:
    if port and port.lower() != "auto":
        return port
    candidates = DISPLAY_SERIAL_CANDIDATES or (
        "/host-dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00",
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00",
    )
    for candidate in candidates:
        for resolved in glob(candidate):
            if Path(resolved).exists():
                return resolved
    return ""


display_port = resolve_display_port(DISPLAY_SERIAL_PORT)
display_resolver = (
    (lambda: resolve_display_port(DISPLAY_SERIAL_PORT))
    if not DISPLAY_SERIAL_PORT or DISPLAY_SERIAL_PORT.lower() == "auto"
    else None
)
display = DisplayClient(
    display_port,
    DISPLAY_SERIAL_BAUD,
    text_max_chars=DISPLAY_TEXT_MAX_CHARS,
    retry_seconds=DISPLAY_SERIAL_RETRY_SECONDS,
    port_resolver=display_resolver,
)
state_lock = threading.Lock()
last_event = {
    "ok": False,
    "state": "idle",
    "text": "",
    "expression": None,
    "seq": 0,
    "changed_at": time.time(),
    "source_status": "not_checked",
}


def apply_event(payload: dict[str, object]) -> None:
    state = str(payload.get("state") or "idle")[:24]
    text = str(payload.get("text") or "")[:DISPLAY_TEXT_MAX_CHARS]
    expression = str(payload.get("expression") or "").strip() or None
    seq_value = payload.get("seq")
    seq = int(seq_value) if isinstance(seq_value, int) else int(last_event["seq"]) + 1
    with state_lock:
        last_event.update(
            {
                "ok": True,
                "state": state,
                "text": text,
                "expression": expression,
                "seq": seq,
                "changed_at": payload.get("changed_at") or time.time(),
                "source_status": "ok",
            }
        )
    display.set_state(state, text, expression=expression)
    log(f"relay state: {state} expression: {expression}", level="debug")


def set_source_error(error: str) -> None:
    with state_lock:
        last_event["ok"] = False
        last_event["source_status"] = error


def poll_speech_state() -> None:
    interval = max(0.1, RELAY_POLL_SECONDS)
    last_seq: int | None = None
    while True:
        try:
            with httpx.Client(timeout=RELAY_SOURCE_TIMEOUT_SECONDS) as client:
                response = client.get(RELAY_SOURCE_URL)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("invalid state payload")
            seq_value = payload.get("seq")
            seq = int(seq_value) if isinstance(seq_value, int) else None
            if seq is None or seq != last_seq:
                apply_event(payload)
                last_seq = seq
        except Exception as exc:
            set_source_error(f"unavailable:{exc.__class__.__name__}")
        time.sleep(interval)


def sync_display_state() -> None:
    while True:
        time.sleep(DISPLAY_SYNC_SECONDS)
        with state_lock:
            state = str(last_event["state"])
            text = str(last_event["text"])
            expression = last_event.get("expression")
            expression_str = str(expression) if expression else None
        display.set_state(state, text, expression=expression_str)


class RelayHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            with state_lock:
                event = dict(last_event)
            self._send_json(200, {"display": display.enabled, "port": display.port, **event})
            return
        self._send_json(404, {"error": "not found"})


def main() -> None:
    log(f"display serial: {display_port or 'disabled'}")
    log(f"relay source: {RELAY_SOURCE_URL}")
    if display.enabled and DISPLAY_SYNC_SECONDS > 0:
        threading.Thread(target=sync_display_state, daemon=True).start()
        log(f"display sync interval: {DISPLAY_SYNC_SECONDS:g}s")
    threading.Thread(target=poll_speech_state, daemon=True).start()
    server = ThreadingHTTPServer((RELAY_HOST, RELAY_PORT), RelayHandler)
    log(f"relay listening on {RELAY_HOST}:{RELAY_PORT}")
    try:
        server.serve_forever()
    finally:
        display.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
    except Exception as exc:
        log(f"fatal: {exc}", level="error")
        sys.exit(1)
