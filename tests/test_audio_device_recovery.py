import ast
from contextlib import contextmanager
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def audio_runtime():
    source = Path(__file__).resolve().parents[1] / "services/speech/app/main.py"
    names = {"refresh_input_devices", "wait_for_input_device", "open_session_input", "open_input_stream"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(SpeechState=object, contextmanager=contextmanager,
                     AUDIO_INPUT_LOCK=threading.RLock(),
                     INPUT_DEVICE="ReSpeaker", INPUT_DEVICE_REQUIRED=True,
                     INPUT_CHANNELS=6, SAMPLE_RATE=16000, SPEECH_WAIT_POLL_SECONDS=.1,
                     time=SimpleNamespace(monotonic=lambda: 0, sleep=lambda seconds: None),
                     log=lambda *args, **kwargs: None,
                     sd=SimpleNamespace(InputStream=object))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("initialization_fails_once", [False, True])
def test_device_appearing_after_startup_is_found(audio_runtime, initialization_fails_once):
    runtime = audio_runtime
    attempts = 0
    cached_device = None
    states = []

    def initialize():
        nonlocal attempts, cached_device
        attempts += 1
        if initialization_fails_once and attempts == 1:
            raise RuntimeError("audio backend temporarily unavailable")
        runtime["sd"]._initialized = 1
        cached_device = 24

    def terminate():
        assert runtime["sd"]._initialized == 1
        runtime["sd"]._initialized = 0

    runtime["sd"] = SimpleNamespace(_initialized=1, _terminate=terminate, _initialize=initialize)
    runtime["select_input_device"] = lambda selector: cached_device
    display = SimpleNamespace(set_state=lambda *args: states.append(args))

    assert runtime["wait_for_input_device"](display) == 24
    assert attempts == (2 if initialization_fails_once else 1)
    assert runtime["sd"]._initialized == 1
    assert bool(states) == initialization_fails_once


def test_stale_stream_is_closed_before_refresh_and_device_is_reselected(audio_runtime):
    runtime = audio_runtime
    selected_device = 7
    streams = []

    class Audio:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]
            self.closed = False
            streams.append(self)

        def start(self):
            if self.device == 7:
                raise RuntimeError("device disconnected")

        def close(self):
            self.closed = True

    def initialize():
        nonlocal selected_device
        assert all(stream.closed for stream in streams)
        runtime["sd"]._initialized = 1
        selected_device = 24

    def terminate():
        assert all(stream.closed for stream in streams)
        runtime["sd"]._initialized = 0

    runtime["sd"] = SimpleNamespace(InputStream=Audio, _initialized=1,
                                     _terminate=terminate, _initialize=initialize)
    runtime["select_input_device"] = lambda selector: selected_device
    display = SimpleNamespace(set_state=lambda *args: None)

    with pytest.raises(RuntimeError, match="session interrupted"):
        with runtime["open_input_stream"](display, 1600) as stream:
            assert stream.device == 24
            assert [item.device for item in streams] == [7, 24]
            assert streams[0].closed
            assert not stream.closed
            raise RuntimeError("session interrupted")
    assert all(stream.closed for stream in streams)


def test_available_microphone_does_not_reset_audio_backend(audio_runtime):
    runtime = audio_runtime
    runtime["select_input_device"] = lambda selector: 24
    runtime["refresh_input_devices"] = lambda: pytest.fail("available input should not refresh")
    display = SimpleNamespace(set_state=lambda *args: pytest.fail("available input should not fail"))
    assert runtime["wait_for_input_device"](display) == 24
