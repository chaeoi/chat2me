import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("initially_busy", [False, True])
def test_wake_listener_releases_microphone_for_session(initially_busy):
    source = Path(__file__).resolve().parents[1] / "services/speech/app/main.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "listen_for_wake")
    lock = threading.Lock()
    events = []
    streams = []
    reads = 0
    if initially_busy:
        lock.acquire()

    class Audio:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            assert not lock.locked(), "session already owns the microphone"
            events.append("open")
            return self

        def __exit__(self, *args):
            events.append("close")

    def read_mono(audio, chunk):
        nonlocal reads
        assert not lock.locked(), "wake listener kept reading during a session"
        reads += 1
        events.append("read")
        if not initially_busy and reads == 1:
            lock.acquire()
        return []

    def sleep(seconds):
        assert lock.locked()
        assert initially_busy or events[-1] == "close"
        events.append("release")
        lock.release()

    def create_stream():
        stream = SimpleNamespace(accept_waveform=lambda *args: None)
        streams.append(stream)
        return stream

    kws = SimpleNamespace(create_stream=create_stream,
                          is_ready=lambda stream: reads > 0 and not lock.locked(),
                          decode_stream=lambda stream: None,
                          get_result=lambda stream: "小江")
    namespace = dict(SpeechState=object, WakeHandler=SimpleNamespace(busy_lock=lock),
                     sd=SimpleNamespace(InputStream=Audio),
                     time=SimpleNamespace(sleep=sleep, monotonic=lambda: 0),
                     log=lambda *args: None, wake_words_display=lambda: "小江",
                     wait_for_input_device=lambda display: 24, read_mono=read_mono,
                     INPUT_CHANNELS=6, SAMPLE_RATE=16000,
                     SPEECH_WAIT_LOG_SECONDS=1, SPEECH_WAIT_POLL_SECONDS=.1)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    display = SimpleNamespace(set_state=lambda *args: pytest.fail("unexpected audio error"))
    assert namespace["listen_for_wake"](kws, 1600, display) == "小江"
    assert events == (["release", "open", "read", "close"] if initially_busy else
                      ["open", "read", "close", "release", "open", "read", "close"])
    assert len(streams) >= 2
