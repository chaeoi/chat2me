import os
from pathlib import Path
import subprocess

import pytest

AUDIO_SH = Path(__file__).resolve().parents[1] / "runtime/entrypoints/audio.sh"
SRC = AUDIO_SH.read_text()
IDX = SRC.index("configure_speech_audio_output()")
FUNC = SRC[IDX : SRC.index("\n}\n", IDX) + 3]

@pytest.mark.parametrize("role,dev,plugin,pactl_rc,exp_rc,exp_dev,exp_aplay", [
    ("chat2me-speech", "auto", False, 0, 0, "default", True),
    ("chat2me-speech", "auto", True, 1, 0, "default", True),
    ("chat2me-speech", "default", False, 0, 0, "default", False),
    ("chat2me-speech", "plughw:ArrayUAC10,0", False, 0, 0, "plughw:ArrayUAC10,0", False),
    ("chat2me-speech", "pulse", True, 1, 1, None, True),
    ("chat2me-speech", "auto", True, 0, 0, "pulse", True),
    ("chat2me-asr", "auto", False, 0, 0, "auto", False),
])
def test_configure_speech_audio_output(tmp_path, role, dev, plugin, pactl_rc, exp_rc, exp_dev, exp_aplay):
    bin_dir, marker = tmp_path / "bin", tmp_path / "marker"
    bin_dir.mkdir()
    (bin_dir / "aplay").write_text(f"#!/bin/sh\ntouch '{marker}'\n[ \"${{M_PLUGIN:-0}}\" = 1 ] && echo pulse\n")
    (bin_dir / "pactl").write_text("#!/bin/sh\nexit ${M_PACTL_RC:-0}\n")
    for f in (bin_dir / "aplay", bin_dir / "pactl"):
        f.chmod(0o755)

    script = (
        "set -eu\n"
        "find_pulse_cookie() { return 1; }\n"
        "copy_pulse_cookie() { return 1; }\n"
        "chat2me_log() { :; }\n"
        f"{FUNC}\n"
        "configure_speech_audio_output\n"
        'printf "%s\\n%s\\n%s\\n" "$AUDIO_OUTPUT_DEVICE" "$PULSE_SERVER" "$PULSE_COOKIE"\n'
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "VOICE_ROLE": role,
        "AUDIO_OUTPUT_DEVICE": dev,
        "PULSE_SERVER": "original_server",
        "PULSE_COOKIE": "original_cookie",
        "M_PLUGIN": "1" if plugin else "0",
        "M_PACTL_RC": str(pactl_rc),
    }
    res = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)

    assert res.returncode == exp_rc
    assert marker.exists() == exp_aplay
    if exp_rc == 0:
        out_dev, srv, cookie = res.stdout.strip().split("\n")
        assert (out_dev, srv, cookie) == (exp_dev, "original_server", "original_cookie")
    else:
        assert "PulseAudio output requested but not available" in res.stderr
