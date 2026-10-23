"""No robot access: HTTP is mocked, or sent to the ROS-free memory-only fixture."""
import asyncio
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import types

import httpx
import pytest
from fastapi import BackgroundTasks

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    os.environ.update(
        RUNTIME_CONFIG_PATH=str(ROOT / "config/runtime.env"),
        PROFILE_PATH=str(ROOT / "config/profile.yaml"),
        SAFETY_PATH=str(ROOT / "config/safety.yaml"),
        MOTION_SOCKET="/tmp/nonexistent-chat2me-test.sock",
    )
    sys.modules["app"] = types.ModuleType("app")
    common = load_module("app.common", ROOT / "runtime/shared/common.py")
    common.LOG_DIR = tmp_path_factory.mktemp("logs")
    return load_module("review_core", ROOT / "services/core/app/main.py")


def run(coroutine):
    return asyncio.run(coroutine)


def accept(request):
    payload = json.loads(request.content)
    return httpx.Response(202, json={"ok": True, "status": "accepted", "id": payload["id"], "action": payload["name"]})


def mock_motion(core, monkeypatch, handler=accept):
    monkeypatch.setattr(core.httpx, "AsyncHTTPTransport", lambda **_: httpx.MockTransport(handler))


@pytest.mark.parametrize("socket_path, expected_error", [(None, "must be set"), ("", "must not be empty")])
def test_socket_path_has_no_code_default(core, monkeypatch, socket_path, expected_error):
    if socket_path is None:
        monkeypatch.delenv("MOTION_SOCKET", raising=False)
    else:
        monkeypatch.setenv("MOTION_SOCKET", socket_path)
    with pytest.raises(RuntimeError, match=f"MOTION_SOCKET {expected_error}"):
        load_module("review_core_missing_socket", ROOT / "services/core/app/main.py")


def test_custom_socket_path_comes_from_environment(core, monkeypatch):
    monkeypatch.setenv("MOTION_SOCKET", "/tmp/custom-motion/commands.sock")
    loaded = load_module("review_core_custom_socket", ROOT / "services/core/app/main.py")
    assert loaded.MOTION_SOCKET == "/tmp/custom-motion/commands.sock"


def test_compose_bootstraps_without_socket_environment(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI is required for configuration validation")
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    env = {key: value for key, value in os.environ.items() if key not in {"MOTION_SOCKET", "MOTION_SOCKET_DIR"}}
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_file), "-f", str(ROOT / "docker-compose.yml"), "config", "--format", "json"],
        env=env, text=True, capture_output=True, check=True,
    )
    core_config = json.loads(result.stdout)["services"]["chat2me-core"]
    assert "MOTION_SOCKET" not in core_config["environment"]
    assert any(volume["source"] == volume["target"] == "/opt/chat2me/motion" for volume in core_config["volumes"])


def test_entrypoint_initializes_runtime_env_and_preserves_existing_values(tmp_path):
    defaults = tmp_path / "defaults"
    config = tmp_path / "config"
    defaults.mkdir()
    (defaults / "runtime.env").write_text("MOTION_SOCKET=/opt/chat2me/motion/motion.sock\n")
    env = {**os.environ, "DEFAULT_CONFIG_DIR": str(defaults), "CONFIG_DIR": str(config),
           "RUNTIME_CONFIG_PATH": str(config / "runtime.env")}
    command = ['sh', '-c', '. "$1"; chat2me_log() { :; }; init_config',
               'sh', str(ROOT / "runtime/entrypoints/lib.sh")]
    subprocess.run(command, env=env, check=True)
    assert (config / "runtime.env").read_text() == "MOTION_SOCKET=/opt/chat2me/motion/motion.sock\n"
    (config / "runtime.env").write_text("MOTION_SOCKET=/opt/chat2me/motion/custom.sock\n")
    subprocess.run(command, env=env, check=True)
    assert (config / "runtime.env").read_text() == "MOTION_SOCKET=/opt/chat2me/motion/custom.sock\n"


@pytest.mark.parametrize("text", ["点头", "点点头吧", "你好，小江，点个头", "你会点头吗", "你能帮我点个头吗"])
def test_explicit_action_commands(core, text):
    matched = core.match_fixed_qa(text)
    assert matched is not None
    assert matched[1]["action"] == "nod"


@pytest.mark.parametrize("text", ["不要点头", "别点头", "我不想让你点头", "如果让你点头会怎样", "解释一下点头是什么意思", "他说点头吧"])
def test_negated_and_hypothetical_inputs_do_not_match_actions(core, text):
    matched = core.match_fixed_qa(text)
    assert matched is None or not matched[1].get("action")


def test_actions_are_exact_even_if_profile_omits_match(core, monkeypatch):
    monkeypatch.setattr(core, "profile", lambda: {"fixed_qa": [{"id": "nod", "action": "nod", "patterns": ["点头"]}]})
    assert core.match_fixed_qa("点头")
    assert core.match_fixed_qa("不要点头") is None


def test_classifier_cannot_dispatch_motion(core, monkeypatch):
    result = core.parse_intent_result(json.dumps({"intent": "fixed_qa", "fixed_qa_id": "nod", "confidence": 1}))
    assert result["intent"] == "chat"
    assert "nod" not in {item["id"] for item in core.fixed_qa_catalog()}

    async def classified(_):
        return {"intent": "fixed_qa", "fixed_qa_id": "nod", "confidence": 1}

    monkeypatch.setattr(core, "call_intent_service", classified)
    tasks = BackgroundTasks()
    assert run(core.intent_chat_response("不要点头", time.perf_counter(), tasks)) is None
    assert not tasks.tasks


def test_plain_fixed_qa_still_uses_intent_fallback(core, monkeypatch):
    async def classified(_):
        return {"intent": "fixed_qa", "fixed_qa_id": "robot_name", "confidence": 1}

    monkeypatch.setattr(core, "call_intent_service", classified)
    tasks = BackgroundTasks()
    response = run(core.intent_chat_response("告诉我你的称呼", time.perf_counter(), tasks))
    assert response.route == "fixed_qa_intent"
    assert response.answer
    assert response.action is None
    assert not tasks.tasks


def test_speech_response_does_not_wait_for_action(core, monkeypatch):
    submitted = []

    async def fake_call(name, item):
        submitted.append(name)
        return True

    monkeypatch.setattr(core, "MOTION_ENABLED", True)
    monkeypatch.setattr(core, "call_motion", fake_call)
    tasks = BackgroundTasks()
    response = run(core.execute_fixed_item("nod", {"action": "nod", "answer": "好的"}, time.perf_counter(), "fixed_qa", tasks))
    assert response.answer == "好的" and response.action_ok is None
    assert not submitted
    run(tasks())
    assert submitted == ["nod"]


def test_motion_disabled_does_not_schedule(core, monkeypatch):
    monkeypatch.setattr(core, "MOTION_ENABLED", False)
    tasks = BackgroundTasks()
    response = run(core.execute_fixed_item("nod", {"action": "nod"}, time.perf_counter(), "fixed_qa", tasks))
    assert response.action_ok is False
    assert not tasks.tasks


def test_motion_params_and_direction_preserved(core, monkeypatch):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return accept(request)

    async def direction():
        return core.DirectionResponse(ok=True, angle_degrees=90, updated_at=0)

    mock_motion(core, monkeypatch, handler)
    monkeypatch.setattr(core, "fetch_direction", direction)
    assert run(core.call_motion("look_at", {"needs_direction": True}))
    assert seen[-1]["params"] == {"direction_deg": 90, "fallback": "speaker"}
    params = {"left_arm": [0] * 7, "right_arm": [.1] * 7, "head": [0, 0, 0], "duration_ms": 1000}
    assert run(core.call_motion("move_joints", {"params": params}))
    assert seen[-1]["params"] == params


@pytest.mark.parametrize("bad", ["wrong_id", "wrong_action", "false_ok", "wrong_status", "list", "invalid_json", "http_200"])
def test_only_matching_acceptance_counts_as_success(core, monkeypatch, bad):
    def handler(request):
        payload = json.loads(request.content)
        data = {"ok": True, "id": payload["id"], "action": payload["name"], "status": "accepted"}
        if bad == "wrong_id": data["id"] = "another-request"
        elif bad == "wrong_action": data["action"] = "shake"
        elif bad == "false_ok": data["ok"] = False
        elif bad == "wrong_status": data["status"] = "failed"
        elif bad == "list": data = []
        elif bad == "invalid_json": return httpx.Response(202, text="not json")
        return httpx.Response(200 if bad == "http_200" else 202, json=data)

    mock_motion(core, monkeypatch, handler)
    assert not run(core.call_motion("nod", {}))


def test_motion_failure_does_not_raise(core, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("fixture unavailable", request=request)

    mock_motion(core, monkeypatch, handler)
    assert not run(core.call_motion("nod", {}))
    assert not run(core.call_motion("move_joints", {"params": []}))


def test_bad_direction_payload_falls_back_to_front(core, monkeypatch):
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"direction": {"ok": True, "angle_degrees": "bad"}}))
    monkeypatch.setattr(core.httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    direction = run(core.fetch_direction())
    assert direction.ok is False
    assert direction.error == "speech_direction_invalid_payload"


def test_direction_poll_preserves_expression_until_state_transition():
    # Load the actual state function without importing audio drivers or devices.
    tree = ast.parse((ROOT / "services/speech/app/main.py").read_text())
    functions = {"current_state", "direction_bucket", "normalize_direction", "update_speech_state"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    namespace = {"threading": threading, "time": time, "_UNSET": object(), "STATE_LOCK": threading.Lock(),
                 "CURRENT_STATE": {"state": "idle", "text": "", "expression": None, "direction": {}, "seq": 0, "changed_at": 0},
                 "direction_label": lambda angle: str(angle)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "speech_state", "exec"), namespace)
    update = namespace["update_speech_state"]
    update(state="speaking", text="你好", expression="happy")
    result = update(direction={"ok": False, "error": "test", "updated_at": 1})
    assert result["expression"] == "happy"
    result = update(state="listening", expression=None)
    assert result["expression"] is None


def test_real_uds_contract_with_memory_only_motion_fixture(core, monkeypatch, tmp_path):
    binary = os.getenv("MOTION_OFFLINE_SERVER")
    if not binary:
        pytest.skip("set MOTION_OFFLINE_SERVER to the ROS-free fixture binary")
    socket_path = str(tmp_path / "motion.sock")
    process = subprocess.Popen([binary, socket_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready (memory only; no ROS)"
        monkeypatch.setattr(core, "MOTION_SOCKET", socket_path)
        assert run(core.call_motion("nod", {}))
        assert run(core.call_motion("move_joints", {"params": {"left_arm": [.1] * 7, "right_arm": [-.1] * 7, "head": [.1, 0, 0], "duration_ms": 1000}}))
        assert not run(core.call_motion("unknown", {}))
    finally:
        process.communicate("stop\n", timeout=5)
    assert process.returncode == 0
