from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

# app.common loads runtime.env at import time.
from app.common import direction_label, env_bool, env_csv, env_float, env_int, env_value, log

CORE_LLM_URL = "http://chat2me-llm:8082/chat"
CORE_INTENT_URL = "http://chat2me-llm:8082/intent"
CORE_LLM_REACHABILITY_URL = "http://chat2me-llm:8082/llm/reachability"
CORE_LLM_HEALTH_URL = "http://chat2me-llm:8082/health"
CORE_LLM_TIMEOUT_SECONDS = 180.0
CORE_LLM_REACHABILITY_TIMEOUT_SECONDS = 2.0
CORE_INTENT_TIMEOUT_SECONDS = env_float("INTENT_TIMEOUT_SECONDS", 15)
INTENT_CLASSIFIER_ENABLED = env_bool("INTENT_CLASSIFIER_ENABLED", True)
INTENT_CONFIDENCE_THRESHOLD = env_float("INTENT_CONFIDENCE_THRESHOLD", 0.70)
EMPTY_ANSWER_RESPONSE = env_value("EMPTY_ANSWER_RESPONSE")
SESSION_END_RESPONSE = env_value("SESSION_END_RESPONSE")
SESSION_END_PHRASES = env_csv("SESSION_END_PHRASES")
SPEECH_STATE_URL = "http://chat2me-speech:8090/state"
WAKE_WORDS = env_csv("WAKE_WORDS")
PROFILE_PATH = Path(os.getenv("PROFILE_PATH", "/app/config/profile.yaml"))
SAFETY_PATH = Path(os.getenv("SAFETY_PATH", "/app/config/safety.yaml"))

MOTION_ENABLED = env_bool("MOTION_ENABLED", False)
MOTION_SOCKET = env_value("MOTION_SOCKET", allow_empty=True, default="/run/chat2me/motion.sock")
MOTION_CONNECT_TIMEOUT_SECONDS = env_float("MOTION_CONNECT_TIMEOUT_SECONDS", 0.3)
MOTION_READ_TIMEOUT_SECONDS = env_float("MOTION_READ_TIMEOUT_SECONDS", 1.0)
MOTION_ACTION_TIMEOUT_MS = env_int("MOTION_ACTION_TIMEOUT_MS", 5000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    route: str
    model: str | None = None
    latency_ms: int
    fallback: bool = False
    online_status: str | None = None
    action: str | None = None
    action_ok: bool | None = None
    expression: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm: str


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


class DirectionResponse(BaseModel):
    ok: bool
    source: str = "speech"
    raw_angle_degrees: int | None = None
    angle_degrees: int | None = None
    sector: str | None = None
    label: str | None = None
    voice_activity: bool | None = None
    coordinate: dict[str, Any] | None = None
    updated_at: float
    error: str | None = None


app = FastAPI(title="Chat2Me Core", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    log(f"core service ready: llm_url={CORE_LLM_URL}")


MATCH_REMOVE_PATTERN = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()【】\[\]{}<>《》]")
# 动作整句匹配时允许的句尾语气字："点点头吧""看我呀"仍算指令。
ACTION_TRAILING_TOKENS = "吧呀啊呢嘛哦哈啦"


def build_match_prefix_pattern() -> re.Pattern[str]:
    prefixes = ("请问", "你好", "您好", *WAKE_WORDS)
    escaped = "|".join(re.escape(prefix) for prefix in sorted(set(prefixes), key=len, reverse=True) if prefix)
    return re.compile(rf"^({escaped})+") if escaped else re.compile(r"a^")


MATCH_PREFIX_PATTERN = build_match_prefix_pattern()
DEFAULT_INVALID_INPUT_UTTERANCES = (
    "嗯",
    "嗯嗯",
    "嗯嗯嗯",
    "恩",
    "恩恩",
    "啊",
    "啊啊",
    "呃",
    "呃呃",
    "额",
    "额额",
    "嗯哼",
    "en",
    "enen",
    "em",
    "emm",
    "um",
    "uh",
)


def load_yaml(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        return default
    return data


def profile() -> dict[str, Any]:
    return load_yaml(PROFILE_PATH, {})


def safety() -> dict[str, Any]:
    return load_yaml(SAFETY_PATH, {"blocked_keywords": [], "blocked_response": "这个问题暂时不能回答。"})


def contains_blocked_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(str(keyword).lower() in lowered for keyword in safety().get("blocked_keywords", []))


def blocked_response() -> str:
    return str(safety().get("blocked_response", "这个问题暂时不能回答。"))


LLM_LEAK_PATTERNS = (
    "内部思考过程",
    "只输出给用户听",
    "输出规则",
    "系统提示",
    "system prompt",
    "我需要用",
    "我需要先",
    "控制在40个中文字符",
    "不要输出",
    "<think",
    "</think>",
)


def contains_llm_leak(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(pattern.lower() in normalized for pattern in LLM_LEAK_PATTERNS)


def leaked_output_response() -> str:
    return "我可以陪你问答交流，也能介绍江淮中心和机器人相关内容。"


def fixed_qa_id(item: dict[str, Any], index: int) -> str:
    value = str(item.get("id") or "").strip()
    return value or f"fixed_qa_{index}"


def fixed_qa_items() -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(profile().get("fixed_qa", [])):
        if isinstance(item, dict):
            items.append((fixed_qa_id(item, index), item))
    return items


def fixed_qa_item_by_id(item_id: str) -> dict[str, Any] | None:
    wanted = item_id.strip()
    if not wanted:
        return None
    for candidate_id, item in fixed_qa_items():
        if candidate_id == wanted:
            return item
    return None


def fixed_qa_answer_by_id(item_id: str) -> str | None:
    item = fixed_qa_item_by_id(item_id)
    if not item:
        return None
    return str(item.get("answer", "")).strip() or None


def normalize_for_match(text: str) -> str:
    normalized = text.strip().lower().replace("您", "你")
    normalized = MATCH_REMOVE_PATTERN.sub("", normalized)
    normalized = MATCH_PREFIX_PATTERN.sub("", normalized)
    return normalized


def invalid_input_utterances() -> frozenset[str]:
    router = profile().get("intent_router", {})
    if not isinstance(router, dict):
        router = {}
    configured = router.get("invalid_input_utterances", [])
    if not isinstance(configured, list):
        configured = []

    intents = router.get("intents", {})
    invalid_input = intents.get("invalid_input", {}) if isinstance(intents, dict) else {}
    examples = invalid_input.get("examples", []) if isinstance(invalid_input, dict) else []
    if not isinstance(examples, list):
        examples = []

    return frozenset(
        normalized
        for normalized in (
            normalize_for_match(str(item))
            for item in (*DEFAULT_INVALID_INPUT_UTTERANCES, *configured, *examples)
        )
        if normalized
    )


def is_invalid_input(message: str) -> bool:
    normalized = normalize_for_match(message)
    if not normalized:
        return True
    if normalized in invalid_input_utterances():
        return True

    chinese_fillers = set("嗯恩啊呃额喔哦噢哼唔")
    if len(normalized) <= 4 and all(char in chinese_fillers for char in normalized):
        return True
    return bool(len(normalized) <= 8 and re.fullmatch(r"(en|em|um|uh|ah|a)+", normalized))


async def call_motion(action_name: str, item: dict[str, Any]) -> bool:
    if not MOTION_SOCKET:
        log("motion socket is not configured; skipping action", level="warning")
        return False

    params: dict[str, Any] = {}
    if bool(item.get("needs_direction")):
        # speech 把声源方向归一化为 0=正前、顺时针递增，这里原样交给 motion 换算颈部 yaw。
        direction = await fetch_direction()
        angle = direction.angle_degrees if direction.ok and direction.angle_degrees is not None else None
        params["direction_deg"] = angle if angle is not None else 0
        params["fallback"] = "speaker" if angle is not None else "front"

    payload = {
        "id": f"act-{uuid.uuid4().hex[:16]}",
        "name": action_name,
        "params": params,
        "preempt": True,
        "timeout_ms": MOTION_ACTION_TIMEOUT_MS,
    }
    timeout = httpx.Timeout(
        connect=MOTION_CONNECT_TIMEOUT_SECONDS,
        read=MOTION_READ_TIMEOUT_SECONDS,
        write=0.5,
        pool=0.5,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=MOTION_SOCKET),
            base_url="http://motion",
            timeout=timeout,
        ) as client:
            response = await client.post("/action", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log(
            f"motion rejected action {action_name}: "
            f"http_{exc.response.status_code} {exc.response.text[:200]}",
            level="warning",
        )
        return False
    except httpx.HTTPError as exc:
        log(f"motion unavailable for action {action_name}: {exc}", level="warning")
        return False
    except OSError as exc:
        log(f"motion socket unavailable for action {action_name}: {exc}", level="warning")
        return False
    return True


async def run_motion_action(item_id: str, action_name: str, item: dict[str, Any]) -> None:
    action_ok = await call_motion(action_name, item)
    log(f"fixed item action triggered: id={item_id} action={action_name} ok={action_ok}")


async def execute_fixed_item(
    item_id: str,
    item: dict[str, Any],
    started_at: float,
    route: str,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    action_name = str(item.get("action") or "").strip() or None
    expression = str(item.get("expression") or "").strip() or None
    action_ok: bool | None = None

    if action_name:
        if MOTION_ENABLED:
            background_tasks.add_task(run_motion_action, item_id, action_name, item)
        else:
            log(f"motion disabled; skipping action {action_name} for fixed item {item_id}", level="debug")

    answer = str(item.get("answer") or "").strip() or "好的"
    return ChatResponse(
        answer=answer,
        route=route,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        action=action_name,
        action_ok=action_ok,
        expression=expression,
    )


def match_fixed_qa(message: str) -> tuple[str, dict[str, Any]] | None:
    # 固定回答匹配：
    # 1. match: exact（如动作指令）：归一化并剥离句尾语气词后整句全等；能力问句需作为单独条目显式配置；
    # 2. 默认（普通问答）：归一化后做子串匹配。
    # 多个条目命中时取 pattern 最长的条目，避免短词遮蔽。
    normalized = normalize_for_match(message)
    if not normalized:
        return None
    normalized_exact = normalized.rstrip(ACTION_TRAILING_TOKENS)

    best: tuple[int, str, dict[str, Any]] | None = None
    for item_id, item in fixed_qa_items():
        is_exact = str(item.get("match") or "").strip().lower() == "exact"
        patterns = [
            norm
            for norm in (normalize_for_match(str(p)) for p in item.get("patterns", []))
            if norm
        ]
        for pattern in patterns:
            if is_exact:
                clean_pattern = pattern.rstrip(ACTION_TRAILING_TOKENS)
                if clean_pattern != normalized_exact:
                    continue
            else:
                if pattern not in normalized:
                    continue

            match_len = len(pattern)
            if best is None or match_len > best[0]:
                best = (match_len, item_id, item)

    if best is None:
        return None
    return best[1], best[2]


def fixed_qa_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for item_id, item in fixed_qa_items():
        patterns = item.get("patterns", [])
        catalog.append(
            {
                "id": item_id,
                "intent": str(item.get("intent") or "").strip(),
                "patterns": [str(pattern) for pattern in patterns if str(pattern).strip()],
            }
        )
    return catalog


def intent_catalog() -> dict[str, Any]:
    data = profile()
    router = data.get("intent_router", {})
    if not isinstance(router, dict):
        router = {}
    return {
        "intents": router.get("intents", {}),
        "fixed_qa": fixed_qa_catalog(),
        "blocked_keywords": [str(keyword) for keyword in safety().get("blocked_keywords", [])],
        "session_end_phrases": list(SESSION_END_PHRASES),
    }


def build_intent_prompt() -> str:
    catalog = json.dumps(intent_catalog(), ensure_ascii=False, separators=(",", ":"))
    allowed_fixed_ids = [item["id"] for item in fixed_qa_catalog()]
    allowed_fixed_ids_json = json.dumps(allowed_fixed_ids, ensure_ascii=False, separators=(",", ":"))
    return (
        "你是 Chat2Me 的本地意图分类器，只做意图分类，不回答用户问题。\n"
        "必须只输出一个 JSON 对象，不要输出解释、Markdown 或额外文本。\n"
        "JSON 字段固定为：intent、fixed_qa_id、confidence。\n"
        "intent 只能是 blocked、fixed_qa、direction、session_end、invalid_input、chat。\n"
        "fixed_qa_id 只能来自 allowed_fixed_qa_ids；非 fixed_qa 时必须为 null。\n"
        "confidence 是 0 到 1 的数字；不确定时返回 intent=chat。\n"
        "如果用户输入只是语气词、应声、口头填充、噪声转写或没有明确问题/指令，必须返回 invalid_input。\n"
        "不要编造 fixed_qa_id；不能确定命中固定问答时返回 chat。\n"
        f"allowed_fixed_qa_ids={allowed_fixed_ids_json}\n"
        f"意图目录={catalog}"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    content = text.strip()
    if not content:
        return None
    try:
        data = json.loads(content)
    except ValueError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def parse_intent_result(content: str) -> dict[str, Any] | None:
    data = extract_json_object(content)
    if data is None:
        return None

    intent = str(data.get("intent") or "chat").strip().lower()
    if intent not in {"blocked", "fixed_qa", "direction", "session_end", "invalid_input", "chat"}:
        return None

    fixed_qa_id_value = data.get("fixed_qa_id")
    fixed_qa_id_text = str(fixed_qa_id_value).strip() if fixed_qa_id_value is not None else ""
    fixed_qa_id_normalized = fixed_qa_id_text if fixed_qa_id_text and fixed_qa_id_text.lower() != "null" else None

    try:
        confidence = float(data.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        return {"intent": "chat", "fixed_qa_id": None, "confidence": confidence}

    if intent == "fixed_qa" and not fixed_qa_answer_by_id(fixed_qa_id_normalized or ""):
        log(f"intent classifier returned invalid fixed_qa_id: {fixed_qa_id_normalized}", level="warning")
        return {"intent": "chat", "fixed_qa_id": None, "confidence": confidence}

    return {
        "intent": intent,
        "fixed_qa_id": fixed_qa_id_normalized,
        "confidence": confidence,
    }


async def call_intent_service(message: str) -> dict[str, Any] | None:
    if not INTENT_CLASSIFIER_ENABLED:
        return None
    payload = {
        "message": message,
        "system_prompt": build_intent_prompt(),
    }
    timeout = httpx.Timeout(connect=3.0, read=CORE_INTENT_TIMEOUT_SECONDS, write=10.0, pool=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(CORE_INTENT_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        log(f"intent classifier unavailable; continuing with normal LLM: {exc}", level="warning")
        return None
    except ValueError:
        log("intent classifier returned invalid JSON envelope; continuing with normal LLM", level="warning")
        return None

    content = str(data.get("content") or "")
    result = parse_intent_result(content)
    if result is None:
        log(f"intent classifier returned invalid payload: {content[:200]}", level="warning")
    return result


async def fetch_direction() -> DirectionResponse:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(SPEECH_STATE_URL)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        return DirectionResponse(
            ok=False,
            source="speech",
            error=f"speech_direction_unavailable:{exc.__class__.__name__}",
            updated_at=time.time(),
        )
    except ValueError:
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_direction_invalid_json",
            updated_at=time.time(),
        )

    if not isinstance(data, dict):
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_direction_invalid_payload",
            updated_at=time.time(),
        )
    direction_data = data.get("direction")
    if not isinstance(direction_data, dict):
        return DirectionResponse(
            ok=False,
            source="speech",
            error="speech_state_missing_direction",
            updated_at=time.time(),
        )
    return DirectionResponse(**direction_data)


async def direction_answer() -> str:
    direction_data = await fetch_direction()
    if not direction_data.ok or direction_data.angle_degrees is None:
        return "我现在读不到麦克风方向信息。"
    return f"您在我的{direction_label(direction_data.angle_degrees)}。"


async def intent_chat_response(
    message: str,
    started_at: float,
    background_tasks: BackgroundTasks,
) -> ChatResponse | None:
    result = await call_intent_service(message)
    if result is None:
        return None

    intent = str(result.get("intent") or "chat")
    if intent == "chat":
        return None
    if intent == "invalid_input":
        return ChatResponse(
            answer="",
            route="invalid_input",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    if intent == "blocked":
        return ChatResponse(
            answer=blocked_response(),
            route="blocked_intent",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    if intent == "fixed_qa":
        fixed_qa_id = str(result.get("fixed_qa_id") or "")
        item = fixed_qa_item_by_id(fixed_qa_id)
        if not item:
            return None
        return await execute_fixed_item(
            item_id=fixed_qa_id,
            item=item,
            started_at=started_at,
            route="fixed_qa_intent",
            background_tasks=background_tasks,
        )
    if intent == "direction":
        return ChatResponse(
            answer=await direction_answer(),
            route="direction_intent",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    if intent == "session_end":
        return ChatResponse(
            answer=SESSION_END_RESPONSE,
            route="session_end",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    return None


async def call_llm_service(request: ChatRequest) -> ChatResponse:
    system_prompt = str(profile().get("system_prompt", "")).strip()
    payload: dict[str, Any] = {
        "message": request.message.strip(),
        "system_prompt": system_prompt,
    }
    timeout = httpx.Timeout(connect=5.0, read=CORE_LLM_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(CORE_LLM_URL, json=payload)
        response.raise_for_status()
    return ChatResponse(**response.json())


@app.get("/llm/reachability", response_model=ReachabilityResponse)
async def llm_reachability() -> ReachabilityResponse:
    try:
        async with httpx.AsyncClient(timeout=CORE_LLM_REACHABILITY_TIMEOUT_SECONDS) as client:
            response = await client.get(CORE_LLM_REACHABILITY_URL)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError:
        return ReachabilityResponse(online=False, provider="unknown", model=None, status="unreachable")
    return ReachabilityResponse(**data)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(CORE_LLM_HEALTH_URL)
            llm_status = "ok" if response.is_success else f"http_{response.status_code}"
    except httpx.HTTPError:
        llm_status = "unreachable"
    return HealthResponse(status="ok" if llm_status == "ok" else "degraded", llm=llm_status)


@app.get("/direction", response_model=DirectionResponse)
async def direction() -> DirectionResponse:
    return await fetch_direction()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    started_at = time.perf_counter()
    message = request.message.strip()

    if is_invalid_input(message):
        return ChatResponse(
            answer="",
            route="invalid_input",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    if contains_blocked_keyword(message):
        return ChatResponse(
            answer=blocked_response(),
            route="blocked_input",
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    matched = match_fixed_qa(message)
    if matched is not None:
        item_id, item = matched
        return await execute_fixed_item(
            item_id=item_id,
            item=item,
            started_at=started_at,
            route="fixed_qa",
            background_tasks=background_tasks,
        )

    intent_response = await intent_chat_response(message, started_at, background_tasks)
    if intent_response is not None:
        return intent_response

    try:
        result = await call_llm_service(request)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        log(f"LLM service returned HTTP {exc.response.status_code}: {detail}", level="warning")
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.ReadTimeout as exc:
        log("LLM service timed out", level="warning")
        raise HTTPException(status_code=504, detail="大模型生成超时。") from exc
    except httpx.HTTPError as exc:
        log(f"LLM service unavailable: {exc}", level="warning")
        raise HTTPException(status_code=502, detail=f"LLM 服务不可用：{exc}") from exc

    if contains_blocked_keyword(result.answer):
        answer = blocked_response()
        route = "blocked_output"
    elif contains_llm_leak(result.answer):
        answer = leaked_output_response()
        route = "filtered_output"
    else:
        answer = result.answer
        route = result.route

    return ChatResponse(
        answer=answer or EMPTY_ANSWER_RESPONSE,
        route=route,
        model=result.model,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        fallback=result.fallback,
        online_status=result.online_status,
    )
