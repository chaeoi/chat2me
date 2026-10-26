from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# app.common loads runtime.env at import time.
from app.common import env_float, env_int, env_value, log

LOCAL_LLM_PROVIDERS = {"ollama", "local"}
ONLINE_LLM_PROVIDERS = {"online"}
ONLINE_LLM_PROVIDER_OPENAI = "openai"
SUPPORTED_ONLINE_LLM_PROVIDERS = {ONLINE_LLM_PROVIDER_OPENAI}
CHAT_COMPLETIONS_PATH = "/chat/completions"
RESPONSES_PATH = "/responses"
MODELS_PATH = "/models"


def normalize_provider(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def provider_display_name(value: str) -> str:
    return value.replace("_", "-")


LLM_ENGINE = normalize_provider(env_value("LLM_ENGINE"))
LLM_ONLINE_PROVIDER = normalize_provider(
    env_value("LLM_ONLINE_PROVIDER", allow_empty=True, default="openai") or "openai"
)
LLM_ONLINE_PROVIDER_DISPLAY = provider_display_name(LLM_ONLINE_PROVIDER)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = env_value("OLLAMA_MODEL", allow_empty=True)
LLM_MODEL = env_value("LLM_MODEL", allow_empty=True)
LLM_BASE_URL = env_value("LLM_BASE_URL", allow_empty=True).rstrip("/")
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE")
LLM_TOP_P = env_float("LLM_TOP_P")
LLM_MAX_TOKENS = env_int("LLM_MAX_TOKENS")
LLM_MAX_TOKENS_FIELD = env_value("LLM_MAX_TOKENS_FIELD", allow_empty=True)
LLM_CONNECT_TIMEOUT_SECONDS = env_float("LLM_CONNECT_TIMEOUT_SECONDS")
LLM_TIMEOUT_SECONDS = env_float("LLM_TIMEOUT_SECONDS")
LLM_REACHABILITY_INTERVAL_SECONDS = 5.0
LLM_REACHABILITY_TIMEOUT_SECONDS = 1.5
OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX")
OLLAMA_NUM_THREAD = env_int("OLLAMA_NUM_THREAD")
OLLAMA_KEEP_ALIVE = env_value("OLLAMA_KEEP_ALIVE", allow_empty=True, default="30m")
INTENT_MODEL = env_value("INTENT_MODEL", allow_empty=True, default="")
INTENT_MAX_TOKENS = env_int("INTENT_MAX_TOKENS", default=48)
INTENT_NUM_CTX = env_int("INTENT_NUM_CTX", default=2048)
INTENT_TEMPERATURE = env_float("INTENT_TEMPERATURE", default=0)
INTENT_TIMEOUT_SECONDS = env_float("INTENT_TIMEOUT_SECONDS", default=15)
EMPTY_ANSWER_RESPONSE = env_value("EMPTY_ANSWER_RESPONSE")
SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "").strip()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    system_prompt: str | None = None


class IntentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    system_prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    route: str
    model: str | None = None
    latency_ms: int
    fallback: bool = False
    online_status: str | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None
    llm: str
    ollama: str
    online: bool
    online_status: str


class ReachabilityResponse(BaseModel):
    online: bool
    provider: str
    model: str | None
    status: str
    checked_at: float | None = None


class IntentResponse(BaseModel):
    content: str
    route: str
    model: str
    latency_ms: int


INTENT_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["blocked", "fixed_qa", "direction", "session_end", "invalid_input", "chat"]},
        "fixed_qa_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["intent", "fixed_qa_id", "confidence"],
    "additionalProperties": False,
}


class LLMConfigError(Exception):
    pass


class LLMResponseError(Exception):
    pass


@dataclass(frozen=True)
class LLMResult:
    answer: str
    route: str
    model: str | None
    fallback: bool = False
    online_status: str | None = None


@dataclass(frozen=True)
class LLMReachability:
    online: bool
    status: str
    checked_at: float | None = None


ONLINE_REACHABILITY = LLMReachability(online=False, status="not_checked")
ONLINE_REACHABILITY_TASK: asyncio.Task[None] | None = None


def chat_messages(message: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    prompt = (system_prompt or SYSTEM_PROMPT).strip()
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": message})
    return messages


def online_base_url() -> str:
    base_url = LLM_BASE_URL
    for suffix in (CHAT_COMPLETIONS_PATH, RESPONSES_PATH):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    return base_url


def online_completion_url() -> str:
    if LLM_BASE_URL.endswith((CHAT_COMPLETIONS_PATH, RESPONSES_PATH)):
        return LLM_BASE_URL
    return f"{online_base_url()}{CHAT_COMPLETIONS_PATH}"


def online_api_key() -> str:
    return env_value("LLM_API_KEY", allow_empty=True)


def online_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = online_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def provider_is_local() -> bool:
    return LLM_ENGINE in LOCAL_LLM_PROVIDERS


def provider_is_online() -> bool:
    return LLM_ENGINE in ONLINE_LLM_PROVIDERS


def require_online_config() -> str:
    if LLM_ONLINE_PROVIDER not in SUPPORTED_ONLINE_LLM_PROVIDERS:
        raise LLMConfigError("在线大模型只支持 LLM_ONLINE_PROVIDER=openai。")
    if not LLM_MODEL:
        raise LLMConfigError("在线大模型未配置 LLM_MODEL。")
    base_url = online_base_url()
    if not base_url:
        raise LLMConfigError("在线大模型未配置 LLM_BASE_URL。")
    return base_url


def require_local_model() -> str:
    if not OLLAMA_MODEL:
        raise LLMConfigError("本地大模型未配置 OLLAMA_MODEL。")
    return OLLAMA_MODEL


def intent_model() -> str:
    return INTENT_MODEL or require_local_model()


def online_uses_responses_api() -> bool:
    return LLM_BASE_URL.endswith(RESPONSES_PATH)


def text_parts(items: list[Any]) -> list[str]:
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
    return parts


def extract_online_answer(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        message_data = choices[0].get("message") or {}
        content = message_data.get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(text_parts(content))
        return str(content)

    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(text_parts(content))
    return "\n".join(parts)


def response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        for line in response.text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
                break
            except ValueError:
                continue
        else:
            raise LLMResponseError("远程大模型返回了非 JSON 响应。") from None
    if not isinstance(data, dict):
        raise LLMResponseError("远程大模型返回了非对象 JSON 响应。")
    return data


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


async def call_ollama_with_options(
    message: str,
    model: str,
    system_prompt: str | None = None,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_ctx: int,
    timeout_seconds: float,
    response_format: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": chat_messages(message, system_prompt),
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_thread": OLLAMA_NUM_THREAD,
        },
    }
    if response_format is not None:
        payload["format"] = response_format
    if OLLAMA_KEEP_ALIVE:
        payload["keep_alive"] = OLLAMA_KEEP_ALIVE
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT_SECONDS, read=timeout_seconds, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    message_data = data.get("message") or {}
    answer = message_data.get("content") or data.get("response") or ""
    return strip_thinking(str(answer))


async def call_ollama(message: str, model: str, system_prompt: str | None = None) -> str:
    return await call_ollama_with_options(
        message,
        model,
        system_prompt,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        max_tokens=LLM_MAX_TOKENS,
        num_ctx=OLLAMA_NUM_CTX,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
    )


async def call_intent_classifier(message: str, system_prompt: str) -> tuple[str, str]:
    model = intent_model()
    content = await call_ollama_with_options(
        message,
        model,
        system_prompt,
        temperature=INTENT_TEMPERATURE,
        top_p=1.0,
        max_tokens=INTENT_MAX_TOKENS,
        num_ctx=INTENT_NUM_CTX,
        timeout_seconds=INTENT_TIMEOUT_SECONDS,
        response_format=INTENT_RESULT_SCHEMA,
    )
    return content, model


async def call_online_llm(message: str, system_prompt: str | None = None) -> str:
    require_online_config()
    if online_uses_responses_api():
        payload: dict[str, Any] = {
            "model": LLM_MODEL,
            "input": message,
            "stream": False,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        prompt = (system_prompt or SYSTEM_PROMPT).strip()
        if prompt:
            payload["instructions"] = prompt
    else:
        payload = {
            "model": LLM_MODEL,
            "messages": chat_messages(message, system_prompt),
            "stream": False,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
    if LLM_MAX_TOKENS_FIELD:
        payload[LLM_MAX_TOKENS_FIELD] = LLM_MAX_TOKENS
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT_SECONDS, read=LLM_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(online_completion_url(), headers=online_headers(), json=payload)
        response.raise_for_status()
    return strip_thinking(extract_online_answer(response_json(response)))


async def call_local_result(
    message: str,
    *,
    system_prompt: str | None = None,
    fallback: bool = False,
    online_status: str | None = None,
) -> LLMResult:
    model = require_local_model()
    answer = await call_ollama(message, model, system_prompt)
    return LLMResult(answer=answer, route="local", model=model, fallback=fallback, online_status=online_status)


async def call_online_result(message: str, system_prompt: str | None = None) -> LLMResult:
    answer = await call_online_llm(message, system_prompt)
    return LLMResult(answer=answer, route="online", model=LLM_MODEL, online_status=ONLINE_REACHABILITY.status)


async def call_llm(
    message: str,
    system_prompt: str | None = None,
) -> LLMResult:
    reachability = ONLINE_REACHABILITY
    online_ok = reachability.online

    if provider_is_local():
        return await call_local_result(message, system_prompt=system_prompt)
    if provider_is_online():
        if online_ok:
            try:
                return await call_online_result(message, system_prompt)
            except httpx.HTTPError as exc:
                log(f"online LLM failed; falling back to local: {exc}", level="warning")
                return await call_local_result(message, system_prompt=system_prompt, fallback=True, online_status="request_failed")
        log(f"online LLM unavailable; falling back to local: {reachability.status}", level="warning")
        return await call_local_result(message, system_prompt=system_prompt, fallback=True, online_status=reachability.status)
    raise LLMConfigError("未配置 LLM_ENGINE。")


async def check_ollama_health() -> str:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return "ok" if response.is_success else f"http_{response.status_code}"
    except httpx.HTTPError:
        return "unreachable"


async def probe_online_health() -> LLMReachability:
    try:
        base_url = require_online_config()
    except LLMConfigError as exc:
        return LLMReachability(online=False, status=f"config_error:{exc}", checked_at=time.time())

    try:
        async with httpx.AsyncClient(timeout=LLM_REACHABILITY_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url}{MODELS_PATH}", headers=online_headers())
            status = "ok" if response.is_success else f"http_{response.status_code}"
            return LLMReachability(online=response.is_success, status=status, checked_at=time.time())
    except httpx.HTTPError:
        return LLMReachability(online=False, status="unreachable", checked_at=time.time())


async def online_reachability_loop() -> None:
    global ONLINE_REACHABILITY

    interval = max(0.5, LLM_REACHABILITY_INTERVAL_SECONDS)
    while True:
        ONLINE_REACHABILITY = await probe_online_health()
        await asyncio.sleep(interval)


async def start_online_reachability_loop() -> None:
    global ONLINE_REACHABILITY_TASK, ONLINE_REACHABILITY
    log(
        f"llm service ready: engine={LLM_ENGINE} "
        f"online_provider={LLM_ONLINE_PROVIDER_DISPLAY} local_model={OLLAMA_MODEL or 'unset'}"
    )
    if provider_is_online():
        ONLINE_REACHABILITY = await probe_online_health()
        ONLINE_REACHABILITY_TASK = asyncio.create_task(online_reachability_loop())


async def stop_online_reachability_loop() -> None:
    if ONLINE_REACHABILITY_TASK is None:
        return
    ONLINE_REACHABILITY_TASK.cancel()
    try:
        await ONLINE_REACHABILITY_TASK
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_online_reachability_loop()
    try:
        yield
    finally:
        await stop_online_reachability_loop()


app = FastAPI(title="Chat2Me LLM", version="0.1.0", lifespan=lifespan)


@app.get("/llm/reachability", response_model=ReachabilityResponse)
async def llm_reachability() -> ReachabilityResponse:
    if not provider_is_online():
        return ReachabilityResponse(
            online=False,
            provider=LLM_ENGINE or "unconfigured",
            model=LLM_MODEL or None,
            status="online_provider_disabled",
        )

    reachability = ONLINE_REACHABILITY
    return ReachabilityResponse(
        online=reachability.online,
        provider=LLM_ONLINE_PROVIDER_DISPLAY,
        model=LLM_MODEL or None,
        status=reachability.status,
        checked_at=reachability.checked_at,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_status = await check_ollama_health()
    reachability = ONLINE_REACHABILITY
    if provider_is_online():
        status = "ok" if reachability.online or ollama_status == "ok" else "degraded"
        return HealthResponse(
            status=status,
            provider=LLM_ONLINE_PROVIDER_DISPLAY,
            model=LLM_MODEL or None,
            llm=reachability.status,
            ollama=ollama_status,
            online=reachability.online,
            online_status=reachability.status,
        )

    status = "ok" if ollama_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        provider=LLM_ENGINE,
        model=OLLAMA_MODEL or None,
        llm=ollama_status,
        ollama=ollama_status,
        online=False,
        online_status="online_provider_disabled",
    )


def short_detail(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    started_at = time.perf_counter()
    try:
        result = await call_llm(
            request.message.strip(),
            request.system_prompt,
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=short_detail(exc.response.text)) from exc
    except httpx.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="大模型生成超时。") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"大模型服务不可用：{exc}") from exc

    return ChatResponse(
        answer=result.answer or EMPTY_ANSWER_RESPONSE,
        route=result.route,
        model=result.model,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        fallback=result.fallback,
        online_status=result.online_status,
    )


@app.post("/intent", response_model=IntentResponse)
async def intent(request: IntentRequest) -> IntentResponse:
    started_at = time.perf_counter()
    try:
        content, model = await call_intent_classifier(request.message.strip(), request.system_prompt)
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=short_detail(exc.response.text)) from exc
    except httpx.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="意图理解超时。") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"本地意图模型不可用：{exc}") from exc

    return IntentResponse(
        content=content,
        route="local_intent",
        model=model,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
