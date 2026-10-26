import json

import httpx
import pytest

from test_motion_integration import ROOT, core, load_module, run


@pytest.mark.parametrize("text", [
    "江淮中心主任是谁", "江淮中心有多少研究员", "谁做的这台相机",
    "介绍一下空间机器人", "不要点头",
])
def test_unrelated_questions_are_not_captured_by_short_patterns(core, text):
    assert core.match_fixed_qa(text) is None


def test_nod_capability_and_command_share_one_entry(core):
    assert core.match_fixed_qa("你会点头吗")[0] == "nod"
    assert core.match_fixed_qa("点个头")[0] == "nod"
    assert len([item for _, item in core.fixed_qa_items() if item.get("action") == "nod"]) == 1


def test_chat_receives_identity_and_facts_without_action_replies(core, monkeypatch):
    monkeypatch.setattr(core, "profile", lambda: {
        "robot": {"persona": "你是小银"},
        "fixed_facts": ["你隶属江淮中心"],
        "fixed_qa": [
            {"answer": "我是机器人"},
            {"action": "nod", "answer": "不要放进资料"},
        ],
        "system_prompt": "通常回答100到150字",
    })
    prompt = core.build_chat_prompt()
    for text in ["你是小银", "你隶属江淮中心", "我是机器人", "通常回答100到150字"]:
        assert text in prompt
    assert "不要放进资料" not in prompt


def test_intent_catalog_omits_fast_match_patterns(core):
    assert all("patterns" not in item for item in core.fixed_qa_catalog())
    assert all(not core.fixed_qa_item_by_id(item["id"]).get("action") for item in core.fixed_qa_catalog())


@pytest.fixture
def llm(core):
    return load_module("review_dialogue_llm", ROOT / "services/llm/app/main.py")


def test_only_intent_requests_constrain_ollama_output(llm, monkeypatch):
    requests = []
    content = json.dumps({"intent": "chat", "fixed_qa_id": None, "confidence": 1})

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": content}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    result, _ = run(llm.call_intent_classifier("讲个笑话", "只输出分类JSON"))
    assert result == content
    schema = requests[-1]["format"]
    assert set(schema["required"]) == {"intent", "fixed_qa_id", "confidence"}
    assert schema["properties"]["intent"]["enum"] == [
        "blocked", "fixed_qa", "direction", "session_end", "invalid_input", "chat",
    ]
    assert requests[-1]["options"]["num_ctx"] == llm.INTENT_NUM_CTX
    assert requests[-1]["options"]["num_predict"] == llm.INTENT_MAX_TOKENS

    run(llm.call_ollama("讲个笑话", "test-model", "你是小银"))
    assert "format" not in requests[-1]
    assert requests[-1]["options"]["num_predict"] == llm.LLM_MAX_TOKENS
    assert requests[-1]["messages"][0] == {"role": "system", "content": "你是小银"}
