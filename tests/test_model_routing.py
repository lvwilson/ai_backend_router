"""GPU-free model resolution and chat proxy regression tests."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from router import create_app
from src.config import RouterConfig

DEFAULT = "qwen3.8-27b"
OTHER = "qwen3.6-27b-instruct"


@pytest.mark.parametrize(("requested", "expected"), [
    (DEFAULT, DEFAULT),
    (OTHER, OTHER),
    ("QWEN3.6-27B-INSTRUCT", OTHER),
    ("qwen-3.8-27b", DEFAULT),
    ("qwen-3.6-27b-instruct", OTHER),
    (" QWEN_3.6_27B_INSTRUCT ", OTHER),
    ("nonexistent-model", DEFAULT),
    ("gpt-4o", DEFAULT),
    ("qwen3.7-27b-instruct", DEFAULT),  # Do not guess a different version
    ("qwen3.6-72b-instruct", DEFAULT),  # Do not guess a different size
    ("", DEFAULT),
    (None, OTHER),  # Existing omitted-model behavior is unchanged
])
def test_resolve_llama(requested, expected):
    config = RouterConfig(llama_backends=[OTHER, DEFAULT])
    assert config.resolve_llama(requested) == expected


def test_exact_match_wins_over_separator_collision():
    config = RouterConfig(llama_backends=["test-model", "test_model", DEFAULT])
    assert config.resolve_llama("test_model") == "test_model"
    assert config.resolve_llama("TEST-MODEL") == "test-model"
    assert config.resolve_llama("test.model") == DEFAULT


def test_fallback_absent():
    config = RouterConfig(llama_backends=[OTHER])
    assert config.resolve_llama("qwen-3.6-27b-instruct") == OTHER
    with pytest.raises(KeyError, match="fallback.*not configured"):
        config.resolve_llama("unknown")


def test_no_chat_backends():
    with pytest.raises(KeyError, match="No llama backends"):
        RouterConfig().resolve_llama("unknown")


def test_embedding_resolution_unchanged():
    config = RouterConfig(
        llama_backends=[DEFAULT], embedding_backends=["qwen3-embedding-8b"]
    )
    with pytest.raises(KeyError, match="Unknown embedding model"):
        config.resolve_embedding("unknown")


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize(("requested", "expected"), [
    ("unknown-model", DEFAULT),
    ("qwen-3.6-27b-instruct", OTHER),
    ("QWEN3.6-27B-INSTRUCT", OTHER),
    (OTHER, OTHER),
    (None, OTHER),
])
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_proxy_rewrites_only_model(monkeypatch, path, requested, expected, stream):
    config = RouterConfig(llama_backends=[OTHER, DEFAULT])
    app = create_app(config)
    orch = app.state.orch
    acquire = AsyncMock()
    release = AsyncMock()
    ensure = AsyncMock(return_value=SimpleNamespace(config=SimpleNamespace(port=18081)))
    monkeypatch.setattr(orch, "acquire_slot", acquire)
    monkeypatch.setattr(orch, "release_slot", release)
    monkeypatch.setattr(orch, "ensure_running", ensure)

    response_body = b'data: {"ok":true}\n\n' if stream else b'{"ok":true}'

    async def chunks():
        yield response_body

    upstream = SimpleNamespace(
        status=200,
        headers={"content-type": "text/event-stream" if stream else "application/json"},
        content=SimpleNamespace(iter_any=chunks),
        release=Mock(),
    )
    send = AsyncMock(return_value=upstream)
    app.state.http = SimpleNamespace(request=send)
    payload = {
        "messages": [{"role": "user", "content": "Hello \u4e16\u754c"}],
        "max_tokens": 8,
        "stream": stream,
        "temperature": 0.2,
    }
    if requested is not None:
        payload["model"] = requested
    raw = json.dumps(payload, indent=2).encode()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            path, content=raw, headers={"content-type": "application/json"}
        )

    assert response.status_code == 200
    assert response.content == response_body
    acquire.assert_awaited_once_with(expected)
    ensure.assert_awaited_once_with(expected)
    release.assert_awaited_once_with(expected)
    upstream.release.assert_called_once()
    assert send.call_args.args == ("POST", f"http://127.0.0.1:18081{path}")
    forwarded = send.call_args.kwargs["data"]
    expected_payload = dict(payload)
    if requested is not None:
        expected_payload["model"] = expected
    assert json.loads(forwarded) == expected_payload
    assert "content-length" not in send.call_args.kwargs["headers"]
    if requested is None or requested == expected:
        assert forwarded == raw


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
async def test_missing_fallback_returns_400_without_launching(path, monkeypatch):
    app = create_app(RouterConfig(llama_backends=[OTHER]))
    ensure = AsyncMock()
    monkeypatch.setattr(app.state.orch, "ensure_running", ensure)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(path, json={"model": "unknown"})
    assert response.status_code == 400
    assert "fallback" in response.json()["error"]["message"]
    ensure.assert_not_awaited()
