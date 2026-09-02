"""Tests for LLM preload and AppState setup lifecycle (Phase 6A.1)."""

import asyncio
import json
import pytest
from unittest.mock import patch

from app import AppState, cleanup_session_async, reset_app_state
from local_anonymizer.llm.provider import (
    preload_ollama_model,
    test_generic_connection as verify_generic_connection,
    parse_iso_expiry,
)
from local_anonymizer.llm.schema import PsModelInfo


class MockResponse:
    def __init__(self, status=200, headers=None, body=b'{"response": ""}'):
        self.status = status
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self._body = body

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self, responses_by_url=None, default_resp=None):
        self._responses = responses_by_url or {}
        self._default = default_resp or MockResponse(200, body=b'{"response": ""}')

    def post(self, url, *args, **kwargs):
        resp = self._responses.get(url, self._default)
        return resp

    def get(self, url, *args, **kwargs):
        resp = self._responses.get(url, self._default)
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_preload_ollama_model_success():
    gen_url = "http://127.0.0.1:11434/api/generate"
    ps_url = "http://127.0.0.1:11434/api/ps"

    ps_payload = {
        "models": [
            {
                "name": "qwen3:8b",
                "model": "qwen3:8b",
                "size": 4900000000,
                "size_vram": 4900000000,
                "expires_at": "2026-09-02T12:00:00Z",
            }
        ]
    }
    responses = {
        gen_url: MockResponse(200, body=b'{"response": ""}'),
        ps_url: MockResponse(200, body=json.dumps(ps_payload).encode("utf-8")),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        info = await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")
        assert isinstance(info, PsModelInfo)
        assert info.name == "qwen3:8b"
        assert info.size_vram == 4900000000
        assert info.expires_at == "2026-09-02T12:00:00Z"


@pytest.mark.asyncio
async def test_preload_ollama_model_not_running_error():
    gen_url = "http://127.0.0.1:11434/api/generate"
    ps_url = "http://127.0.0.1:11434/api/ps"

    responses = {
        gen_url: MockResponse(200, body=b'{"response": ""}'),
        ps_url: MockResponse(200, body=b'{"models": []}'),  # Empty running models list
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="nicht als aktiv in Ollama gemeldet"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_preload_cloud_model_rejected():
    with pytest.raises(ValueError, match="Cloud-Modelle"):
        await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b:cloud")


@pytest.mark.asyncio
async def test_verify_generic_connection_success():
    session = MockSession(default_resp=MockResponse(200, body=b'{"data": [{"id": "local-model"}]}'))

    with patch("aiohttp.ClientSession", return_value=session):
        ok = await verify_generic_connection("http://127.0.0.1:1234/v1")
        assert ok is True


@pytest.mark.asyncio
async def test_app_state_setup_lifecycle():
    import time
    st = AppState()
    assert st.llm_setup_state == "idle"
    assert not st.is_model_ready()

    # Simulate loaded model
    st.config.llm_base_url = "http://127.0.0.1:11434/v1"
    st.config.llm_model_name = "qwen3:8b"
    st.llm_provider_type = "ollama"
    st.llm_ready_info = PsModelInfo(name="qwen3:8b", model="qwen3:8b")
    st.llm_ready_bound_url = "http://127.0.0.1:11434/v1"
    st.llm_ready_bound_model = "qwen3:8b"
    st.llm_ready_expires_at = time.time() + 300.0
    assert st.is_model_ready() is True

    # Invalidate if model name changes
    st.config.llm_model_name = "ministral-3:8b"
    assert st.is_model_ready() is False

    st.invalidate_llm_ready()
    assert st.llm_ready_info is None

    # Test cancel_setup_task
    async def dummy_long_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(dummy_long_task())
    st.llm_setup_task = task
    st.llm_setup_state = "preloading"
    await st.cancel_setup_task()
    assert st.llm_setup_state == "idle"
    assert st.llm_setup_task is None
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cleanup_session_async():
    class DummyProvider:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    st = AppState()
    provider = DummyProvider()
    st.llm_provider = provider
    await cleanup_session_async(st)
    assert st.llm_provider is None
    assert provider.closed is True


@pytest.mark.asyncio
async def test_preload_ollama_prefix_mismatch_rejected():
    """Ensure that prefix matches (e.g. qwen3:8b-preview matching qwen3:8b) are strictly rejected."""
    ps_resp = {
        "models": [
            {
                "name": "qwen3:8b-preview",
                "model": "qwen3:8b-preview",
                "size_vram": 4500000000,
            }
        ]
    }
    generate_url = "http://127.0.0.1:11434/api/generate"
    ps_url = "http://127.0.0.1:11434/api/ps"
    responses = {
        generate_url: MockResponse(200, body=b'{"response": ""}'),
        ps_url: MockResponse(200, body=json.dumps(ps_resp).encode("utf-8")),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="nicht als aktiv in Ollama gemeldet"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


def test_app_state_is_model_ready_expiration():
    """Verify that is_model_ready auto-invalidates when keep-alive expiry timestamp has passed."""
    import time
    st = AppState()
    st.config.llm_base_url = "http://127.0.0.1:11434/v1"
    st.config.llm_model_name = "qwen3:8b"
    st.llm_provider_type = "ollama"
    st.llm_ready_info = PsModelInfo(name="qwen3:8b", model="qwen3:8b")
    st.llm_ready_bound_url = "http://127.0.0.1:11434/v1"
    st.llm_ready_bound_model = "qwen3:8b"

    # Future expiration: model is ready
    st.llm_ready_expires_at = time.time() + 300.0
    assert st.is_model_ready() is True

    # Past expiration: model is no longer ready and invalidated
    st.llm_ready_expires_at = time.time() - 10.0
    assert st.is_model_ready() is False
    assert st.llm_ready_info is None
    assert st.llm_ready_expires_at == 0.0


def test_app_state_mutual_exclusion_is_busy():
    """Verify mutual exclusion busy lock across analysis, triage, and setup states."""
    st = AppState()
    assert st.is_busy is False

    st.is_analyzing = True
    assert st.is_busy is True
    st.is_analyzing = False

    st.is_llm_running = True
    assert st.is_busy is True
    st.is_llm_running = False

    st.llm_setup_state = "discovering"
    assert st.is_busy is True
    st.llm_setup_state = "preloading"
    assert st.is_busy is True
    st.llm_setup_state = "testing"
    assert st.is_busy is True
    st.llm_setup_state = "idle"
    assert st.is_busy is False


@pytest.mark.asyncio
async def test_load_document_into_state_async_cleans_provider_and_readiness():
    """Verify that switching document asynchronously closes provider and resets readiness."""
    import time
    class DummyProvider:
        def __init__(self):
            self.closed = False
        async def close(self):
            self.closed = True

    from app import load_document_into_state_async

    st = AppState()
    provider = DummyProvider()
    st.llm_provider = provider
    st.config.llm_base_url = "http://127.0.0.1:11434/v1"
    st.config.llm_model_name = "qwen3:8b"
    st.llm_provider_type = "ollama"
    st.llm_ready_info = PsModelInfo(name="qwen3:8b", model="qwen3:8b")
    st.llm_ready_bound_url = "http://127.0.0.1:11434/v1"
    st.llm_ready_bound_model = "qwen3:8b"
    st.llm_ready_expires_at = time.time() + 300.0
    st.llm_setup_request_id = "req-12345"

    assert st.is_model_ready() is True

    await load_document_into_state_async(st, "Neuer Text", "neu.txt")

    assert provider.closed is True
    assert st.llm_provider is None
    assert st.llm_setup_request_id == ""
    assert st.is_model_ready() is False
    assert st.filename == "neu.txt"
    assert st.raw_text == "Neuer Text"


@pytest.mark.asyncio
async def test_preload_ollama_generate_invalid_mime_rejected():
    """Verify that invalid MIME on /api/generate is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, headers={"Content-Type": "text/html"}, body=b"<html>error</html>"),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="Ungültiger Content-Type bei /api/generate"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_preload_ollama_generate_error_response_rejected():
    """Verify that explicit error JSON from /api/generate is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, body=b'{"error": "model \'qwen3:8b\' not found"}'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="Ollama meldete Fehler beim Vorladen"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_preload_ollama_generate_invalid_json_rejected():
    """Verify that non-JSON response from /api/generate is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, body=b'NOT JSON AT ALL'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="Antwort ist kein gültiges JSON"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


def test_parse_iso_expiry():
    """Verify strict ISO expiry parsing without fallback on invalid input."""
    from datetime import datetime, timezone
    now_iso = "2026-09-02T12:00:00Z"
    ts = parse_iso_expiry(now_iso)
    assert ts > 0.0

    # Invalid / empty
    assert parse_iso_expiry(None) == 0.0
    assert parse_iso_expiry("") == 0.0
    assert parse_iso_expiry("invalid-date-format") == 0.0
