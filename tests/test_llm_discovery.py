"""Tests for Ollama and Generic LLM discovery endpoints (Phase 6A.1)."""

import asyncio
import json
import pytest
from unittest.mock import patch

from local_anonymizer.llm.provider import (
    derive_ollama_base_url,
    fetch_ollama_models,
    fetch_generic_models,
    MAX_RESPONSE_BYTES,
)
from local_anonymizer.llm.schema import DiscoveryResult


class MockResponse:
    def __init__(self, status=200, headers=None, body=b""):
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
    def __init__(self, get_resp=None, get_exc=None):
        self._get_resp = get_resp
        self._get_exc = get_exc

    def get(self, *args, **kwargs):
        if self._get_exc:
            raise self._get_exc
        return self._get_resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_derive_ollama_base_url():
    assert derive_ollama_base_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert derive_ollama_base_url("http://localhost:11434/v1/chat/completions") == "http://localhost:11434"
    assert derive_ollama_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert derive_ollama_base_url("http://[::1]:11434/api") == "http://[::1]:11434"

    # Non loopback blocked
    with pytest.raises(ValueError, match="keine lokale Loopback-Adresse"):
        derive_ollama_base_url("http://remote-server.com:11434/v1")


@pytest.mark.asyncio
async def test_fetch_ollama_models_success():
    mock_tags_response = {
        "models": [
            {"name": "qwen3:8b", "size": 4900000000},
            {"name": "ministral-3:8b", "size": 5100000000},
            {"name": "cloud-model:cloud", "size": 0},  # Filtered out
        ]
    }
    resp = MockResponse(200, body=json.dumps(mock_tags_response).encode("utf-8"))
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "success"
        assert res.models == ["qwen3:8b", "ministral-3:8b"]
        assert "cloud-model:cloud" not in res.models


@pytest.mark.asyncio
async def test_fetch_ollama_models_empty():
    resp = MockResponse(200, body=b'{"models": []}')
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "empty"
        assert res.models == []


@pytest.mark.asyncio
async def test_fetch_ollama_models_http_error():
    resp = MockResponse(500, body=b"Internal error")
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "invalid_response"
        assert "500" in res.message


@pytest.mark.asyncio
async def test_fetch_ollama_models_timeout():
    session = MockSession(get_exc=asyncio.TimeoutError())

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "timeout"


@pytest.mark.asyncio
async def test_fetch_ollama_models_oversized():
    resp = MockResponse(200, body=b"x" * (MAX_RESPONSE_BYTES + 100))
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "invalid_response"
        assert "Gr\xc3\xb6\xc3\x9fenlimit" in res.message or "Größenlimit" in res.message


@pytest.mark.asyncio
async def test_fetch_generic_models_success():
    mock_generic_response = {
        "data": [
            {"id": "local-qwen"},
            {"id": "local-mistral"},
        ]
    }
    resp = MockResponse(200, body=json.dumps(mock_generic_response).encode("utf-8"))
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_generic_models("http://127.0.0.1:1234/v1")
        assert res.status == "success"
        assert res.models == ["local-qwen", "local-mistral"]


@pytest.mark.asyncio
async def test_fetch_ollama_models_invalid_content_type():
    resp = MockResponse(200, headers={"Content-Type": "text/html"}, body=b"<html>error</html>")
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "invalid_response"
        assert "Content-Type" in res.message


@pytest.mark.asyncio
async def test_fetch_ollama_models_missing_models_key():
    resp = MockResponse(200, body=b'{"error": "not found"}')
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "invalid_response"
        assert "models" in res.message


@pytest.mark.asyncio
async def test_fetch_ollama_models_not_a_list():
    resp = MockResponse(200, body=b'{"models": "not-a-list"}')
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_ollama_models("http://127.0.0.1:11434/v1")
        assert res.status == "invalid_response"
        assert "Array" in res.message or "models" in res.message


@pytest.mark.asyncio
async def test_fetch_generic_models_invalid_content_type():
    resp = MockResponse(200, headers={"Content-Type": "text/plain"}, body=b"plain text")
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_generic_models("http://127.0.0.1:1234/v1")
        assert res.status == "invalid_response"
        assert "Content-Type" in res.message


@pytest.mark.asyncio
async def test_fetch_generic_models_missing_data_key():
    resp = MockResponse(200, body=b'{"object": "list"}')
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_generic_models("http://127.0.0.1:1234/v1")
        assert res.status == "invalid_response"
        assert "data" in res.message


@pytest.mark.asyncio
async def test_fetch_generic_models_data_not_a_list():
    resp = MockResponse(200, body=b'{"data": {"id": "single-model"}}')
    session = MockSession(get_resp=resp)

    with patch("aiohttp.ClientSession", return_value=session):
        res = await fetch_generic_models("http://127.0.0.1:1234/v1")
        assert res.status == "invalid_response"
        assert "Array" in res.message or "data" in res.message


@pytest.mark.asyncio
async def test_read_limited_body_chunked_streaming_abort():
    """Verify that _read_limited_body aborts mid-stream without buffering all data when limit is exceeded."""
    from local_anonymizer.llm.provider import _read_limited_body

    class ChunkedStream:
        def __init__(self, chunk_count=500, chunk_size=10000):
            self.chunk_count = chunk_count
            self.chunk_size = chunk_size
            self.yielded = 0

        async def iter_chunked(self, n):
            for _ in range(self.chunk_count):
                self.yielded += 1
                yield b"x" * self.chunk_size

    class ChunkedResp:
        def __init__(self):
            self.content = ChunkedStream()

    resp = ChunkedResp()
    with pytest.raises(ValueError, match="Größenlimit|Limit"):
        await _read_limited_body(resp, max_bytes=50000)

    # Aborted early after ~6 chunks, NOT reading through all 500 chunks
    assert resp.content.yielded < 10
