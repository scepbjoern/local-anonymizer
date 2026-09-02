"""Tests for LLM preload and AppState setup lifecycle (Phase 6A.1)."""

import asyncio
import json
import pytest
from unittest.mock import patch

from app import (
    AppState,
    cleanup_session_async,
    reset_app_state,
    launch_llm_triage_for_state,
    run_llm_triage_for_state,
    EntityGroup,
    EntityOccurrence,
)
from local_anonymizer.llm.provider import (
    preload_ollama_model,
    test_generic_connection as verify_generic_connection,
    verify_ollama_model_running,
    parse_iso_expiry,
    LocalApiProvider,
)
from local_anonymizer.llm.schema import PsModelInfo, TriageEnvelope, TriageKeepItem


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
        gen_url: MockResponse(200, body=b'{"model": "qwen3:8b", "response": "", "done": true, "done_reason": "load"}'),
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
        gen_url: MockResponse(200, body=b'{"model": "qwen3:8b", "response": "", "done": true, "done_reason": "load"}'),
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
        generate_url: MockResponse(200, body=b'{"model": "qwen3:8b", "response": "", "done": true, "done_reason": "load"}'),
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
    """Verify strict ISO expiry parsing requiring explicit timezone without fallback on invalid input."""
    now_iso = "2026-09-02T12:00:00Z"
    ts = parse_iso_expiry(now_iso)
    assert ts > 0.0

    offset_iso = "2026-09-02T14:00:00+02:00"
    ts_offset = parse_iso_expiry(offset_iso)
    assert ts_offset > 0.0

    # Naive timestamp without timezone MUST be rejected
    assert parse_iso_expiry("2026-09-02T12:00:00") == 0.0

    # Invalid / empty
    assert parse_iso_expiry(None) == 0.0
    assert parse_iso_expiry("") == 0.0
    assert parse_iso_expiry("invalid-date-format") == 0.0


@pytest.mark.asyncio
async def test_preload_ollama_empty_response_rejected():
    """Verify that empty JSON object from /api/generate is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, body=b'{}'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="nicht abgeschlossen"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_preload_ollama_not_done_rejected():
    """Verify that preload response with done=false is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, body=b'{"model": "qwen3:8b", "done": false}'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="nicht abgeschlossen"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_preload_ollama_wrong_model_rejected():
    """Verify that preload response with conflicting model name is rejected."""
    gen_url = "http://127.0.0.1:11434/api/generate"
    responses = {
        gen_url: MockResponse(200, body=b'{"model": "wrong-model:1b", "done": true}'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(RuntimeError, match="Modell-Identit.+tskonflikt"):
            await preload_ollama_model("http://127.0.0.1:11434/v1", "qwen3:8b")


@pytest.mark.asyncio
async def test_verify_ollama_model_running_active():
    """Verify that verify_ollama_model_running detects active unexpired running model."""
    import time
    ps_resp = {
        "models": [
            {
                "name": "qwen3:8b",
                "model": "qwen3:8b",
                "size_vram": 4900000000,
                "expires_at": "2026-09-02T12:00:00Z",
            }
        ]
    }
    responses = {
        "http://127.0.0.1:11434/api/ps": MockResponse(200, body=json.dumps(ps_resp).encode("utf-8")),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("local_anonymizer.llm.provider.parse_iso_expiry", return_value=time.time() + 300.0):
            info = await verify_ollama_model_running("http://127.0.0.1:11434/v1", "qwen3:8b")
            assert info is not None
            assert info.name == "qwen3:8b"


@pytest.mark.asyncio
async def test_verify_ollama_model_running_externally_unloaded():
    """Verify that verify_ollama_model_running returns None when model is unloaded in Ollama."""
    responses = {
        "http://127.0.0.1:11434/api/ps": MockResponse(200, body=b'{"models": []}'),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        info = await verify_ollama_model_running("http://127.0.0.1:11434/v1", "qwen3:8b")
        assert info is None


@pytest.mark.asyncio
async def test_verify_ollama_model_running_expired():
    """Verify that verify_ollama_model_running returns None when model expiry timestamp has passed."""
    ps_resp = {
        "models": [
            {
                "name": "qwen3:8b",
                "model": "qwen3:8b",
                "expires_at": "2020-01-01T00:00:00Z",
            }
        ]
    }
    responses = {
        "http://127.0.0.1:11434/api/ps": MockResponse(200, body=json.dumps(ps_resp).encode("utf-8")),
    }
    session = MockSession(responses_by_url=responses)

    with patch("aiohttp.ClientSession", return_value=session):
        info = await verify_ollama_model_running("http://127.0.0.1:11434/v1", "qwen3:8b")
        assert info is None


@pytest.mark.asyncio
async def test_launch_llm_triage_manual_end_to_end():
    """Verify that manual triage launch executes worker, hits provider, and populates results."""
    st = AppState()
    st.raw_text = "Dr. Anna Keller arbeitet am Spital Zürich."
    occ = EntityOccurrence(
        start=0,
        end=15,
        score=1.0,
        context_html="<b>Dr. Anna Keller</b> arbeitet am Spital Zürich.",
        needs_review=False,
    )
    grp = EntityGroup(
        original_text="Dr. Anna Keller",
        entity_type="PERSON",
        group_id="grp-1",
    )
    grp.occurrences.append(occ)
    st.entity_groups = [grp]
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"
    st.config.llm_base_url = "http://127.0.0.1:11434/v1"

    # Mock provider generate
    mock_resp_json = {
        "schema_version": "1.0.0",
        "request_id": "",
        "document_revision": st.document_revision,
        "snapshot_hash": "",
        "items": [
            {
                "occ_id": occ.occ_id,
                "action": "keep",
                "confidence": "high",
                "reasoning": "Echte Person",
                "descriptor_suggestion": "ÄRZTIN",
            }
        ],
    }

    class MockProvider:
        model_name = "qwen3:8b"
        base_url = "http://127.0.0.1:11434/v1"
        call_count = 0

        async def generate(self, prompt, system_prompt=""):
            self.call_count += 1
            import re
            from local_anonymizer.llm.apply_service import compute_triage_snapshot
            snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
            req_match = re.search(r'"request_id":\s*"([^"]+)"', prompt)
            req_id = req_match.group(1) if req_match else "req-1"
            return json.dumps({
                "schema_version": "1.0",
                "request_id": req_id,
                "document_revision": st.document_revision,
                "document_hash": snap,
                "items": [
                    {
                        "occ_id": occ.occ_id,
                        "action": "keep",
                        "confidence": "high",
                        "reasoning": "Echte Person",
                        "descriptor_suggestion": "ÄRZTIN",
                    }
                ],
            })

        async def close(self):
            pass

    provider = MockProvider()
    st.llm_provider = provider

    notifications = []
    task = launch_llm_triage_for_state(
        st,
        triggered_from_analysis=False,
        notify_cb=lambda msg, t: notifications.append((msg, t)),
    )
    assert task is not None
    assert st.is_llm_running is True

    await task

    assert st.is_llm_running is False
    assert st.llm_active_task is None
    assert provider.call_count == 1
    assert len(st.llm_triage_results) == 1
    res = st.llm_triage_results[occ.occ_id]
    assert res.action == "keep"
    assert res.descriptor_suggestion == "ÄRZTIN"
    assert any("abgeschlossen" in m[0] for m in notifications)


@pytest.mark.asyncio
async def test_launch_llm_triage_auto_review_end_to_end():
    """Verify that auto-review launch seamlessly takes over ownership from analysis and finishes cleanly."""
    st = AppState()
    st.raw_text = "Prof. Müller hält einen Vortrag."
    occ = EntityOccurrence(
        start=0,
        end=12,
        score=1.0,
        context_html="<b>Prof. Müller</b> hält einen Vortrag.",
        needs_review=False,
    )
    grp = EntityGroup(
        original_text="Prof. Müller",
        entity_type="PERSON",
        group_id="grp-2",
    )
    grp.occurrences.append(occ)
    st.entity_groups = [grp]
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"
    st.config.llm_base_url = "http://127.0.0.1:11434/v1"
    st.is_analyzing = True

    class MockProvider:
        model_name = "qwen3:8b"
        base_url = "http://127.0.0.1:11434/v1"

        async def generate(self, prompt, system_prompt=""):
            import re
            from local_anonymizer.llm.apply_service import compute_triage_snapshot
            snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
            req_match = re.search(r'"request_id":\s*"([^"]+)"', prompt)
            req_id = req_match.group(1) if req_match else "req-2"
            return json.dumps({
                "schema_version": "1.0",
                "request_id": req_id,
                "document_revision": st.document_revision,
                "document_hash": snap,
                "items": [
                    {
                        "occ_id": occ.occ_id,
                        "action": "keep",
                        "confidence": "high",
                        "reasoning": "Dozent",
                        "descriptor_suggestion": "DOZENT",
                    }
                ],
            })

        async def close(self):
            pass

    st.llm_provider = MockProvider()

    # Launch with triggered_from_analysis=True
    task = launch_llm_triage_for_state(st, triggered_from_analysis=True)
    assert task is not None
    assert st.is_analyzing is False
    assert st.is_llm_running is True

    await task

    assert st.is_llm_running is False
    assert len(st.llm_triage_results) == 1


@pytest.mark.asyncio
async def test_launch_llm_triage_double_click_protection():
    """Verify that rapid double-clicks return existing active task without starting duplicate runner."""
    st = AppState()
    st.raw_text = "Test Text"
    occ = EntityOccurrence(start=0, end=4, score=1.0, context_html="Test", needs_review=False)
    grp = EntityGroup(original_text="Test", entity_type="MISC", group_id="grp-3")
    grp.occurrences.append(occ)
    st.entity_groups = [grp]
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"

    class SlowProvider:
        model_name = "qwen3:8b"
        base_url = "http://127.0.0.1:11434/v1"
        call_count = 0

        async def generate(self, prompt, system_prompt=""):
            self.call_count += 1
            await asyncio.sleep(0.05)
            import re
            from local_anonymizer.llm.apply_service import compute_triage_snapshot
            snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
            req_match = re.search(r'"request_id":\s*"([^"]+)"', prompt)
            req_id = req_match.group(1) if req_match else "req-3"
            return json.dumps({
                "schema_version": "1.0",
                "request_id": req_id,
                "document_revision": st.document_revision,
                "document_hash": snap,
                "items": [{"occ_id": occ.occ_id, "action": "discard", "confidence": "high", "reasoning": "Generic"}],
            })

        async def close(self):
            pass

    provider = SlowProvider()
    st.llm_provider = provider

    task1 = launch_llm_triage_for_state(st, triggered_from_analysis=False)
    task2 = launch_llm_triage_for_state(st, triggered_from_analysis=False)

    assert task1 is not None
    assert task2 is task1  # Exact same task returned

    await task1
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_launch_llm_triage_rejected_when_setup_active():
    """Verify that launch is rejected when setup is discovering/preloading/testing."""
    st = AppState()
    st.raw_text = "Test"
    occ = EntityOccurrence(start=0, end=4, score=1.0, context_html="Test", needs_review=False)
    grp = EntityGroup(original_text="Test", entity_type="MISC", group_id="grp-4")
    grp.occurrences.append(occ)
    st.entity_groups = [grp]
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"

    st.llm_setup_state = "preloading"
    notifications = []
    task = launch_llm_triage_for_state(st, notify_cb=lambda msg, t: notifications.append((msg, t)))

    assert task is None
    assert st.is_llm_running is False
    assert any("Setup" in n[0] for n in notifications)


@pytest.mark.asyncio
async def test_launch_llm_triage_exception_resets_busy_state():
    """Verify that provider exception resets busy state and active task gracefully."""
    st = AppState()
    st.raw_text = "Test Text"
    occ = EntityOccurrence(start=0, end=4, score=1.0, context_html="Test", needs_review=False)
    grp = EntityGroup(original_text="Test", entity_type="MISC", group_id="grp-5")
    grp.occurrences.append(occ)
    st.entity_groups = [grp]
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"

    class FailingProvider:
        model_name = "qwen3:8b"
        base_url = "http://127.0.0.1:11434/v1"

        async def generate(self, prompt, system_prompt=""):
            raise RuntimeError("Connection dropped")

        async def close(self):
            pass

    st.llm_provider = FailingProvider()
    notifications = []
    task = launch_llm_triage_for_state(st, notify_cb=lambda msg, t: notifications.append((msg, t)))
    assert task is not None

    await task

    assert st.is_llm_running is False
    assert st.llm_active_task is None
    assert st.llm_partial_failure is True
    assert occ.occ_id in st.llm_unprocessed_occ_ids
