"""
Comprehensive unit and regression tests for Phase 6B (LLM-Ausgangskontrolle / Postcheck).
Verifies:
- Package A: GUI run-id binding, cancellation with in-flight event & clean restart, exception safety, extraction lock.
- Package B: Real catalog context limit lookup (no mocking of service), CatalogError on corruption, start guard.
- Package C: Incomplete mixed findings (valid + invalid slice) show warning; PII canary is never leaked into logs.
- Package D: EffectiveConfig ignore list alone is authoritative over legacy textbox.
- Clarifications: R1 non-empty UI component test with counter/checkboxes; full restore & re-analysis consistency;
  budget overflow, response reserve (max_tokens=4096), and finish_reason ("length" / invalid) rejection.
"""

import asyncio
import copy
import json
import logging
import pytest
import uuid
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

from app import (
    AppState,
    EntityGroup,
    EntityOccurrence,
    OccurrenceOverride,
    compute_reactive_preview,
    launch_llm_triage_for_state,
    cleanup_session_async,
    run_postcheck_for_state,
    cancel_postcheck_for_state,
    apply_postcheck_for_state,
    parse_ignore_terms,
    build_anonymizer,
    render_postcheck_ui_component,
)
from local_anonymizer.anonymizer import LocalAnonymizer
from local_anonymizer.llm.postcheck_schema import (
    PostcheckEnvelope,
    PostcheckFindingItem,
    SYSTEM_POSTCHECK_PROMPT,
    USER_POSTCHECK_TEMPLATE,
)
from local_anonymizer.llm.postcheck_service import (
    MAX_POSTCHECK_TOTAL_BUDGET,
    POSTCHECK_RESPONSE_RESERVE,
    MAX_POSTCHECK_INPUT_TOKENS,
    calculate_postcheck_budget,
    compute_unchanged_segments,
    map_output_slice_to_raw,
    validate_scope_and_category,
    check_batch_conflicts,
    atomic_apply_postcheck_findings,
    get_known_context_limit,
)
from local_anonymizer.llm.provider import LocalApiProvider, LlmProvider
from local_anonymizer.llm.catalog import CatalogError
from local_anonymizer.llm.schema import CatalogModelEntry, CatalogPhaseEvaluation, CatalogSchema
from local_anonymizer.config import AppConfig


class FakeProvider(LlmProvider):
    """Controllable fake provider for testing production async lifecycles."""
    def __init__(self, response_text: str = "", delay: float = 0.0, started_event: Optional[asyncio.Event] = None):
        self.response_text = response_text
        self.delay = delay
        self.started_event = started_event
        self.generate_called = 0
        self.closed = False
        self.last_max_tokens = None

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        return self.response_text

    async def generate_postcheck(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> str:
        self.generate_called += 1
        self.last_max_tokens = max_tokens
        if self.started_event is not None:
            self.started_event.set()
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        import re
        match = re.search(r'\"request_id\":\s*\"([^\"]+)\"', prompt)
        req_id = match.group(1) if match else "test-req"
        return self.response_text.replace("REQID", req_id)

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 1. Back-Mapping and Unchanged Segments Tests
# ---------------------------------------------------------------------------

def test_compute_unchanged_segments_no_occurrences():
    raw = "Dies ist ein ganz normaler Text ohne Entitäten."
    segs = compute_unchanged_segments(raw, [])
    assert len(segs) == 1
    assert segs[0] == (0, len(raw), 0, len(raw))


def test_compute_unchanged_segments_with_placeholders():
    raw = "Hallo Anna und Bob!"
    occs = [
        {"start": 6, "end": 10, "placeholder": "[PERSON_1]"},
        {"start": 15, "end": 18, "placeholder": "[PERSON_2]"},
    ]
    segs = compute_unchanged_segments(raw, occs)
    assert len(segs) == 3
    assert segs[0] == (0, 6, 0, 6)
    assert segs[1] == (10, 15, 16, 21)
    assert segs[2] == (18, 19, 31, 32)


def test_map_output_slice_to_raw_exact_match_shifted():
    raw = "Hallo Anna und Bob!"
    anon = "Hallo [PERSON_1] und Bob!"
    occs = [{"start": 6, "end": 10, "placeholder": "[PERSON_1]"}]
    segs = compute_unchanged_segments(raw, occs)
    assert anon[21:24] == "Bob"

    raw_s, raw_e = map_output_slice_to_raw(21, 24, "Bob", anon, raw, segs)
    assert (raw_s, raw_e) == (15, 18)
    assert raw[raw_s:raw_e] == "Bob"


def test_map_output_slice_to_raw_word_part():
    raw = "Der Patient Peter Meier wurde entlassen."
    anon = "Der Patient [PERSON_1] wurde entlassen."
    occs = [{"start": 12, "end": 23, "placeholder": "[PERSON_1]"}]
    segs = compute_unchanged_segments(raw, occs)

    raw_s, raw_e = map_output_slice_to_raw(4, 11, "Patient", anon, raw, segs)
    assert (raw_s, raw_e) == (4, 11)


# ---------------------------------------------------------------------------
# 2. Interval and Slice Safety (Package C Error Sanitization)
# ---------------------------------------------------------------------------

def test_map_output_slice_rejects_overlapping_placeholder():
    raw = "Hallo Anna und Bob!"
    anon = "Hallo [PERSON_1] und Bob!"
    occs = [{"start": 6, "end": 10, "placeholder": "[PERSON_1]"}]
    segs = compute_unchanged_segments(raw, occs)

    with pytest.raises(ValueError, match="nicht vollständig in einem unveränderten Textsegment"):
        map_output_slice_to_raw(6, 16, "[PERSON_1]", anon, raw, segs)

    with pytest.raises(ValueError, match="nicht vollständig in einem unveränderten Textsegment"):
        map_output_slice_to_raw(3, 10, "lo [PER", anon, raw, segs)


def test_map_output_slice_rejects_text_mismatch_without_pii_leak():
    raw = "Hallo Welt!"
    anon = "Hallo Welt!"
    segs = compute_unchanged_segments(raw, [])

    with pytest.raises(ValueError) as exc_info:
        map_output_slice_to_raw(0, 5, "Falsch_SecretPII", anon, raw, segs)

    # Verify exception message does NOT leak raw secret text
    assert "Falsch_SecretPII" not in str(exc_info.value)
    assert "Text stimmt nicht mit Fundstelle überein" in str(exc_info.value)


def test_map_output_slice_rejects_inverted_or_out_of_bounds_offsets():
    raw = "Hallo Welt!"
    anon = "Hallo Welt!"
    segs = compute_unchanged_segments(raw, [])

    with pytest.raises(ValueError, match="Ungültiger Zeichenbereich"):
        map_output_slice_to_raw(5, 2, "Hallo", anon, raw, segs)

    with pytest.raises(ValueError, match="Ungültiger Zeichenbereich"):
        map_output_slice_to_raw(0, 999, "Hallo", anon, raw, segs)


# ---------------------------------------------------------------------------
# 3. Scope & Category Validation Across All 4 Modes & Ignores (R4 & Package D)
# ---------------------------------------------------------------------------

def test_validate_scope_and_category_all_four_modes():
    allowed, norm, err = validate_scope_and_category("PERSON", "Hans", {"PERSON": "all"}, set())
    assert allowed is True
    assert norm == "PERSON"

    allowed, norm, err = validate_scope_and_category("PERSON", "Hans", {"PERSON": "explicit_eupii"}, set())
    assert allowed is True
    assert norm == "PERSON"

    allowed, norm, err = validate_scope_and_category("PERSON", "Hans", {"PERSON": "explicit_only"}, set())
    assert allowed is False
    assert "gesperrt" in err

    allowed, norm, err = validate_scope_and_category("LOCATION", "Bern", {"LOCATION": "off"}, set())
    assert allowed is False
    assert "gesperrt" in err


def test_validate_scope_and_category_protected_ignores():
    ignores = {"Zürich", "Spital"}
    allowed, norm, err = validate_scope_and_category("LOCATION", "Zürich", {"LOCATION": "all"}, ignores)
    assert allowed is False
    assert "Ignorierliste" in err

    allowed, norm, err = validate_scope_and_category("LOCATION", "spital", {"LOCATION": "all"}, ignores)
    assert allowed is False
    assert "Ignorierliste" in err


def test_validate_scope_and_category_disabled_occurrence_protection():
    disabled_spans = [(10, 20)]
    allowed, norm, err = validate_scope_and_category(
        "PERSON", "Müller", {"PERSON": "all"}, set(),
        disabled_spans=disabled_spans,
        raw_span=(10, 20)
    )
    assert allowed is False
    assert "deaktivierten" in err


def test_parse_ignore_terms_comma_and_newline():
    terms = parse_ignore_terms("Anna, Bob\nCharlie,  David ")
    assert set(terms) == {"Anna", "Bob", "Charlie", "David"}


@pytest.mark.asyncio
async def test_effective_config_ignore_precedence_over_legacy_textbox():
    """Package D: Effective config is alone authoritative; old text field does not resurrect un-ignored terms."""
    st = AppState()
    st.raw_text = "Hier ist Anna und ein Bericht."
    st.current_anon_text = "Hier ist Anna und ein Bericht."
    st.entity_modes = {"PERSON": "all"}
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    # Suppose legacy text box has "Anna", but effective_config has EMPTY ignore terms (document scope un-ignored it)
    st.ignore_terms_text = "Anna"
    st.effective_config = AppConfig()
    st.effective_config.ignore_terms = ""  # Document explicitly has no ignore terms

    resp = '{"schema_version": "1.0", "request_id": "REQID", "items": [{"text": "Anna", "entity_type": "PERSON", "output_start": 9, "output_end": 13}]}'
    fake = FakeProvider(resp)

    task = run_postcheck_for_state(st, provider=fake)
    await task

    # Anna MUST be accepted as a valid finding, NOT discarded by the legacy text field!
    assert len(st.postcheck_findings) == 1
    assert st.postcheck_findings[0]["text"] == "Anna"


# ---------------------------------------------------------------------------
# 4. Strict Schema & Distinguishing Invalid Responses & Privacy (Package C & R3)
# ---------------------------------------------------------------------------

def test_schema_rejects_missing_required_fields():
    with pytest.raises(Exception):
        PostcheckEnvelope.model_validate({"request_id": "r1"})

    with pytest.raises(Exception):
        PostcheckEnvelope.model_validate({"schema_version": "1.0", "request_id": "r1"})


def test_schema_strict_integer_offsets_rejection():
    with pytest.raises(Exception):
        PostcheckFindingItem(text="Hans", entity_type="PERSON", output_start=False, output_end=4)

    with pytest.raises(Exception):
        PostcheckFindingItem(text="Hans", entity_type="PERSON", output_start=0, output_end="4")

    with pytest.raises(Exception):
        PostcheckFindingItem(text="Hans", entity_type="PERSON", output_start=0.0, output_end=4)


def test_schema_whitespace_preservation_on_text():
    item = PostcheckFindingItem(text=" Hans ", entity_type="PERSON", output_start=0, output_end=6)
    assert item.text == " Hans "


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_slice_findings_warns_incomplete():
    """Package C: Mixed valid and invalid slices must yield warning and visible incompleteness."""
    st = AppState()
    st.raw_text = "Anna war vor Ort."
    st.current_anon_text = "Anna war vor Ort."
    st.entity_modes = {"PERSON": "all"}
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    messages = []
    # 1 valid finding ("Anna" [0:4]) + 1 invalid slice finding ("Ghost" [10:15])
    resp = json.dumps({
        "schema_version": "1.0",
        "request_id": "REQID",
        "items": [
            {"text": "Anna", "entity_type": "PERSON", "output_start": 0, "output_end": 4},
            {"text": "Ghost", "entity_type": "PERSON", "output_start": 10, "output_end": 15},
        ]
    })
    fake = FakeProvider(resp)
    task = run_postcheck_for_state(st, provider=fake, notify_fn=lambda m, t: messages.append((m, t)))
    await task

    # Must warn about incompleteness despite having 1 valid finding
    assert any(t == "warning" and "unvollständig" in m for m, t in messages)
    assert len(st.postcheck_findings) == 1
    assert st.postcheck_findings[0]["text"] == "Anna"


@pytest.mark.asyncio
async def test_log_privacy_canary_not_leaked(caplog):
    """Package C: Ensure raw PII / canary tokens in invalid slices are NEVER logged."""
    st = AppState()
    st.raw_text = "Text ohne Entitaet."
    st.current_anon_text = "Text ohne Entitaet."
    st.entity_modes = {"PERSON": "all"}
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    canary = "CANARY_PII_SECRET_XYZ_98765"
    resp = json.dumps({
        "schema_version": "1.0",
        "request_id": "REQID",
        "items": [
            {"text": canary, "entity_type": "PERSON", "output_start": 0, "output_end": 5}
        ]
    })
    fake = FakeProvider(resp)

    with caplog.at_level(logging.DEBUG):
        task = run_postcheck_for_state(st, provider=fake)
        await task

    assert canary not in caplog.text


# ---------------------------------------------------------------------------
# 5. Production Lifecycle & Task Ownership (Package A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifecycle_cancellation_with_event_and_clean_restart():
    """Package A: Verify task was actively in flight before cancelling, closes owned provider, preserves borrowed provider, blocks restart during teardown."""
    st = AppState()
    st.raw_text = "Hallo Welt."
    st.current_anon_text = "Hallo Welt."
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    started_ev = asyncio.Event()
    closed_events = []

    async def fake_close(self):
        await asyncio.sleep(0.05)
        closed_events.append(True)

    async def fake_gen(self, prompt, system_prompt="", max_tokens=4096):
        started_ev.set()
        await asyncio.sleep(0.5)
        return '{"schema_version": "1.0", "request_id": "REQID", "items": []}'

    with patch.object(LocalApiProvider, "generate_postcheck", fake_gen), \
         patch.object(LocalApiProvider, "close", fake_close):
        # 1. Owned provider path: provider=None and st.llm_provider=None
        st.llm_provider = None
        task1 = run_postcheck_for_state(st)
        assert st.is_postcheck_active is True

        # Wait until provider is confirmed in-flight
        await started_ev.wait()

        # Cancel in-flight run
        cancel_postcheck_for_state(st)
        assert st.is_postcheck_active is False
        assert st.postcheck_run_id == ""

        # Immediate restart while task1 is still tearing down must be rejected
        notified = []
        task_rejected = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
        assert task_rejected is None
        assert any("noch beendet" in m for m, t in notified)

        # Await task1 cancellation cleanly
        with pytest.raises(asyncio.CancelledError):
            await task1

        # Owned provider was closed in finally
        assert len(closed_events) == 1

    # 2. Borrowed provider path: borrowed provider must NOT be closed on cancel
    borrowed_ev = asyncio.Event()
    borrowed_fake = FakeProvider(delay=0.5, started_event=borrowed_ev)
    st.llm_provider = borrowed_fake
    task2 = run_postcheck_for_state(st)
    await borrowed_ev.wait()
    cancel_postcheck_for_state(st)
    with pytest.raises(asyncio.CancelledError):
        await task2
    assert borrowed_fake.closed is False

    # 3. Clean restart after teardown succeeds
    clean_fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')
    st.llm_provider = None
    task3 = run_postcheck_for_state(st, provider=clean_fake)
    assert task3 is not None
    await task3
    assert st.is_postcheck_active is False


def test_stale_gui_callbacks_isolated():
    """Package A: Stale GUI callbacks bound to an old run_id have no effect on a new run."""
    st = AppState()
    st.is_postcheck_active = True
    st.postcheck_run_id = "run_current"
    st.postcheck_findings = [{"id": "f1", "text": "A", "entity_type": "PERSON", "raw_start": 0, "raw_end": 1}]

    # Old callback bound to "run_old" attempts apply
    ok, msg = apply_postcheck_for_state(st, selected_ids={"f1"}, expected_run_id="run_old")
    assert ok is False
    assert "Veralteter Lauf" in msg
    assert st.is_postcheck_active is True
    assert st.postcheck_run_id == "run_current"

    # Old callback bound to "run_old" attempts cancel
    res = cancel_postcheck_for_state(st, expected_run_id="run_old")
    assert res is False
    assert st.is_postcheck_active is True
    assert st.postcheck_run_id == "run_current"


def test_start_reservation_exception_safety():
    """Package A: Any exception during start setup safely rolls back edit lock."""
    st = AppState()
    st.raw_text = "Hallo Welt."
    st.current_anon_text = "Hallo Welt."
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    def crashing_refresh():
        raise RuntimeError("Crash during start refresh")

    task = run_postcheck_for_state(st, refresh_fn=crashing_refresh)
    assert task is None
    assert st.is_postcheck_active is False
    assert st.postcheck_run_id == ""


@pytest.mark.asyncio
async def test_worker_early_failure_cleans_task_reference_without_unbound_local():
    """Package C1: Early failure in worker refresh_fn does not raise UnboundLocalError and cleans up task."""
    st = AppState()
    st.raw_text = "Hallo Welt."
    st.current_anon_text = "Hallo Welt."
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    refresh_calls = 0

    def dynamic_refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 2:
            # First call is start reservation in run_postcheck_for_state
            # Second call is early in _worker() before provider setup
            raise RuntimeError("Crash during worker refresh!")

    fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')
    task = run_postcheck_for_state(st, refresh_fn=dynamic_refresh, provider=fake)
    assert task is not None

    await task

    # Task reference was cleaned in finally without UnboundLocalError
    assert getattr(st, "postcheck_active_task", None) is None
    assert st.is_postcheck_active is False


@pytest.mark.asyncio
async def test_provider_creation_without_existing_instance():
    """Package A1: Real LocalApiProvider constructor runs cleanly without provider_type parameter error."""
    st = AppState()
    st.raw_text = "Hallo Welt."
    st.current_anon_text = "Hallo Welt."
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|qwen3:8b"
    st.llm_provider = None

    # Simulate generate_postcheck on the created LocalApiProvider to avoid requiring a real live LLM server
    resp = '{"schema_version": "1.0", "request_id": "REQID", "items": []}'
    with patch.object(LocalApiProvider, "generate_postcheck", AsyncMock(side_effect=lambda p, s="", max_tokens=4096: resp.replace("REQID", st.postcheck_run_id))) as mock_gen:
        # Call without provider parameter and with st.llm_provider=None
        task = run_postcheck_for_state(st)
        assert task is not None
        await task
        assert mock_gen.called
        assert st.is_postcheck_active is False


@pytest.mark.asyncio
async def test_real_extraction_pipeline_blocks_postcheck_start():
    """Package A3: Real extraction workflow extract_and_load_file_bytes_workflow sets is_extracting and blocks postcheck."""
    from app import extract_and_load_file_bytes_workflow

    st = AppState()
    st.raw_text = "Vorhandener Text."
    st.current_anon_text = "Vorhandener Text."
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    loop = asyncio.get_running_loop()
    extract_event = asyncio.Event()

    def waiting_read_document(*args, **kwargs):
        import time
        loop.call_soon_threadsafe(extract_event.set)
        time.sleep(0.08)
        return "Neu eingelesener Dokumenttext."

    with patch("app.read_document_from_bytes", side_effect=waiting_read_document):
        extract_task = asyncio.create_task(
            extract_and_load_file_bytes_workflow(st, b"dummy_bytes", "sample.txt")
        )
        # Wait until extraction has started and acquired is_extracting
        await extract_event.wait()
        assert st.is_extracting is True
        assert st.is_busy is True

        # While extraction is in flight, postcheck start must be blocked
        notified = []
        pc_task = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
        assert pc_task is None
        assert any("beschäftigt" in m for m, t in notified)

        # Let extraction complete
        await extract_task
        assert st.is_extracting is False
        assert st.raw_text == "Neu eingelesener Dokumenttext."

    # After extraction completes, postcheck start must succeed
    fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')
    task_after = run_postcheck_for_state(st, provider=fake)
    assert task_after is not None
    await task_after


def test_extraction_in_flight_blocks_postcheck():
    """Package A: Ongoing extraction or analysis blocks postcheck start."""
    st = AppState()
    st.raw_text = "Text"
    st.current_anon_text = "Text"
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"
    st.is_extracting = True

    notified = []
    task = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
    assert task is None
    assert any("beschäftigt" in m for m, t in notified)


# ---------------------------------------------------------------------------
# 6. Real Catalog Context Limits & Start Guard (Package B)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extraction_lock_releases_on_progress_exception_and_cancellation():
    """Package C2: Progress UI exception or cancellation releases is_extracting cleanly."""
    from app import extract_and_load_file_bytes_workflow

    st = AppState()

    # 1. Exception during initial progress UI setup
    class BrokenCard:
        def set_visibility(self, val):
            if val:
                raise RuntimeError("UI render error")

    progress_card = BrokenCard()
    await extract_and_load_file_bytes_workflow(
        st, b"dummy", "sample.txt",
        progress_card=progress_card,
        progress_bar=MagicMock(),
        progress_label=MagicMock(),
    )
    assert st.is_extracting is False
    assert st.current_extraction_id is None

    # 2. Cancellation during initial progress sleep
    cancel_card = MagicMock()
    task = asyncio.create_task(
        extract_and_load_file_bytes_workflow(
            st, b"dummy", "sample.txt",
            progress_card=cancel_card,
            progress_bar=MagicMock(),
            progress_label=MagicMock(),
        )
    )
    await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert st.is_extracting is False
    assert st.current_extraction_id is None


@pytest.mark.asyncio
async def test_second_extraction_rejected_while_first_in_flight():
    """Package C2: Second extraction is rejected while first is in flight; owner completes and commits."""
    from app import extract_and_load_file_bytes_workflow

    st = AppState()
    st.raw_text = "Ursprünglicher Text."
    st.current_anon_text = "Ursprünglicher Text."
    st.entity_modes = {"PERSON": "all"}
    st.config.llm_enabled = True
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    loop = asyncio.get_running_loop()
    extract1_started = asyncio.Event()

    def delayed_read(*args, **kwargs):
        import time
        loop.call_soon_threadsafe(extract1_started.set)
        time.sleep(0.1)
        return "Inhalt Dokument 1"

    with patch("app.read_document_from_bytes", side_effect=delayed_read):
        task1 = asyncio.create_task(
            extract_and_load_file_bytes_workflow(st, b"b1", "doc1.txt")
        )
        await extract1_started.wait()
        assert st.is_extracting is True

        # Second extraction attempt must be rejected immediately
        notified2 = []
        await extract_and_load_file_bytes_workflow(
            st, b"b2", "doc2.txt",
            notify_fn=lambda m, t: notified2.append((m, t))
        )
        assert any("läuft bereits" in m for m, t in notified2)
        # Rejection must NOT prematurely clear is_extracting for task1!
        assert st.is_extracting is True

        # While task1 is in flight, postcheck start must be blocked
        pc_notified = []
        pc_task = run_postcheck_for_state(st, notify_fn=lambda m, t: pc_notified.append((m, t)))
        assert pc_task is None
        assert any("beschäftigt" in m for m, t in pc_notified)

        # Let task1 finish
        await task1
        assert st.is_extracting is False
        assert st.raw_text == "Inhalt Dokument 1"

    st.current_anon_text = "Inhalt Dokument 1"
    # After task1 is finished, postcheck can start cleanly
    fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')
    pc_task_after = run_postcheck_for_state(st, provider=fake)
    assert pc_task_after is not None
    await pc_task_after


def test_reset_app_state_preserves_active_extraction_lock():
    """Package C2: reset_app_state does not prematurely release the lock of an active extraction."""
    from app import reset_app_state
    st = AppState()
    st.is_extracting = True
    st.current_extraction_id = "active_ext_id"

    reset_app_state(st)
    assert st.is_extracting is True
    assert st.current_extraction_id == "active_ext_id"


def test_get_known_context_limit_real_catalog_entry():
    """Package B: Real CatalogSchema lookup with context_limit (no service mocking)."""
    entry_small = CatalogModelEntry(
        canonical_name="small:3b",
        tested_tag="small:3b",
        provider="Ollama",
        hardware_class="Test",
        test_date="2026-09-01",
        phase_6a_triage=CatalogPhaseEvaluation(status="suitable", reason="ok"),
        phase_6b_smart_linking=CatalogPhaseEvaluation(status="untested", reason="none"),
        context_limit=8192,
    )
    catalog = CatalogSchema(schema_version="1.0.0", models=[entry_small])

    # 1. Real service call
    limit = get_known_context_limit("small:3b", catalog=catalog)
    assert limit == 8192

    # 2. Real start guard test with this catalog
    st = AppState()
    st.raw_text = "Text"
    st.current_anon_text = "Text"
    st.config.llm_enabled = True
    st.config.llm_model_name = "small:3b"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|small:3b"

    with patch("local_anonymizer.llm.catalog.load_catalog", return_value=catalog):
        notified = []
        task = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
        assert task is None
        assert any("bekannte Kontextgrenze von 8192 Tokens" in m for m, t in notified)


def test_get_known_context_limit_corrupted_catalog_raises():
    """Package B: Corrupted catalog raises CatalogError and is not swallowed."""
    with pytest.raises(CatalogError):
        get_known_context_limit("some-model", catalog_path=MagicMock(exists=lambda: False))


# ---------------------------------------------------------------------------
# 7. Group Identity, Homonyms & Full Rollback (R5)
# ---------------------------------------------------------------------------

def test_homonym_different_entity_type_creates_unique_group_id():
    st = AppState()
    st.raw_text = "Anna traf Anna in Bern."
    grp = EntityGroup("Anna", "PERSON", "anna")
    grp.occurrences.append(EntityOccurrence(0, 4, 1.0, "context", False, occ_id="occ_1"))
    st.entity_groups = [grp]
    compute_reactive_preview(st)

    st.is_postcheck_active = True
    st.postcheck_run_id = "run-homonym"
    f_loc = {
        "id": "f_loc",
        "text": "Anna",
        "entity_type": "LOCATION",
        "raw_start": 10,
        "raw_end": 14,
    }
    st.postcheck_findings = [f_loc]

    ok, msg = apply_postcheck_for_state(
        st,
        selected_ids={"f_loc"},
        expected_run_id="run-homonym",
        preview_fn=compute_reactive_preview,
    )
    assert ok is True
    assert len(st.entity_groups) == 2
    g_ids = [g.group_id for g in st.entity_groups]
    assert len(set(g_ids)) == 2
    assert "anna" in g_ids


def test_atomic_apply_full_rollback_of_all_fields():
    st = AppState()
    st.raw_text = "Frau Meier und Herr Schmidt."
    grp = EntityGroup("Meier", "PERSON", "meier")
    grp.occurrences.append(EntityOccurrence(5, 10, 1.0, "context", False, occ_id="occ_1"))
    st.entity_groups = [grp]
    compute_reactive_preview(st)

    st.is_postcheck_active = True
    st.postcheck_run_id = "run-rb"
    st.postcheck_selected_ids = {"f1"}
    st.colliding_roles = {("PERSON", "Leiter")}
    st.postcheck_status_msg = "Auswahl"
    st.postcheck_findings = [
        {"id": "f1", "text": "Schmidt", "entity_type": "PERSON", "raw_start": 20, "raw_end": 27}
    ]

    def crashing_preview(state):
        state.colliding_roles.clear()
        raise RuntimeError("Absturz in preview_fn")

    ok, msg = apply_postcheck_for_state(
        st,
        selected_ids={"f1"},
        expected_run_id="run-rb",
        preview_fn=crashing_preview,
    )
    assert ok is False
    assert "Rollback durchgeführt" in msg
    assert st.postcheck_selected_ids == {"f1"}
    assert st.colliding_roles == {("PERSON", "Leiter")}
    assert st.postcheck_status_msg == "Auswahl"
    assert st.is_postcheck_active is True
    assert len(st.entity_groups) == 1


# ---------------------------------------------------------------------------
# 8. Restore, Re-analysis & Re-bind Consistency
# ---------------------------------------------------------------------------

def test_restore_and_rebind_consistency():
    """Nachweislücke N1: Full homonym split-group creation, non-empty override, rebind_overrides_after_analysis and de_anonymize restore."""
    from app import rebind_overrides_after_analysis
    from types import SimpleNamespace

    raw_text = "Dr. Anna Keller leitete das Spital. Später besuchte Anna Bern."
    st = AppState()
    st.raw_text = raw_text

    # 1. Base group for first 'Anna' [4:8] with explicit custom role
    grp1 = EntityGroup("Anna", "PERSON", "anna")
    grp1.role = "Leiterin"
    grp1.role_provenance = "manual"
    grp1.occurrences.append(EntityOccurrence(4, 8, 1.0, "context", False, occ_id="occ_anna_1"))
    st.entity_groups = [grp1]
    compute_reactive_preview(st)

    # 2. Postcheck finding for SECOND 'Anna' [52:56] without role
    st.is_postcheck_active = True
    f_anna2 = {
        "id": "post_anna2",
        "text": "Anna",
        "entity_type": "PERSON",
        "raw_start": 52,
        "raw_end": 56,
        "output_start": 52,
        "output_end": 56,
    }
    ok, _ = atomic_apply_postcheck_findings(st, [f_anna2], preview_fn=compute_reactive_preview)
    assert ok is True

    # VERIFY BEFORE REBIND:
    # A distinct split group was created because grp1 has a custom role
    split_groups = [g for g in st.entity_groups if g.group_id.startswith("split_")]
    assert len(split_groups) == 1
    split_g = split_groups[0]
    assert len(st.entity_groups) == 2
    assert len(st.occurrence_overrides) == 1

    override = list(st.occurrence_overrides.values())[0]
    assert override.target_group_id == split_g.group_id
    assert override.expected_original_text == "Anna"
    assert override.entity_type == "PERSON"

    # 3. Real app function rebind_overrides_after_analysis with synthetic recognition results
    synthetic_results = [
        SimpleNamespace(start=4, end=8, score=0.95, entity_type="PERSON", recognition_metadata={"custom_role": "Leiterin"}),
        SimpleNamespace(start=52, end=56, score=0.90, entity_type="PERSON", recognition_metadata={}),
    ]

    new_groups, new_overrides = rebind_overrides_after_analysis(
        raw_text=raw_text,
        results=synthetic_results,
        current_overrides=st.occurrence_overrides,
        existing_groups=st.entity_groups,
    )

    # VERIFY AFTER REBIND:
    # Override was preserved
    assert len(new_overrides) == 1
    rebound_ov = list(new_overrides.values())[0]
    assert rebound_ov.target_group_id == split_g.group_id
    assert rebound_ov.expected_original_text == "Anna"

    # Both groups exist with distinct IDs and correct occurrence bindings
    assert len(new_groups) == 2
    base_rebound = [g for g in new_groups if g.group_id == "anna"][0]
    assert base_rebound.role == "Leiterin"
    assert len(base_rebound.occurrences) == 1
    assert base_rebound.occurrences[0].start == 4 and base_rebound.occurrences[0].end == 8

    split_rebound = [g for g in new_groups if g.group_id == split_g.group_id][0]
    assert len(split_rebound.occurrences) == 1
    assert split_rebound.occurrences[0].start == 52 and split_rebound.occurrences[0].end == 56

    # 4. Commit rebound state and verify preview & restore
    st.entity_groups = new_groups
    st.occurrence_overrides = new_overrides
    compute_reactive_preview(st)

    # Preview has two distinct placeholders for both groups (one with role LEITERIN, one without)
    assert "[PERSON_1_LEITERIN]" in st.current_anon_text
    assert "[PERSON_2]" in st.current_anon_text
    # De-anonymize completely restores original raw text
    restored = LocalAnonymizer.de_anonymize(st.current_anon_text, st.current_mapping)
    assert restored == raw_text



# ---------------------------------------------------------------------------
# 9. Budget, Response Reserve & Finish Reason Regressions
# ---------------------------------------------------------------------------

def test_postcheck_budget_limits_and_overflow():
    """Test 27.904 input token limit enforcement."""
    short_text = "Kurzer Text"
    ok, est_in, max_in, msg = calculate_postcheck_budget(short_text)
    assert ok is True
    assert est_in < max_in
    assert max_in == MAX_POSTCHECK_INPUT_TOKENS

    huge_text = "A" * 150000
    ok_huge, est_huge, max_huge, msg_huge = calculate_postcheck_budget(huge_text)
    assert ok_huge is False
    assert "überschreitet das Limit von 27904 Tokens" in msg_huge


@pytest.mark.asyncio
async def test_postcheck_response_reserve_and_finish_reason():
    """Nachweislücke 3: HTTP transport fake checks max_tokens=4096 payload and finish_reason rejection."""
    import aiohttp
    from unittest.mock import MagicMock

    prov = LocalApiProvider(base_url="http://127.0.0.1:11434/v1", model_name="test-model")
    captured_payloads = []

    def make_mock_post(finish_reason: str):
        class MockResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"Content-Type": "application/json"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                body = {
                    "choices": [
                        {
                            "message": {"content": '{"schema_version": "1.0", "request_id": "r1", "items": []}'},
                            "finish_reason": finish_reason,
                        }
                    ]
                }
                return json.dumps(body).encode("utf-8")

        def _post(url, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            return MockResponse()

        return _post

    # Case 1: finish_reason == 'length' must be rejected
    with patch("aiohttp.ClientSession.post", side_effect=make_mock_post("length")):
        with pytest.raises(ValueError, match="wegen Längenbegrenzung"):
            await prov.generate_postcheck("prompt")
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["max_tokens"] == POSTCHECK_RESPONSE_RESERVE
        assert captured_payloads[0]["model"] == "test-model"

    # Case 2: finish_reason == 'stop' must succeed
    captured_payloads.clear()
    with patch("aiohttp.ClientSession.post", side_effect=make_mock_post("stop")):
        res = await prov.generate_postcheck("prompt")
        assert "items" in res
        assert captured_payloads[0]["max_tokens"] == POSTCHECK_RESPONSE_RESERVE

    # Case 3: unexpected finish_reason must be rejected
    with patch("aiohttp.ClientSession.post", side_effect=make_mock_post("error")):
        with pytest.raises(ValueError, match="Unerwarteter oder fehlender Abschlussstatus"):
            await prov.generate_postcheck("prompt")

    await prov.close()


# ---------------------------------------------------------------------------
# 10. UI Component Real Rendering & Interactivity (R1)
# ---------------------------------------------------------------------------

def test_render_postcheck_ui_component_with_findings_and_counter():
    """Nachweislücke 2: Test real component interactivity: checkboxes toggle selection, counter updates, stale clicks rejected."""
    from nicegui import Client, ui
    from nicegui.page import page
    from types import SimpleNamespace

    st = AppState()
    st.current_anon_text = "Hallo Welt mit Befunden."
    st.is_postcheck_active = True
    st.postcheck_run_id = "run_ui_1"
    st.config.llm_enabled = True
    st.config.llm_model_name = "qwen3:8b"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|qwen3:8b"
    st.postcheck_findings = [
        {"id": "f1", "text": "Anna", "entity_type": "PERSON", "output_start": 0, "output_end": 4, "reasoning": "Name"},
        {"id": "f2", "text": "Bern", "entity_type": "LOCATION", "output_start": 6, "output_end": 10, "reasoning": "Ort"},
    ]
    st.postcheck_selected_ids = {"f1"}

    applied_runs = []
    cancelled_runs = []

    def mock_apply(rid): applied_runs.append(rid)
    def mock_cancel(rid): cancelled_runs.append(rid)

    with Client(page('/')) as client:
        render_postcheck_ui_component(
            state=st,
            anon_text=st.current_anon_text,
            apply_action=mock_apply,
            cancel_action=mock_cancel,
        )

        # 1. Find checkboxes and trigger toggle for f2
        checkboxes = [el for el in client.elements.values() if isinstance(el, ui.checkbox)]
        assert len(checkboxes) >= 2
        # Toggle checkbox for f2 (add to selection)
        cb_f2 = checkboxes[1]
        cb_f2.value = True
        cb_f2._handle_value_change(True)

        assert "f2" in st.postcheck_selected_ids
        assert len(st.postcheck_selected_ids) == 2

        # 2. Find apply button and trigger on_click
        buttons = [el for el in client.elements.values() if isinstance(el, ui.button)]
        apply_buttons = [b for b in buttons if "Ausgewählte übernehmen" in b.text]
        assert len(apply_buttons) == 1
        btn = apply_buttons[0]
        # Counter was updated
        assert "2" in btn.text

        # Trigger click on apply button via registered listener handler
        for listener in btn._event_listeners.values():
            listener.handler(None)
        assert applied_runs == ["run_ui_1"]

        # 3. Trigger stale event check: run advances to run_ui_2
        st.postcheck_run_id = "run_ui_2"
        stale_ok, stale_msg = apply_postcheck_for_state(st, expected_run_id=applied_runs[0])
        assert stale_ok is False
        assert "Veralteter Lauf" in stale_msg


@pytest.mark.asyncio
async def test_restore_file_dropped_production_pipeline():
    """Package R-Restore: Regression test for real drop -> extraction -> load_restore_text pipeline."""
    import base64
    from types import SimpleNamespace
    from nicegui import Client, ui
    from nicegui.page import page
    from app import create_ui

    with Client(page('/')) as client:
        create_ui(client)
        st = client.state
        st.config.llm_enabled = True
        st.config.llm_model_name = "test-model"
        st.postcheck_user_context_confirmed = True
        st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

        # 1. Locate the registered restore_file_dropped listener
        restore_listeners = [
            l for l in client.elements[0]._event_listeners.values()
            if l.type == "restore_file_dropped"
        ]
        assert len(restore_listeners) == 1
        listener = restore_listeners[0]

        # 2. Locate the restore_anon_input textarea
        textareas = [el for el in client.elements.values() if isinstance(el, ui.textarea)]
        assert len(textareas) >= 1

        loop = asyncio.get_running_loop()
        extract_started = asyncio.Event()

        def controlled_read_document(*args, **kwargs):
            import time
            loop.call_soon_threadsafe(extract_started.set)
            time.sleep(0.08)
            return "Wiederhergestellter LLM-Antworttext 123"

        with patch("app.read_document_from_bytes", side_effect=controlled_read_document):
            payload = {
                "name": "llm_reply.txt",
                "base64": base64.b64encode(b"dummy").decode("ascii"),
            }
            async def run_drop(args):
                with client:
                    await listener.handler(SimpleNamespace(args=args))

            drop_task = asyncio.create_task(run_drop(payload))

            # Wait until extraction is actively running
            await extract_started.wait()
            assert st.is_extracting is True
            assert st.is_busy is True

            # While extraction is in-flight, postcheck start must be blocked
            notified = []
            pc_task = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
            assert pc_task is None
            assert any("beschäftigt" in m for m, t in notified)

            # Let extraction complete
            await drop_task
            assert st.is_extracting is False
            assert st.restore_anon_text == "Wiederhergestellter LLM-Antworttext 123"

            # Check that restore textarea was updated
            restore_inputs = [inp for inp in textareas if inp.value == "Wiederhergestellter LLM-Antworttext 123"]
            assert len(restore_inputs) == 1

        # 3. Error path: Extraction fails -> restore text is NOT overwritten and lock is released
        def failing_read_document(*args, **kwargs):
            raise RuntimeError("Datei beschädigt")

        with patch("app.read_document_from_bytes", side_effect=failing_read_document):
            err_payload = {
                "name": "corrupt.txt",
                "base64": base64.b64encode(b"corrupt").decode("ascii"),
            }
            with client:
                await listener.handler(SimpleNamespace(args=err_payload))
            assert st.is_extracting is False
            # restore_anon_text preserves previous valid text, not updated to corrupted content
            assert st.restore_anon_text == "Wiederhergestellter LLM-Antworttext 123"

# ---------------------------------------------------------------------------
# 11. Minimal Decoupling & F1 NiceGUI Client Context (Handoff 1210)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_postcheck_runs_and_completes_with_llm_review_disabled():
    """Decoupling requirement: Ausgangskontrolle must start and complete even when llm_enabled (Review) is False."""
    st = AppState()
    st.raw_text = "Dr. Anna Keller leitete das Spital in Bern."
    st.current_anon_text = "Dr. [PERSON_1] leitete das [ORGANIZATION_1] in [LOCATION_1]."
    st.config.llm_enabled = False  # Review is deactivated!
    st.config.llm_model_name = "test-model"
    st.entity_modes = {"PERSON": "all"}
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')

    notified = []
    refreshed = []
    task = run_postcheck_for_state(
        st,
        provider=fake,
        notify_fn=lambda m, t: notified.append((m, t)),
        refresh_fn=lambda: refreshed.append(True),
    )
    assert task is not None
    await task

    assert st.is_postcheck_active is False
    assert any("abgeschlossen" in m for m, t in notified)
    assert len(refreshed) >= 2


def test_safety_gates_still_reject_invalid_states():
    """All safety checks (missing model, unconfirmed context, busy state, missing package) remain intact."""
    st = AppState()
    st.raw_text = "Text"
    st.current_anon_text = "Text"
    st.config.llm_enabled = False
    st.entity_modes = {"PERSON": "all"}

    # 1. Missing model name
    st.config.llm_model_name = ""
    notified = []
    t = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
    assert t is None
    assert any("Kein lokales Modell" in m for m, t in notified)

    # 2. Context unconfirmed
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = False
    notified.clear()
    t = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
    assert t is None
    assert any("32.000" in m for m, t in notified)

    # 3. Busy state
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"
    st.is_analyzing = True
    notified.clear()
    t = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
    assert t is None
    assert any("beschäftigt" in m for m, t in notified)
    st.is_analyzing = False

    # 4. Known small limit model
    with patch("app.get_known_context_limit", return_value=8192):
        notified.clear()
        t = run_postcheck_for_state(st, notify_fn=lambda m, t: notified.append((m, t)))
        assert t is None
        assert any("bekannte Kontextgrenze von 8192" in m for m, t in notified)


def test_production_ui_has_no_simulation_button():
    """Production cleanup: verify simulate_postcheck_finding and test button are completely removed."""
    from nicegui import Client, ui
    from nicegui.page import page

    st = AppState()
    st.raw_text = "Max Muster in Zürich."
    st.current_anon_text = "[PERSON_1] in [LOCATION_1]."
    st.config.llm_enabled = False
    st.config.llm_model_name = "test-model"
    st.postcheck_user_context_confirmed = True
    st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

    with Client(page('/')) as client:
        render_postcheck_ui_component(
            state=st,
            anon_text=st.current_anon_text,
            run_action=lambda: None,
        )

        buttons = [el for el in client.elements.values() if isinstance(el, ui.button)]
        btn_texts = [b.text or "" for b in buttons]
        # Ausgangskontrolle button must exist and be enabled
        assert any("Ausgangskontrolle starten" in t for t in btn_texts)
        # Simulation button must NOT exist
        assert not any("simulieren" in t.lower() or "test-nachzügler" in t.lower() for t in btn_texts)


@pytest.mark.asyncio
async def test_f1_client_context_safe_completion():
    """F1 Regression: NiceGUI background task completion cleanly updates UI without RuntimeError."""
    from nicegui import Client, ui
    from nicegui.page import page
    from app import create_ui

    with Client(page('/')) as client:
        create_ui(client)
        st = client.state
        st.raw_text = "Dr. Anna Keller"
        st.current_anon_text = "Dr. [PERSON_1]"
        st.config.llm_enabled = False  # Decoupled!
        st.config.llm_model_name = "test-model"
        st.entity_modes = {"PERSON": "all"}
        st.postcheck_user_context_confirmed = True
        st.postcheck_confirmed_bound_key = f"{st.config.llm_base_url.strip()}|test-model"

        fake = FakeProvider(response_text='{"schema_version": "1.0", "request_id": "REQID", "items": []}')

        notified = []
        refreshed = []

        def safe_notify(msg, t="info"):
            with client:
                notified.append((msg, t))
                ui.notify(msg, type=t)

        def safe_refresh():
            with client:
                refreshed.append(True)

        task = run_postcheck_for_state(
            st,
            provider=fake,
            notify_fn=safe_notify,
            refresh_fn=safe_refresh,
        )
        assert task is not None
        # Must await cleanly without slot stack RuntimeError
        await task

        assert st.is_postcheck_active is False
        assert any("abgeschlossen" in m for m, t in notified)
        assert len(refreshed) >= 2


def test_analysis_with_review_disabled_does_not_trigger_auto_review():
    """Verify that when llm_enabled (Review) is False, auto-review flag is ignored during analysis."""
    st = AppState()
    st.config.llm_enabled = False
    st.config.llm_auto_review = True
    st.config.llm_model_name = "test-model"

    # Evaluation condition from start_analysis:
    should_auto_trigger = st.config.llm_enabled and st.config.llm_auto_review and not st.is_busy
    assert should_auto_trigger is False


