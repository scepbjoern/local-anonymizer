import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock
from local_anonymizer.llm.provider import validate_loopback_url, LocalApiProvider


def test_validate_loopback_url_allowed():
    assert validate_loopback_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"
    assert validate_loopback_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert validate_loopback_url("http://[::1]:11434/v1") == "http://[::1]:11434/v1"
    assert validate_loopback_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert validate_loopback_url("https://127.0.0.1:11434/v1") == "https://127.0.0.1:11434/v1"


def test_validate_loopback_url_disallowed_remote_and_lan():
    with pytest.raises(ValueError, match="keine lokale Loopback-Adresse"):
        validate_loopback_url("https://api.openai.com/v1")

    with pytest.raises(ValueError, match="keine lokale Loopback-Adresse"):
        validate_loopback_url("http://192.168.1.50:11434/v1")

    with pytest.raises(ValueError, match="keine lokale Loopback-Adresse"):
        validate_loopback_url("http://10.0.0.1:11434/v1")

    with pytest.raises(ValueError, match="keine lokale Loopback-Adresse"):
        validate_loopback_url("http://0.0.0.0:11434/v1")


def test_validate_loopback_url_disallowed_credentials():
    with pytest.raises(ValueError, match="Benutzerdaten"):
        validate_loopback_url("http://admin:secret@127.0.0.1:11434/v1")


def test_validate_loopback_url_disallowed_query_and_fragment():
    with pytest.raises(ValueError, match="Abfrageparameter"):
        validate_loopback_url("http://127.0.0.1:11434/v1?key=123")

    with pytest.raises(ValueError, match="Fragmente"):
        validate_loopback_url("http://127.0.0.1:11434/v1#section")


def test_validate_loopback_url_disallowed_scheme():
    with pytest.raises(ValueError, match="Ungültiges URL-Schema"):
        validate_loopback_url("ftp://127.0.0.1:11434/v1")


@pytest.mark.asyncio
async def test_local_api_provider_generate_success():
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="phi4:latest",
    )

    mock_resp_json = {
        "choices": [
            {
                "message": {
                    "content": '{"schema_version": "1.0", "request_id": "r1", "document_revision": 1, "document_hash": "h", "items": []}'
                }
            }
        ]
    }

    class MockContent:
        def __init__(self, data: bytes):
            self.data = data

        async def iter_chunked(self, n: int):
            yield self.data

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Type": "application/json; charset=utf-8"}
    mock_resp.content = MockContent(json.dumps(mock_resp_json).encode("utf-8"))

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_cm
    mock_session.close = AsyncMock()
    mock_session.closed = False

    provider._session = mock_session

    res = await provider.generate("Test user prompt", "Test system prompt")
    assert '{"schema_version": "1.0"' in res

    # Verify max_tokens and reasoning_effort are sent in payload
    call_kwargs = mock_session.post.call_args[1]
    assert call_kwargs["json"]["max_tokens"] == 4096
    assert call_kwargs["json"]["reasoning_effort"] == "none"

    await provider.close()


@pytest.mark.asyncio
async def test_local_api_provider_content_type_validation_negative():
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="phi4:latest",
    )

    invalid_content_types = [
        "text/html",
        "text/application/json",
        "application/jsonp",
        "application/javascript",
        "text/plain",
    ]

    for ct in invalid_content_types:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": ct}
        mock_resp.content = MagicMock()

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm
        mock_session.close = AsyncMock()
        mock_session.closed = False

        provider._session = mock_session

        with pytest.raises(ValueError, match="Unerwarteter Content-Type"):
            await provider.generate("prompt", "sys")

    await provider.close()


@pytest.mark.asyncio
async def test_local_api_provider_content_type_validation_positive():
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="phi4:latest",
    )

    mock_resp_json = {
        "choices": [{"message": {"content": "{}"}}]
    }

    valid_content_types = [
        "application/json",
        "application/json; charset=utf-8",
        "application/problem+json",
        "application/geo+json; charset=utf-8",
    ]

    class MockContent:
        async def iter_chunked(self, n: int):
            yield json.dumps(mock_resp_json).encode("utf-8")

    for ct in valid_content_types:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": ct}
        mock_resp.content = MockContent()

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm
        mock_session.close = AsyncMock()
        mock_session.closed = False

        provider._session = mock_session

        res = await provider.generate("prompt", "sys")
        assert res == "{}"

    await provider.close()


@pytest.mark.asyncio
async def test_local_api_provider_size_limit_exceeded():
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="phi4:latest",
    )

    class GiantMockContent:
        async def iter_chunked(self, n: int):
            chunk = b"x" * (1024 * 1024)
            for _ in range(3):  # 3MB > 2MB limit
                yield chunk

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.content = GiantMockContent()

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_cm
    mock_session.close = AsyncMock()
    mock_session.closed = False

    provider._session = mock_session

    with pytest.raises(ValueError, match="überschreitet das Limit"):
        await provider.generate("prompt", "sys")
    await provider.close()


@pytest.mark.asyncio
async def test_two_separate_provider_sessions_run_concurrently():
    # Verify that concurrency lock is session-local and does not globally block other sessions
    p1 = LocalApiProvider("http://127.0.0.1:11434/v1", "phi4:latest")
    p2 = LocalApiProvider("http://127.0.0.1:11434/v1", "phi4:latest")

    events = []

    async def run_p(provider: LocalApiProvider, tag: str):
        async with provider._lock:
            events.append(f"{tag}_start")
            await asyncio.sleep(0.05)
            events.append(f"{tag}_end")

    await asyncio.gather(run_p(p1, "p1"), run_p(p2, "p2"))

    # Both should have started before both ended (concurrent execution across sessions)
    assert events[0].endswith("_start")
    assert events[1].endswith("_start")

    await p1.close()
    await p2.close()


@pytest.mark.asyncio
async def test_local_api_provider_retry_without_reasoning_effort_on_400_422():
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="qwen3:8b",
    )

    mock_resp_400 = MagicMock()
    mock_resp_400.status = 400

    mock_resp_200 = MagicMock()
    mock_resp_200.status = 200
    mock_resp_200.headers = {"Content-Type": "application/json"}

    class MockContent:
        async def iter_chunked(self, n: int):
            yield b'{"choices": [{"message": {"content": "{\\"schema_version\\": \\"1.0\\", \\"items\\": []}"}}]}'

    mock_resp_200.content = MockContent()

    cm_400 = MagicMock()
    cm_400.__aenter__ = AsyncMock(return_value=mock_resp_400)
    cm_400.__aexit__ = AsyncMock(return_value=None)

    cm_200 = MagicMock()
    cm_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
    cm_200.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.side_effect = [cm_400, cm_200]
    mock_session.close = AsyncMock()
    mock_session.closed = False

    provider._session = mock_session

    res = await provider.generate("Test user prompt", "Test system prompt")
    assert '{"schema_version": "1.0"' in res

    # Verify 2 calls: first with reasoning_effort, second without reasoning_effort
    assert mock_session.post.call_count == 2
    first_call_json = mock_session.post.call_args_list[0][1]["json"]
    second_call_json = mock_session.post.call_args_list[1][1]["json"]

    assert first_call_json.get("reasoning_effort") == "none"
    assert "reasoning_effort" not in second_call_json

    await provider.close()


@pytest.mark.asyncio
async def test_local_api_provider_no_pii_leakage_in_exceptions_and_logs(caplog):
    provider = LocalApiProvider(
        base_url="http://127.0.0.1:11434/v1",
        model_name="qwen3:8b",
    )

    sensitive_text = "SECRET_PII_PATIENT_JULIA_MEIER_DIAGNOSIS_DATA"

    mock_session = MagicMock()
    mock_session.post.side_effect = Exception(f"Internal connection crash with {sensitive_text}")
    mock_session.close = AsyncMock()
    mock_session.closed = False

    provider._session = mock_session

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as exc_info:
            await provider.generate(f"Please check {sensitive_text}")

        # The error message surfaced to callers must not contain the sensitive PII
        assert sensitive_text not in str(exc_info.value)
        # Logged records must also not leak the prompt or sensitive text
        for record in caplog.records:
            assert sensitive_text not in record.message

    await provider.close()


def test_clean_tag_and_anonymization_umlauts():
    from local_anonymizer.anonymizer import clean_tag, LocalAnonymizer

    # 1. Verify clean_tag preserves German umlauts
    assert clean_tag("Chefärztin") == "CHEFÄRZTIN"
    assert clean_tag("Führungskraft") == "FÜHRUNGSKRAFT"
    assert clean_tag("Förderverein e.V.") == "FÖRDERVEREIN_E_V"

    # 2. Verify de-anonymization roundtrip with umlaut tag
    orig_text = "Frau Dr. Julia Meier leitet die Abteilung."
    anon_text = "Frau [PERSON_1_CHEFÄRZTIN] leitet die Abteilung."
    mapping = {"[PERSON_1_CHEFÄRZTIN]": "Dr. Julia Meier"}

    restored = LocalAnonymizer.de_anonymize(anon_text, mapping)
    assert restored == orig_text
