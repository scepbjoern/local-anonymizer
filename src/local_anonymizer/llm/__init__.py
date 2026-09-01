"""Local LLM Triage Layer (Phase 6A)."""

try:
    import aiohttp
    import pydantic
except ImportError as err:
    raise ImportError("Dependencies for [llm] extra (aiohttp, pydantic) are not installed.") from err

from local_anonymizer.llm.schema import (
    TriageAction,
    TriageConfidence,
    TriageItem,
    TriageKeepItem,
    TriageRecategorizeItem,
    TriageDiscardItem,
    TriageEnvelope,
    validate_batch_response,
)
from local_anonymizer.llm.provider import LlmProvider, LocalApiProvider
from local_anonymizer.llm.batching import TriageBatch, prepare_triage_batches
from local_anonymizer.llm.apply_service import ApplyCommand, ApplyService

__all__ = [
    "TriageAction",
    "TriageConfidence",
    "TriageItem",
    "TriageKeepItem",
    "TriageRecategorizeItem",
    "TriageDiscardItem",
    "TriageEnvelope",
    "validate_batch_response",
    "LlmProvider",
    "LocalApiProvider",
    "TriageBatch",
    "prepare_triage_batches",
    "ApplyCommand",
    "ApplyService",
]
