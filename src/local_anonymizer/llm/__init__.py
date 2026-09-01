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
    extract_json_from_llm_response,
)
from local_anonymizer.llm.provider import LlmProvider, LocalApiProvider
from local_anonymizer.llm.batching import TriageBatch, prepare_triage_batches
from local_anonymizer.llm.apply_service import (
    ApplyCommand,
    ApplyService,
    compute_triage_snapshot,
    normalize_entity_type,
    check_mutation_allowed,
    CANONICAL_APP_ENTITY_TYPES,
    ENTITY_TYPE_ALIASES,
)

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
    "compute_triage_snapshot",
    "normalize_entity_type",
    "check_mutation_allowed",
    "CANONICAL_APP_ENTITY_TYPES",
    "ENTITY_TYPE_ALIASES",
]
