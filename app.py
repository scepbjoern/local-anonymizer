"""
Privacy-First Local Anonymizer - NiceGUI Interactive Review Application.
Runs completely locally and offline with in-memory processing.
Instant startup (< 0.5s) with background asynchronous model warmup.
"""

import argparse
import asyncio
import base64
import html
import json
import logging
import os
import re
import atexit
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from nicegui import Client, app, ui

from local_anonymizer.anonymizer import AVAILABLE_ENTITIES, REVIEW_SCORE_THRESHOLD, clean_tag
from local_anonymizer.recognizers import (
    EUPII_MODEL_NAME,
    EUPII_MODEL_SIZE_MB,
    GLINER_MODEL_NAME,
    GLINER_MODEL_SIZE_MB,
    is_model_cached,
)
from local_anonymizer.config import (
    AppConfig,
    CONFIG_DIR,
    ENTITY_MODE_ALL,
    ENTITY_MODE_EXPLICIT_ONLY,
    ENTITY_MODE_EXPLICIT_EUPII,
    ENTITY_MODE_OFF,
    LOG_FILE,
)
from local_anonymizer.profiles import (
    CategoryTemplate,
    DocumentProfileOverlay,
    EffectiveConfig,
    ProfileController,
    ProfileStore,
    RevisionConflictError,
    ScopedTerm,
    ScopeLevel,
    ScopeResolutionEngine,
    get_builtin_templates,
    normalize_term_key,
)
from local_anonymizer.extractors import (
    UnsupportedFileFormatError,
    create_docx_from_markdown,
    extract_text_from_txt_bytes,
    is_pdf_worker,
    read_document_from_bytes,
    safe_read_bytes,
    save_markdown_to_docx_bytes,
    strip_html_markup,
)

# Optional LLM Triage Layer (Phase 6A & 6A.1)
try:
    from local_anonymizer.llm import (
        TriageItem,
        TriageEnvelope,
        TriageBatch,
        validate_batch_response,
        extract_json_from_llm_response,
        validate_model_name,
        LocalApiProvider,
        prepare_triage_batches,
        ApplyCommand,
        ApplyService,
        CANONICAL_APP_ENTITY_TYPES,
        ENTITY_TYPE_ALIASES,
        normalize_entity_type,
        compute_triage_snapshot,
        fetch_ollama_models,
        fetch_generic_models,
        preload_ollama_model,
        test_generic_connection,
        load_catalog,
        find_catalog_entry,
        get_model_suitability_badge,
        CatalogError,
        DiscoveryResult,
        PsModelInfo,
        parse_iso_expiry,
        verify_ollama_model_running,
        PROVIDER_TYPE_OLLAMA,
        PROVIDER_TYPE_GENERIC,
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    compute_triage_snapshot = None  # type: ignore
    TriageBatch = None  # type: ignore
    extract_json_from_llm_response = None  # type: ignore
    validate_model_name = None  # type: ignore
    fetch_ollama_models = None  # type: ignore
    fetch_generic_models = None  # type: ignore
    preload_ollama_model = None  # type: ignore
    test_generic_connection = None  # type: ignore
    load_catalog = None  # type: ignore
    find_catalog_entry = None  # type: ignore
    get_model_suitability_badge = None  # type: ignore
    CatalogError = Exception  # type: ignore
    parse_iso_expiry = lambda s: 0.0  # type: ignore
    verify_ollama_model_running = None  # type: ignore
    PROVIDER_TYPE_OLLAMA = "ollama"
    PROVIDER_TYPE_GENERIC = "generic"

# Silence presidio analyzer language mismatch warnings
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# Shared temp upload directory for large file Drag & Drop (cross-process, zero WebSocket limits)
UPLOAD_DIR = CONFIG_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit


class ProfileMutationAborted(RuntimeError):
    """Raised when a delayed profile action is no longer safe to execute."""


def _reload_profile_controller_preserving_overlay(
    controller: ProfileController,
    overlay: DocumentProfileOverlay,
    active_project_id: Optional[str] = None,
) -> None:
    """Reload durable profiles while retaining the current document overlay."""
    controller.reload(active_project_id=active_project_id)
    controller.document_overlay = overlay


def cleanup_temp_uploads(max_age_seconds: int = 1800):
    """Clean up any stale uploaded temporary files older than max_age_seconds (default 30 min).
    Never deletes recently active or newly created files from other running app instances or tabs."""
    if is_pdf_worker():
        return
    try:
        if UPLOAD_DIR.exists():
            cutoff = time.time() - max_age_seconds
            for f in UPLOAD_DIR.glob("*"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


if not is_pdf_worker():
    cleanup_temp_uploads()
    atexit.register(cleanup_temp_uploads)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """FastAPI endpoint to receive large dropped files via HTTP streaming to disk with size limits."""
    bin_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse({"error": "Dateigröße überschreitet das Limit von 50 MB."}, status_code=413)

        file_id = str(uuid.uuid4())
        bin_path = UPLOAD_DIR / f"{file_id}.bin"
        meta_path = UPLOAD_DIR / f"{file_id}.json"
        bin_path.write_bytes(content)
        meta_path.write_text(json.dumps({"filename": file.filename or "upload", "size": len(content)}), encoding="utf-8")
        logging.info(f"api_upload: Saved {file.filename} ({len(content)} bytes) as {file_id}")
        return JSONResponse({"file_id": file_id, "filename": file.filename, "size": len(content)})
    except Exception as e:
        cleanup_upload_paths(bin_path, meta_path)
        logging.error(f"api_upload failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def validate_and_resolve_upload_paths(file_id: Any, upload_dir: Path = UPLOAD_DIR) -> Optional[Tuple[Path, Path]]:
    """
    Strictly validate that file_id is a valid UUID string and return the canonical (bin_path, meta_path)
    located directly within upload_dir.
    Returns None if file_id is missing, invalid, or contains path traversal attempts.
    """
    if not file_id or not isinstance(file_id, str):
        return None
    file_id_str = file_id.strip()
    try:
        val = uuid.UUID(file_id_str)
        canonical_id = str(val)
    except (ValueError, AttributeError, TypeError):
        return None

    resolved_upload_dir = upload_dir.resolve()
    bin_path = (resolved_upload_dir / f"{canonical_id}.bin").resolve()
    meta_path = (resolved_upload_dir / f"{canonical_id}.json").resolve()

    # Defense-in-depth: Verify paths reside directly inside resolved_upload_dir
    if bin_path.parent != resolved_upload_dir or meta_path.parent != resolved_upload_dir:
        return None

    return bin_path, meta_path


def cleanup_upload_paths(bin_path: Optional[Path], meta_path: Optional[Path]) -> None:
    """Safely delete uploaded temporary files immediately."""
    if bin_path is not None:
        try:
            bin_path.unlink(missing_ok=True)
        except Exception:
            pass
    if meta_path is not None:
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass


def extract_upload_payload(
    data: Any,
    upload_dir: Path = UPLOAD_DIR,
) -> Tuple[Optional[bytes], str, Optional[Tuple[Path, Path]]]:
    """
    Extract raw_bytes and filename from an upload event payload (dict).
    Validates file_id as strict UUID and resolves temp paths under upload_dir.
    Returns (raw_bytes, filename, temp_paths_tuple_or_None).
    Guarantees immediate cleanup of any resolved temp files if extraction fails before returning.
    Raises ValueError if data is invalid or cannot be decoded.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Ungültige Eventdaten: {type(data)}")

    filename = data.get("name") or "dokument"
    filepath = data.get("path", "")
    file_id = data.get("file_id", "")

    temp_paths: Optional[Tuple[Path, Path]] = None
    if file_id:
        temp_paths = validate_and_resolve_upload_paths(file_id, upload_dir=upload_dir)
        if temp_paths is None:
            raise ValueError(f"Ungültige oder unzulässige file_id: {file_id}")

    bin_path, meta_path = temp_paths if temp_paths else (None, None)
    raw_bytes: Optional[bytes] = None

    try:
        if bin_path:
            if not bin_path.exists():
                raise ValueError(f"Upload-Datei {bin_path.name} existiert nicht.")
            raw_bytes = bin_path.read_bytes()
            if meta_path and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    filename = meta.get("filename") or filename
                except Exception:
                    pass
        elif filepath and Path(filepath).is_file():
            raw_bytes = safe_read_bytes(filepath)
        elif "base64" in data and data["base64"]:
            raw_bytes = base64.b64decode(data["base64"])
        else:
            raise ValueError(f"Keine Dateidaten im Event empfangen (keys={list(data.keys())})")
    except Exception:
        if temp_paths:
            cleanup_upload_paths(*temp_paths)
        raise

    return raw_bytes, filename, temp_paths


import hashlib

# --- Data Models for Grouped Review ---
def compute_context_fingerprint(text: str, start: int, end: int, window: int = 50) -> str:
    """Compute deterministic context fingerprint for reattachment across re-analyses."""
    pre = text[max(0, start - window):start]
    term = text[start:end]
    post = text[end:min(len(text), end + window)]
    raw = f"{pre}|{term}|{post}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class EntityOccurrence:
    start: int
    end: int
    score: float
    context_html: str
    needs_review: bool
    source: str = "automatic"
    method: str = "ai"
    method_detail: Optional[str] = None
    occ_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    context_fingerprint: str = ""


@dataclass
class OccurrenceOverride:
    target_group_id: str
    context_fingerprint: str
    expected_original_text: str
    entity_type: Optional[str] = None
    role: Optional[str] = None
    enabled: Optional[bool] = None


class EntityGroup:
    def __init__(self, original_text: str, entity_type: str, group_id: Optional[str] = None):
        self.original_text: str = original_text
        self.entity_type: str = entity_type
        self.group_id: str = group_id if group_id is not None else original_text.strip().lower()
        self.enabled: bool = True
        self.role: str = ""
        self.role_provenance: str = "auto"
        self.parent_group_id: Optional[str] = None
        self.surface_tag: str = ""
        self.placeholder: str = ""
        self.suggested_parent: Optional[str] = None
        self.suggested_tag: Optional[str] = None
        self.suggested_candidates: List[str] = []
        self.occurrences: List[EntityOccurrence] = []

    @property
    def text_key(self) -> str:
        return self.original_text.strip().lower()

    @property
    def key(self) -> str:
        return self.group_id

    @property
    def parent_group_text(self) -> Optional[str]:
        return self.parent_group_id

    @parent_group_text.setter
    def parent_group_text(self, value: Optional[str]) -> None:
        self.parent_group_id = value

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def score_range(self) -> Tuple[float, float]:
        scores = [occ.score for occ in self.occurrences]
        return (min(scores), max(scores)) if scores else (0.0, 0.0)

    @property
    def score_display(self) -> str:
        """Display one score or the complete score range across all occurrences."""
        if not self.occurrences:
            return "–"
        low, high = self.score_range
        if len(self.occurrences) == 1:
            return f"{high:.2f}"
        return f"{low:.2f}–{high:.2f}"

    @property
    def needs_review(self) -> bool:
        return any(occ.needs_review for occ in self.occurrences)

    @property
    def first_start(self) -> int:
        return self.occurrences[0].start if self.occurrences else 0


def split_occurrence_to_new_group(st: "AppState", grp: EntityGroup, occ: EntityOccurrence) -> EntityGroup:
    """Split a single occurrence into its own separate EntityGroup with an active override."""
    target_group_id = f"split_{uuid.uuid4().hex[:8]}"
    exact_text = (
        st.raw_text[occ.start:occ.end]
        if (st.raw_text and 0 <= occ.start < occ.end <= len(st.raw_text))
        else grp.original_text
    )
    override = OccurrenceOverride(
        target_group_id=target_group_id,
        context_fingerprint=occ.context_fingerprint,
        expected_original_text=exact_text,
        entity_type=grp.entity_type,
        role=grp.role,
        enabled=grp.enabled,
    )
    st.occurrence_overrides[occ.occ_id] = override
    if occ in grp.occurrences:
        grp.occurrences.remove(occ)
    new_grp = EntityGroup(
        original_text=exact_text,
        entity_type=grp.entity_type,
        group_id=target_group_id,
    )
    new_grp.role = grp.role
    new_grp.role_provenance = getattr(grp, "role_provenance", "auto")
    new_grp.enabled = grp.enabled
    new_grp.occurrences.append(occ)
    st.entity_groups.append(new_grp)
    return new_grp


def revert_occurrence_to_base(st: "AppState", grp: EntityGroup, occ: EntityOccurrence) -> None:
    """Revert a split occurrence back to its canonical base group."""
    st.occurrence_overrides.pop(occ.occ_id, None)
    base_key = grp.text_key
    base_grp = next((g for g in st.entity_groups if g.group_id == base_key), None)
    if not base_grp:
        base_grp = EntityGroup(
            original_text=grp.original_text,
            entity_type=grp.entity_type,
            group_id=base_key,
        )
        base_grp.role = grp.role
        base_grp.enabled = grp.enabled
        st.entity_groups.append(base_grp)

    if occ in grp.occurrences:
        grp.occurrences.remove(occ)
    if occ not in base_grp.occurrences:
        base_grp.occurrences.append(occ)
        base_grp.occurrences.sort(key=lambda o: o.start)

    # Clean up empty split group
    if grp.group_id != base_key and len(grp.occurrences) == 0:
        if grp in st.entity_groups:
            st.entity_groups.remove(grp)
        for other in st.entity_groups:
            if other.parent_group_id == grp.group_id:
                other.parent_group_id = base_grp.group_id


def sync_group_overrides(st: "AppState", grp: EntityGroup) -> None:
    """Sync group property changes (type, role, enabled) to any active overrides in this group."""
    for occ in grp.occurrences:
        if occ.occ_id in st.occurrence_overrides:
            ov = st.occurrence_overrides[occ.occ_id]
            ov.entity_type = grp.entity_type
            ov.role = grp.role
            ov.enabled = grp.enabled


def rebind_overrides_after_analysis(
    raw_text: str,
    results: List[Any],
    current_overrides: Dict[str, OccurrenceOverride],
    existing_groups: Optional[List[EntityGroup]] = None,
) -> Tuple[List[EntityGroup], Dict[str, OccurrenceOverride]]:
    """
    Rebind occurrence overrides fail-safely after an analysis run.

    1. Builds base EntityGroup instances and occurrences with fresh occ_id and context_fingerprint.
       Preserves user customizations (role, custom type, enabled, surface_tag, parent_group_id)
       on base groups if they existed prior to re-analysis.
    2. Re-attaches overrides only on strict 1:1 match of context_fingerprint and exact original_text.
    3. Reconstructs target_group_id split EntityGroups with expected_original_text.
    4. Discards ambiguous (1:N, N:1, N:M), mismatched, invalid target_id, base-collision,
       or conflicting overrides fail-safely.

    Returns:
        (entity_groups, new_occurrence_overrides)
    """
    existing_base_map: Dict[str, EntityGroup] = {}
    if existing_groups:
        for g in existing_groups:
            if g.group_id == g.text_key:
                existing_base_map[g.group_id] = g

    base_groups_dict: Dict[str, EntityGroup] = {}
    all_new_occurrences: List[Tuple[EntityOccurrence, str, str]] = []

    for res in results:
        orig = raw_text[res.start:res.end]
        norm = orig.strip()
        key = norm.lower()
        needs_rev = res.score < REVIEW_SCORE_THRESHOLD
        ctx_html = extract_context_snippet(raw_text, res.start, res.end)
        fingerprint = compute_context_fingerprint(raw_text, res.start, res.end)
        occ_id = uuid.uuid4().hex

        source, method = classify_recognition_result(res)
        occ = EntityOccurrence(
            start=res.start,
            end=res.end,
            score=res.score,
            context_html=ctx_html,
            needs_review=needs_rev,
            source=source,
            method=method,
            occ_id=occ_id,
            context_fingerprint=fingerprint,
        )
        if key not in base_groups_dict:
            if key in existing_base_map:
                prev = existing_base_map[key]
                base_grp = EntityGroup(
                    original_text=norm,
                    entity_type=prev.entity_type or res.entity_type,
                    group_id=key,
                )
                base_grp.role = prev.role
                base_grp.role_provenance = getattr(prev, "role_provenance", "auto")
                base_grp.enabled = prev.enabled
                base_grp.surface_tag = prev.surface_tag
                base_grp.parent_group_id = prev.parent_group_id
            else:
                base_grp = EntityGroup(original_text=norm, entity_type=res.entity_type, group_id=key)
            base_groups_dict[key] = base_grp
        base_group = base_groups_dict[key]
        profile_role = (res.recognition_metadata or {}).get("custom_role")
        profile_role = profile_role.strip() if isinstance(profile_role, str) and profile_role.strip() else None
        if key not in existing_base_map:
            if profile_role is not None:
                base_group.role = profile_role
                base_group.role_provenance = "profile"
        else:
            previous_provenance = getattr(base_group, "role_provenance", "auto")
            if previous_provenance not in ("manual", "llm") and not (
                previous_provenance == "auto" and base_group.role
            ):
                base_group.role = profile_role or ""
                base_group.role_provenance = "profile" if profile_role else "auto"
        base_group.occurrences.append(occ)
        all_new_occurrences.append((occ, orig, res.entity_type))

    old_overrides = list(current_overrides.values())
    new_overrides: Dict[str, OccurrenceOverride] = {}

    old_by_fp: Dict[str, List[OccurrenceOverride]] = {}
    for ov in old_overrides:
        old_by_fp.setdefault(ov.context_fingerprint, []).append(ov)

    new_by_fp: Dict[str, List[Tuple[EntityOccurrence, str, str]]] = {}
    for occ, orig, ent_type in all_new_occurrences:
        new_by_fp.setdefault(occ.context_fingerprint, []).append((occ, orig, ent_type))

    # Phase 1: Collect candidates that pass 1:1 fingerprint match and exact original text
    candidates_by_tgt: Dict[str, List[Tuple[OccurrenceOverride, EntityOccurrence, str, str]]] = {}

    for fp, ov_list in old_by_fp.items():
        cand_list = new_by_fp.get(fp, [])
        if len(ov_list) == 1 and len(cand_list) == 1:
            ov = ov_list[0]
            new_occ, actual_orig, actual_type = cand_list[0]
            if actual_orig == ov.expected_original_text:
                tgt_id = (ov.target_group_id or "").strip()
                if tgt_id:
                    candidates_by_tgt.setdefault(tgt_id, []).append((ov, new_occ, actual_orig, actual_type))

    # Phase 2: Validate target_group_id integrity and consistency
    split_groups_dict: Dict[str, EntityGroup] = {}

    for tgt_id, matches in candidates_by_tgt.items():
        # Reject empty target_group_id or collision with any canonical base group key
        if not tgt_id or tgt_id.lower() in base_groups_dict:
            continue

        # Check if all overrides for this target_group_id have consistent expected_original_text and metadata
        first_ov, _, first_orig, first_type = matches[0]
        is_consistent = all(
            (
                ov.expected_original_text == first_ov.expected_original_text
                and ov.entity_type == first_ov.entity_type
                and ov.role == first_ov.role
                and ov.enabled == first_ov.enabled
                and orig == first_orig
            )
            for ov, occ, orig, typ in matches
        )
        if not is_consistent:
            # Conflicting overrides pointing to the same target_group_id -> fail-safe drop
            continue

        restored_type = first_ov.entity_type if first_ov.entity_type is not None else first_type
        restored_grp = EntityGroup(
            original_text=first_ov.expected_original_text,
            entity_type=restored_type,
            group_id=tgt_id,
        )
        if first_ov.role is not None:
            restored_grp.role = first_ov.role
            restored_grp.role_provenance = "manual"
        if first_ov.enabled is not None:
            restored_grp.enabled = first_ov.enabled

        for ov, new_occ, actual_orig, _ in matches:
            new_overrides[new_occ.occ_id] = OccurrenceOverride(
                target_group_id=tgt_id,
                context_fingerprint=new_occ.context_fingerprint,
                expected_original_text=ov.expected_original_text,
                entity_type=ov.entity_type,
                role=ov.role,
                enabled=ov.enabled,
            )
            base_k = actual_orig.strip().lower()
            if base_k in base_groups_dict and new_occ in base_groups_dict[base_k].occurrences:
                base_groups_dict[base_k].occurrences.remove(new_occ)
            restored_grp.occurrences.append(new_occ)

        split_groups_dict[tgt_id] = restored_grp

    active_base_groups = [g for g in base_groups_dict.values() if g.occurrences]
    final_entity_groups = active_base_groups + list(split_groups_dict.values())

    # Fail-safe cleanup of parent_group_id if referenced target group no longer exists
    active_group_ids = {g.group_id for g in final_entity_groups}
    for g in final_entity_groups:
        if g.parent_group_id and g.parent_group_id not in active_group_ids:
            g.parent_group_id = None
            g.surface_tag = ""

    return final_entity_groups, new_overrides


async def cleanup_session_async(st: "AppState") -> None:
    """Asynchronously cancel any active LLM task, await teardown, and close provider session."""
    if hasattr(st, "cancel_setup_task"):
        await st.cancel_setup_task()
    active_task = getattr(st, "llm_active_task", None)
    if active_task and not active_task.done():
        active_task.cancel()
        try:
            await active_task
        except (asyncio.CancelledError, Exception):
            pass
        if getattr(st, "llm_active_task", None) is active_task:
            st.llm_active_task = None
    st.is_llm_running = False
    await st.close_llm_provider()


def reset_app_state(st: "AppState") -> None:
    """Reset all document-specific state in AppState cleanly."""
    if getattr(st, "llm_setup_task", None) and not st.llm_setup_task.done():
        st.llm_setup_task.cancel()
    st.llm_setup_task = None
    st.llm_setup_state = "idle"
    st.llm_setup_request_id = ""
    if getattr(st, "llm_active_task", None) and not st.llm_active_task.done():
        st.llm_active_task.cancel()
    st.llm_active_task = None
    st.filename = ""
    st.raw_text = ""
    st.last_raw_bytes = None
    st.entity_groups = []
    st.occurrence_overrides = {}
    st.current_mapping = {}
    st.current_anon_text = ""
    st.current_report = {}
    st.document_revision += 1
    st.llm_triage_results.clear()
    st.llm_triage_snapshot = ""
    st.llm_staged_selections.clear()
    st.is_llm_running = False
    st.llm_partial_failure = False
    st.llm_unprocessed_occ_ids.clear()
    st.llm_provider = None
    st.invalidate_llm_ready()
    # A document reset starts a fresh transient scope. Durable project/system
    # profiles remain untouched.
    if getattr(st, "profile_controller", None) is not None:
        st.document_overlay = st.profile_controller.discard_overlay()
    else:
        st.document_overlay = DocumentProfileOverlay()
    st.refresh_effective_config()


async def reset_app_state_async(st: "AppState") -> None:
    """Asynchronously clean up active task and provider, then reset state."""
    await cleanup_session_async(st)
    reset_app_state(st)


async def run_triage_batch_loop(
    state: "AppState",
    batches: List[Any],
    snapshot_hash: str,
    on_batch_complete: Optional[Callable[[], None]] = None,
) -> None:
    """
    Production batch execution loop.
    Validates snapshot and document_revision before and after every batch await.
    On drift or batch failure, marks all remaining batches (current and subsequent) as unprocessed.
    """
    for batch_idx, batch in enumerate(batches):
        current_snap = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups) if compute_triage_snapshot else ""
        if current_snap != snapshot_hash or state.document_revision != batch.document_revision:
            logging.info("Dokument- oder Reviewzustand vor Inferenz geändert -> Verbleibende Batches verworfen.")
            state.llm_partial_failure = True
            for rem_b in batches[batch_idx:]:
                state.llm_unprocessed_occ_ids.update(rem_b.occ_id_set)
            break

        try:
            raw_json = await state.llm_provider.generate(
                prompt=batch.user_prompt,
                system_prompt=batch.system_prompt,
            )
            # Re-validate snapshot after await
            post_snap = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups) if compute_triage_snapshot else ""
            if post_snap != snapshot_hash or state.document_revision != batch.document_revision:
                logging.info("Dokument- oder Reviewzustand nach Inferenz geändert -> Verbleibende Batches verworfen.")
                state.llm_partial_failure = True
                for rem_b in batches[batch_idx:]:
                    state.llm_unprocessed_occ_ids.update(rem_b.occ_id_set)
                break

            clean_json = extract_json_from_llm_response(raw_json) if extract_json_from_llm_response else raw_json
            envelope = TriageEnvelope.model_validate_json(clean_json)
            validate_batch_response(
                envelope,
                batch.occ_id_set,
                state.document_revision,
                snapshot_hash,
                expected_request_id=batch.request_id,
                strict_count=False,
            )

            batch_received_occ_ids = set()
            for item in envelope.items:
                if item.occ_id in batch.occ_id_set:
                    state.llm_triage_results[item.occ_id] = item
                    batch_received_occ_ids.add(item.occ_id)

            missing_in_batch = batch.occ_id_set - batch_received_occ_ids
            if missing_in_batch:
                state.llm_partial_failure = True
                state.llm_unprocessed_occ_ids.update(missing_in_batch)

        except asyncio.CancelledError:
            raise
        except Exception as ex:
            err_cls = type(ex).__name__
            logging.warning(f"Batch {batch.batch_index}/{batch.total_batches} fehlgeschlagen ({err_cls})")
            try:
                ui.notify(
                    f"Batch {batch.batch_index}/{batch.total_batches} konnte nicht verarbeitet werden ({err_cls}).",
                    type="warning",
                    close_button=True,
                    timeout=10000,
                )
            except Exception:
                pass

            state.llm_partial_failure = True
            for rem_b in batches[batch_idx:]:
                state.llm_unprocessed_occ_ids.update(rem_b.occ_id_set)
            break

        if on_batch_complete:
            on_batch_complete()
        await asyncio.sleep(0.01)


async def run_llm_triage_for_state(
    state: "AppState",
    triggered_from_analysis: bool = False,
    notify_cb: Optional[Callable[[str, str], None]] = None,
    on_batch_complete: Optional[Callable[[], None]] = None,
) -> None:
    """
    Direct asynchronous execution worker for LLM triage.
    Prepares batches, re-verifies live Ollama model presence, and executes the batch loop.
    """
    def notify(msg: str, type_: str = "info"):
        if notify_cb:
            notify_cb(msg, type_)
        else:
            try:
                ui.notify(msg, type=type_)
            except Exception:
                pass

    if not LLM_AVAILABLE:
        notify("LLM-Paket nicht verfügbar. Bitte `pip install local-anonymizer[llm]` installieren.", "warning")
        return

    if not state.config.llm_enabled:
        if not triggered_from_analysis:
            notify("Lokale LLM-Review-Assistenz ist deaktiviert. Bitte in den Einstellungen aktivieren.", "info")
        return

    if not state.config.llm_model_name or not state.config.llm_model_name.strip():
        notify("Bitte geben Sie einen Modellnamen in den LLM-Einstellungen an (z. B. qwen3:8b).", "warning")
        return

    if not state.raw_text or not state.entity_groups:
        if not triggered_from_analysis:
            notify("Keine analysierten Fundstellen vorhanden.", "info")
        return

    candidates = []
    for g in state.entity_groups:
        if not g.enabled:
            continue
        for occ in g.occurrences:
            ctx_snippet = strip_html_markup(occ.context_html) if occ.context_html else g.original_text
            candidates.append({
                "occ_id": occ.occ_id,
                "original_text": g.original_text,
                "entity_type": g.entity_type,
                "role": g.role,
                "context_snippet": ctx_snippet,
            })

    if not candidates:
        if not triggered_from_analysis:
            notify("Keine aktiven Fundstellen zum Prüfen gefunden.", "info")
        return

    snapshot_hash = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups) if compute_triage_snapshot else ""
    batches = prepare_triage_batches(candidates, state.document_revision, snapshot_hash) if prepare_triage_batches else []

    # Re-verify readiness against /api/ps if claimed ready
    if state.llm_ready_info is not None and state.llm_provider_type == "ollama" and verify_ollama_model_running is not None:
        ready_info_before = state.llm_ready_info
        bound_url_before = state.llm_ready_bound_url
        bound_model_before = state.llm_ready_bound_model
        provider_type_before = state.llm_provider_type
        url_snap = state.config.llm_base_url.strip()
        model_snap = state.config.llm_model_name.strip()

        try:
            active_info = await verify_ollama_model_running(url_snap, model_snap)
            if (
                state.llm_ready_info is ready_info_before
                and state.llm_ready_bound_url == bound_url_before
                and state.llm_ready_bound_model == bound_model_before
                and state.llm_provider_type == provider_type_before
            ):
                if active_info is None:
                    state.invalidate_llm_ready()
                elif active_info.expires_at:
                    new_exp = parse_iso_expiry(active_info.expires_at) if parse_iso_expiry else 0.0
                    if new_exp > 0.0:
                        state.llm_ready_expires_at = new_exp
                        state.llm_ready_info = active_info
        except Exception:
            pass

    try:
        if (
            state.llm_provider is None
            or getattr(state.llm_provider, "model_name", "") != state.config.llm_model_name.strip()
            or getattr(state.llm_provider, "base_url", "") != state.config.llm_base_url.strip()
        ):
            if state.llm_provider is not None:
                await state.close_llm_provider()
            state.llm_provider = LocalApiProvider(
                base_url=state.config.llm_base_url,
                model_name=state.config.llm_model_name,
            )
    except Exception as e:
        notify(f"Fehler bei LLM-Initialisierung: {e}", "negative")
        return

    state.llm_partial_failure = False
    state.llm_unprocessed_occ_ids.clear()
    state.llm_triage_results.clear()
    state.llm_staged_selections.clear()
    state.llm_triage_snapshot = snapshot_hash

    if on_batch_complete:
        on_batch_complete()

    notify(f"Starte LLM-Triage ({len(candidates)} Fundstellen in {len(batches)} Batches)...", "info")

    try:
        await run_triage_batch_loop(
            state,
            batches,
            snapshot_hash,
            on_batch_complete=on_batch_complete,
        )

        if state.llm_partial_failure:
            notify(
                f"LLM-Triage unvollständig ({len(state.llm_triage_results)} geprüft, {len(state.llm_unprocessed_occ_ids)} ungeprüft).",
                "warning",
            )
        else:
            notify(f"LLM-Triage abgeschlossen ({len(state.llm_triage_results)} Fundstellen geprüft).", "positive")

    except asyncio.CancelledError:
        notify("LLM-Triage abgebrochen.", "info")
        state.llm_triage_results.clear()
        state.llm_triage_snapshot = ""
        state.llm_staged_selections.clear()
    except Exception as e:
        notify(f"Fehler bei LLM-Triage: {e}", "negative")


def launch_llm_triage_for_state(
    state: "AppState",
    triggered_from_analysis: bool = False,
    notify_cb: Optional[Callable[[str, str], None]] = None,
    on_update_ui: Optional[Callable[[], None]] = None,
    client: Any = None,
) -> Optional[asyncio.Task]:
    """
    Synchronously validate preconditions and atomically launch LLM triage in a background task.
    """
    def notify(msg: str, type_: str = "info"):
        if notify_cb:
            notify_cb(msg, type_)
        else:
            try:
                ui.notify(msg, type=type_)
            except Exception:
                pass

    if state.is_llm_running or (state.llm_active_task and not state.llm_active_task.done()):
        if not triggered_from_analysis:
            notify("Eine LLM-Prüfung läuft bereits.", "info")
        return state.llm_active_task

    if state.llm_setup_state != "idle":
        if not triggered_from_analysis:
            notify("LLM-Prüfung kann während laufendem Setup nicht gestartet werden.", "warning")
        return None

    if state.is_analyzing and not triggered_from_analysis:
        notify("LLM-Prüfung kann während laufender Textanalyse nicht gestartet werden.", "warning")
        return None

    if not LLM_AVAILABLE:
        notify("LLM-Paket nicht verfügbar. Bitte `pip install local-anonymizer[llm]` installieren.", "warning")
        return None

    if not state.config.llm_enabled:
        if not triggered_from_analysis:
            notify("Lokale LLM-Review-Assistenz ist deaktiviert. Bitte in den Einstellungen aktivieren.", "info")
        return None

    if not state.config.llm_model_name or not state.config.llm_model_name.strip():
        notify("Bitte geben Sie einen Modellnamen in den LLM-Einstellungen an (z. B. qwen3:8b).", "warning")
        return None

    if not state.raw_text or not state.entity_groups:
        if not triggered_from_analysis:
            notify("Keine analysierten Fundstellen vorhanden.", "info")
        return None

    if state.analyzed_config_hash is not None and (
        state.preview_stale
        or (
            state.effective_config is not None
            and state.analyzed_config_hash != state.effective_config.snapshot_hash
        )
    ):
        notify("LLM-Triage gesperrt: Die Konfiguration wurde seit der Analyse geändert. Bitte zuerst neu analysieren.", "warning")
        return None

    # Synchronously lock busy state and transfer ownership from analysis
    state.is_analyzing = False
    state.is_llm_running = True

    try:
        if on_update_ui:
            on_update_ui()
    except Exception:
        state.is_llm_running = False
        raise

    def _cleanup_task(t: asyncio.Task) -> None:
        """Identity-bound idempotent cleanup callback that safely releases reservation even if cancelled prior to worker start."""
        if state.llm_active_task is t or state.llm_active_task is None:
            state.is_llm_running = False
            state.llm_active_task = None
            if on_update_ui:
                try:
                    on_update_ui()
                except Exception:
                    pass

    async def _runner():
        current_task = asyncio.current_task()
        try:
            if client is not None:
                with client:
                    await run_llm_triage_for_state(
                        state,
                        triggered_from_analysis=triggered_from_analysis,
                        notify_cb=notify_cb,
                        on_batch_complete=on_update_ui,
                    )
            else:
                await run_llm_triage_for_state(
                    state,
                    triggered_from_analysis=triggered_from_analysis,
                    notify_cb=notify_cb,
                    on_batch_complete=on_update_ui,
                )
        finally:
            if current_task is not None:
                _cleanup_task(current_task)

    task = asyncio.create_task(_runner())
    state.llm_active_task = task
    task.add_done_callback(_cleanup_task)
    return task


async def load_document_into_state_async(
    state: "AppState",
    text: str,
    filename: str,
    raw_bytes: Optional[bytes] = None,
) -> None:
    """
    Asynchronously clean up active LLM session and load new document text/bytes into AppState.
    """
    await reset_app_state_async(state)
    state.filename = filename
    state.raw_text = text
    state.last_raw_bytes = raw_bytes


def load_document_into_state(
    state: "AppState",
    text: str,
    filename: str,
    raw_bytes: Optional[bytes] = None,
) -> None:
    """
    Synchronous fallback to load extracted document text and optional raw bytes into AppState.
    Clears previous analysis and overrides via reset_app_state while preserving the new raw_bytes.
    """
    reset_app_state(state)
    state.filename = filename
    state.raw_text = text
    state.last_raw_bytes = raw_bytes


def format_link_dropdown_label(grp: EntityGroup) -> str:
    """
    Generate an unambiguous visible label for an EntityGroup in link dropdowns.
    Includes text, type, optional role, base/split marker with short ID, and count.
    """
    is_split = grp.group_id != grp.text_key
    desc_parts = [grp.entity_type]
    if grp.role:
        desc_parts.append(grp.role)
    if is_split:
        short_id = grp.group_id.replace("split_", "")[:6]
        desc_parts.append(f"Ausgliederung · {short_id}")
    else:
        desc_parts.append("Basis")
    desc_parts.append(f"{grp.count}x")
    return f"{grp.original_text} ({', '.join(desc_parts)})"


@dataclass
class HomonymCluster:
    text_key: str
    primary_text: str
    nodes: List[Any]


def group_tree_nodes_by_homonym(tree: List[Any]) -> List[HomonymCluster]:
    """
    Cluster root tree nodes by normalized text_key for visual grouping in the UI.
    Maintains relative sorting order of the clusters according to the first seen node.
    Does NOT modify or introduce any semantic parent_group_id links.
    """
    clusters_dict: Dict[str, HomonymCluster] = {}
    ordered_keys: List[str] = []

    for root_node in tree:
        item = root_node.item
        t_key = getattr(item, "text_key", getattr(item, "original_text", str(item)).strip().lower())
        if t_key not in clusters_dict:
            clusters_dict[t_key] = HomonymCluster(
                text_key=t_key,
                primary_text=getattr(item, "original_text", str(item)),
                nodes=[],
            )
            ordered_keys.append(t_key)
        clusters_dict[t_key].nodes.append(root_node)

    return [clusters_dict[k] for k in ordered_keys]


# --- App State ---
class AppState:
    def __init__(self):
        self.config: AppConfig = AppConfig.load()
        self.profile_store = ProfileStore(CONFIG_DIR)
        self.profile_controller: Optional[ProfileController] = None
        try:
            self.profile_store.initialize_or_migrate()
            self.profile_controller = ProfileController(self.profile_store)
            self.system_profile = self.profile_controller.system_profile
            self.project_profile = self.profile_controller.project_profile
        except Exception as ex:
            logging.warning("Could not load profile scopes; using AppConfig compatibility state: %s", ex)
            self.system_profile = None
            self.project_profile = None
        self.document_overlay = (
            self.profile_controller.document_overlay
            if self.profile_controller is not None
            else DocumentProfileOverlay()
        )
        self.effective_config = None
        self.filename: str = ""
        self.raw_text: str = ""
        self.entity_groups: List[EntityGroup] = []
        self.occurrence_overrides: Dict[str, OccurrenceOverride] = {}
        self.entity_modes: Dict[str, str] = resolve_entity_modes(self.config)
        # Legacy compatibility for config files and code paths that still expose active_entities.
        self.active_entities: List[str] = [
            entity for entity, mode in self.entity_modes.items()
            if mode in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
        ]
        self.format_mode: str = self.config.format_mode  # "numbered", "numbered_role", "role_only"
        self.export_format: str = self.config.export_format  # "txt", "md"
        self.gliner_model_name: str = getattr(self.config, "gliner_model_name", GLINER_MODEL_NAME)
        self.gliner_threshold: float = self.config.gliner_threshold
        self.enable_eupii: bool = self.config.enable_eupii
        self.eupii_threshold: float = self.config.eupii_threshold
        self.eupii_model_name: str = self.config.eupii_model_name
        self.sort_by: str = "Alphabetisch (A–Z)"  # Default alphabetical
        self.ignore_terms_text: str = self.config.ignore_terms
        self.glossary_text: str = self.config.glossary
        self.colliding_roles: Set[Tuple[str, str]] = set()
        self.analyzed_config_hash: Optional[str] = None
        self.preview_stale: bool = False

        self.refresh_effective_config()

        # Shared mapping for Tab 1 -> Tab 2 handoff
        self.current_mapping: Dict[str, str] = {}
        self.current_report: Dict[str, Any] = {}
        self.current_anon_text: str = ""

        # Tab 2 state
        self.restore_anon_text: str = ""
        self.restore_mapping: Dict[str, str] = {}
        self.restored_text: str = ""

        # Document extraction settings
        self.include_headers_footers: bool = False
        self.extract_picture_text: bool = True
        self.last_raw_bytes: Optional[bytes] = None

        # Document and review revision tracking (PR 1 / Phase 6A)
        self.document_revision: int = 0
        self.analysis_revision: int = 0
        self.preview_revision: int = 0

        # LLM Triage Layer State
        self.llm_provider: Optional[Any] = None
        self.llm_triage_results: Dict[str, Any] = {}
        self.llm_triage_snapshot: str = ""
        self.llm_staged_selections: Set[str] = set()
        self.is_llm_running: bool = False
        self.llm_partial_failure: bool = False
        self.llm_unprocessed_occ_ids: Set[str] = set()
        self.llm_active_task: Optional[asyncio.Task] = None

        # LLM Setup & Preloading State (Phase 6A.1)
        self.llm_provider_type: str = getattr(self.config, "llm_provider_type", "ollama")
        self.llm_setup_state: str = "idle"  # idle, discovering, preloading, testing
        self.llm_setup_task: Optional[asyncio.Task] = None
        self.llm_setup_request_id: str = ""
        self.llm_setup_endpoint_snapshot: str = ""
        self.llm_setup_model_snapshot: str = ""
        self.llm_discovered_models: List[str] = []
        self.llm_discovery_status: Optional[str] = None
        self.llm_custom_model_mode: bool = False
        self.llm_custom_model_name: str = ""
        self.llm_ready_info: Optional[Any] = None
        self.llm_ready_timestamp: float = 0.0
        self.llm_ready_bound_url: str = ""
        self.llm_ready_bound_model: str = ""
        self.llm_setup_status_msg: str = ""
        self.llm_show_details: bool = True
        # Analysis Lock State
        self.is_analyzing: bool = False
        self.llm_ready_expires_at: float = 0.0

        self.mutating_ui_zones: Dict[str, List[Any]] = {
            "sidebar": [],
            "table": [],
            "workspace": [],
            "llm": [],
            "llm_settings": [],
            "ignore": [],
            "glossary": [],
        }

    @property
    def is_busy(self) -> bool:
        """Central check whether any mutating async operation is in-flight."""
        return self.is_analyzing or self.is_llm_running or self.llm_setup_state != "idle"

    def refresh_effective_config(self, reload_profiles: bool = False) -> Any:
        """Rebuild the immutable profile snapshot and update stale-result state."""
        if self.profile_controller is not None:
            self.profile_controller.system_profile = self.system_profile
            self.profile_controller.project_profile = self.project_profile
            self.profile_controller.document_overlay = self.document_overlay
        if reload_profiles and self.project_profile is not None:
            try:
                manifest = self.profile_store.load_manifest()
                self.system_profile = self.profile_store.load_system_profile()
                self.project_profile = self.profile_store.load_project_profile(manifest["active_project_id"])
                if self.profile_controller is not None:
                    self.profile_controller.manifest = manifest
                    self.profile_controller.system_profile = self.system_profile
                    self.profile_controller.project_profile = self.project_profile
            except Exception as ex:
                logging.warning("Could not reload profile scopes: %s", ex)
        if self.system_profile is not None and self.project_profile is not None:
            self.effective_config = ScopeResolutionEngine.resolve(
                self.system_profile,
                self.project_profile,
                self.document_overlay,
                set(AVAILABLE_ENTITIES),
            )
            self.entity_modes = dict(self.effective_config.entity_modes)
            self.active_entities = [
                entity for entity, mode in self.entity_modes.items()
                if mode in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
            ]
        if self.effective_config is not None and self.analyzed_config_hash is not None:
            self.preview_stale = self.effective_config.snapshot_hash != self.analyzed_config_hash
        return self.effective_config

    @property
    def mutating_ui_elements(self) -> List[Any]:
        """Flattened list of all registered mutating UI elements across all zones."""
        res: List[Any] = []
        for elems in self.mutating_ui_zones.values():
            res.extend(elems)
        return res

    def register_mutating_element(self, elem: Any, zone: str = "table") -> None:
        """Register a mutating UI control within a specific UI zone."""
        self.mutating_ui_zones.setdefault(zone, []).append(elem)
        if self.is_busy:
            try:
                elem.disable()
            except Exception:
                pass

    def clear_mutating_zone(self, zone: str = "table") -> None:
        """Clear registered elements for a specific UI zone (e.g. before re-rendering review table)."""
        if zone in self.mutating_ui_zones:
            self.mutating_ui_zones[zone].clear()

    def set_all_mutating_elements_disabled(self, disabled: bool) -> None:
        """Disable or enable all registered mutating UI controls across all zones."""
        for zone, elements in self.mutating_ui_zones.items():
            for elem in elements:
                try:
                    if disabled:
                        elem.disable()
                    else:
                        elem.enable()
                except Exception:
                    pass

    def invalidate_llm_ready(self) -> None:
        """Reset transient model readiness status."""
        self.llm_ready_info = None
        self.llm_ready_timestamp = 0.0
        self.llm_ready_expires_at = 0.0
        self.llm_ready_bound_url = ""
        self.llm_ready_bound_model = ""
        if self.llm_setup_status_msg == "Modell bereit":
            self.llm_setup_status_msg = ""

    def is_model_ready(self) -> bool:
        """
        Check if active model was verified via /api/ps, config matches,
        and keep-alive expiration time has not been exceeded.
        """
        if (
            self.llm_ready_info is not None
            and self.llm_provider_type == "ollama"
            and self.llm_ready_bound_url == self.config.llm_base_url.strip()
            and self.llm_ready_bound_model == self.config.llm_model_name.strip()
            and self.llm_ready_expires_at > 0.0
        ):
            if time.time() >= self.llm_ready_expires_at:
                self.invalidate_llm_ready()
                return False
            return True
        return False

    async def cancel_setup_task(self) -> None:
        """Cancel and await currently active discovery or preload setup task."""
        self.llm_setup_request_id = ""
        task = self.llm_setup_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self.llm_setup_task = None
        self.llm_setup_state = "idle"
        self.invalidate_llm_ready()

    async def close_llm_provider(self) -> None:
        """Close provider HTTP session safely."""
        if self.llm_provider is not None:
            try:
                await self.llm_provider.close()
            except Exception:
                pass
            self.llm_provider = None


# Background model warmup state
_model_ready: bool = False
_cached_anonymizer: Any = None
_model_lock = threading.Lock()


def parse_glossary(text: str) -> Dict[str, str]:
    """Parse key-value glossary lines: 'Term: ENTITY_TYPE' or JSON."""
    glossary = {}
    text = text.strip()
    if not text:
        return glossary
    if text.startswith("{"):
        try:
            raw = json.loads(text)
            if isinstance(raw, dict):
                for term, value in raw.items():
                    if isinstance(value, str):
                        glossary[term] = value.split("|", 1)[0].strip().upper()
                return glossary
        except Exception:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            parts = line.split(":", 1)
            glossary[parts[0].strip()] = parts[1].split("|", 1)[0].strip().upper()
        elif "=" in line:
            parts = line.split("=", 1)
            glossary[parts[0].strip()] = parts[1].split("|", 1)[0].strip().upper()
    return glossary


def parse_glossary_roles(text: str) -> Dict[str, str]:
    """Parse optional ``| role`` suffixes while keeping the legacy glossary API intact."""
    roles: Dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            continue
        term, value = (part.strip() for part in line.split(separator, 1))
        if "|" in value:
            _, role = (part.strip() for part in value.split("|", 1))
            if term and role:
                roles[term] = role
    return roles


def parse_ignore_terms(text: str) -> List[str]:
    """Parse ignore terms separated by comma or newlines."""
    terms = []
    for line in text.replace(",", "\n").splitlines():
        t = line.strip()
        if t:
            terms.append(t)
    return terms


def build_anonymizer(
    app_state: Optional[AppState] = None,
    effective_config: Optional[EffectiveConfig] = None,
):
    """Build LocalAnonymizer instance with specified state settings or loaded config."""
    from local_anonymizer.anonymizer import LocalAnonymizer

    if app_state is not None:
        snapshot = effective_config or app_state.effective_config
        if snapshot is None:
            raise RuntimeError("No EffectiveConfig snapshot available for anonymizer construction")
        general_entities, glossary_entities = get_recognizer_entities(dict(snapshot.entity_modes))
        return LocalAnonymizer(
            language="de",
            glossary=dict(snapshot.glossary),
            glossary_roles=dict(snapshot.glossary_roles),
            ignore_terms=list(snapshot.ignore_terms),
            enabled_entities=general_entities,
            enabled_glossary_entities=glossary_entities,
            gliner_model=app_state.gliner_model_name,
            gliner_threshold=app_state.gliner_threshold,
            enable_eupii=app_state.enable_eupii,
            eupii_threshold=app_state.eupii_threshold,
            eupii_model=app_state.eupii_model_name,
            entity_modes=dict(snapshot.entity_modes),
        )
    else:
        cfg = AppConfig.load()
        entity_modes = resolve_entity_modes(cfg)
        general_entities, glossary_entities = get_recognizer_entities(entity_modes)
        return LocalAnonymizer(
            language="de",
            glossary=parse_glossary(cfg.glossary),
            glossary_roles=parse_glossary_roles(cfg.glossary),
            ignore_terms=parse_ignore_terms(cfg.ignore_terms),
            enabled_entities=general_entities,
            enabled_glossary_entities=glossary_entities,
            gliner_model=getattr(cfg, "gliner_model_name", GLINER_MODEL_NAME),
            gliner_threshold=cfg.gliner_threshold,
            enable_eupii=cfg.enable_eupii,
            eupii_threshold=cfg.eupii_threshold,
            eupii_model=cfg.eupii_model_name,
            entity_modes=entity_modes,
        )


def sync_cached_anonymizer_settings(
    anon,
    app_state: "AppState",
    effective_config: Optional[EffectiveConfig] = None,
) -> None:
    """
    Push the current UI-configured settings (entity modes, threshold, ignore terms, glossary)
    onto an already-built LocalAnonymizer instance, so a cached instance stays in sync with
    sidebar edits instead of only reflecting the settings it was first constructed with.

    Single source of truth for this sync -- previously duplicated inline, which is how the
    glossary update silently wrote to a dead `.terms` attribute instead of the real `.glossary`
    for a while without anyone noticing.
    """
    snapshot = effective_config or app_state.effective_config
    if snapshot is None:
        raise RuntimeError("No EffectiveConfig snapshot available for anonymizer synchronization")
    general_entities, glossary_entities = get_recognizer_entities(dict(snapshot.entity_modes))
    anon.enabled_entities = general_entities
    anon.enabled_glossary_entities = glossary_entities
    anon.gliner_recognizer.threshold = app_state.gliner_threshold
    anon.set_eupii_enabled(app_state.enable_eupii, app_state.eupii_threshold)
    anon.set_entity_modes(dict(snapshot.entity_modes))
    anon.set_ignore_terms(list(snapshot.ignore_terms))
    anon.set_glossary(dict(snapshot.glossary), glossary_roles=dict(snapshot.glossary_roles))


def get_synced_cached_anonymizer(
    app_state: "AppState",
    effective_config: Optional[EffectiveConfig] = None,
):
    """Return the shared cached LocalAnonymizer, building it if needed and syncing it to the
    current UI settings. Must be called while holding `_model_lock`."""
    global _cached_anonymizer
    if _cached_anonymizer is None:
        _cached_anonymizer = build_anonymizer(app_state, effective_config=effective_config)
    else:
        sync_cached_anonymizer_settings(_cached_anonymizer, app_state, effective_config=effective_config)
    return _cached_anonymizer


_SOURCE_KIND_LABELS: Dict[str, Tuple[str, str]] = {
    "prompt": ("🤖 KI-Prompt", "blue"),
    "model": ("🤖 KI-Modell", "indigo"),
    "regex": ("🔤 Regex", "purple"),
    "library": ("📚 Bibliothek", "grey-7"),
    "glossary": ("📖 Begriffsliste", "teal"),
}


def render_entity_source_overview(overview: List[Dict[str, Any]]) -> None:
    """Render the entity-source transparency overview (see LocalAnonymizer.get_entity_source_overview)
    as a read-only list: per category, whether it's active and exactly how it's detected."""
    if not overview:
        ui.label("Keine Kategorien gefunden.").classes("text-xs text-slate-400")
        return

    for row in overview:
        active = row["active"]
        mode = row.get("mode")
        with ui.card().classes("w-full p-2 " + ("bg-white" if active else "bg-slate-50 opacity-60")):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.icon(
                    "check_circle" if active else "radio_button_unchecked",
                    color="positive" if active else "grey",
                ).classes("text-sm")
                ui.label(row["category"]).classes("text-sm font-mono font-bold text-slate-800")
                if mode == ENTITY_MODE_EXPLICIT_EUPII:
                    ui.badge("Glossar, manuell, deterministisch & EU-PII (ohne GLiNER)", color="blue").props("dense")
                elif mode == ENTITY_MODE_EXPLICIT_ONLY:
                    ui.badge("nur explizite Einträge & manuell", color="orange-8").props("dense")
                elif mode == ENTITY_MODE_ALL:
                    ui.badge("alle Quellen (inkl. GLiNER)", color="green-8").props("dense")
                elif mode == "automatic_only":
                    ui.badge("nur automatische Erkennung", color="purple").props("dense")
                elif not active or mode == ENTITY_MODE_OFF:
                    ui.badge("inaktiv", color="grey-5").props("dense")

            with ui.column().classes("w-full gap-1 mt-1 pl-6"):
                for src in row["sources"]:
                    src_active = src.get("active", active)
                    kind_label, kind_color = _SOURCE_KIND_LABELS.get(src["kind"], (src["kind"], "grey"))
                    with ui.row().classes("items-start gap-2 flex-wrap" + ("" if src_active else " opacity-40")):
                        ui.badge(kind_label, color=kind_color if src_active else "grey-5").props("dense outline")
                        if src["kind"] == "prompt":
                            prompts_str = ", ".join(f'"{p}"' for p in src["prompts"])
                            status_suffix = "" if src_active else " (inaktiv in diesem Modus)"
                            ui.label(f"GLiNER Zero-Shot: {prompts_str}{status_suffix}").classes(
                                "text-xs text-slate-600 font-mono"
                            )
                        elif src["kind"] == "model":
                            model_name = src.get("model_name", "bardsai/eu-pii-anonimization-multilang")
                            thresh = src.get("threshold", 0.5)
                            status_suffix = "" if src_active else " (inaktiv in diesem Modus)"
                            ui.label(f"EU-PII Token-Klassifikator ({model_name}, Schwellenwert: {thresh:.2f}){status_suffix}").classes(
                                "text-xs text-slate-600 font-mono"
                            )
                        elif src["kind"] == "regex":
                            with ui.column().classes("gap-0"):
                                for p in src["patterns"]:
                                    ui.label(f'{p["name"]}: {p["regex"]}').classes(
                                        "text-xs text-slate-600 font-mono break-all"
                                    )
                        elif src["kind"] == "glossary":
                            count = src["entry_count"]
                            noun = "Eintrag" if count == 1 else "Einträge"
                            ui.label(f"{count} {noun} in deiner Begriffsliste").classes("text-xs text-slate-600")
                        elif src["kind"] == "library":
                            rec_name = src.get("recognizer", "")
                            if "Phone" in rec_name:
                                ui.label("Google phonenumbers (libphonenumber für CH/intl. Rufnummern)").classes("text-xs text-slate-600")
                            else:
                                ui.label(f'externe Bibliothek ({rec_name})').classes("text-xs text-slate-600")


def _warmup_background_thread():
    """Worker to initialize cached AI models in the background without blocking the UI or triggering unannounced downloads."""
    global _cached_anonymizer, _model_ready
    try:
        logging.info("Background warming up local AI models...")
        anon = build_anonymizer()

        # 1. Warmup GLiNER only if already cached locally
        if is_model_cached(anon.gliner_recognizer.model_name, "gliner"):
            anon.gliner_recognizer.load()
        else:
            logging.info(f"GLiNER model '{anon.gliner_recognizer.model_name}' (~1.10 GB) is not in local cache; skipping silent warmup download.")

        # 2. Warmup EU-PII only if enabled and already cached locally
        if anon.enable_eupii:
            if is_model_cached(anon.eupii_recognizer.model_name, "transformers"):
                anon.eupii_recognizer.load()
            else:
                logging.info(f"EU-PII model '{anon.eupii_recognizer.model_name}' (~1.07 GB) is not in local cache; skipping silent warmup download.")

        with _model_lock:
            _cached_anonymizer = anon
            _model_ready = True
        logging.info("AI models warmup check finished.")
    except Exception as e:
        logging.error(f"Error during background model warmup: {e}", exc_info=True)
        with _model_lock:
            _model_ready = True


# Surface tag options as a clean dictionary
SURFACE_TAG_OPTIONS: Dict[str, str] = {
    "VOLLNAME": "Vollname (z. B. Julia Meier)",
    "VORNAME": "Vorname (z. B. Julia)",
    "NACHNAME": "Nachname (z. B. Meier)",
    "ANREDE": "Anrede / Titel (z. B. Frau Meier, Mr. Smith)",
    "GENITIV": "Genitiv (z. B. Julias)",
    "KURZFORM": "Kurzform / Kürzel (z. B. JM)",
}

def get_entity_mode_options(ent: str) -> Dict[str, str]:
    if ent in ["PERSON", "LOCATION", "ID_NUMBER", "HEALTH_DATA"]:
        return {
            ENTITY_MODE_OFF: "Aus – nichts anonymisieren",
            ENTITY_MODE_EXPLICIT_ONLY: "Nur Glossar & manuell",
            ENTITY_MODE_EXPLICIT_EUPII: "Nur Glossar, manuell, deterministisch & EU-PII (ohne GLiNER)",
            ENTITY_MODE_ALL: "Alle Quellen (inkl. GLiNER)",
        }
    return {
        ENTITY_MODE_OFF: "Aus – nichts anonymisieren",
        ENTITY_MODE_EXPLICIT_ONLY: "Nur Glossar & manuell",
        ENTITY_MODE_ALL: "Alle Quellen",
    }

ENTITY_MODE_COLORS: Dict[str, str] = {
    ENTITY_MODE_OFF: "bg-red-100 text-red-900 border-red-300",
    ENTITY_MODE_EXPLICIT_ONLY: "bg-orange-100 text-orange-900 border-orange-300",
    ENTITY_MODE_EXPLICIT_EUPII: "bg-blue-100 text-blue-900 border-blue-300",
    ENTITY_MODE_ALL: "bg-green-100 text-green-900 border-green-300",
}


def entity_mode_classes(mode: str) -> str:
    """Return stable base and state color classes for one category mode selector."""
    return f"w-64 text-xs {ENTITY_MODE_COLORS.get(mode, ENTITY_MODE_COLORS[ENTITY_MODE_OFF])}"

RECOGNIZER_METHODS: Dict[str, str] = {
    "GLiNERRecognizer": "gliner",
    "EUPiiRecognizer": "eupii",
    "AddressPatternRecognizer": "regex",
    "AHVNumberRecognizer": "regex",
    "UIDNumberRecognizer": "regex",
    "IbanRecognizer": "regex",
    "EmailRecognizer": "regex",
    "UrlRecognizer": "regex",
    "DateRecognizer": "regex",
    "PhoneRecognizer": "library",
}

METHOD_DISPLAY: Dict[str, Tuple[str, str, str]] = {
    "gliner": ("🤖 GLiNER", "blue", "Durch das lokale GLiNER-Zero-Shot-Modell gefunden"),
    "eupii": ("🤖 EU-PII", "indigo", "Durch das lokale EU-PII-Token-Klassifikationsmodell gefunden"),
    "ai": ("🤖 KI", "blue", "Durch ein lokales KI-Modell gefunden"),
    "regex": ("🔤 Regex", "purple", "Durch einen regulären Ausdruck gefunden"),
    "library": ("📚 Bibliothek", "grey-7", "Durch eine lokale Presidio-/Python-Bibliothek (z. B. phonenumbers für Telefon) gefunden"),
    "glossary_direct": ("📖 Glossar · direkt", "teal", "Direkter Treffer in der eigenen Begriffsliste"),
    "glossary_fuzzy": ("📖 Glossar · Fuzzy", "orange", "Ähnlichkeitstreffer in der eigenen Begriffsliste"),
    "manual": ("✍ Manuell", "green", "Manuell für diesen Durchlauf markiert"),
}


def classify_recognition_result(result: Any) -> Tuple[str, str]:
    """Return the coarse source and display method for one Presidio result."""
    metadata = result.recognition_metadata or {}
    recognizer_name = metadata.get("recognizer_name", "")
    if recognizer_name == "FuzzyGlossaryRecognizer":
        match_kind = metadata.get("glossary_match", "direct")
        return "glossary", "glossary_fuzzy" if match_kind == "fuzzy" else "glossary_direct"
    method = metadata.get("detection_method") or RECOGNIZER_METHODS.get(recognizer_name, "library")
    return "automatic", method


def method_display(method: str) -> Tuple[str, str, str]:
    """Return label, badge color, and tooltip for an occurrence method."""
    return METHOD_DISPLAY.get(method, (method, "grey-7", "Erkennungsmethode"))


def resolve_entity_modes(config: AppConfig) -> Dict[str, str]:
    """Load source-aware category modes, migrating older active_entities settings safely."""
    saved_modes = getattr(config, "entity_modes", {}) or {}
    legacy_active = set(getattr(config, "active_entities", []) or [])

    def get_legacy_default(ent: str) -> str:
        if ent in legacy_active:
            if ent in ["PERSON", "LOCATION", "ID_NUMBER", "HEALTH_DATA"]:
                return ENTITY_MODE_EXPLICIT_EUPII
            return ENTITY_MODE_ALL
        return ENTITY_MODE_OFF

    return {
        entity: saved_modes.get(entity, get_legacy_default(entity))
        for entity in AVAILABLE_ENTITIES
    }


def get_recognizer_entities(entity_modes: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Translate UI modes into general-recognizer and glossary category filters."""
    general_entities = [
        entity for entity in AVAILABLE_ENTITIES
        if entity_modes.get(entity, ENTITY_MODE_OFF) in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
    ]
    glossary_entities = [
        entity for entity in AVAILABLE_ENTITIES
        if entity_modes.get(entity, ENTITY_MODE_OFF) in (ENTITY_MODE_EXPLICIT_ONLY, ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
    ]
    return general_entities, glossary_entities


def is_category_enabled(st: AppState, entity_type: str) -> bool:
    """Return whether a category may contribute to the anonymized output at all."""
    return st.entity_modes.get(entity_type, ENTITY_MODE_OFF) != ENTITY_MODE_OFF


def get_active_occurrences(st: AppState, group: EntityGroup) -> List[EntityOccurrence]:
    """Filter a group's occurrences according to its category mode and detection source."""
    mode = st.entity_modes.get(group.entity_type, ENTITY_MODE_OFF)
    if mode == ENTITY_MODE_OFF:
        return []
    if mode == ENTITY_MODE_EXPLICIT_ONLY:
        return [occ for occ in group.occurrences if occ.source in ("glossary", "manual")]
    if mode == ENTITY_MODE_EXPLICIT_EUPII:
        return [
            occ for occ in group.occurrences
            if occ.source in ("glossary", "manual") or occ.method != "gliner"
        ]
    return list(group.occurrences)


# Start background warmup only after all configuration helpers are defined, and never in PDF worker subprocesses.
if not is_pdf_worker():
    threading.Thread(target=_warmup_background_thread, daemon=True).start()


def save_current_config(st: AppState):
    """Save user modifications back to persistent config."""
    manifest = st.profile_store.load_manifest()
    warning_required = int(manifest.get("warning_acknowledged_version", 0)) < 1
    if warning_required and not getattr(st, "_profile_warning_confirmed", False):
        if getattr(st, "_profile_warning_dialog_open", False):
            return False
        st._profile_warning_dialog_open = True
        understood = {"value": False}
        with ui.dialog() as warning_dialog, ui.card().classes("p-5 max-w-xl bg-white rounded-xl shadow-xl"):
            ui.label("⚠️ Wichtiger Hinweis zur lokalen Datenspeicherung").classes("text-lg font-bold text-slate-800")
            ui.markdown(
                "Profile, Begriffe, Rollen und Ignore-Einträge werden lokal im Benutzerordner gespeichert, "
                "standardmässig unverschlüsselt. BitLocker/FileVault schützt nur das Gerät im ausgeschalteten "
                "Zustand; andere Programme bei entsperrter Sitzung können die Dateien lesen. Eine Cloud-Synchronisation "
                "kann Kopien ausserhalb dieses Geräts erzeugen."
            ).classes("text-sm text-slate-700 leading-relaxed mt-2")
            confirmation = ui.checkbox("Ich habe diesen Hinweis verstanden und möchte fortfahren.")
            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                def cancel_warning() -> None:
                    st._profile_warning_dialog_open = False
                    warning_dialog.close()

                def confirm_warning() -> None:
                    if not confirmation.value:
                        ui.notify("Bitte bestätige zuerst den Datenschutzhinweis.", type="warning")
                        return
                    try:
                        current = st.profile_store.load_manifest()
                        saved_manifest = dict(current)
                        saved_manifest["warning_acknowledged_version"] = 1
                        saved_manifest = st.profile_store.save_manifest(
                            saved_manifest, expected_revision=int(current["revision"])
                        )
                        if st.profile_controller is not None:
                            st.profile_controller.manifest = saved_manifest
                        st._profile_warning_confirmed = True
                        st._profile_warning_dialog_open = False
                        warning_dialog.close()
                        save_current_config(st)
                    except RevisionConflictError:
                        st._profile_warning_dialog_open = False
                        warning_dialog.close()
                        ui.notify("Speichern abgebrochen: Der Hinweisstatus wurde extern geändert. Bitte erneut versuchen.", type="warning")
                    except Exception as ex:
                        st._profile_warning_dialog_open = False
                        warning_dialog.close()
                        ui.notify(f"Datenschutzhinweis konnte nicht bestätigt werden: {ex}", type="negative")

                ui.button("Abbrechen", on_click=cancel_warning).props("flat")
                ui.button("Bestätigen & Fortfahren", on_click=confirm_warning, color="primary").props("unelevated")
        warning_dialog.open()
        return False

    st._profile_warning_confirmed = False
    controller = getattr(st, "profile_controller", None)
    if controller is None or st.project_profile is None:
        st.config.format_mode = st.format_mode
        st.config.entity_modes = dict(st.entity_modes)
        st.config.ignore_terms = st.ignore_terms_text
        st.config.glossary = st.glossary_text
        return bool(st.config.save())

    # Keep the loaded controller revisions as the session's CAS baseline. Do
    # not create a new ProfileStore or reload immediately before saving.
    controller.system_profile = st.system_profile
    controller.project_profile = st.project_profile
    controller.document_overlay = st.document_overlay
    project = controller.project_profile
    project.entity_modes = dict(st.entity_modes)
    parsed_glossary = parse_glossary(st.glossary_text)
    parsed_roles = parse_glossary_roles(st.glossary_text)
    project.glossary_terms = {
        normalize_term_key(term): ScopedTerm(
            term=term,
            term_key=normalize_term_key(term),
            entity_type=entity_type,
            role=parsed_roles.get(term),
        )
        for term, entity_type in parsed_glossary.items()
        if normalize_term_key(term)
    }
    project.ignore_terms = {
        normalize_term_key(term): ScopedTerm(term=term, term_key=normalize_term_key(term))
        for term in parse_ignore_terms(st.ignore_terms_text)
        if normalize_term_key(term)
    }
    config = getattr(st, "config", None)
    manifest_updates = {
        "format_mode": st.format_mode,
        "gliner_model_name": st.gliner_model_name,
        "gliner_threshold": st.gliner_threshold,
        "enable_eupii": st.enable_eupii,
        "eupii_threshold": st.eupii_threshold,
        "eupii_model_name": st.eupii_model_name,
        "export_format": st.export_format,
        "llm_enabled": getattr(config, "llm_enabled", False),
        "llm_base_url": getattr(config, "llm_base_url", "http://127.0.0.1:11434/v1"),
        "llm_model_name": getattr(config, "llm_model_name", "qwen3:8b"),
        "llm_provider_type": getattr(config, "llm_provider_type", "ollama"),
        "llm_auto_review": getattr(config, "llm_auto_review", True),
    }
    try:
        controller.save_project(expected_revision=project.revision)
        controller.save_manifest(manifest_updates, expected_revision=int(controller.manifest["revision"]))
        st.system_profile = controller.system_profile
        st.project_profile = controller.project_profile
        st.refresh_effective_config()
        return True
    except RevisionConflictError:
        overlay = st.document_overlay
        _reload_profile_controller_preserving_overlay(
            controller,
            overlay,
            active_project_id=project.project_id,
        )
        st.system_profile = controller.system_profile
        st.project_profile = controller.project_profile
        st.document_overlay = controller.document_overlay
        st.refresh_effective_config()
        try:
            ui.notify("Speichern fehlgeschlagen: Das Profil wurde extern geändert. Die aktuelle Version wurde neu geladen.", type="warning", close_button=True)
        except Exception:
            pass
        return False
    except Exception as ex:
        logging.error("Sessiongebundenes Profilspeichern fehlgeschlagen: %s", ex)
        return False


async def ensure_models_downloaded_with_dialog(
    state: AppState,
    effective_config: Optional[EffectiveConfig] = None,
) -> bool:
    """
    Ensure all AI models required by the current configuration are available in the local cache.
    If a model needs to be downloaded for the first time, prompts the user with an explicit
    confirmation dialog detailing download size, cache location, and offline privacy guarantees.
    Returns True if models are ready to use, False if cancelled or download failed.
    """
    snapshot = effective_config or state.effective_config
    if snapshot is None:
        raise RuntimeError("No EffectiveConfig snapshot available for model selection")
    needs_gliner = any(m == ENTITY_MODE_ALL for m in snapshot.entity_modes.values())
    needs_eupii = state.enable_eupii and any(
        snapshot.entity_modes.get(e) in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
        for e in ["PERSON", "LOCATION", "ID_NUMBER", "HEALTH_DATA"]
    )

    models_to_download = []
    if needs_gliner and not is_model_cached(state.gliner_model_name, "gliner"):
        models_to_download.append({
            "name": "GLiNER Zero-Shot Modell",
            "repo": state.gliner_model_name,
            "type": "gliner",
            "size": f"ca. {GLINER_MODEL_SIZE_MB / 1000:.2f} GB",
            "desc": "Universelles Zero-Shot NER-Modell für flexible Erkennung von Personen, Organisationen, Rollen und IT-Systemen.",
        })
    if needs_eupii and not is_model_cached(state.eupii_model_name, "transformers"):
        models_to_download.append({
            "name": "EU-PII Multilingual Modell",
            "repo": state.eupii_model_name,
            "type": "transformers",
            "size": f"ca. {EUPII_MODEL_SIZE_MB / 1000:.2f} GB",
            "desc": "Spezialisiertes RoBERTa-Token-Klassifikationsmodell für europäische PII (Personen, Adressen, Ausweisnummern, Gesundheitsdaten).",
        })

    if not models_to_download:
        return True

    # Show confirmation dialog
    confirmed = False
    with ui.dialog() as dlg, ui.card().classes("p-5 max-w-lg bg-white rounded-xl shadow-xl"):
        ui.label("Einmaliger Modell-Download erforderlich").classes("text-base font-bold text-slate-800 mb-1")

        md_lines = ["Für die gewählte Konfiguration ist ein einmaliger Download lokaler KI-Modelle erforderlich:\n"]
        total_mb = sum(
            GLINER_MODEL_SIZE_MB if m["type"] == "gliner" else EUPII_MODEL_SIZE_MB
            for m in models_to_download
        )
        for m in models_to_download:
            md_lines.append(f"- **{m['name']}** (`{m['repo']}`): **{m['size']}**\n  _{m['desc']}_")

        md_lines.append(
            "\n- **Speicherort:** Lokaler HuggingFace-Cache (`~/.cache/huggingface`)\n"
            "- **Datenschutz:** Nach dem Download arbeiten alle Modelle zu **100% lokal und offline**.\n\n"
            "Möchtest du den Download jetzt starten?"
        )
        ui.markdown("\n".join(md_lines)).classes("text-xs text-slate-600 leading-relaxed mb-3")

        with ui.row().classes("w-full justify-end gap-2"):
            def on_cancel():
                dlg.close()
            def on_confirm():
                nonlocal confirmed
                confirmed = True
                dlg.close()
            ui.button("Abbrechen", on_click=on_cancel).props("flat text-color=slate")
            ui.button(f"Jetzt herunterladen ({total_mb / 1000:.2f} GB)", icon="cloud_download", on_click=on_confirm, color="primary").props("unelevated")

    await dlg
    if not confirmed:
        return False

    # Perform download with user notification
    for m in models_to_download:
        ui.notify(f"Lade {m['name']} ({m['size']}) herunter... Bitte warten.", type="info", timeout=15000)
        try:
            def load_model():
                with _model_lock:
                    anon = get_synced_cached_anonymizer(state, effective_config=snapshot)
                    if m["type"] == "gliner":
                        anon.gliner_recognizer.load()
                    else:
                        anon.eupii_recognizer.load()
            await asyncio.to_thread(load_model)
            ui.notify(f"{m['name']} erfolgreich heruntergeladen und einsatzbereit.", type="positive")
        except Exception as ex:
            ui.notify(f"Fehler beim Download von {m['name']}: {ex}", type="negative", timeout=12000, close_button=True)
            return False

    return True


def extract_context_snippet(raw_text: str, start: int, end: int, window: int = 150) -> str:
    """Extract contextual snippet around entity with highlighted keyword, prioritizing full sentences."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(raw_text), end + window)

    # Versuche, den Satzanfang zu finden (bis max. 250 Zeichen zurück)
    while ctx_start > 0 and raw_text[ctx_start - 1] not in ".!?\n" and (start - ctx_start) < 250:
        ctx_start -= 1
    # Wenn wir an einem Leerzeichen/Satzzeichen gestoppt haben, dieses überspringen
    if ctx_start > 0 and raw_text[ctx_start] in " \n":
        ctx_start += 1

    # Versuche, das Satzende zu finden (bis max. 250 Zeichen nach vorn)
    while ctx_end < len(raw_text) and raw_text[ctx_end] not in ".!?\n" and (ctx_end - end) < 250:
        ctx_end += 1
    if ctx_end < len(raw_text) and raw_text[ctx_end] in ".!?":
        ctx_end += 1

    before = raw_text[ctx_start:start].replace("\r", " ").replace("\n", " ")
    match = raw_text[start:end].replace("\r", " ").replace("\n", " ")
    after = raw_text[end:ctx_end].replace("\r", " ").replace("\n", " ")

    prefix = "…" if ctx_start > 0 else ""
    suffix = "…" if ctx_end < len(raw_text) else ""

    return f"{prefix}{html.escape(before)}<b class='text-blue-700 bg-blue-100 px-1 rounded'>{html.escape(match)}</b>{html.escape(after)}{suffix}"


def compute_reactive_preview(st: AppState) -> Tuple[str, Dict[str, str], Dict]:
    """
    Recalculate placeholder substitution based on current active groups, roles, format mode, and entity links.
    """
    st.preview_revision += 1
    if not st.raw_text:
        return "", {}, {}

    active_groups = [g for g in st.entity_groups if g.enabled and get_active_occurrences(st, g)]
    if not active_groups:
        for group in st.entity_groups:
            group.placeholder = "(ignoriert / inaktiv)"
        st.current_mapping = {}
        st.current_anon_text = st.raw_text
        st.current_report = {"source_file": st.filename, "entity_count": 0, "mapping": {}, "entities": []}
        return st.raw_text, {}, st.current_report

    # 1. Assign entity numbering to master groups
    category_counters: Dict[str, int] = {}
    group_info: Dict[str, Dict[str, Any]] = {}

    # Pass 1: Masters
    for g in active_groups:
        is_child = bool(g.parent_group_id and g.parent_group_id != g.group_id)
        if not is_child:
            count = category_counters.get(g.entity_type, 0) + 1
            category_counters[g.entity_type] = count
            group_info[g.group_id] = {
                "id": count,
                "role": g.role,
                "surface_tag": g.surface_tag,
            }

    # Pass 2: Linked children
    for g in active_groups:
        is_child = bool(g.parent_group_id and g.parent_group_id != g.group_id)
        if is_child:
            parent_id = g.parent_group_id
            if parent_id in group_info:
                p_id = group_info[parent_id]["id"]
                p_role = group_info[parent_id]["role"] or g.role
            else:
                count = category_counters.get(g.entity_type, 0) + 1
                category_counters[g.entity_type] = count
                p_id = count
                p_role = g.role

            group_info[g.group_id] = {
                "id": p_id,
                "role": p_role,
                "surface_tag": g.surface_tag or "VORNAME",
            }

    # 2. Check collisions for Mode 3 (role_only)
    role_type_groups: Dict[Tuple[str, str], Set[int]] = {}
    for g in active_groups:
        info = group_info.get(g.group_id, {})
        role_str = clean_tag(info.get("role", ""))
        if role_str:
            pair = (g.entity_type, role_str)
            role_type_groups.setdefault(pair, set()).add(info.get("id", 1))

    st.colliding_roles = {pair for pair, ids in role_type_groups.items() if len(ids) > 1}

    # 3. Generate placeholders
    mapping: Dict[str, str] = {}
    final_entities_report = []
    flat_occurrences = []

    for g in st.entity_groups:
        active_occurrences = get_active_occurrences(st, g)
        if not g.enabled or not active_occurrences:
            g.placeholder = "(ignoriert / inaktiv)"
            continue

        info = group_info.get(g.group_id, {"id": 1, "role": "", "surface_tag": ""})
        ent_id = info["id"]
        role_str = clean_tag(info["role"])
        tag_str = clean_tag(info["surface_tag"])
        suffix_tag = f"_{tag_str}" if tag_str else ""
        pair = (g.entity_type, role_str)
        is_colliding = pair in st.colliding_roles

        if st.format_mode == "role_only" and role_str and not is_colliding:
            placeholder = f"[{g.entity_type}_{role_str}{suffix_tag}]"
        elif (st.format_mode == "numbered_role" or (st.format_mode == "role_only" and is_colliding)) and role_str:
            placeholder = f"[{g.entity_type}_{ent_id}_{role_str}{suffix_tag}]"
        else:
            # Modus 1 (Numbered)
            placeholder = f"[{g.entity_type}_{ent_id}{suffix_tag}]"

        g.placeholder = placeholder
        mapping[placeholder] = g.original_text

        for occ in active_occurrences:
            flat_occurrences.append({
                "start": occ.start,
                "end": occ.end,
                "score": occ.score,
                "original": g.original_text,
                "type": g.entity_type,
                "placeholder": placeholder,
                "needs_review": occ.needs_review,
                "role": info["role"] or None,
                "surface_tag": info["surface_tag"] or None,
            })
            final_entities_report.append({
                "type": g.entity_type,
                "original": g.original_text,
                "placeholder": placeholder,
                "score": round(occ.score, 3),
                "needs_review": occ.needs_review,
                "role": info["role"] or None,
                "surface_tag": info["surface_tag"] or None,
            })

    # 4. Substitute in reverse character order
    sorted_for_sub = sorted(flat_occurrences, key=lambda x: x["start"], reverse=True)
    chars = list(st.raw_text)
    for item in sorted_for_sub:
        start, end = item["start"], item["end"]
        chars[start:end] = list(item["placeholder"])

    anonymized_text = "".join(chars)

    audit_report = {
        "source_file": st.filename,
        "format_mode": st.format_mode,
        "entity_count": len(final_entities_report),
        "unique_entities_count": len(mapping),
        "mapping": mapping,
        "entities": final_entities_report,
    }

    st.current_mapping = mapping
    st.current_report = audit_report
    st.current_anon_text = anonymized_text

    return anonymized_text, mapping, audit_report


def native_save_file(default_filename: str, content: Union[str, bytes], title: str = "Datei speichern") -> Optional[str]:
    """Open native OS save dialog on Windows/Desktop to save a file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        suffix = Path(default_filename).suffix
        filepath = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_filename,
            defaultextension=suffix,
            filetypes=[("Dateien", f"*{suffix}"), ("Alle Dateien", "*.*")],
        )
        root.destroy()
        if filepath:
            out_p = Path(filepath)
            if isinstance(content, bytes):
                out_p.write_bytes(content)
            else:
                out_p.write_text(content, encoding="utf-8")
            return filepath
    except Exception as e:
        logging.error(f"Native save dialog error: {e}")
    return None


def native_export_folder(stem: str, anon_text: str, mapping: dict, report: dict, ext: str = "txt") -> Optional[str]:
    """Export all three files into a folder chosen by the user."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Zielordner für Export auswählen")
        root.destroy()
        if folder:
            out_dir = Path(folder)
            (out_dir / f"{stem}_anonymized.{ext}").write_text(anon_text, encoding="utf-8")
            (out_dir / f"{stem}_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(out_dir)
    except Exception as e:
        logging.error(f"Native folder export error: {e}")
    return None


# --- UI Construction ---
@ui.page("/")
def create_ui(client: Optional[Client] = None):
    state = AppState()
    if client is not None:
        client.on_disconnect(lambda: cleanup_session_async(state))
    ui.colors(primary="#1976D2", secondary="#26A69A", accent="#9C27B0", positive="#2E7D32", warning="#F57C00", negative="#C62828")

    # Header
    with ui.header().classes("bg-slate-800 text-white p-4 shadow-md flex items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("lock", size="md").classes("text-teal-400")
            ui.label("Privacy-First Local Anonymizer").classes("text-xl font-bold")
            ui.badge("100% Lokal & Offline", color="teal").props("outline")
        
        # Real-time background AI model warmup status
        with ui.row().classes("items-center gap-2"):
            model_spinner = ui.spinner(size="xs", color="teal")
            model_status_label = ui.label("KI-Modell lädt im Hintergrund...").classes("text-xs text-teal-200")

            def check_warmup_status():
                if _model_ready:
                    model_spinner.set_visibility(False)
                    model_status_label.set_text("✅ KI-Modell bereit (100% Lokal)")
                    model_status_label.classes("text-teal-300 font-bold")
                    warmup_timer.cancel()

            warmup_timer = ui.timer(0.3, check_warmup_status)

    # UI container references
    preview_holder = None
    table_holder = None
    export_holder = None
    raw_text_area = None
    progress_holder = None
    file_badge_card = None
    file_badge_label = None
    analyze_btn = None
    reset_btn = None
    reanalysis_warning_card = None
    extraction_progress_card = None
    extraction_progress_label = None
    extraction_progress_bar = None
    ignore_container = None
    glossary_container = None
    llm_setup_holder = None
    llm_panel_holder = None
    manual_input = None
    manual_type = None
    save_perm_check = None
    add_manual_btn = None
    restore_anon_input = None
    map_json_input = None
    restored_preview = None
    sidebar_entity_mode_selects: Dict[str, Any] = {}

    def ask_discard_document_overlay(action: Callable[[], Any]) -> None:
        """Require an explicit discard before replacing the current document scope."""
        if not state.document_overlay.dirty:
            result = action()
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return
        with ui.dialog() as discard_dialog, ui.card().classes("p-4 max-w-lg"):
            ui.label("Ungespeicherte Dokumentänderungen").classes("text-lg font-bold")
            ui.label("Die aktuelle Dokumentebene enthält flüchtige Änderungen. Beim Fortfahren gehen diese Änderungen verloren.").classes("text-sm text-slate-700")
            with ui.row().classes("justify-end w-full mt-3 gap-2"):
                ui.button("Abbrechen", on_click=discard_dialog.close).props("flat")
                async def discard_and_continue() -> None:
                    discard_dialog.close()
                    state.document_overlay = DocumentProfileOverlay(
                        overlay_revision=state.document_overlay.overlay_revision + 1
                    )
                    state.refresh_effective_config()
                    result = action()
                    if asyncio.iscoroutine(result):
                        await result
                ui.button("Änderungen verwerfen", on_click=discard_and_continue, color="negative").props("unelevated")
        discard_dialog.open()

    async def load_content_into_workspace(text: str, filename: str, raw_bytes: Optional[bytes] = None):
        """Unified asynchronous workspace loader."""
        if not check_mutation_allowed():
            return
        if state.document_overlay.dirty:
            ask_discard_document_overlay(
                lambda: load_content_into_workspace(text, filename, raw_bytes=raw_bytes)
            )
            return
        await load_document_into_state_async(state, text, filename, raw_bytes=raw_bytes)
        if raw_text_area is not None:
            raw_text_area.value = text
        if analyze_btn is not None:
            analyze_btn.set_enabled(bool(text and text.strip()))
        if reset_btn is not None:
            reset_btn.set_visibility(bool(text and text.strip()))
        if file_badge_card is not None and file_badge_label is not None:
            if filename:
                file_badge_label.set_text(f"{filename} ({len(text)} Zeichen)")
                file_badge_card.set_visibility(True)
            else:
                file_badge_card.set_visibility(False)
        if llm_panel_holder is not None:
            build_llm_panel()
        ui.notify(f"Datei '{filename}' geladen ({len(text)} Zeichen).", type="positive")

    async def extract_and_load_file_bytes(raw_bytes: bytes, filename: str):
        """Asynchronously extract structured text from document bytes with live UI progress."""
        if not check_mutation_allowed():
            return
        if extraction_progress_card is not None and extraction_progress_bar is not None and extraction_progress_label is not None:
            extraction_progress_card.set_visibility(True)
            extraction_progress_bar.set_value(0.0)
            extraction_progress_label.set_text(f"Lese '{filename}' ein...")
            await asyncio.sleep(0.02)

        loop = asyncio.get_running_loop()

        def progress_cb(curr: int, total: int, msg: str):
            if extraction_progress_bar is not None and extraction_progress_label is not None:
                val = curr / max(1, total)
                loop.call_soon_threadsafe(extraction_progress_bar.set_value, val)
                loop.call_soon_threadsafe(extraction_progress_label.set_text, f"{msg} ({int(val * 100)}%)")

        try:
            text = await asyncio.to_thread(
                read_document_from_bytes,
                raw_bytes,
                filename,
                progress_cb,
                state.include_headers_footers,
                state.extract_picture_text,
            )
            await load_content_into_workspace(text, filename, raw_bytes=raw_bytes)
        except Exception as ex:
            err_msg = f"{type(ex).__name__}: {str(ex)}"
            logging.error(f"File extraction error: {err_msg}", exc_info=True)
            ui.notify(f"Fehler beim Einlesen von '{filename}': {err_msg}", type="negative", timeout=15000)
        finally:
            if extraction_progress_card is not None:
                extraction_progress_card.set_visibility(False)

    async def open_native_file_dialog():
        """Open native OS file picker with full Win32 lock-sharing support."""
        if not check_mutation_allowed():
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            filepath = filedialog.askopenfilename(
                title="Dokument zum Anonymisieren auswählen",
                filetypes=[
                    ("Unterstützte Dokumente (*.docx, *.pdf, *.txt, *.md, *.csv, *.json)", "*.docx;*.pdf;*.txt;*.md;*.csv;*.json"),
                    ("Word-Dokumente (*.docx)", "*.docx"),
                    ("PDF-Dokumente (*.pdf)", "*.pdf"),
                    ("Text & Markdown (*.txt, *.md)", "*.txt;*.md"),
                    ("Tabellen & Daten (*.csv, *.json)", "*.csv;*.json"),
                    ("Alle Dateien (*.*)", "*.*"),
                ]
            )
            root.destroy()
            if filepath:
                p = Path(filepath)
                data = safe_read_bytes(p)
                await extract_and_load_file_bytes(data, p.name)
        except Exception as ex:
            err_msg = f"{type(ex).__name__}: {str(ex)}"
            logging.error(f"Native file open error: {err_msg}", exc_info=True)
            ui.notify(f"Fehler beim Laden: {err_msg}", type="negative", timeout=15000)

    async def reset_workspace():
        """Reset raw text, filename, and analysis table."""
        if not check_mutation_allowed():
            return
        if state.document_overlay.dirty:
            ask_discard_document_overlay(reset_workspace)
            return
        await reset_app_state_async(state)
        if raw_text_area is not None:
            raw_text_area.value = ""
        if preview_holder is not None:
            preview_holder.clear()
        if export_holder is not None:
            export_holder.clear()
        if table_holder is not None:
            table_holder.clear()
        if file_badge_card is not None:
            file_badge_card.set_visibility(False)
        if analyze_btn is not None:
            analyze_btn.set_enabled(False)
        if reset_btn is not None:
            reset_btn.set_visibility(False)
        if reanalysis_warning_card is not None:
            reanalysis_warning_card.set_visibility(False)
        if llm_panel_holder is not None:
            build_llm_panel()
        ui.notify("Workspace zurückgesetzt.", type="info", icon="delete_sweep")


    def render_ignore_list_ui():
        if not ignore_container:
            return
        state.clear_mutating_zone("ignore")
        ignore_container.clear()
        with ignore_container:
            raw_ignores = parse_ignore_terms(state.ignore_terms_text)
            unique_ignores = sorted(list({t.strip() for t in raw_ignores if t.strip()}), key=lambda s: s.lower())

            with ui.row().classes("w-full items-center gap-1 mb-2"):
                new_ig_input = ui.input(placeholder="Neuer Begriff...").props("dense outlined bg-white").classes("flex-grow text-xs")
                state.register_mutating_element(new_ig_input, "ignore")
                def add_ig():
                    if not check_mutation_allowed():
                        return
                    val = new_ig_input.value.strip()
                    if val:
                        curr = parse_ignore_terms(state.ignore_terms_text)
                        if val not in curr:
                            curr.append(val)
                            state.ignore_terms_text = ", ".join(curr)
                            save_current_config(state)
                            render_ignore_list_ui()
                            refresh_preview_and_exports()
                            ui.notify(f"'{val}' zur Ignore-Liste hinzugefügt.", type="info")
                        new_ig_input.value = ""
                add_ig_btn = ui.button(icon="add", on_click=add_ig, color="primary").props("dense flat size=sm")
                state.register_mutating_element(add_ig_btn, "ignore")

            with ui.column().classes("w-full max-h-48 overflow-y-auto gap-1 pr-1"):
                if not unique_ignores:
                    ui.label("Keine Begriffe auf der Ignore-Liste.").classes("text-[11px] text-slate-400 italic")
                else:
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        for term in unique_ignores:
                            def make_remove_ig(t):
                                def on_remove():
                                    if not check_mutation_allowed():
                                        return
                                    curr = [x for x in parse_ignore_terms(state.ignore_terms_text) if x.lower() != t.lower()]
                                    state.ignore_terms_text = ", ".join(curr)
                                    save_current_config(state)
                                    render_ignore_list_ui()
                                    refresh_preview_and_exports()
                                    ui.notify(f"'{t}' aus Ignore-Liste entfernt.", type="info")
                                return on_remove

                            with ui.row().classes("items-center gap-1 bg-slate-100 border border-slate-300 rounded px-2 py-0.5 shadow-none"):
                                ui.label(term).classes("font-mono font-semibold text-xs text-slate-900")
                                del_btn = ui.button(icon="close", on_click=make_remove_ig(term)).props("flat round dense size=xs color=negative").classes("p-0 min-h-0 min-w-0 ml-0.5 hover:bg-red-100")
                                state.register_mutating_element(del_btn, "ignore")

    def render_glossary_list_ui():
        if not glossary_container:
            return
        state.clear_mutating_zone("glossary")
        glossary_container.clear()
        with glossary_container:
            glossary_dict = parse_glossary(state.glossary_text)
            sorted_keys = sorted(glossary_dict.keys(), key=lambda s: s.lower())

            with ui.row().classes("w-full items-center gap-1 mb-2 flex-wrap"):
                new_g_term = ui.input(placeholder="Begriff...").props("dense outlined bg-white").classes("flex-grow text-xs")
                new_g_type = ui.select(options=AVAILABLE_ENTITIES, value="ORGANIZATION").props("dense outlined bg-white").classes("w-28 text-xs")
                state.register_mutating_element(new_g_term, "glossary")
                state.register_mutating_element(new_g_type, "glossary")
                def add_g():
                    if not check_mutation_allowed():
                        return
                    t = new_g_term.value.strip()
                    if t:
                        lines = [l.strip() for l in state.glossary_text.splitlines() if l.strip()]
                        lines = [l for l in lines if not (l.lower().startswith(f"{t.lower()}:") or l.lower().startswith(f"{t.lower()}="))]
                        lines.append(f"{t}: {new_g_type.value}")
                        state.glossary_text = "\n".join(lines)
                        save_current_config(state)
                        render_glossary_list_ui()
                        refresh_preview_and_exports()
                        ui.notify(f"'{t}' ({new_g_type.value}) zur Begriffsliste hinzugefügt.", type="positive")
                        new_g_term.value = ""
                add_g_btn = ui.button(icon="add", on_click=add_g, color="positive").props("dense flat size=sm")
                state.register_mutating_element(add_g_btn, "glossary")

            with ui.column().classes("w-full max-h-48 overflow-y-auto gap-1 pr-1"):
                if not sorted_keys:
                    ui.label("Keine eigenen Begriffe in der Begriffsliste.").classes("text-[11px] text-slate-400 italic")
                else:
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        for term in sorted_keys:
                            ent_type = glossary_dict[term]
                            def make_remove_g(t):
                                def on_remove():
                                    if not check_mutation_allowed():
                                        return
                                    lines = [l for l in state.glossary_text.splitlines() if not (l.strip().lower().startswith(f"{t.lower()}:") or l.strip().lower().startswith(f"{t.lower()}="))]
                                    state.glossary_text = "\n".join(lines)
                                    save_current_config(state)
                                    render_glossary_list_ui()
                                    refresh_preview_and_exports()
                                    ui.notify(f"'{t}' aus Begriffsliste entfernt.", type="info")
                                return on_remove

                            with ui.row().classes("items-center gap-1.5 bg-blue-50 border border-blue-300 rounded px-2 py-0.5 shadow-none"):
                                ui.label(term).classes("font-mono font-bold text-xs text-slate-900")
                                ui.label(f"({ent_type})").classes("text-[10px] font-semibold text-blue-800 bg-blue-100 px-1 rounded")
                                del_btn = ui.button(icon="close", on_click=make_remove_g(term)).props("flat round dense size=xs color=negative").classes("p-0 min-h-0 min-w-0 ml-0.5 hover:bg-red-100")
                                state.register_mutating_element(del_btn, "glossary")

    def build_llm_setup_panel():
        if llm_setup_holder is None:
            return
        state.clear_mutating_zone("llm_setup")
        llm_setup_holder.clear()
        with llm_setup_holder:
            if not LLM_AVAILABLE:
                with ui.card().classes("w-full p-3 bg-slate-50 border border-slate-200 rounded-lg"):
                    with ui.row().classes("items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2 text-slate-600 text-xs"):
                            ui.icon("info", size="sm").classes("text-slate-400")
                            ui.label("Lokale LLM-Review-Assistenz: Optionales Zusatzpaket [llm] nicht installiert. Installieren mit: pip install local-anonymizer[llm]").classes("font-medium")
                return

            with ui.card().classes("w-full p-3 bg-slate-50 border border-slate-200 rounded-xl mb-1 shadow-none"):
                # Header Row
                with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.icon("psychology", size="sm").classes("text-indigo-700")
                        ui.label("Lokale LLM-Review-Assistenz (Optional)").classes("font-bold text-xs text-slate-800")
                        if state.config.llm_enabled:
                            if state.llm_setup_state == "discovering":
                                ui.spinner(size="xs", color="primary")
                                ui.badge("Suche Modelle...", color="primary").props("dense")
                            elif state.llm_setup_state in ("preloading", "testing"):
                                ui.spinner(size="xs", color="primary")
                                ui.badge("Lade / Teste...", color="primary").props("dense")
                            elif state.is_model_ready():
                                ps_inf = state.llm_ready_info
                                tooltip_text = f"Modell '{state.config.llm_model_name}' ist in Ollama geladen."
                                if ps_inf and getattr(ps_inf, "size_vram", None):
                                    vram_mb = ps_inf.size_vram // (1024 * 1024)
                                    tooltip_text += f" (VRAM: ca. {vram_mb} MB)"
                                if ps_inf and getattr(ps_inf, "expires_at", None):
                                    tooltip_text += f" (Ablauf: {ps_inf.expires_at})"
                                ui.badge(f"✓ Bereit ({state.config.llm_model_name})", color="positive").props("dense outline").tooltip(tooltip_text)
                            elif state.llm_provider_type == "generic" and state.llm_ready_bound_url == state.config.llm_base_url.strip():
                                ui.badge("✓ Verbindung OK", color="positive").props("dense outline").tooltip("Generischer Server erreichbar")
                            elif state.llm_setup_status_msg:
                                ui.badge(state.llm_setup_status_msg, color="grey-7").props("dense outline")

                    with ui.row().classes("items-center gap-2"):
                        if state.config.llm_enabled:
                            def toggle_details():
                                state.llm_show_details = not state.llm_show_details
                                build_llm_setup_panel()

                            det_btn = ui.button(
                                "Details ausblenden" if state.llm_show_details else "Einstellungen anpassen",
                                icon="expand_less" if state.llm_show_details else "tune",
                                on_click=toggle_details,
                            ).props("flat dense size=xs color=slate").classes("text-xs")
                            state.register_mutating_element(det_btn, "llm_setup")

                        async def on_llm_toggle(e):
                            if not check_mutation_allowed():
                                return
                            state.config.llm_enabled = bool(e.value)
                            save_current_config(state)
                            if not state.config.llm_enabled:
                                await state.cancel_setup_task()
                                await state.close_llm_provider()
                                state.invalidate_llm_ready()
                            else:
                                state.llm_show_details = True
                                if state.llm_provider_type == "ollama" and not state.llm_discovered_models:
                                    await trigger_model_discovery()
                            build_llm_setup_panel()
                            build_llm_panel()
                            build_review_table()

                        llm_switch = ui.switch("Aktivieren", value=state.config.llm_enabled, on_change=on_llm_toggle).props("dense size=sm").classes("text-xs font-semibold")
                        state.register_mutating_element(llm_switch, "llm_setup")

                # Expanded Body when enabled and details shown
                if state.config.llm_enabled and state.llm_show_details:
                    ui.separator().classes("my-2")

                    # Row 1: Main Controls on single line (Modellauswahl first, then Provider, then API-Endpunkt)
                    with ui.row().classes("w-full items-center gap-3 flex-wrap mb-2"):
                        if state.llm_provider_type == "ollama":
                            model_options: Dict[str, str] = {}
                            if state.llm_discovered_models:
                                for m in state.llm_discovered_models:
                                    lbl, col, tt = get_model_suitability_badge(m)
                                    model_options[m] = f"{m} ({lbl})" if lbl != "Nicht evaluiert" else m

                            curr_m = state.config.llm_model_name
                            if curr_m and curr_m not in model_options and not state.llm_custom_model_mode:
                                lbl, col, tt = get_model_suitability_badge(curr_m)
                                model_options[curr_m] = f"{curr_m} ({lbl})" if lbl != "Nicht evaluiert" else curr_m

                            model_options["__custom__"] = "Anderes Modell …"

                            async def on_model_select_change(e):
                                if not check_mutation_allowed():
                                    return
                                val = e.value
                                if val == "__custom__":
                                    state.llm_custom_model_mode = True
                                    build_llm_setup_panel()
                                    return
                                state.llm_custom_model_mode = False
                                try:
                                    valid_name = validate_model_name(val)
                                    state.config.llm_model_name = valid_name
                                    state.invalidate_llm_ready()
                                    await state.close_llm_provider()
                                    save_current_config(state)
                                    build_llm_setup_panel()
                                    build_llm_panel()
                                except ValueError as ve:
                                    ui.notify(str(ve), type="negative")

                            selected_val = "__custom__" if state.llm_custom_model_mode else (state.config.llm_model_name or "qwen3:8b")

                            model_dropdown = ui.select(
                                options=model_options,
                                value=selected_val if selected_val in model_options else "__custom__",
                                on_change=on_model_select_change,
                                label="Modellauswahl",
                            ).props("dense outlined bg-white").classes("w-64 text-xs")
                            state.register_mutating_element(model_dropdown, "llm_setup")

                            if state.llm_custom_model_mode or not state.llm_discovered_models:
                                async def on_custom_name_change(e):
                                    if not check_mutation_allowed():
                                        return
                                    raw_val = (e.value or "").strip()
                                    try:
                                        if raw_val:
                                            valid_name = validate_model_name(raw_val)
                                            state.config.llm_model_name = valid_name
                                            state.invalidate_llm_ready()
                                            await state.close_llm_provider()
                                            save_current_config(state)
                                            build_llm_panel()
                                    except ValueError as ve:
                                        ui.notify(str(ve), type="negative")

                                custom_input = ui.input(
                                    label="Freier Modellname",
                                    value=state.config.llm_model_name,
                                    placeholder="z. B. qwen3:8b",
                                    on_change=on_custom_name_change,
                                ).props("dense outlined bg-white").classes("w-44 text-xs")
                                state.register_mutating_element(custom_input, "llm_setup")

                        else:
                            async def on_generic_model_change(e):
                                if not check_mutation_allowed():
                                    return
                                raw_val = (e.value or "").strip()
                                try:
                                    if raw_val:
                                        valid_name = validate_model_name(raw_val)
                                        state.config.llm_model_name = valid_name
                                        state.invalidate_llm_ready()
                                        await state.close_llm_provider()
                                        save_current_config(state)
                                        build_llm_panel()
                                except ValueError as ve:
                                    ui.notify(str(ve), type="negative")

                            gen_model_input = ui.input(
                                label="Modellname",
                                value=state.config.llm_model_name,
                                placeholder="z. B. local-model",
                                on_change=on_generic_model_change,
                            ).props("dense outlined bg-white").classes("w-52 text-xs")
                            state.register_mutating_element(gen_model_input, "llm_setup")

                        # Provider Selector
                        async def on_provider_type_change(e):
                            if not check_mutation_allowed():
                                return
                            new_type = e.value
                            state.llm_provider_type = new_type
                            state.config.llm_provider_type = new_type
                            state.invalidate_llm_ready()
                            await state.close_llm_provider()
                            save_current_config(state)
                            if new_type == "ollama" and not state.llm_discovered_models:
                                await trigger_model_discovery()
                            build_llm_setup_panel()
                            build_llm_panel()

                        prov_select = ui.select(
                            options={"ollama": "Ollama (Lokal)", "generic": "OpenAI-kompatibel"},
                            value=state.llm_provider_type,
                            on_change=on_provider_type_change,
                            label="Provider",
                        ).props("dense outlined bg-white").classes("w-48 text-xs")
                        state.register_mutating_element(prov_select, "llm_setup")

                        # Compact Endpoint input
                        async def on_base_url_change(e):
                            if not check_mutation_allowed():
                                return
                            new_url = (e.value or "").strip()
                            if new_url != state.config.llm_base_url:
                                state.config.llm_base_url = new_url
                                state.invalidate_llm_ready()
                                await state.close_llm_provider()
                                save_current_config(state)
                                build_llm_panel()

                        base_url_input = ui.input(
                            label="API-Endpunkt (Loopback)",
                            value=state.config.llm_base_url,
                            placeholder="http://127.0.0.1:11434/v1",
                            on_change=on_base_url_change,
                        ).props("dense outlined bg-white").classes("w-60 text-xs")
                        state.register_mutating_element(base_url_input, "llm_setup")

                    # Row 2: Action Buttons & Auto-Review Checkbox (All on one line)
                    with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap mb-1"):
                        with ui.row().classes("items-center gap-2 flex-wrap"):
                            if state.llm_provider_type == "ollama":
                                async def on_preload_click():
                                    if not check_mutation_allowed():
                                        return
                                    await trigger_ollama_preload()

                                preload_btn = ui.button("Laden", icon="memory", on_click=on_preload_click, color="primary").props("unelevated dense size=sm").tooltip("Lädt das Modell vorab in den Arbeitsspeicher / VRAM")
                                state.register_mutating_element(preload_btn, "llm_setup")

                                async def on_refresh_click():
                                    if not check_mutation_allowed():
                                        return
                                    await trigger_model_discovery()

                                refresh_btn = ui.button("Liste aktualisieren", icon="refresh", on_click=on_refresh_click, color="slate").props("outline dense size=sm")
                                state.register_mutating_element(refresh_btn, "llm_setup")

                            else:
                                async def on_test_conn_click():
                                    if not check_mutation_allowed():
                                        return
                                    await trigger_generic_test()

                                test_btn = ui.button("Verbindung testen", icon="wifi", on_click=on_test_conn_click, color="primary").props("unelevated dense size=sm")
                                state.register_mutating_element(test_btn, "llm_setup")

                                async def on_refresh_generic():
                                    if not check_mutation_allowed():
                                        return
                                    await trigger_model_discovery()

                                refresh_gen_btn = ui.button("Modelle abfragen", icon="refresh", on_click=on_refresh_generic, color="slate").props("outline dense size=sm")
                                state.register_mutating_element(refresh_gen_btn, "llm_setup")

                            if state.llm_setup_state in ("discovering", "preloading", "testing"):
                                async def on_cancel_setup():
                                    await state.cancel_setup_task()
                                    state.llm_setup_status_msg = "Abgebrochen"
                                    ui.notify("Vorgang abgebrochen.", type="info")
                                    set_mutating_controls_disabled(False)
                                    build_llm_setup_panel()
                                cancel_setup_btn = ui.button("Abbrechen", icon="cancel", on_click=on_cancel_setup, color="negative").props("flat dense size=sm")

                        def on_auto_review_toggle(e):
                            if not check_mutation_allowed():
                                return
                            state.config.llm_auto_review = bool(e.value)
                            save_current_config(state)

                        auto_review_checkbox = ui.checkbox(
                            "LLM-Review direkt an die Textanalyse anschließen",
                            value=state.config.llm_auto_review,
                            on_change=on_auto_review_toggle,
                        ).props("dense size=sm").classes("text-xs text-slate-700 font-medium").tooltip(
                            "Wenn aktiviert, startet nach der lokalen Erkennung automatisch die LLM-Triage der Fundstellen."
                        )
                        state.register_mutating_element(auto_review_checkbox, "llm_setup")

                    # Row 3: Catalog Info & Privacy Notice (Discreet and compact)
                    catalog_err_msg: Optional[str] = None
                    curr_entry: Optional[Any] = None
                    try:
                        curr_entry = find_catalog_entry(state.config.llm_model_name)
                    except CatalogError as ce:
                        catalog_err_msg = str(ce)

                    with ui.row().classes("w-full items-center justify-between gap-2 px-2.5 py-1 bg-slate-100/70 border border-slate-200 rounded text-xs text-slate-600 flex-wrap"):
                        with ui.row().classes("items-center gap-2"):
                            if catalog_err_msg is not None:
                                ui.badge("Katalog-Fehler", color="negative").props("dense").tooltip(f"Modellkatalog nicht verfügbar: {catalog_err_msg}")
                                ui.label("Katalog-Integritätsfehler").classes("text-[11px] text-red-600 font-medium")
                            elif curr_entry is not None:
                                lbl, col, tt = get_model_suitability_badge(state.config.llm_model_name)
                                ui.badge(f"Katalog: {lbl}", color=col).props("dense").tooltip(tt)
                                ui.label(f"{curr_entry.phase_6a_triage.reason}").classes("text-[11px] text-slate-600")
                                if curr_entry.hardware_class:
                                    ui.label(f"({curr_entry.hardware_class})").classes("text-[10px] text-slate-400 font-mono")
                            else:
                                ui.badge("Katalog: Nicht evaluiert", color="grey-7").props("dense")
                                ui.label("Frei gewähltes Modell").classes("text-[11px] text-slate-500")

                        with ui.row().classes("items-center gap-1 text-[11px] text-slate-500"):
                            ui.icon("lock", size="xs").classes("text-slate-400")
                            ui.label("Empfehlung: Ollama im Local-Only-Modus betreiben (OLLAMA_NO_CLOUD=1) · ':cloud' gesperrt").classes("text-[10px]")

    async def trigger_model_discovery():
        if not check_mutation_allowed():
            return
        await state.cancel_setup_task()

        req_id = uuid.uuid4().hex
        url_snap = state.config.llm_base_url.strip()
        state.llm_setup_request_id = req_id
        state.llm_setup_endpoint_snapshot = url_snap
        state.llm_setup_state = "discovering"
        set_mutating_controls_disabled(True)
        build_llm_setup_panel()

        async def _run_discovery():
            try:
                if state.llm_provider_type == "ollama":
                    res = await fetch_ollama_models(url_snap)
                else:
                    res = await fetch_generic_models(url_snap)

                if state.llm_setup_request_id != req_id or state.config.llm_base_url.strip() != url_snap:
                    return

                if res.status == "success":
                    state.llm_discovered_models = res.models
                    state.llm_discovery_status = "success"
                    state.llm_setup_status_msg = f"{len(res.models)} Modelle gefunden"
                    ui.notify(f"{len(res.models)} lokale Modelle gefunden.", type="positive")
                elif res.status == "empty":
                    state.llm_discovered_models = []
                    state.llm_discovery_status = "empty"
                    state.llm_custom_model_mode = True
                    state.llm_setup_status_msg = "Keine Modelle gefunden"
                    ui.notify("Keine installierten Modelle gefunden.", type="warning")
                else:
                    state.llm_discovery_status = res.status
                    state.llm_custom_model_mode = True
                    state.llm_setup_status_msg = res.message or "Discovery fehlgeschlagen"
                    ui.notify(f"Discovery-Hinweis: {res.message}", type="warning")
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                state.llm_setup_status_msg = "Fehler bei Discovery"
                ui.notify(f"Discovery-Fehler: {ex}", type="negative")
            finally:
                if state.llm_setup_request_id == req_id:
                    state.llm_setup_state = "idle"
                    state.llm_setup_task = None
                    set_mutating_controls_disabled(False)
                    build_llm_setup_panel()

        async def _runner_discovery():
            if client is not None:
                with client:
                    await _run_discovery()
            else:
                await _run_discovery()

        state.llm_setup_task = asyncio.create_task(_runner_discovery())

    async def trigger_ollama_preload():
        if not check_mutation_allowed():
            return
        await state.cancel_setup_task()

        req_id = uuid.uuid4().hex
        url_snap = state.config.llm_base_url.strip()
        model_snap = state.config.llm_model_name.strip()
        state.llm_setup_request_id = req_id
        state.llm_setup_endpoint_snapshot = url_snap
        state.llm_setup_model_snapshot = model_snap
        state.llm_setup_state = "preloading"
        set_mutating_controls_disabled(True)
        build_llm_setup_panel()

        async def _run_preload():
            try:
                ps_info = await preload_ollama_model(url_snap, model_snap)
                if (
                    state.llm_setup_request_id != req_id
                    or state.config.llm_base_url.strip() != url_snap
                    or state.config.llm_model_name.strip() != model_snap
                ):
                    return

                expiry_ts = parse_iso_expiry(getattr(ps_info, "expires_at", None))
                if expiry_ts <= 0.0:
                    state.invalidate_llm_ready()
                    state.llm_setup_status_msg = "Bereitschaft nicht verifizierbar"
                    ui.notify(f"Modell '{model_snap}' geladen, aber Ablaufzeit nicht verifizierbar.", type="warning")
                else:
                    state.llm_ready_info = ps_info
                    state.llm_ready_timestamp = time.time()
                    state.llm_ready_expires_at = expiry_ts
                    state.llm_ready_bound_url = url_snap
                    state.llm_ready_bound_model = model_snap
                    state.llm_setup_status_msg = "Modell bereit"
                    ui.notify(f"Modell '{model_snap}' erfolgreich in Ollama geladen.", type="positive")
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                state.invalidate_llm_ready()
                state.llm_setup_status_msg = "Laden fehlgeschlagen"
                ui.notify(f"Fehler beim Vorladen: {ex}", type="negative")
            finally:
                if state.llm_setup_request_id == req_id:
                    state.llm_setup_state = "idle"
                    state.llm_setup_task = None
                    set_mutating_controls_disabled(False)
                    build_llm_setup_panel()
                    build_llm_panel()

        async def _runner_preload():
            if client is not None:
                with client:
                    await _run_preload()
            else:
                await _run_preload()

        state.llm_setup_task = asyncio.create_task(_runner_preload())

    async def trigger_generic_test():
        if not check_mutation_allowed():
            return
        await state.cancel_setup_task()

        req_id = uuid.uuid4().hex
        url_snap = state.config.llm_base_url.strip()
        state.llm_setup_request_id = req_id
        state.llm_setup_endpoint_snapshot = url_snap
        state.llm_setup_state = "testing"
        set_mutating_controls_disabled(True)
        build_llm_setup_panel()

        async def _run_test():
            try:
                ok = await test_generic_connection(url_snap)
                if state.llm_setup_request_id != req_id or state.config.llm_base_url.strip() != url_snap:
                    return
                if ok:
                    state.llm_ready_bound_url = url_snap
                    state.llm_setup_status_msg = "Verbindung OK"
                    ui.notify("Verbindung zum Server erfolgreich.", type="positive")
                else:
                    state.invalidate_llm_ready()
                    state.llm_setup_status_msg = "Verbindung fehlgeschlagen"
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                state.invalidate_llm_ready()
                state.llm_setup_status_msg = "Verbindung fehlgeschlagen"
                ui.notify(f"Fehler bei Verbindungstest: {ex}", type="negative")
            finally:
                if state.llm_setup_request_id == req_id:
                    state.llm_setup_state = "idle"
                    state.llm_setup_task = None
                    set_mutating_controls_disabled(False)
                    build_llm_setup_panel()
                    build_llm_panel()

        async def _runner_test():
            if client is not None:
                with client:
                    await _run_test()
            else:
                await _run_test()

        state.llm_setup_task = asyncio.create_task(_runner_test())

    def refresh_preview_and_exports():
        if not preview_holder or not export_holder:
            return
        state.refresh_effective_config()
        if state.preview_stale:
            preview_holder.clear()
            export_holder.clear()
            with preview_holder:
                with ui.row().classes("w-full items-center gap-2 p-3 bg-amber-50 border border-amber-300 rounded text-amber-900 text-sm"):
                    ui.icon("warning", size="sm").classes("text-amber-600")
                    ui.label("⚠️ Konfiguration geändert. Bisherige Ergebnisse sind veraltet. Bitte analysieren Sie den Text mit dem neuen Profil erneut.").classes("font-medium")
            return
        anon_text, mapping, audit_report = compute_reactive_preview(state)
        stem = Path(state.filename).stem or "dokument"
        ext = state.export_format

        # Sync the current mapping to Tab 2 immediately so it's pre-filled when the user switches tabs
        if mapping and map_json_input is not None:
            state.restore_mapping = dict(mapping)
            map_json_input.value = json.dumps(mapping, indent=2, ensure_ascii=False)
            logging.debug(f"[Tab2 sync] mapping synced: {len(mapping)} entries")
        else:
            logging.debug(f"[Tab2 sync] SKIPPED: mapping={bool(mapping)}, map_json_input is None={map_json_input is None}")


        preview_holder.clear()
        with preview_holder:
            if state.format_mode == "role_only" and state.colliding_roles:
                with ui.row().classes("w-full items-center gap-2 p-3 mb-2 bg-amber-50 border border-amber-300 rounded text-amber-900 text-xs"):
                    ui.icon("warning", size="sm").classes("text-amber-600")
                    coll_str = ", ".join(f"'{pair[1]}' ({pair[0]})" for pair in state.colliding_roles)
                    ui.label(f"Rollenkollision erkannt: Rolle {coll_str} ist mehrfach vergeben. Automatischer Fallback auf Modus 2 (nummeriert) für diese Entitäten.").classes("font-medium")

            ui.label(f"Anonymisierte Vorschau ({ext.upper()} / Markdown):").classes("font-semibold text-slate-700 mb-1")
            ui.textarea(value=anon_text).props("readonly rows=12").classes("w-full font-mono text-sm bg-slate-50 border rounded p-2")

        export_holder.clear()
        with export_holder:
            with ui.row().classes("gap-3 mt-2 flex-wrap items-center"):
                # 1. Copy to clipboard
                async def copy_clipboard():
                    await ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(anon_text)});')
                    ui.notify("Anonymisierter Text in Zwischenablage kopiert!", type="positive", icon="content_copy")

                ui.button(
                    "📋 In Zwischenablage kopieren",
                    icon="content_copy",
                    color="secondary",
                    on_click=copy_clipboard,
                ).props("unelevated")

                # 2. Save Anonymized Text
                def save_anon():
                    path = native_save_file(f"{stem}_anonymized.{ext}", anon_text, f"Anonymisierten Text speichern (.{ext})")
                    if path:
                        ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                    else:
                        ui.download(anon_text.encode("utf-8"), filename=f"{stem}_anonymized.{ext}")

                ui.button(
                    f"💾 Text speichern (.{ext})",
                    icon="save",
                    color="primary",
                    on_click=save_anon,
                ).props("unelevated")

                # 3. Save Mapping
                def save_map():
                    map_str = json.dumps(mapping, indent=2, ensure_ascii=False)
                    path = native_save_file(f"{stem}_mapping.json", map_str, "Mapping-Tabelle speichern")
                    if path:
                        ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                    else:
                        ui.download(map_str.encode("utf-8"), filename=f"{stem}_mapping.json")

                ui.button(
                    "💾 Mapping speichern (.json)",
                    icon="key",
                    color="primary",
                    on_click=save_map,
                ).props("unelevated")

                # 4. Save Report
                def save_rep():
                    rep_str = json.dumps(audit_report, indent=2, ensure_ascii=False)
                    path = native_save_file(f"{stem}_report.json", rep_str, "Audit-Bericht speichern")
                    if path:
                        ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                    else:
                        ui.download(rep_str.encode("utf-8"), filename=f"{stem}_report.json")

                ui.button(
                    "💾 Bericht speichern (.json)",
                    icon="assessment",
                    color="slate",
                    on_click=save_rep,
                ).props("outline")

                # 5. Export all to folder
                def export_all():
                    out_path = native_export_folder(stem, anon_text, mapping, audit_report, ext=ext)
                    if out_path:
                        with ui.dialog() as dlg, ui.card().classes("p-4"):
                            ui.label("✅ Export erfolgreich!").classes("text-lg font-bold text-slate-800")
                            ui.label(f"3 Dateien wurden exportiert nach:\n{out_path}").classes("text-sm text-slate-600 font-mono my-2")
                            with ui.row().classes("gap-2 justify-end w-full mt-2"):
                                if hasattr(os, "startfile"):
                                    ui.button("Ordner im Explorer öffnen", icon="folder_open", on_click=lambda: os.startfile(out_path), color="primary").props("unelevated")
                                ui.button("Schliessen", on_click=dlg.close).props("flat")
                        dlg.open()

                ui.button(
                    "📁 Alle 3 Dateien in Ordner exportieren",
                    icon="folder_zip",
                    color="teal",
                    on_click=export_all,
                ).props("unelevated")

    PIPELINE_STEPS = [
        "1. Vorverarbeitung & Text-Filterung",
        "2. Lokale KI-Erkennung & Presidio NER",
        "3. Span-Deduplizierung & Genitiv-Erweiterung",
        "4. Smart-Linking (Rollen- & Namensbezüge)",
        "5. Vorschau-Berechnung & Mapping-Export",
    ]

    async def run_analysis():
        if state.is_busy:
            ui.notify("Aktion nicht möglich: Ein Analyse- oder Setup-Vorgang läuft bereits.", type="warning")
            return
        if not check_mutation_allowed():
            return
        if not state.raw_text or not state.raw_text.strip():
            ui.notify("Bitte laden Sie zuerst ein Dokument hoch oder fügen Sie Text ein.", type="warning")
            return

        # Freeze the complete scope resolution before any model work starts.
        # The same immutable object drives model selection, recognizer sync and
        # the analyzed hash; later UI edits can only make the preview stale.
        analysis_snapshot = state.refresh_effective_config()
        if analysis_snapshot is None:
            ui.notify("Analyse abgebrochen: Keine gültige Konfiguration verfügbar.", type="negative")
            return
        analysis_snapshot_hash = analysis_snapshot.snapshot_hash
        state.is_analyzing = True
        set_mutating_controls_disabled(True)
        try:
            # Ensure required AI models are confirmed and downloaded before starting analysis
            ready = await ensure_models_downloaded_with_dialog(state, effective_config=analysis_snapshot)
            if not ready:
                ui.notify("Analyse abgebrochen: Modell-Download wurde nicht bestätigt.", type="warning")
                return

            if reanalysis_warning_card is not None:
                reanalysis_warning_card.set_visibility(False)

            # Show visual indicators immediately (< 20ms)
            if analyze_btn:
                analyze_btn.props("loading")

            progress_bar = None
            progress_label = None
            step_labels = []
            if progress_holder:
                progress_holder.clear()
                with progress_holder:
                    progress_bar = ui.linear_progress(value=0.0, show_value=False).props("color=primary stripe rounded instant-feedback").classes("w-full mb-1")
                    progress_label = ui.label("0% - Lokale Analyse gestartet...").classes("text-xs text-slate-700 font-bold mb-1")
                    with ui.card().classes("w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg shadow-none flex flex-col gap-1 mb-2"):
                        for idx, name in enumerate(PIPELINE_STEPS):
                            if idx == 0:
                                lbl = ui.label(f"⏳ {name}").classes("text-[11px] text-blue-700 font-bold")
                            else:
                                lbl = ui.label(f"○ {name}").classes("text-[11px] text-rose-600 font-normal")
                            step_labels.append(lbl)

            if table_holder:
                table_holder.clear()
                with table_holder:
                    with ui.row().classes("items-center gap-3 p-4 bg-blue-50 rounded border border-blue-200"):
                        ui.spinner(size="md", color="primary")
                        ui.label("Dokument wird lokal analysiert (NER, Markdown, Struktur)...").classes("text-slate-700 text-sm font-medium")

            # Yield to event loop so DOM updates render immediately in the browser
            await asyncio.sleep(0.02)

            def update_step_ui(active_idx: int, val: float, msg: str):
                if progress_bar:
                    progress_bar.set_value(val)
                if progress_label:
                    progress_label.set_text(f"{int(val * 100)}% - {msg}")
                for i, lbl in enumerate(step_labels):
                    name = PIPELINE_STEPS[i]
                    if i < active_idx:
                        lbl.set_text(f"✓ {name}")
                        lbl.classes(replace="text-[11px] text-emerald-700 font-semibold")
                    elif i == active_idx:
                        lbl.set_text(f"⏳ {name} ({msg})")
                        lbl.classes(replace="text-[11px] text-blue-700 font-bold")
                    else:
                        lbl.set_text(f"○ {name}")
                        lbl.classes(replace="text-[11px] text-rose-600 font-normal")

            # Wait for background warmup if still running
            while not _model_ready:
                if progress_label:
                    progress_label.set_text("Warte auf KI-Modell-Initialisierung...")
                await asyncio.sleep(0.1)

            # Step 1: Preprocessing & Ignore filter
            update_step_ui(0, 0.15, "Ignore-Filterung...")
            await asyncio.sleep(0.05)

            # Step 2: Local AI & Presidio NER (with periodic live ticker updates every 2s)
            update_step_ui(1, 0.25, "Inferenz läuft...")

            def do_analysis(text, snapshot: EffectiveConfig):
                with _model_lock:
                    anon = get_synced_cached_anonymizer(state, effective_config=snapshot)
                return anon.analyze(text)

            loop = asyncio.get_running_loop()
            analysis_task = asyncio.create_task(
                asyncio.to_thread(do_analysis, state.raw_text, analysis_snapshot)
            )

            start_t = loop.time()
            ticker_messages = [
                "KI-Modell & Presidio Inferenz...",
                "Texteinbettungen & NER-Modell aktiv...",
                "Erkennung von Personen, Rollen & Orten...",
                "Kontextuelle Analyse längerer Passagen...",
                "Muster- & Wörterbuchabgleich...",
            ]
            msg_idx = 0
            cur_pct = 0.25
            while not analysis_task.done():
                await asyncio.sleep(2.0)
                if analysis_task.done():
                    break
                elapsed = int(loop.time() - start_t)
                cur_pct = min(0.65, cur_pct + 0.08)
                sub_msg = f"{ticker_messages[msg_idx % len(ticker_messages)]} ({elapsed}s)"
                msg_idx += 1
                update_step_ui(1, cur_pct, sub_msg)

            results = await analysis_task

            # Step 3: Deduplication & Genitives
            update_step_ui(2, 0.75, "Span-Deduplizierung & Genitiv-Erweiterung...")
            await asyncio.sleep(0.05)

            state.entity_groups, state.occurrence_overrides = rebind_overrides_after_analysis(
                state.raw_text,
                results,
                state.occurrence_overrides,
                existing_groups=state.entity_groups,
            )

            # Step 4: Smart-Linking proposals
            update_step_ui(3, 0.90, "Namensbezüge & Rollen-Kandidaten...")
            await asyncio.sleep(0.05)

            from local_anonymizer.anonymizer import compute_smart_link_proposals
            compute_smart_link_proposals(state.entity_groups)

            # Step 5: Reactive Preview & Mapping
            update_step_ui(4, 0.98, "Generiere Vorschau & Mapping...")
            await asyncio.sleep(0.05)

            state.analysis_revision += 1
            state.llm_triage_results.clear()
            state.llm_triage_snapshot = ""
            state.llm_staged_selections.clear()

            compute_reactive_preview(state)
            state.analyzed_config_hash = analysis_snapshot_hash
            state.refresh_effective_config()
            total_occurrences = sum(g.count for g in state.entity_groups)
            build_llm_panel()
            build_review_table()
            refresh_preview_and_exports()

            # Mark all complete (green)
            update_step_ui(5, 1.0, f"Abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen)")
            ui.notify(f"Analyse abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen).", type="positive")

            # Check if LLM review should immediately chain into execution (Decision 1 from Handoff 1316)
            if state.config.llm_enabled and state.config.llm_auto_review and LLM_AVAILABLE and state.entity_groups:
                task = launch_llm_triage(triggered_from_analysis=True)
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Keep visible for 1.2s so user sees all checkmarks, then auto-disappear
            await asyncio.sleep(1.2)

        except Exception as e:
            if table_holder:
                table_holder.clear()
            logging.error(f"Analysis error: {e}", exc_info=True)
            ui.notify(f"Fehler bei der Analyse: {str(e)}", type="negative", close_button=True)
        finally:
            state.is_analyzing = False
            if not state.is_llm_running and state.llm_setup_state == "idle":
                set_mutating_controls_disabled(False)
            if analyze_btn:
                analyze_btn.props(remove="loading")
            if progress_holder:
                progress_holder.clear()

    def update_triage_ui():
        set_mutating_controls_disabled(state.is_busy)
        if llm_panel_holder is not None:
            build_llm_panel()
        if table_holder is not None:
            build_review_table()

    def launch_llm_triage(triggered_from_analysis: bool = False) -> Optional[asyncio.Task]:
        """Central launcher for LLM triage runs, creating exactly one tracked task in state.llm_active_task."""
        return launch_llm_triage_for_state(
            state,
            triggered_from_analysis=triggered_from_analysis,
            notify_cb=lambda msg, t: ui.notify(msg, type=t),
            on_update_ui=update_triage_ui,
            client=client,
        )

    def check_mutation_allowed() -> bool:
        """Central guard against mutating state while analysis, LLM triage or setup is running."""
        if state.is_analyzing:
            ui.notify("Aktion während laufender Textanalyse gesperrt.", type="warning")
            return False
        if state.is_llm_running:
            ui.notify("Aktion während laufender LLM-Prüfung gesperrt.", type="warning")
            return False
        if state.llm_setup_state != "idle":
            ui.notify("Aktion während laufendem Modell-Setup gesperrt.", type="warning")
            return False
        return True

    def set_mutating_controls_disabled(disabled: bool):
        """Disable/enable all mutating UI controls during active analysis, LLM inference or setup."""
        is_blocked = disabled or state.is_busy
        if analyze_btn:
            analyze_btn.set_enabled(not is_blocked and bool(state.raw_text and state.raw_text.strip()))
        if reset_btn:
            reset_btn.set_enabled(not is_blocked)
        if raw_text_area:
            if is_blocked:
                raw_text_area.props("readonly")
            else:
                raw_text_area.props(remove="readonly")
        if add_manual_btn:
            add_manual_btn.set_enabled(not is_blocked)
        if manual_input:
            manual_input.set_enabled(not is_blocked)
        if manual_type:
            manual_type.set_enabled(not is_blocked)
        for elem in state.mutating_ui_elements:
            try:
                elem.set_enabled(not is_blocked)
            except Exception:
                pass

    def open_apply_dialog(selected_occ_ids: List[str]):
        """Open modal impact preview dialog and execute confirmed mutations atomically via ApplyService."""
        if not check_mutation_allowed():
            return
        if not selected_occ_ids:
            ui.notify("Keine Fundstellen ausgewählt.", type="warning")
            return

        commands: List[ApplyCommand] = []
        for occ_id in selected_occ_ids:
            item = state.llm_triage_results.get(occ_id)
            if item is None:
                continue
            cmd = ApplyCommand(
                occ_id=occ_id,
                action=item.action,
                new_entity_type=getattr(item, "new_entity_type", None),
                descriptor_suggestion=getattr(item, "descriptor_suggestion", None),
            )
            commands.append(cmd)

        if not commands:
            ui.notify("Keine gültigen Befehle gefunden.", type="warning")
            return

        is_valid, err, impacts = ApplyService.prevalidate_and_preview_impact(
            state,
            state.llm_triage_snapshot,
            commands,
        )

        if not is_valid:
            ui.notify(f"Übernahme nicht möglich: {err}", type="warning", timeout=8000)
            build_llm_panel()
            return

        with ui.dialog() as dialog, ui.card().classes("w-[650px] max-w-full p-4"):
            ui.label("Vorschau: Ausgewählte Änderungen übernehmen").classes("text-base font-bold text-slate-800 mb-1")
            ui.label(
                f"Es werden {len(commands)} Fundstellen atomar über den kanonischen Override-Service angepasst:"
            ).classes("text-xs text-slate-600 mb-3")

            with ui.column().classes("w-full max-h-[300px] overflow-y-auto gap-2 mb-3 border rounded p-2 bg-slate-50"):
                for imp in impacts:
                    with ui.row().classes("w-full items-center justify-between text-xs p-1.5 bg-white border rounded"):
                        with ui.row().classes("items-center gap-1.5"):
                            ui.label(imp["original_text"]).classes("font-mono font-bold text-slate-800")
                            if imp["will_split"]:
                                ui.badge("✂️ Eigene Gruppe", color="purple-7").props("dense")
                            if imp["action"] == "discard":
                                ui.badge("Ignorieren", color="grey-8")
                            elif imp["action"] == "recategorize":
                                ui.badge(f"{imp['old_type']} ➔ {imp['new_type']}", color="amber-8")
                            elif imp["action"] == "keep":
                                ui.badge("Bestätigt", color="green-7").props("outline")

                            if imp.get("new_role") != imp.get("old_role"):
                                ui.badge(f"Rolle: {imp['new_role']}", color="blue-7")

                        # Target category/descriptor preview badge in apply dialog
                        clean_r = clean_tag(imp.get("new_role") or "")
                        typ = imp.get("new_type") or imp.get("old_type")
                        preview_tag = f"[{typ} · {clean_r}]" if clean_r else f"[{typ}]"
                        if imp["action"] == "discard":
                            preview_tag = "(ignoriert)"
                        ui.badge(preview_tag, color="indigo-8").props("dense").classes("font-mono font-semibold").tooltip("Unverbindliche Kategorie-/Deskriptorvorschau (Nummerierung wird bei Übernahme ermittelt)")

            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")

                def confirm_apply():
                    success, msg = ApplyService.apply_mutations(
                        state,
                        state.llm_triage_snapshot,
                        commands,
                        split_fn=split_occurrence_to_new_group,
                        sync_fn=sync_group_overrides,
                        preview_fn=compute_reactive_preview,
                    )
                    dialog.close()
                    if success:
                        ui.notify(msg, type="positive", icon="check")
                        build_llm_panel()
                        build_review_table()
                        refresh_preview_and_exports()
                    else:
                        ui.notify(f"Fehler bei Übernahme: {msg}", type="negative", timeout=8000)
                        build_llm_panel()
                        build_review_table()

                ui.button("Bestätigen & Übernehmen", icon="check", color="positive", on_click=confirm_apply).props("unelevated dense")

        dialog.open()

    def render_proposal_card(item: Any, tone: str):
        grp, occ = ApplyService._find_occurrence_and_group(state.entity_groups, item.occ_id)
        if grp is None or occ is None:
            return

        is_staged = item.occ_id in state.llm_staged_selections

        def on_stage_toggle(e):
            if not check_mutation_allowed():
                return
            if e.value:
                state.llm_staged_selections.add(item.occ_id)
            else:
                state.llm_staged_selections.discard(item.occ_id)
            build_llm_panel()

        def compute_card_target_placeholder(custom_role: Optional[str] = None) -> str:
            if item.action == "discard":
                return "(wird ignoriert)"
            target_type = item.new_entity_type if (item.action == "recategorize" and getattr(item, "new_entity_type", None)) else grp.entity_type
            target_role = custom_role if custom_role is not None else getattr(item, "descriptor_suggestion", None)
            if target_role is None:
                target_role = grp.role or ""

            clean_r = clean_tag(target_role) if target_role else ""
            if clean_r:
                return f"➔ [{target_type} · {clean_r}]"
            else:
                return f"➔ [{target_type}]"

        border_cls = "border-primary ring-1 ring-primary/40 bg-white" if is_staged else "border-slate-200 bg-white"

        with ui.card().props(f'id="llm_card_{item.occ_id}"').classes(f"w-full p-2.5 my-1 border rounded shadow-none {border_cls}"):
            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                with ui.row().classes("items-center gap-2 flex-1 min-w-[280px]"):
                    stage_chk = ui.checkbox(value=is_staged, on_change=on_stage_toggle).props("dense").tooltip("Für Sammelübernahme vormerken")
                    state.register_mutating_element(stage_chk, "llm")
                    ui.label(grp.original_text).classes("font-mono font-bold text-xs text-slate-900")
                    ui.badge(grp.entity_type, color="blue-grey").props("outline")
                    if grp.role:
                        ui.badge(f"Ist: {grp.role}", color="teal").props("outline")
                    if occ.context_html:
                        ui.html(occ.context_html).classes("text-xs text-slate-700 ml-2 flex-1")

                with ui.row().classes("items-center gap-2 flex-wrap"):
                    if item.action == "discard":
                        ui.badge("Ignorieren", color="grey-8")
                    elif item.action == "recategorize":
                        ui.badge(f"➔ {item.new_entity_type}", color="amber-8")
                    elif item.action == "keep":
                        ui.badge("Bestätigt", color="green-7").props("outline")

                    current_suggestion = getattr(item, "descriptor_suggestion", None) or ""

                    # Target placeholder live preview badge
                    target_badge = ui.badge(
                        compute_card_target_placeholder(current_suggestion),
                        color="indigo-8",
                    ).props("dense").classes("font-mono text-xs font-bold").tooltip("Unverbindliche Kategorie-/Deskriptorvorschau (Nummerierung wird bei Übernahme ermittelt)")

                    # Inline editable input for descriptor/role
                    def on_role_edit(e):
                        val = (e.value or "").strip()
                        item.descriptor_suggestion = val if val else None
                        target_badge.set_text(compute_card_target_placeholder(val))

                    if item.action != "discard":
                        role_inp = ui.input(
                            value=current_suggestion,
                            placeholder="Rolle anpassen...",
                            on_change=on_role_edit,
                        ).props("dense outlined dense-input").classes("w-36 text-xs h-7").tooltip("Deskriptor / Rolle direkt anpassen")
                        state.register_mutating_element(role_inp, "llm")

                    conf_color = "positive" if item.confidence == "high" else ("warning" if item.confidence == "medium" else "grey")
                    ui.badge(f"{item.confidence}", color=conf_color).props("outline").tooltip(f"Modell-Konfidenz: {item.confidence}")

                    def single_apply():
                        open_apply_dialog([item.occ_id])

                    single_btn = ui.button("Übernehmen", icon="check", color="positive", on_click=single_apply).props("outline dense size=xs")
                    state.register_mutating_element(single_btn, "llm")

            if getattr(item, "reasoning", None):
                with ui.row().classes("w-full mt-1 text-[11px] text-slate-600 italic px-7"):
                    ui.label(f"Begründung: {item.reasoning}")

    def render_proposal_categories(keep_items: List[Any], recat_items: List[Any], discard_items: List[Any]):
        recat_and_desc = []
        pure_keeps = []
        for item in keep_items:
            if getattr(item, "descriptor_suggestion", None):
                recat_and_desc.append(item)
            else:
                pure_keeps.append(item)
        recat_and_desc = recat_items + recat_and_desc

        with ui.expansion(
            f"🟡 Änderung vorgeschlagen – menschlich ungeprüft ({len(recat_and_desc)})",
            value=bool(recat_and_desc),
        ).classes("w-full bg-amber-50/50 border border-amber-200 rounded-lg mb-2"):
            if not recat_and_desc:
                ui.label("Keine Änderungsvorschläge.").classes("text-xs text-slate-500 italic p-2")
            else:
                for item in recat_and_desc:
                    render_proposal_card(item, "amber")

        with ui.expansion(
            f"⚪ Ignorieren vorgeschlagen – menschlich ungeprüft ({len(discard_items)})",
            value=bool(discard_items),
        ).classes("w-full bg-slate-50 border border-slate-200 rounded-lg mb-2"):
            if not discard_items:
                ui.label("Keine Ignorier-Vorschläge.").classes("text-xs text-slate-500 italic p-2")
            else:
                for item in discard_items:
                    render_proposal_card(item, "slate")

        with ui.expansion(
            f"🟢 LLM bestätigt – menschlich ungeprüft ({len(pure_keeps)})",
            value=False,
        ).classes("w-full bg-emerald-50/40 border border-emerald-200 rounded-lg mb-2"):
            if not pure_keeps:
                ui.label("Keine reinen Bestätigungen.").classes("text-xs text-slate-500 italic p-2")
            else:
                for item in pure_keeps:
                    render_proposal_card(item, "emerald")

        if state.llm_unprocessed_occ_ids:
            unprocessed_items = []
            for occ_id in state.llm_unprocessed_occ_ids:
                grp, occ = ApplyService._find_occurrence_and_group(state.entity_groups, occ_id)
                if grp and occ:
                    unprocessed_items.append((grp, occ))

            with ui.expansion(
                f"⚠️ Ungeprüft / Vom Modell übersprungen ({len(unprocessed_items)})",
                value=True,
            ).classes("w-full bg-amber-50/70 border border-amber-300 rounded-lg mb-2"):
                for grp, occ in unprocessed_items:
                    with ui.card().classes("w-full p-2.5 my-1 border border-amber-200 rounded bg-white shadow-none"):
                        with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                            with ui.row().classes("items-center gap-2 flex-1 min-w-[280px]"):
                                ui.label(grp.original_text).classes("font-mono font-bold text-xs text-slate-900")
                                ui.badge(grp.entity_type, color="blue-grey").props("outline")
                                if occ.context_html:
                                    ui.html(occ.context_html).classes("text-xs text-slate-700 ml-2 flex-1")
                            with ui.row().classes("items-center gap-2"):
                                ui.badge("Vom LLM nicht beantwortet", color="amber-9").props("dense outline").tooltip("Das lokale Modell hat für diese Fundstelle im JSON-Output keinen Eintrag generiert.")

    def build_llm_panel():
        if not llm_panel_holder:
            return
        state.clear_mutating_zone("llm")
        llm_panel_holder.clear()
        with llm_panel_holder:
            if not LLM_AVAILABLE:
                with ui.card().classes("w-full p-3 bg-slate-50 border border-slate-200 rounded-lg"):
                    with ui.row().classes("items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2 text-slate-600 text-xs"):
                            ui.icon("info", size="sm").classes("text-slate-400")
                            ui.label("Lokale LLM-Review-Assistenz: Optionales Zusatzpaket `[llm]` nicht installiert. Zur Aktivierung: `pip install local-anonymizer[llm]`.").classes("font-medium")
                return

            if not state.config.llm_enabled:
                with ui.card().classes("w-full p-3 bg-slate-50 border border-slate-200 rounded-lg"):
                    with ui.row().classes("items-center justify-between gap-2 flex-wrap"):
                        with ui.row().classes("items-center gap-2 text-slate-700 text-xs"):
                            ui.icon("psychology", size="sm").classes("text-slate-400")
                            ui.label("Lokale LLM-Review-Assistenz ist deaktiviert.").classes("font-semibold")
                            ui.label("Aktivieren Sie die Option in der LLM-Konfiguration (oberer Bereich), um Fundstellen automatisch durch ein lokales LLM prüfen zu lassen.").classes("text-slate-500")
                        def enable_in_config():
                            state.config.llm_enabled = True
                            save_current_config(state)
                            render_llm_settings_ui()
                            build_llm_panel()
                        ui.button("In Konfiguration aktivieren", icon="toggle_on", on_click=enable_in_config).props("flat dense size=sm color=primary")
                return

            has_entities = bool(state.entity_groups)
            has_results = bool(state.llm_triage_results)
            has_model = bool(state.config.llm_model_name and state.config.llm_model_name.strip())

            card_bg = "bg-indigo-50/50 border-indigo-200" if has_results else "bg-slate-50 border-slate-200"
            with ui.card().classes(f"w-full p-3.5 border rounded-lg {card_bg}"):
                with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("psychology", size="md").classes("text-indigo-700")
                        ui.label("Lokale LLM-Review-Assistenz").classes("font-bold text-sm text-indigo-950")
                        if state.is_llm_running:
                            ui.spinner(size="xs", color="primary")
                            ui.badge("Prüfung läuft...", color="primary").props("dense")
                        elif has_results:
                            if state.llm_partial_failure:
                                ui.badge("⚠️ Prüfung unvollständig", color="warning").props("dense")
                            else:
                                ui.badge("✓ Prüfung abgeschlossen", color="positive").props("dense outline")
                        elif not has_entities:
                            ui.badge("Keine Fundstellen", color="grey-6").props("dense outline")
                        elif not has_model:
                            ui.badge("Modellname fehlt", color="warning").props("dense outline")
                        else:
                            ui.badge("Bereit", color="positive").props("dense outline")

                    with ui.row().classes("items-center gap-2"):
                        if state.is_llm_running:
                            async def cancel_triage():
                                if state.llm_active_task and not state.llm_active_task.done():
                                    state.llm_active_task.cancel()
                                await state.close_llm_provider()
                                ui.notify("LLM-Triage wird abgebrochen...", type="info")
                            ui.button("⏹️ Abbrechen", icon="cancel", color="negative", on_click=cancel_triage).props("dense outline size=sm")
                        else:
                            btn_label = "🔄 LLM-Review erneut starten" if has_results else "🤖 Fundstellen mit lokalem LLM prüfen"
                            def start_triage_click():
                                if not check_mutation_allowed():
                                    return
                                launch_llm_triage(triggered_from_analysis=False)
                            triage_start_btn = ui.button(btn_label, icon="auto_awesome", color="primary", on_click=start_triage_click).props("unelevated dense size=sm")
                            state.register_mutating_element(triage_start_btn, "llm")
                            is_start_enabled = has_entities and has_model and not state.is_llm_running
                            triage_start_btn.set_enabled(is_start_enabled)
                            if not has_entities:
                                triage_start_btn.tooltip("Führen Sie zuerst eine Textanalyse durch, um Fundstellen zu ermitteln.")
                            elif not has_model:
                                triage_start_btn.tooltip("Geben Sie im oberen LLM-Bereich einen Modellnamen an (z. B. qwen3:8b).")

                if state.is_llm_running:
                    with ui.column().classes("w-full mt-2"):
                        ui.label("Überprüfe Fundstellen sequenziell in Batches über lokalen LLM-Endpunkt...").classes("text-xs text-slate-600 italic")
                elif has_results:
                    keep_items = [item for item in state.llm_triage_results.values() if item.action == "keep"]
                    recat_items = [item for item in state.llm_triage_results.values() if item.action == "recategorize"]
                    discard_items = [item for item in state.llm_triage_results.values() if item.action == "discard"]
                    descriptor_changes = [item for item in keep_items if getattr(item, "descriptor_suggestion", None)]
                    pure_keeps = [item for item in keep_items if not getattr(item, "descriptor_suggestion", None)]

                    if state.llm_partial_failure:
                        unprocessed_names = []
                        for occ_id in list(state.llm_unprocessed_occ_ids)[:3]:
                            grp, occ = ApplyService._find_occurrence_and_group(state.entity_groups, occ_id)
                            if grp:
                                unprocessed_names.append(f"„{grp.original_text}“")
                        extra_info = f" (u. a. {', '.join(unprocessed_names)})" if unprocessed_names else ""

                        with ui.row().classes("w-full items-center gap-2 p-2 bg-amber-100/70 border border-amber-300 rounded text-amber-950 text-xs my-2"):
                            ui.icon("warning", size="xs").classes("text-amber-700")
                            ui.label(f"Einige Batches konnten nicht verarbeitet werden ({len(state.llm_unprocessed_occ_ids)} Fundstellen ungeprüft{extra_info}). Sammelübernahme ist gesperrt; Einzelübernahmen sind möglich.").classes("font-medium")

                    with ui.row().classes("w-full items-center justify-between p-2 bg-white/80 border rounded text-xs my-2 flex-wrap gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"Geprüft: {len(state.llm_triage_results)} Fundstellen ({len(pure_keeps)} bestätigt, {len(recat_items) + len(descriptor_changes)} Änderungsvorschläge, {len(discard_items)} False Positives)").classes("text-slate-700 font-semibold")

                        with ui.row().classes("items-center gap-2"):
                            def stage_all():
                                if not check_mutation_allowed():
                                    return
                                state.llm_staged_selections = set(state.llm_triage_results.keys())
                                build_llm_panel()
                            def unstage_all():
                                if not check_mutation_allowed():
                                    return
                                state.llm_staged_selections.clear()
                                build_llm_panel()

                            stage_all_btn = ui.button("Alle vormerken", on_click=stage_all).props("flat dense size=xs color=slate")
                            unstage_all_btn = ui.button("Auswahl leeren", on_click=unstage_all).props("flat dense size=xs color=slate")
                            state.register_mutating_element(stage_all_btn, "llm")
                            state.register_mutating_element(unstage_all_btn, "llm")

                            selected_count = len(state.llm_staged_selections)
                            apply_bulk_btn = ui.button(
                                f"Ausgewählte Änderungen übernehmen ({selected_count})",
                                icon="done_all",
                                color="positive",
                                on_click=lambda: open_apply_dialog(list(state.llm_staged_selections)),
                            ).props("unelevated dense size=sm")
                            state.register_mutating_element(apply_bulk_btn, "llm")

                            if selected_count == 0 or state.llm_partial_failure:
                                apply_bulk_btn.disable()
                                if state.llm_partial_failure:
                                    apply_bulk_btn.tooltip("Sammelübernahme bei unvollständigem Gesamtlauf gesperrt. Bitte Einzelübernahmen nutzen.")

                    render_proposal_categories(keep_items, recat_items, discard_items)

    def get_sorted_groups() -> List[EntityGroup]:
        """Return entity groups sorted according to user selection."""
        if state.sort_by == "Alphabetisch (A–Z)":
            return sorted(state.entity_groups, key=lambda g: (g.text_key, g.group_id))
        elif state.sort_by == "Erstes Auftreten im Text":
            return sorted(state.entity_groups, key=lambda g: g.first_start)
        elif state.sort_by == "Häufigkeit (meiste Treffer zuerst)":
            return sorted(state.entity_groups, key=lambda g: (-g.count, g.first_start))
        elif state.sort_by == "Entitätstyp (PERSON, ORG, ...)":
            return sorted(state.entity_groups, key=lambda g: (g.entity_type, g.text_key, g.group_id))
        elif state.sort_by == "⚠️ Review-Bedarf zuerst":
            return sorted(state.entity_groups, key=lambda g: (not g.needs_review, -g.count, g.first_start))
        else:
            return sorted(state.entity_groups, key=lambda g: (g.text_key, g.group_id))

    def build_review_table():
        if not table_holder:
            return
        state.clear_mutating_zone("table")
        table_holder.clear()
        with table_holder:
            if not state.entity_groups:
                ui.label("Keine zu anonymisierenden Entitäten im Text erkannt.").classes("text-slate-500 italic p-4")
                return

            total_hits = sum(g.count for g in state.entity_groups)
            unique_count = len(state.entity_groups)

            ui.label(f"Erkannte Entitäten ({unique_count} Begriffe, {total_hits} Fundstellen gesamt)").classes("text-lg font-bold text-slate-800 mb-1")
            ui.label("Vergeben Sie optionale Rollen oder übernehmen Sie Verknüpfungsvorschläge. Jede Zeile zeigt den zugeordneten Platzhalter und klappt Fundstellen auf:").classes("text-sm text-slate-600 mb-3")

            # Toolbar: Sorting & Bulk actions
            all_expansions: List[Any] = []
            toggle_expand_btn = None

            def update_expand_btn():
                if not toggle_expand_btn:
                    return
                any_open = any(bool(exp.value) for exp in all_expansions)
                if any_open:
                    toggle_expand_btn.set_text("Alle zuklappen")
                    toggle_expand_btn.props("icon=unfold_less")
                else:
                    toggle_expand_btn.set_text("Alle aufklappen")
                    toggle_expand_btn.props("icon=unfold_more")

            def toggle_all_expansions():
                any_open = any(bool(exp.value) for exp in all_expansions)
                for exp in all_expansions:
                    if any_open:
                        exp.close()
                    else:
                        exp.open()
                update_expand_btn()

            with ui.row().classes("w-full items-center justify-between bg-slate-100 p-2.5 rounded-lg border mb-3 flex-wrap gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sort", size="sm").classes("text-slate-600")
                    ui.label("Sortierung:").classes("text-xs font-semibold text-slate-700")

                    def on_sort_change(e):
                        if not check_mutation_allowed():
                            return
                        state.sort_by = e.value
                        build_review_table()

                    sort_sel = ui.select(
                        options=[
                            "Alphabetisch (A–Z)",
                            "Erstes Auftreten im Text",
                            "Häufigkeit (meiste Treffer zuerst)",
                            "Entitätstyp (PERSON, ORG, ...)",
                            "⚠️ Review-Bedarf zuerst",
                        ],
                        value=state.sort_by,
                        on_change=on_sort_change,
                    ).props("dense outlined bg-white").classes("w-64 text-xs")
                    state.register_mutating_element(sort_sel, "table")

                with ui.row().classes("items-center gap-2"):
                    def select_all():
                        if not check_mutation_allowed():
                            return
                        for g in state.entity_groups:
                            g.enabled = True
                            sync_group_overrides(state, g)
                        refresh_preview_and_exports()
                        build_review_table()

                    def deselect_all():
                        if not check_mutation_allowed():
                            return
                        for g in state.entity_groups:
                            g.enabled = False
                            sync_group_overrides(state, g)
                        refresh_preview_and_exports()
                        build_review_table()

                    sel_all_btn = ui.button("Alle aktivieren", icon="select_all", on_click=select_all, color="slate").props("outline dense size=sm")
                    desel_all_btn = ui.button("Alle abwählen", icon="deselect", on_click=deselect_all, color="slate").props("outline dense size=sm")
                    state.register_mutating_element(sel_all_btn, "table")
                    state.register_mutating_element(desel_all_btn, "table")

                    toggle_expand_btn = ui.button(
                        "Alle aufklappen",
                        icon="unfold_more",
                        on_click=toggle_all_expansions,
                        color="slate",
                    ).props("outline dense size=sm")

            sorted_groups = get_sorted_groups()
            from local_anonymizer.anonymizer import build_entity_tree
            tree = build_entity_tree(sorted_groups)
            clusters = group_tree_nodes_by_homonym(tree)

            # Render Tree Nodes within Homonym Clusters (Roots & Indented Children)
            for cluster_idx, cluster in enumerate(clusters):
                is_multi_homonym = len(cluster.nodes) > 1
                cluster_container_classes = "w-full mb-3 p-2 bg-indigo-50/40 border border-indigo-200 rounded-lg" if is_multi_homonym else "w-full mb-2"

                with ui.column().classes(cluster_container_classes):
                    if is_multi_homonym:
                        total_cluster_hits = sum(node.item.count for node in cluster.nodes)
                        with ui.row().classes("w-full items-center justify-between px-2 py-1 bg-indigo-100/70 border border-indigo-200 rounded text-xs mb-1.5"):
                            with ui.row().classes("items-center gap-1.5"):
                                ui.icon("layers", size="xs").classes("text-indigo-800")
                                ui.label(f"Homonym-Bündel: '{cluster.primary_text}'").classes("font-bold text-indigo-950")
                                ui.badge(f"{len(cluster.nodes)} Varianten", color="indigo-8").props("dense")
                            ui.badge(f"{total_cluster_hits} Fundstellen gesamt", color="indigo-6").props("dense outline")

                    for node_idx, root_node in enumerate(cluster.nodes):
                        master: EntityGroup = root_node.item
                        has_children = len(root_node.children) > 0
                        row_bg = "bg-amber-50" if master.needs_review else ("bg-white" if node_idx % 2 == 0 else "bg-slate-50")
                        is_split = master.group_id != master.text_key

                        with ui.column().classes("w-full mb-1"):
                            child_badge_refs: List[Tuple[EntityGroup, Any]] = []

                            # Master Row
                            with ui.expansion().classes(f"w-full border rounded {row_bg}") as exp:
                                all_expansions.append(exp)
                                exp.on_value_change(lambda _: update_expand_btn())
                                with exp.add_slot("header"):
                                    with ui.row().classes("w-full items-center justify-between gap-3 pr-2 flex-wrap"):
                                        # 1. Checkbox + Name + Count + Assigned Placeholder Badge
                                        with ui.row().classes("items-center gap-2 min-w-[280px]"):
                                            def make_group_check(grp):
                                                def on_change(e):
                                                    if not check_mutation_allowed():
                                                        return
                                                    grp.enabled = e.value
                                                    sync_group_overrides(state, grp)
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                return on_change

                                            master_check = ui.checkbox(value=master.enabled, on_change=make_group_check(master)).props("dense")
                                            state.register_mutating_element(master_check, "table")
                                            if not is_category_enabled(state, master.entity_type):
                                                master_check.disable()
                                            ui.label(master.original_text).classes("font-mono text-sm font-bold text-slate-800")
                                            ui.badge(f"{master.count}x", color="primary" if master.count > 1 else "grey-6").props("dense")

                                            # Master Placeholder Badge
                                            master_badge = ui.badge(master.placeholder, color="blue-9").props("outline dense").classes("text-xs font-mono font-bold")

                                            if is_split:
                                                ui.badge("✂️ Ausgliederung", color="purple-7").props("dense").tooltip("Abgespaltene Homonym-Gruppe mit eigener Zuweisung")
                                            elif is_multi_homonym:
                                                ui.badge("Basis", color="indigo-7").props("dense outline").tooltip("Hauptgruppe dieses Homonyms")

                                            if has_children:
                                                ui.badge(f"🔗 {len(root_node.children)} verknüpft", color="teal").props("dense").tooltip("Hauptperson mit verknüpften Schreibweisen")

                                        # 2. Type Selector
                                        with ui.row().classes("items-center gap-1"):
                                            def make_group_select(grp):
                                                def on_change(e):
                                                    if not check_mutation_allowed():
                                                        return
                                                    grp.entity_type = e.value
                                                    sync_group_overrides(state, grp)
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                return on_change

                                            m_sel = ui.select(
                                                options=AVAILABLE_ENTITIES,
                                                value=master.entity_type,
                                                on_change=make_group_select(master),
                                            ).props("dense outlined bg-white").classes("w-36 text-xs")
                                            state.register_mutating_element(m_sel, "table")

                                        # 3. Role Input Field (In-place badge update WITHOUT losing focus!)
                                        with ui.row().classes("items-center gap-1"):
                                            def make_role_change(grp, m_badge, c_badges):
                                                def on_change(e):
                                                    if not check_mutation_allowed():
                                                        return
                                                    grp.role = (e.value or "").strip()
                                                    grp.role_provenance = "manual"
                                                    sync_group_overrides(state, grp)
                                                    compute_reactive_preview(state)
                                                    m_badge.set_text(grp.placeholder)
                                                    for c_grp, c_badge in c_badges:
                                                        c_badge.set_text(c_grp.placeholder)
                                                    refresh_preview_and_exports()
                                                return on_change

                                            m_role_inp = ui.input(
                                                label="Rolle (optional)",
                                                value=master.role,
                                                placeholder="z.B. Student",
                                                on_change=make_role_change(master, master_badge, child_badge_refs),
                                            ).props("dense outlined bg-white debounce=300").classes("w-32 text-xs")
                                            state.register_mutating_element(m_role_inp, "table")

                                        # 4. Interactive Smart-Linking Proposal (Confirmation Model) or Link Dropdown
                                        if not has_children:
                                            if master.suggested_parent and not master.parent_group_id:
                                                with ui.row().classes("items-center gap-1"):
                                                    def make_apply_suggestion(grp, p_target_id, tag):
                                                        def on_click():
                                                            if not check_mutation_allowed():
                                                                return
                                                            grp.parent_group_id = p_target_id
                                                            grp.surface_tag = tag
                                                            grp.suggested_parent = None
                                                            grp.suggested_parent_text = None
                                                            grp.suggested_tag = None
                                                            grp.suggested_candidates = []
                                                            p_match = next((x for x in state.entity_groups if x.group_id == p_target_id), None)
                                                            if p_match and not p_match.surface_tag:
                                                                p_match.surface_tag = "VOLLNAME"
                                                            target_name = p_match.original_text if p_match else p_target_id
                                                            ui.notify(f"'{grp.original_text}' mit '{target_name}' ({tag}) verknüpft.", type="positive", icon="link")
                                                            refresh_preview_and_exports()
                                                            build_review_table()
                                                        return on_click

                                                    p_target_id = master.suggested_parent
                                                    p_target_text = getattr(master, "suggested_parent_text", None)
                                                    if not p_target_text:
                                                        p_match = next((x for x in state.entity_groups if x.group_id == p_target_id), None)
                                                        p_target_text = p_match.original_text if p_match else p_target_id

                                                    tag_label = SURFACE_TAG_OPTIONS.get(master.suggested_tag, master.suggested_tag or "Vorname")
                                                    sugg_btn = ui.button(
                                                        f"💡 Mit '{p_target_text}' verknüpfen",
                                                        icon="auto_awesome",
                                                        color="teal-8",
                                                        on_click=make_apply_suggestion(master, p_target_id, master.suggested_tag or "VORNAME"),
                                                    ).props("outline dense size=sm").tooltip(f"Vorschlag übernehmen: Als {tag_label} verknüpfen")
                                                    state.register_mutating_element(sugg_btn, "table")

                                            elif master.suggested_candidates:
                                                ui.badge(f"💡 {len(master.suggested_candidates)} Namenskandidaten", color="amber-8").props("outline dense").tooltip("Mehrere passende Personen gefunden. Bitte im Dropdown rechts auswählen.")

                                            # Manual Link Dropdown: All other recognized entities in the workspace
                                            link_options = {"": "Eigenständig"}
                                            for other in state.entity_groups:
                                                if other.group_id != master.group_id and not other.parent_group_id:
                                                    link_options[other.group_id] = format_link_dropdown_label(other)

                                            if len(link_options) > 1:
                                                with ui.row().classes("items-center gap-1"):
                                                    def make_link_to_master(grp):
                                                        def on_change(e):
                                                            if not check_mutation_allowed():
                                                                return
                                                            val = (e.value or "").strip()
                                                            if not val or val == "Eigenständig":
                                                                grp.parent_group_id = None
                                                                grp.surface_tag = ""
                                                                grp.suggested_parent = None
                                                                grp.suggested_parent_text = None
                                                                grp.suggested_candidates = []
                                                            else:
                                                                p_match = next((x for x in state.entity_groups if x.group_id == val), None)
                                                                if p_match:
                                                                    grp.parent_group_id = p_match.group_id
                                                                    if not grp.surface_tag:
                                                                        grp.surface_tag = "VORNAME"
                                                                    grp.suggested_parent = None
                                                                    grp.suggested_parent_text = None
                                                                    grp.suggested_candidates = []
                                                                    if not p_match.surface_tag:
                                                                        p_match.surface_tag = "VOLLNAME"
                                                                    ui.notify(f"'{grp.original_text}' verknüpft mit '{p_match.original_text}'.", type="info")
                                                                    refresh_preview_and_exports()
                                                                    build_review_table()
                                                                else:
                                                                    ui.notify(f"Ungültige Verknüpfung: Zielgruppe existiert nicht.", type="warning")
                                                        return on_change

                                                    current_val = master.parent_group_id if (master.parent_group_id and master.parent_group_id in link_options) else ""
                                                    link_sel = ui.select(
                                                        options=link_options,
                                                        value=current_val,
                                                        label="Verknüpfen mit:",
                                                        on_change=make_link_to_master(master),
                                                    ).props('dense outlined bg-white options-dense menu-props="{ maxHeight: \'300px\' }"').classes("min-w-[200px] max-w-[320px] text-xs").tooltip("Zielperson / Bezug auswählen")
                                                    state.register_mutating_element(link_sel, "table")

                                        # 5. Score & Action
                                        with ui.row().classes("items-center gap-2"):
                                            ui.label(master.score_display).classes("text-xs font-mono text-slate-600").tooltip(
                                                f"Score-Bereich über {master.count} Fundstellen"
                                            )
                                            if master.needs_review:
                                                ui.badge("⚠️ Review", color="warning").props("dense")
                                            else:
                                                ui.badge("✓ Sicher", color="positive").props("dense outline")

                                            def make_group_ignore(term, grp):
                                                def on_click():
                                                    if not check_mutation_allowed():
                                                        return
                                                    current_ignores = parse_ignore_terms(state.ignore_terms_text)
                                                    if term not in current_ignores:
                                                        current_ignores.append(term)
                                                        state.ignore_terms_text = ", ".join(current_ignores)
                                                        render_ignore_list_ui()
                                                        save_current_config(state)
                                                    grp.enabled = False
                                                    sync_group_overrides(state, grp)
                                                    ui.notify(f"'{term}' zur Ignore-Liste hinzugefügt.", type="info")
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                return on_click

                                            def make_add_to_glossary(term, grp):
                                                def on_click():
                                                    if not check_mutation_allowed():
                                                        return
                                                    lines = [line.strip() for line in state.glossary_text.splitlines() if line.strip()]
                                                    lines = [
                                                        line for line in lines
                                                        if not (
                                                            line.lower().startswith(f"{term.lower()}:")
                                                            or line.lower().startswith(f"{term.lower()}=")
                                                        )
                                                    ]
                                                    lines.append(f"{term}: {grp.entity_type}")
                                                    state.glossary_text = "\n".join(lines)
                                                    for occurrence in grp.occurrences:
                                                        occurrence.source = "glossary"
                                                        occurrence.method = "glossary_direct"
                                                        occurrence.method_detail = "direct"
                                                    grp.enabled = True
                                                    sync_group_overrides(state, grp)
                                                    save_current_config(state)
                                                    render_glossary_list_ui()
                                                    compute_reactive_preview(state)
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                    ui.notify(f"'{term}' als {grp.entity_type} zur Begriffsliste hinzugefügt.", type="positive", icon="library_add")
                                                return on_click

                                            def make_mark_manual(term, grp):
                                                def on_click():
                                                    if not check_mutation_allowed():
                                                        return
                                                    for occurrence in grp.occurrences:
                                                        occurrence.source = "manual"
                                                        occurrence.method = "manual"
                                                        occurrence.method_detail = "manual"
                                                        occurrence.needs_review = False
                                                    grp.enabled = True
                                                    sync_group_overrides(state, grp)
                                                    compute_reactive_preview(state)
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                    ui.notify(f"'{term}' nur für diesen Durchlauf manuell markiert.", type="info", icon="edit_note")
                                                return on_click

                                            ign_btn = ui.button(
                                                icon="block",
                                                on_click=make_group_ignore(master.original_text, master),
                                            ).props("flat round dense size=sm color=grey-8").tooltip("Begriff ignorieren und zur Ignore-Liste hinzufügen")
                                            gloss_btn = ui.button(
                                                icon="library_add",
                                                on_click=make_add_to_glossary(master.original_text, master),
                                            ).props("flat round dense size=sm color=teal-8").tooltip("Diesen Begriff dauerhaft mit diesem Entitätstyp zum Glossar hinzufügen")
                                            manual_button = ui.button(
                                                icon="edit_note",
                                                on_click=make_mark_manual(master.original_text, master),
                                            ).props("flat round dense size=sm color=orange-8").tooltip("Diesen Begriff nur für den aktuellen Durchlauf als manuell markiert behandeln")
                                            state.register_mutating_element(ign_btn, "table")
                                            state.register_mutating_element(gloss_btn, "table")
                                            state.register_mutating_element(manual_button, "table")
                                            if not is_category_enabled(state, master.entity_type):
                                                manual_button.disable()

                                # Expanded content: Context occurrences
                                with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                                    ui.label(f"Fundstellen im Text ({master.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                                    for occ_idx, occ in enumerate(master.occurrences, start=1):
                                        with ui.row().classes("items-center justify-between gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                            with ui.row().classes("items-center gap-2 flex-1"):
                                                ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                                ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                                method_label, method_color, method_tooltip = method_display(occ.method)
                                                ui.badge(method_label, color=method_color).props("dense outline").tooltip(method_tooltip)
                                                ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

                                                # Occurrence-bound LLM Action Badge
                                                if occ.occ_id in state.llm_triage_results:
                                                    occ_v = state.llm_triage_results[occ.occ_id]
                                                    if occ_v.action == "recategorize" or (occ_v.action == "keep" and getattr(occ_v, "descriptor_suggestion", None)):
                                                        b = ui.badge("LLM: Änderungsvorschlag", color="amber-8").props("dense").classes("cursor-pointer")
                                                        b.tooltip("Klicken, um zum LLM-Vorschlag zu springen")
                                                        b.on("click", lambda _, oid=occ.occ_id: ui.run_javascript(f"document.getElementById('llm_card_{oid}')?.scrollIntoView({{behavior: 'smooth'}});"))
                                                    elif occ_v.action == "discard":
                                                        b = ui.badge("LLM: Ignorieren vorgeschlagen", color="grey-8").props("dense").classes("cursor-pointer")
                                                        b.tooltip("Klicken, um zum LLM-Vorschlag zu springen")
                                                        b.on("click", lambda _, oid=occ.occ_id: ui.run_javascript(f"document.getElementById('llm_card_{oid}')?.scrollIntoView({{behavior: 'smooth'}});"))

                                            with ui.row().classes("items-center gap-1 shrink-0"):
                                                is_split_occ = occ.occ_id in state.occurrence_overrides or master.group_id != master.text_key
                                                if is_split_occ:
                                                    def make_revert_occ(grp, o):
                                                        def on_click():
                                                            if not check_mutation_allowed():
                                                                return
                                                            revert_occurrence_to_base(state, grp, o)
                                                            compute_reactive_preview(state)
                                                            refresh_preview_and_exports()
                                                            build_review_table()
                                                            ui.notify(f"Fundstelle wieder mit der Hauptgruppe '{grp.original_text}' zusammengeführt.", type="info", icon="merge")
                                                        return on_click

                                                    rev_btn = ui.button("↩️ Zusammenführen", on_click=make_revert_occ(master, occ)).props("flat dense size=sm color=teal-8").tooltip("Ausgliederung rückgängig machen und mit Basisgruppe vereinen")
                                                    state.register_mutating_element(rev_btn, "table")
                                                elif master.count > 1:
                                                    def make_split_occ(grp, o):
                                                        def on_click():
                                                            if not check_mutation_allowed():
                                                                return
                                                            split_occurrence_to_new_group(state, grp, o)
                                                            compute_reactive_preview(state)
                                                            refresh_preview_and_exports()
                                                            build_review_table()
                                                            ui.notify(f"Fundstelle als eigene Gruppe ausgegliedert.", type="info", icon="call_split")
                                                        return on_click

                                                    split_btn = ui.button("✂️ Ausgliedern", on_click=make_split_occ(master, occ)).props("flat dense size=sm color=primary").tooltip("Als eigene Gruppe ausgliedern (Homonym-Behandlung)")
                                                    state.register_mutating_element(split_btn, "table")

                            # Render Indented Linked Children
                            for child_node in root_node.children:
                                child: EntityGroup = child_node.item
                                with ui.expansion().classes("w-full ml-6 my-1 bg-teal-50/50 border-l-4 border-teal-500 border rounded shadow-none") as child_exp:
                                    all_expansions.append(child_exp)
                                    child_exp.on_value_change(lambda _: update_expand_btn())
                                    with child_exp.add_slot("header"):
                                        with ui.row().classes("w-full items-center justify-between gap-3 pr-2 flex-wrap"):
                                            with ui.row().classes("items-center gap-2 min-w-[260px]"):
                                                ui.label("↳").classes("text-teal-700 font-bold text-base")
                                                child_check = ui.checkbox(value=child.enabled, on_change=make_group_check(child)).props("dense")
                                                state.register_mutating_element(child_check, "table")
                                                if not is_category_enabled(state, child.entity_type):
                                                    child_check.disable()
                                                ui.label(child.original_text).classes("font-mono text-sm font-bold text-teal-900")
                                                ui.badge(f"{child.count}x", color="teal-6").props("dense")
                                                c_badge = ui.badge(child.placeholder, color="teal-9").props("outline dense").classes("text-xs font-mono font-bold")
                                                child_badge_refs.append((child, c_badge))

                                            # Surface Form Selector
                                            with ui.row().classes("items-center gap-1"):
                                                def make_surface_change(c_grp):
                                                    def on_change(e):
                                                        if not check_mutation_allowed():
                                                            return
                                                        c_grp.surface_tag = e.value
                                                        refresh_preview_and_exports()
                                                        build_review_table()
                                                    return on_change

                                                surf_sel = ui.select(
                                                    options=SURFACE_TAG_OPTIONS,
                                                    value=child.surface_tag or "VORNAME",
                                                    label="Schreibweise:",
                                                    on_change=make_surface_change(child),
                                                ).props("dense outlined bg-white").classes("w-44 text-xs")
                                                state.register_mutating_element(surf_sel, "table")

                                            # Explicit UNLINK Action
                                            with ui.row().classes("items-center gap-2"):
                                                def make_unlink_action(c_grp):
                                                    def on_click():
                                                        if not check_mutation_allowed():
                                                            return
                                                        c_grp.parent_group_id = None
                                                        c_grp.surface_tag = ""
                                                        ui.notify(f"'{c_grp.original_text}' getrennt und als eigenständige Entität gesetzt.", type="info")
                                                        refresh_preview_and_exports()
                                                        build_review_table()
                                                    return on_click

                                                unlink_btn = ui.button(
                                                    "✕ Trennen",
                                                    icon="link_off",
                                                    color="negative",
                                                    on_click=make_unlink_action(child),
                                                ).props("flat dense size=sm").tooltip("Verknüpfung aufheben und als separate Entität führen")
                                                state.register_mutating_element(unlink_btn, "table")

                                    # Child Occurrences
                                    with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                                        ui.label(f"Fundstellen von '{child.original_text}' ({child.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                                        for occ_idx, occ in enumerate(child.occurrences, start=1):
                                            with ui.row().classes("items-center justify-between gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                                with ui.row().classes("items-center gap-2 flex-1"):
                                                    ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                                    ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                                    method_label, method_color, method_tooltip = method_display(occ.method)
                                                    ui.badge(method_label, color=method_color).props("dense outline").tooltip(method_tooltip)
                                                    ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

                                                    # Occurrence-bound LLM Action Badge for child
                                                    if occ.occ_id in state.llm_triage_results:
                                                        c_occ_v = state.llm_triage_results[occ.occ_id]
                                                        if c_occ_v.action == "recategorize" or (c_occ_v.action == "keep" and getattr(c_occ_v, "descriptor_suggestion", None)):
                                                            b = ui.badge("LLM: Änderungsvorschlag", color="amber-8").props("dense").classes("cursor-pointer")
                                                            b.tooltip("Klicken, um zum LLM-Vorschlag zu springen")
                                                            b.on("click", lambda _, oid=occ.occ_id: ui.run_javascript(f"document.getElementById('llm_card_{oid}')?.scrollIntoView({{behavior: 'smooth'}});"))
                                                        elif c_occ_v.action == "discard":
                                                            b = ui.badge("LLM: Ignorieren vorgeschlagen", color="grey-8").props("dense").classes("cursor-pointer")
                                                            b.tooltip("Klicken, um zum LLM-Vorschlag zu springen")
                                                            b.on("click", lambda _, oid=occ.occ_id: ui.run_javascript(f"document.getElementById('llm_card_{oid}')?.scrollIntoView({{behavior: 'smooth'}});"))

                                                with ui.row().classes("items-center gap-1 shrink-0"):
                                                    is_split_occ = occ.occ_id in state.occurrence_overrides or child.group_id != child.text_key
                                                    if is_split_occ:
                                                        def make_revert_child_occ(grp, o):
                                                            def on_click():
                                                                if not check_mutation_allowed():
                                                                    return
                                                                revert_occurrence_to_base(state, grp, o)
                                                                compute_reactive_preview(state)
                                                                refresh_preview_and_exports()
                                                                build_review_table()
                                                                ui.notify(f"Fundstelle wieder mit der Hauptgruppe '{grp.original_text}' zusammengeführt.", type="info", icon="merge")
                                                            return on_click

                                                        rev_child_btn = ui.button("↩️ Zusammenführen", on_click=make_revert_child_occ(child, occ)).props("flat dense size=sm color=teal-8").tooltip("Ausgliederung rückgängig machen und mit Basisgruppe vereinen")
                                                        state.register_mutating_element(rev_child_btn, "table")
                                                    elif child.count > 1:
                                                        def make_split_child_occ(grp, o):
                                                            def on_click():
                                                                if not check_mutation_allowed():
                                                                    return
                                                                split_occurrence_to_new_group(state, grp, o)
                                                                compute_reactive_preview(state)
                                                                refresh_preview_and_exports()
                                                                build_review_table()
                                                                ui.notify(f"Fundstelle als eigene Gruppe ausgegliedert.", type="info", icon="call_split")
                                                            return on_click

                                                        split_child_btn = ui.button("✂️ Ausgliedern", on_click=make_split_child_occ(child, occ)).props("flat dense size=sm color=primary").tooltip("Als eigene Gruppe ausgliedern (Homonym-Behandlung)")
                                                        state.register_mutating_element(split_child_btn, "table")

    # --- Main Layout ---
    with ui.row().classes("w-full no-wrap p-4 gap-6"):
        # Sidebar: Configuration
        with ui.card().classes("w-80 p-4 shrink-0 bg-slate-50 border shadow-sm"):
            ui.label("⚙️ Konfiguration").classes("text-base font-bold text-slate-800 mb-3")

            # Format Mode Selector
            ui.label("Platzhalter-Format:").classes("text-xs font-semibold text-slate-700 mb-1")
            def on_mode_change(e):
                if not check_mutation_allowed():
                    return
                state.format_mode = e.value
                save_current_config(state)
                refresh_preview_and_exports()
                build_review_table()

            mode_radio = ui.radio(
                {
                    "numbered": "Modus 1: [TYP_NR] (Klassisch)",
                    "numbered_role": "Modus 2: [TYP_NR_ROLLE] (Empfohlen)",
                    "role_only": "Modus 3: [TYP_ROLLE] (Kompakt)",
                },
                value=state.format_mode,
                on_change=on_mode_change,
            ).props("dense").classes("text-xs mb-3")
            state.register_mutating_element(mode_radio, "sidebar")

            # Export Format Choice (.txt vs .md)
            ui.separator().classes("my-2")
            ui.label("Standard Export-Format:").classes("text-xs font-semibold text-slate-700 mb-1")
            def on_export_fmt_change(e):
                if not check_mutation_allowed():
                    return
                state.export_format = e.value
                save_current_config(state)
                refresh_preview_and_exports()

            export_fmt_radio = ui.radio(
                {
                    "txt": ".txt (Reiner Text)",
                    "md": ".md (Markdown-Formatierung)",
                },
                value=state.export_format,
                on_change=on_export_fmt_change,
            ).props("dense inline").classes("text-xs mb-3")
            state.register_mutating_element(export_fmt_radio, "sidebar")

            ui.separator().classes("my-2")

            ui.label("Erkennung je Entitätstyp:").classes("text-xs font-semibold text-slate-700 mb-1")
            ui.label("Aus = Kategorie vollständig aus. Nur Glossar & manuell = keine automatische Erkennung. Alle Quellen = KI, Regex, Bibliothek und explizite Einträge.").classes("text-[11px] text-slate-500 mb-2")
            for ent in AVAILABLE_ENTITIES:
                with ui.row().classes("w-full items-center justify-between gap-1"):
                    entity_label = ui.label(ent).classes("text-xs font-mono text-slate-700")
                    if ent == "ROLE":
                        entity_label.tooltip(
                            "Optionale Erkennung von Funktionsbezeichnungen wie CEO, CFO oder Leiter Prozessmanagement. "
                            "Standardmässig aus, weil Rollen häufig fachlich erhalten bleiben sollen."
                        )

                    def make_entity_mode_change(e_name):
                        selector_ref: List[Any] = []

                        def change_mode(e):
                            if not check_mutation_allowed():
                                return
                            mode = e.value or ENTITY_MODE_OFF
                            state.entity_modes[e_name] = mode
                            state.active_entities = [
                                entity for entity, m in state.entity_modes.items()
                                if m in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
                            ]
                            if selector_ref:
                                selector_ref[0].classes(replace=entity_mode_classes(mode))
                            save_current_config(state)
                            # Existing groups are re-filtered immediately by source; a new analysis
                            # is still needed when automatic detection has just been enabled.
                            compute_reactive_preview(state)
                            refresh_preview_and_exports()
                            if state.entity_groups:
                                build_review_table()
                                if reanalysis_warning_card is not None:
                                    reanalysis_warning_card.set_visibility(True)
                        return change_mode, selector_ref

                    mode_change, selector_ref = make_entity_mode_change(ent)
                    mode_select = ui.select(
                        options=get_entity_mode_options(ent),
                        value=state.entity_modes.get(ent, ENTITY_MODE_OFF),
                        on_change=mode_change,
                    ).props("dense outlined options-dense").classes(
                        entity_mode_classes(state.entity_modes.get(ent, ENTITY_MODE_OFF))
                    )
                    selector_ref.append(mode_select)
                    sidebar_entity_mode_selects[ent] = mode_select
                    state.register_mutating_element(mode_select, "sidebar")

            ui.separator().classes("my-2")

            ui.label("Erkennungs-Schwellenwert (GLiNER):").classes("text-xs font-semibold text-slate-700")
            def on_thresh_change(e):
                if not check_mutation_allowed():
                    return
                state.gliner_threshold = e.value
                save_current_config(state)
                if state.entity_groups and reanalysis_warning_card is not None:
                    reanalysis_warning_card.set_visibility(True)
            thresh_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.gliner_threshold, on_change=on_thresh_change)
            state.register_mutating_element(thresh_slider, "sidebar")
            ui.label().bind_text_from(thresh_slider, "value", lambda v: f"GLiNER Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

            ui.separator().classes("my-2")

            # EU-PII Multilingual Model Toggle & Threshold
            ui.label("EU-PII Multilingual Modell:").classes("text-xs font-semibold text-slate-700")
            with ui.row().classes("w-full items-center justify-between gap-1 mb-1"):
                eupii_spinner = ui.spinner(size="xs", color="primary")
                eupii_spinner.set_visibility(False)

                async def on_eupii_toggle(e):
                    if not check_mutation_allowed():
                        return
                    target_val = bool(e.value)
                    if target_val:
                        # Check if model is already available in local cache
                        is_cached = is_model_cached(state.eupii_model_name, "transformers")
                        if not is_cached:
                            confirmed = False
                            with ui.dialog() as dlg, ui.card().classes("p-5 max-w-lg bg-white rounded-xl shadow-xl"):
                                ui.label("Einmaliger Modell-Download erforderlich").classes("text-base font-bold text-slate-800 mb-1")
                                ui.markdown(
                                    f"Das spezialisierte EU-PII-Modell (`{state.eupii_model_name}`) ist noch nicht lokal gespeichert.\n\n"
                                    f"- **Downloadgrösse:** ca. **1.07 GB** (einmalig)\n"
                                    f"- **Speicherort:** Lokaler HuggingFace-Cache (`~/.cache/huggingface`)\n"
                                    f"- **Datenschutz:** Nach dem Download arbeitet das Modell zu **100% lokal und offline**.\n\n"
                                    f"Möchtest du das Modell jetzt herunterladen?"
                                ).classes("text-xs text-slate-600 leading-relaxed mb-3")
                                with ui.row().classes("w-full justify-end gap-2"):
                                    def on_cancel():
                                        dlg.close()
                                    def on_confirm():
                                        nonlocal confirmed
                                        confirmed = True
                                        dlg.close()
                                    ui.button("Abbrechen", on_click=on_cancel).props("flat text-color=slate")
                                    ui.button("Jetzt herunterladen (1.07 GB)", icon="cloud_download", on_click=on_confirm, color="primary").props("unelevated")

                            await dlg
                            if not confirmed:
                                eupii_switch.value = False
                                return

                        eupii_spinner.set_visibility(True)
                        eupii_switch.disable()
                        try:
                            def load_eupii_model():
                                with _model_lock:
                                    anon = get_synced_cached_anonymizer(state)
                                    anon.set_eupii_enabled(True, state.eupii_threshold)
                                    anon.eupii_recognizer.load()
                            await asyncio.to_thread(load_eupii_model)
                            state.enable_eupii = True
                            save_current_config(state)
                            ui.notify("EU-PII Modell einsatzbereit und aktiviert.", type="positive")
                        except Exception as ex:
                            state.enable_eupii = False
                            eupii_switch.value = False
                            with _model_lock:
                                if _cached_anonymizer is not None:
                                    _cached_anonymizer.set_eupii_enabled(False)
                            save_current_config(state)
                            ui.notify(f"Fehler beim Laden des EU-PII Modells: {ex}", type="negative", timeout=12000, close_button=True)
                        finally:
                            eupii_spinner.set_visibility(False)
                            eupii_switch.enable()
                    else:
                        state.enable_eupii = False
                        with _model_lock:
                            if _cached_anonymizer is not None:
                                _cached_anonymizer.set_eupii_enabled(False)
                        save_current_config(state)
                        ui.notify("EU-PII Modell deaktiviert.", type="info")

                    if state.entity_groups and reanalysis_warning_card is not None:
                        reanalysis_warning_card.set_visibility(True)

                eupii_switch = ui.switch("EU-PII Modell aktivieren", value=state.enable_eupii, on_change=on_eupii_toggle).props("dense").classes("text-xs")
                eupii_switch.tooltip("Aktiviert das spezialisierte Token-Klassifikationsmodell bardsai/eu-pii für erweiterte Erkennung von Personen, Orten, IDs und Gesundheitsdaten.")
                state.register_mutating_element(eupii_switch, "sidebar")

            def on_eupii_thresh_change(e):
                if not check_mutation_allowed():
                    return
                state.eupii_threshold = e.value
                save_current_config(state)
                with _model_lock:
                    if _cached_anonymizer is not None:
                        _cached_anonymizer.set_eupii_enabled(state.enable_eupii, state.eupii_threshold)
                if state.entity_groups and reanalysis_warning_card is not None:
                    reanalysis_warning_card.set_visibility(True)

            eupii_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.eupii_threshold, on_change=on_eupii_thresh_change)
            state.register_mutating_element(eupii_slider, "sidebar")
            ui.label().bind_text_from(eupii_slider, "value", lambda v: f"EU-PII Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

            ui.separator().classes("my-2")

            # Interactive Ignore List (Alphabetical with [x] delete buttons, high-contrast)
            with ui.expansion("Ignore-Liste (Nicht ersetzen)", icon="visibility_off").classes("w-full text-xs"):
                ignore_container = ui.column().classes("w-full")
                render_ignore_list_ui()

            # Interactive Glossary / Word List (Alphabetical with [x] delete buttons, high-contrast)
            with ui.expansion("Eigene Begriffsliste (Wortliste & Namen)", icon="library_books").classes("w-full text-xs mt-1"):
                ui.label("Begriffe oder Namen, die bei jeder Analyse immer zuverlässig erkannt werden sollen:").classes("text-[11px] text-slate-500 mb-1")
                glossary_container = ui.column().classes("w-full")
                render_glossary_list_ui()

    # Phase 5b.1 profile controls live at the top of the workspace.  The
    # document overlay remains transient and is deliberately reset on a
    # project switch.
    def profile_terms_text(terms: Mapping[str, ScopedTerm], include_role: bool = False) -> str:
        rows = []
        for term in terms.values():
            suffix = f" | {term.role}" if include_role and term.role else ""
            rows.append(f"{term.term}: {term.entity_type}{suffix}" if term.entity_type else term.term)
        return "\n".join(rows)

    def refresh_profile_select_options(select: Any) -> None:
        options = {p.project_id: p.project_name for p in state.profile_store.list_projects()}
        select.options = options
        select.update()

    def switch_project(project_id: str, confirmed_discard: bool = False) -> None:
        if not check_mutation_allowed():
            return
        if state.document_overlay.dirty and not confirmed_discard:
            ask_discard_document_overlay(lambda: switch_project(project_id, confirmed_discard=True))
            return
        try:
            controller = _profile_controller()
            project = controller.switch_project(project_id, discard_overlay=confirmed_discard)
            _sync_profile_controller()
            refresh_profile_select_options(project_select)
            project_select.value = project.project_id
            if state.entity_groups:
                refresh_preview_and_exports()
                build_review_table()
            ui.notify(f"Projekt „{project.project_name}“ aktiviert.", type="positive")
        except Exception as ex:
            ui.notify(f"Projektwechsel nicht möglich: {ex}", type="negative")

    def open_new_project_dialog() -> None:
        if not check_mutation_allowed():
            return
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Neues Projekt").classes("text-lg font-bold")
            name_input = ui.input("Projektname").props("autofocus outlined dense").classes("w-80")
            with ui.row().classes("justify-end w-full mt-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def create_project() -> None:
                    try:
                        created: Dict[str, Any] = {}
                        def action() -> None:
                            created["project"] = state.profile_store.create_project(name_input.value or "")
                            dialog.close()
                            switch_project(created["project"].project_id)
                        _run_durable_profile_action(action)
                    except Exception as ex:
                        ui.notify(f"Projekt konnte nicht angelegt werden: {ex}", type="negative")
                ui.button("Anlegen", on_click=create_project, color="primary").props("unelevated")
        dialog.open()

    def open_rename_project_dialog() -> None:
        if not check_mutation_allowed() or state.project_profile is None:
            return
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Projekt umbenennen").classes("text-lg font-bold")
            name_input = ui.input("Projektname", value=state.project_profile.project_name).props("autofocus outlined dense").classes("w-80")
            with ui.row().classes("justify-end w-full mt-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def rename_project() -> None:
                    try:
                        controller = _profile_controller()
                        def action() -> None:
                            controller.project_profile.project_name = (name_input.value or "").strip()
                            controller.save_project(expected_revision=controller.project_profile.revision)
                            _sync_profile_controller()
                            dialog.close()
                            refresh_profile_select_options(project_select)
                            ui.notify("Projekt umbenannt.", type="positive")
                        _run_durable_profile_action(action)
                    except Exception as ex:
                        ui.notify(f"Projekt konnte nicht umbenannt werden: {ex}", type="negative")
                ui.button("Speichern", on_click=rename_project, color="primary").props("unelevated")
        dialog.open()

    def delete_active_project(confirmed: bool = False) -> None:
        if not check_mutation_allowed() or state.project_profile is None:
            return
        if not confirmed:
            with ui.dialog() as delete_dialog, ui.card().classes("p-4"):
                ui.label("Projekt löschen?").classes("text-lg font-bold")
                ui.label(f"„{state.project_profile.project_name}“ wird dauerhaft gelöscht. Das Standardprojekt bleibt erhalten.").classes("text-sm text-slate-700")
                with ui.row().classes("justify-end w-full mt-3 gap-2"):
                    ui.button("Abbrechen", on_click=delete_dialog.close).props("flat")
                    ui.button("Dauerhaft löschen", on_click=lambda: (delete_dialog.close(), delete_active_project(True)), color="negative").props("unelevated")
            delete_dialog.open()
            return
        try:
            controller = _profile_controller()
            controller.delete_project(expected_revision=controller.project_profile.revision)
            manifest = state.profile_store.load_manifest()
            state.project_profile = state.profile_store.load_project_profile(manifest["active_project_id"])
            state.document_overlay = DocumentProfileOverlay()
            if state.profile_controller is not None:
                state.profile_controller.reload(active_project_id=manifest["active_project_id"])
            state.refresh_effective_config()
            state.preview_stale = bool(state.entity_groups)
            refresh_profile_select_options(project_select)
            project_select.value = state.project_profile.project_id
            ui.notify("Projekt gelöscht; das Standardprojekt ist aktiv.", type="positive")
        except RevisionConflictError:
            if state.profile_controller is not None:
                state.profile_controller.reload()
                state.system_profile = state.profile_controller.system_profile
                state.project_profile = state.profile_controller.project_profile
                state.document_overlay = state.profile_controller.document_overlay
                state.refresh_effective_config()
            ui.notify("Löschen abgebrochen: Das Projekt wurde extern geändert und neu geladen.", type="warning", close_button=True)
        except Exception as ex:
            ui.notify(f"Projekt konnte nicht gelöscht werden: {ex}", type="negative")

    def _template_is_custom(template_id: Optional[str]) -> bool:
        return bool(template_id) and template_id not in {t.template_id for t in get_builtin_templates(set(AVAILABLE_ENTITIES)).values()}

    def _refresh_template_select() -> None:
        templates = state.profile_store.list_all_templates()
        template_select.options = {
            tpl.template_id: (f"{tpl.name} · eingebaut" if tpl.is_builtin else f"{tpl.name} · eigene")
            for tpl in templates
        }
        template_select.update()
        if state.project_profile is not None and state.project_profile.template_id:
            if state.profile_store.load_template(state.project_profile.template_id) is None:
                template_reference_label.set_text("⚠️ Früher verwendete Vorlage nicht verfügbar · Modi bleiben als Snapshot erhalten")
                template_reference_label.set_visibility(True)
            else:
                template_reference_label.set_visibility(False)

    def open_save_template_dialog() -> None:
        if not check_mutation_allowed():
            return
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Eigene Vorlage speichern").classes("text-lg font-bold")
            name_input = ui.input("Vorlagenname").props("autofocus outlined dense").classes("w-80")
            description_input = ui.input("Beschreibung (optional)").props("outlined dense").classes("w-80")
            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def save_template() -> None:
                    try:
                        template = CategoryTemplate(
                            template_id=str(uuid.uuid4()),
                            name=(name_input.value or "").strip(),
                            description=(description_input.value or "").strip(),
                            entity_modes=dict(state.entity_modes),
                        )
                        def action() -> None:
                            _profile_controller().save_custom_template(template)
                            _refresh_template_select()
                            template_select.value = template.template_id
                            ui.notify("Eigene Vorlage gespeichert.", type="positive")
                        _run_durable_profile_action(action)
                        dialog.close()
                    except Exception as ex:
                        ui.notify(f"Vorlage konnte nicht gespeichert werden: {ex}", type="negative")
                ui.button("Speichern", on_click=save_template, color="primary").props("unelevated")
        dialog.open()

    def update_selected_template() -> None:
        template_id = template_select.value
        if not _template_is_custom(template_id) or not check_mutation_allowed():
            ui.notify("Bitte zuerst eine eigene Vorlage auswählen.", type="info")
            return
        current = state.profile_store.load_template(template_id)
        if current is None:
            ui.notify("Diese Vorlage ist nicht mehr verfügbar.", type="warning")
            _refresh_template_select()
            return
        def action() -> None:
            current.entity_modes = dict(state.entity_modes)
            _profile_controller().save_custom_template(current, expected_revision=current.revision)
            _refresh_template_select()
            ui.notify("Eigene Vorlage aktualisiert.", type="positive")
        _run_durable_profile_action(action)

    def open_rename_template_dialog() -> None:
        template_id = template_select.value
        if not _template_is_custom(template_id) or not check_mutation_allowed():
            ui.notify("Bitte zuerst eine eigene Vorlage auswählen.", type="info")
            return
        current = state.profile_store.load_template(template_id)
        if current is None:
            ui.notify("Diese Vorlage ist nicht mehr verfügbar.", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Eigene Vorlage umbenennen").classes("text-lg font-bold")
            name_input = ui.input("Vorlagenname", value=current.name).props("autofocus outlined dense").classes("w-80")
            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def rename_template() -> None:
                    def action() -> None:
                        current.name = (name_input.value or "").strip()
                        _profile_controller().save_custom_template(current, expected_revision=current.revision)
                        _refresh_template_select()
                        ui.notify("Eigene Vorlage umbenannt.", type="positive")
                    _run_durable_profile_action(action)
                    dialog.close()
                ui.button("Umbenennen", on_click=rename_template, color="primary").props("unelevated")
        dialog.open()

    def delete_selected_template() -> None:
        template_id = template_select.value
        if not _template_is_custom(template_id) or not check_mutation_allowed():
            ui.notify("Bitte zuerst eine eigene Vorlage auswählen.", type="info")
            return
        selected_template = state.profile_store.load_template(template_id)
        if selected_template is None:
            ui.notify("Diese Vorlage ist nicht mehr verfügbar.", type="warning")
            _refresh_template_select()
            return
        expected_template_revision = selected_template.revision
        with ui.dialog() as dialog, ui.card().classes("p-4"):
            ui.label("Eigene Vorlage löschen?").classes("text-lg font-bold")
            ui.label("Bereits angewendete Projektmodi bleiben unverändert; nur die Vorlage wird entfernt.").classes("text-sm text-slate-700")
            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def confirm_delete() -> None:
                    try:
                        _profile_controller().delete_custom_template(
                            template_id,
                            expected_revision=expected_template_revision,
                        )
                        dialog.close()
                        _refresh_template_select()
                        ui.notify("Eigene Vorlage gelöscht.", type="positive")
                    except RevisionConflictError:
                        _refresh_template_select()
                        ui.notify("Löschen abgebrochen: Die Vorlage wurde extern geändert. Die aktuelle Liste wurde neu geladen.", type="warning", close_button=True)
                    except Exception as ex:
                        ui.notify(f"Vorlage konnte nicht gelöscht werden: {ex}", type="negative")
                ui.button("Löschen", on_click=confirm_delete, color="negative").props("unelevated")
        dialog.open()

    def open_backup_cleanup_dialog() -> None:
        if not check_mutation_allowed():
            return
        backup_paths = list(state.profile_store.backups_dir.glob("config.v1.backup.*.json"))
        migrated_path = state.profile_store.root_dir / "config.v1.migrated.json"
        if migrated_path.exists():
            backup_paths.append(migrated_path)
        count = len(backup_paths)
        with ui.dialog() as dialog, ui.card().classes("p-4 max-w-lg"):
            ui.label("Alte Migrationsbackups löschen?").classes("text-lg font-bold")
            ui.label(f"{count} Datei(en) würden aus dem lokalen Backup-Ordner entfernt.").classes("text-sm text-slate-700")
            ui.label("Das ist keine sichere Löschung. Für eine vollständige Entfernung müssen zusätzlich Betriebssystem-Backups und Synchronisationskopien berücksichtigt werden.").classes("text-xs text-amber-800 mt-2")
            with ui.row().classes("justify-end w-full mt-3 gap-2"):
                ui.button("Abbrechen", on_click=dialog.close).props("flat")
                def confirm_cleanup() -> None:
                    try:
                        removed = state.profile_store.delete_migration_backups()
                        dialog.close()
                        ui.notify(f"{removed} Migrationsbackup(s) entfernt.", type="positive")
                    except Exception as ex:
                        ui.notify(f"Backups konnten nicht entfernt werden: {ex}", type="negative")
                ui.button("Entfernen", on_click=confirm_cleanup, color="negative").props("unelevated")
        dialog.open()

    def _profile_controller() -> ProfileController:
        """Return the controller bound to this session's current scope objects."""
        if state.profile_controller is None:
            state.profile_controller = ProfileController(state.profile_store)
        return state.profile_controller

    def _sync_profile_controller() -> None:
        """Copy the controller's authoritative result into the session state."""
        controller = _profile_controller()
        state.system_profile = controller.system_profile
        state.project_profile = controller.project_profile
        state.document_overlay = controller.document_overlay
        state.glossary_text = profile_terms_text(state.project_profile.glossary_terms, include_role=True)
        state.ignore_terms_text = ", ".join(term.term for term in state.project_profile.ignore_terms.values())
        state.refresh_effective_config()
        state.preview_stale = bool(state.entity_groups)
        for entity, selector in sidebar_entity_mode_selects.items():
            try:
                mode = state.entity_modes.get(entity, ENTITY_MODE_OFF)
                selector.set_value(mode)
                selector.classes(replace=entity_mode_classes(mode))
            except Exception:
                pass

    def _run_durable_profile_action(
        action: Callable[[], None],
        system_mutation: bool = False,
        system_confirmed: bool = False,
        on_abort: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run a profile/template write only after the one-time local-storage warning."""
        controller = _profile_controller()

        def abort_action() -> None:
            if on_abort is not None:
                try:
                    on_abort()
                except Exception:
                    logging.exception("Profilaktion konnte nach Abbruch nicht zurückgesetzt werden")

        if system_mutation and not system_confirmed:
            with ui.dialog() as system_dialog, ui.card().classes("p-4 max-w-lg"):
                ui.label("Systemweite Änderung bestätigen").classes("text-lg font-bold")
                ui.label("Diese Glossar-/Ignore-Änderung gilt für alle Projekte und kann dort die wirksame Konfiguration beeinflussen.").classes("text-sm text-slate-700")
                with ui.row().classes("w-full justify-end gap-2 mt-3"):
                    ui.button("Abbrechen", on_click=lambda: (system_dialog.close(), abort_action())).props("flat")
                    ui.button(
                        "Systemweit anwenden",
                        on_click=lambda: (
                            system_dialog.close(),
                            _run_durable_profile_action(
                                action,
                                True,
                                True,
                                on_abort=on_abort,
                            ),
                        ),
                        color="primary",
                    ).props("unelevated")
            system_dialog.open()
            return
        if not controller.warning_required():
            try:
                action()
            except ProfileMutationAborted as ex:
                abort_action()
                ui.notify(str(ex), type="warning", close_button=True)
            except RevisionConflictError:
                overlay = state.document_overlay
                _reload_profile_controller_preserving_overlay(
                    controller,
                    overlay,
                    active_project_id=state.project_profile.project_id if state.project_profile else None,
                )
                _sync_profile_controller()
                abort_action()
                ui.notify("Speichern fehlgeschlagen: Das Profil wurde extern geändert und neu geladen.", type="warning", close_button=True)
            except Exception as ex:
                abort_action()
                ui.notify(f"Profiländerung konnte nicht gespeichert werden: {ex}", type="negative")
            return

        with ui.dialog() as warning_dialog, ui.card().classes("p-5 max-w-xl bg-white rounded-xl shadow-xl"):
            ui.label("⚠️ Wichtiger Hinweis zur lokalen Datenspeicherung").classes("text-lg font-bold text-slate-800")
            ui.markdown(
                "Profile, Begriffe, Rollen und Ignore-Einträge werden lokal im Benutzerordner gespeichert, "
                "standardmässig unverschlüsselt. BitLocker/FileVault schützt nur das Gerät im ausgeschalteten "
                "Zustand; bei entsperrter Sitzung können andere Programme die Dateien lesen. Eine Cloud-Synchronisation "
                "kann Kopien ausserhalb dieses Geräts erzeugen."
            ).classes("text-sm text-slate-700 leading-relaxed mt-2")
            understood = ui.checkbox("Ich habe diesen Hinweis verstanden und möchte fortfahren.")
            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("Abbrechen", on_click=lambda: (warning_dialog.close(), abort_action())).props("flat")
                def confirm_profile_warning() -> None:
                    if not understood.value:
                        ui.notify("Bitte bestätige zuerst den Datenschutzhinweis.", type="warning")
                        return
                    try:
                        controller.acknowledge_warning(expected_revision=int(controller.manifest["revision"]))
                        _run_durable_profile_action(
                            action,
                            system_mutation=system_mutation,
                            system_confirmed=True,
                            on_abort=on_abort,
                        )
                        warning_dialog.close()
                    except RevisionConflictError:
                        overlay = state.document_overlay
                        _reload_profile_controller_preserving_overlay(
                            controller,
                            overlay,
                            active_project_id=state.project_profile.project_id if state.project_profile else None,
                        )
                        _sync_profile_controller()
                        abort_action()
                        warning_dialog.close()
                        ui.notify("Speichern abgebrochen: Der Profilstand wurde extern geändert und neu geladen.", type="warning")
                    except Exception as ex:
                        abort_action()
                        warning_dialog.close()
                        ui.notify(f"Profiländerung konnte nicht gespeichert werden: {ex}", type="negative")
                ui.button("Bestätigen & Fortfahren", on_click=confirm_profile_warning, color="primary").props("unelevated")
        warning_dialog.open()

    def open_profile_manager() -> None:
        if not check_mutation_allowed():
            return
        controller = _profile_controller()
        with ui.dialog() as dialog, ui.card().classes("w-[1000px] max-w-full p-4"):
            ui.label("Profile & Begriffe verwalten").classes("text-lg font-bold text-slate-800")
            ui.label("Systemänderungen gelten global, Projektänderungen dauerhaft für das Projekt, Dokumentänderungen nur für diese Sitzung.").classes("text-xs text-slate-600 mb-2")
            scope_select = ui.select(
                {"document": "Dokument (flüchtig)", "project": "Projekt (dauerhaft)", "system": "System (dauerhaft, global)"},
                value="project",
                label="Bearbeitungsebene",
            ).props("dense outlined").classes("w-72")
            with ui.tabs().classes("w-full") as profile_tabs:
                categories_tab = ui.tab("Kategorien")
                glossary_tab = ui.tab("Glossar")
                ignore_tab = ui.tab("Ignore-Liste")
            category_holder = None
            glossary_holder = None
            ignore_holder = None

            def mutate(
                scope: str,
                action: Callable[[], None],
                after_success: Optional[Callable[[], None]] = None,
                expected_project_id: Optional[str] = None,
            ) -> None:
                if not check_mutation_allowed():
                    return
                controller = _profile_controller()
                if expected_project_id and (
                    state.project_profile is None
                    or state.project_profile.project_id != expected_project_id
                ):
                    ui.notify("Änderung abgebrochen: Das aktive Projekt wurde inzwischen gewechselt.", type="warning")
                    return
                if scope == "document":
                    action()
                    _sync_profile_controller()
                    render_all()
                    if after_success:
                        after_success()
                    return
                def durable_action() -> None:
                    if not check_mutation_allowed():
                        raise ProfileMutationAborted("Änderung abgebrochen: Die Anwendung ist inzwischen beschäftigt.")
                    if expected_project_id and (
                        state.project_profile is None
                        or state.project_profile.project_id != expected_project_id
                    ):
                        raise ProfileMutationAborted("Änderung abgebrochen: Das aktive Projekt wurde inzwischen gewechselt.")
                    if scope == "system":
                        controller.run_system_mutation(action, confirmed=True)
                        controller.save_system(expected_revision=controller.system_profile.revision)
                    else:
                        action()
                        controller.save_project(expected_revision=controller.project_profile.revision)
                    _sync_profile_controller()
                    render_all()
                    if after_success:
                        after_success()
                _run_durable_profile_action(durable_action, system_mutation=(scope == "system"))

            def render_categories() -> None:
                category_holder.clear()
                cfg = controller.effective_config()
                scope = scope_select.value or "project"
                with category_holder:
                    for entity in AVAILABLE_ENTITIES:
                        with ui.row().classes("w-full items-center gap-2 mb-1"):
                            ui.label(entity).classes("font-mono text-xs w-44")
                            if scope == "system":
                                ui.badge("System: nur Begriffe", color="grey-7").props("dense")
                                continue
                            selector = ui.select(get_entity_mode_options(entity), value=cfg.entity_modes.get(entity, ENTITY_MODE_OFF)).props("dense outlined").classes("w-80 text-xs")
                            def on_mode(event: Any, ent: str = entity, selected: Any = selector) -> None:
                                mode_value = event.value
                                selected.set_value(mode_value)
                                selected_scope = str(scope_select.value or "project")
                                selected_project_id = state.project_profile.project_id if state.project_profile else None
                                mutate(
                                    selected_scope,
                                    lambda: controller.set_entity_mode(selected_scope, ent, mode_value),
                                    expected_project_id=selected_project_id,
                                )
                            selector.on_value_change(on_mode)
                            state.register_mutating_element(selector, "sidebar")

            def render_terms() -> None:
                glossary_holder.clear()
                ignore_holder.clear()
                cfg = controller.effective_config()
                scope = scope_select.value or "project"
                with glossary_holder:
                    with ui.row().classes("w-full items-end gap-2 mb-2"):
                        g_term = ui.input("Begriff").props("dense outlined").classes("flex-grow")
                        g_type = ui.select({e: e for e in AVAILABLE_ENTITIES}, label="Kategorie").props("dense outlined").classes("w-48")
                        g_role = ui.input("Rolle (optional)").props("dense outlined").classes("w-44")
                        def add_glossary() -> None:
                            term_value = str(g_term.value or "").strip()
                            entity_value = str(g_type.value or "").strip()
                            role_value = str(g_role.value or "").strip()
                            selected_scope = str(scope_select.value or "project")
                            selected_project_id = state.project_profile.project_id if state.project_profile else None
                            if not term_value or not entity_value:
                                ui.notify("Begriff und Kategorie sind erforderlich.", type="warning")
                                return
                            mutate(
                                selected_scope,
                                lambda: controller.upsert_glossary(selected_scope, term_value, entity_value, role_value),
                                after_success=lambda: (setattr(g_term, "value", ""), setattr(g_role, "value", "")),
                                expected_project_id=selected_project_id,
                            )
                        ui.button("Hinzufügen", icon="add", on_click=add_glossary, color="primary").props("dense")
                    with ui.column().classes("w-full max-h-64 overflow-y-auto gap-1 pr-1"):
                        for key, value in sorted(cfg.glossary.items(), key=lambda item: item[0].casefold()):
                            provenance = cfg.glossary_provenance.get(normalize_term_key(key))
                            label = provenance[1] if provenance else "wirksam"
                            role = cfg.glossary_roles.get(key)
                            role_suffix = f" · Rolle: {role}" if role else ""
                            with ui.row().classes("w-full items-center gap-2 mb-1"):
                                ui.label(f"Begriff: {key} · Typ: {value}{role_suffix}").classes("font-mono text-xs flex-grow")
                                ui.badge(label, color="teal" if provenance and provenance[0] == ScopeLevel.PROJECT else "grey-7").props("dense")
                                source_scope = provenance[0] if provenance else ScopeLevel.PROJECT
                                selected_scope = ScopeLevel(scope)
                                if source_scope == selected_scope:
                                    ui.button("Löschen", on_click=lambda k=key: mutate(scope, lambda: controller.remove_term(scope, k, True))).props("dense flat color=negative")
                                elif selected_scope in (ScopeLevel.PROJECT, ScopeLevel.DOCUMENT):
                                    ui.button("Ausblenden", on_click=lambda k=key: mutate(scope, lambda: controller.disable_inherited(scope, k, True))).props("dense flat")
                    ui.label("Begriffe mit Badge «System» oder «Projekt» sind geerbt bzw. wirksam aus dieser Ebene.").classes("text-[11px] text-slate-500 mt-2")
                with ignore_holder:
                    with ui.row().classes("w-full items-end gap-2 mb-2"):
                        i_term = ui.input("Ignorierter Begriff").props("dense outlined").classes("flex-grow")
                        def add_ignore() -> None:
                            term_value = str(i_term.value or "").strip()
                            selected_scope = str(scope_select.value or "project")
                            selected_project_id = state.project_profile.project_id if state.project_profile else None
                            if not term_value:
                                return
                            mutate(
                                selected_scope,
                                lambda: controller.upsert_ignore(selected_scope, term_value),
                                after_success=lambda: setattr(i_term, "value", ""),
                                expected_project_id=selected_project_id,
                            )
                        ui.button("Hinzufügen", icon="add", on_click=add_ignore, color="primary").props("dense")
                    with ui.column().classes("w-full max-h-64 overflow-y-auto gap-1 pr-1"):
                        for key in sorted(cfg.ignore_provenance, key=str.casefold):
                            provenance = cfg.ignore_provenance.get(key)
                            with ui.row().classes("w-full items-center gap-2 mb-1"):
                                ui.label(key).classes("font-mono text-xs flex-grow")
                                ui.badge(provenance[1] if provenance else "wirksam", color="grey-7").props("dense")
                                source_scope = provenance[0] if provenance else ScopeLevel.PROJECT
                                selected_scope = ScopeLevel(scope)
                                if source_scope == selected_scope:
                                    ui.button("Löschen", on_click=lambda k=key: mutate(scope, lambda: controller.remove_term(scope, k, False))).props("dense flat color=negative")
                                elif selected_scope in (ScopeLevel.PROJECT, ScopeLevel.DOCUMENT):
                                    ui.button("Ausblenden", on_click=lambda k=key: mutate(scope, lambda: controller.disable_inherited(scope, k, False))).props("dense flat")

            def render_all() -> None:
                render_categories()
                render_terms()

            scope_select.on_value_change(lambda _: render_all())
            with ui.tab_panels(profile_tabs, value=categories_tab).classes("w-full"):
                with ui.tab_panel(categories_tab):
                    category_holder = ui.column().classes("w-full")
                with ui.tab_panel(glossary_tab):
                    glossary_holder = ui.column().classes("w-full")
                with ui.tab_panel(ignore_tab):
                    ignore_holder = ui.column().classes("w-full")
            with ui.row().classes("justify-end w-full mt-3"):
                ui.button("Schliessen", on_click=dialog.close).props("flat")
            render_all()
        dialog.open()

    with ui.card().classes("w-full mb-3 p-3 bg-white border border-slate-200 rounded-lg shadow-sm"):
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            ui.label("Projekt").classes("text-sm font-semibold text-slate-700")
            project_options = {p.project_id: p.project_name for p in state.profile_store.list_projects()}
            project_select = ui.select(
                options=project_options,
                value=(state.project_profile.project_id if state.project_profile else None),
                on_change=lambda event: switch_project(event.value),
            ).props("dense outlined").classes("min-w-48")
            state.register_mutating_element(project_select, "sidebar")
            new_project_btn = ui.button("➕ Neues Projekt", on_click=lambda: open_new_project_dialog()).props("outline dense")
            rename_project_btn = ui.button("✏️ Umbenennen", on_click=lambda: open_rename_project_dialog()).props("outline dense")
            delete_project_btn = ui.button("🗑️ Löschen", on_click=lambda: delete_active_project()).props("outline dense color=negative")
            for control in (new_project_btn, rename_project_btn, delete_project_btn):
                state.register_mutating_element(control, "sidebar")
            ui.separator().props("vertical")
            ui.label("Vorlage").classes("text-sm font-semibold text-slate-700")
            template_options = {
                tpl.template_id: (f"{tpl.name} · eingebaut" if tpl.is_builtin else f"{tpl.name} · eigene")
                for tpl in state.profile_store.list_all_templates()
            }

            def apply_template(template_id: str, confirmed: bool = False) -> None:
                if not template_id or not check_mutation_allowed() or state.project_profile is None:
                    return
                previous_template_id = state.project_profile.template_id

                def restore_template_selection() -> None:
                    template_select.value = previous_template_id
                    template_select.update()

                template = state.profile_store.load_template(template_id)
                if template is None:
                    restore_template_selection()
                    ui.notify("Diese Vorlage ist nicht mehr verfügbar; die Projektmodi bleiben unverändert.", type="warning")
                    return
                changed = dict(state.project_profile.entity_modes) != dict(template.entity_modes)
                if changed and not confirmed:
                    with ui.dialog() as confirm_dialog, ui.card().classes("p-4 max-w-lg"):
                        ui.label("Projektmodi überschreiben?").classes("text-lg font-bold")
                        ui.label("Die aktuelle Projektkonfiguration wird durch eine Kopie der Vorlage ersetzt. Eigene Begriffe bleiben erhalten.").classes("text-sm text-slate-700")
                        with ui.row().classes("justify-end w-full mt-3 gap-2"):
                            ui.button("Abbrechen", on_click=lambda: (confirm_dialog.close(), restore_template_selection())).props("flat")
                            ui.button("Anwenden", on_click=lambda: (confirm_dialog.close(), apply_template(template_id, True)), color="primary").props("unelevated")
                    confirm_dialog.open()
                    return
                try:
                    controller = _profile_controller()
                    def action() -> None:
                        controller.apply_template(template_id, expected_revision=controller.project_profile.revision)
                        _sync_profile_controller()
                        template_select.value = template_id
                        template_select.update()
                        ui.notify("Vorlage als Snapshot auf das Projekt angewendet.", type="positive")
                        if state.entity_groups:
                            state.preview_stale = True
                            if reanalysis_warning_card is not None:
                                reanalysis_warning_card.set_visibility(True)
                            refresh_preview_and_exports()
                    _run_durable_profile_action(action, on_abort=restore_template_selection)
                except Exception as ex:
                    restore_template_selection()
                    ui.notify(f"Vorlage konnte nicht angewendet werden: {ex}", type="negative")

            template_select = ui.select(options=template_options, on_change=lambda event: apply_template(event.value)).props("dense outlined").classes("min-w-64")
            state.register_mutating_element(template_select, "sidebar")
            template_reference_label = ui.label().classes("text-[11px] text-amber-700")
            template_reference_label.set_visibility(
                bool(state.project_profile and state.project_profile.template_id and state.profile_store.load_template(state.project_profile.template_id) is None)
            )
            template_save_btn = ui.button("💾 Als Vorlage speichern", on_click=lambda: open_save_template_dialog()).props("outline dense")
            template_rename_btn = ui.button("✏️ Vorlage umbenennen", on_click=lambda: open_rename_template_dialog()).props("outline dense")
            template_update_btn = ui.button("↻ Vorlage aktualisieren", on_click=lambda: update_selected_template()).props("outline dense")
            template_delete_btn = ui.button("🗑️ Eigene Vorlage löschen", on_click=lambda: delete_selected_template()).props("outline dense color=negative")
            for control in (template_save_btn, template_rename_btn, template_update_btn, template_delete_btn):
                state.register_mutating_element(control, "sidebar")
            profile_manager_btn = ui.button("⚙️ Profile & Begriffe verwalten", on_click=open_profile_manager).props("outline dense")
            state.register_mutating_element(profile_manager_btn, "sidebar")
            backup_cleanup_btn = ui.button("🧹 Migrationsbackups", on_click=open_backup_cleanup_dialog).props("outline dense")
            state.register_mutating_element(backup_cleanup_btn, "sidebar")
        with ui.row().classes("items-center gap-2 mt-2"):
            ui.badge("Projekt-Scope", color="teal").props("dense outline")
            ui.label("Projektänderungen dauerhaft · Dokumentoverlay nur für diese Sitzung").classes("text-xs text-slate-600")

    # Main Workspace
        with ui.column().classes("flex-grow"):
            with ui.tabs().classes("w-full border-b") as tabs:
                tab_anonymize = ui.tab("🔒 Anonymisieren & Review")
                tab_restore = ui.tab("🔄 Wiederherstellen (De-Anonymize)")
                tab_transparency = ui.tab("🔍 Erkennungslogik")

            _transparency_loaded = []  # mutable one-shot flag: lazy-load once, not on every page build

            def on_tab_change(e):
                """Auto-preload mapping into Tab 2 when user switches there after an analysis.
                Also lazy-loads Tab 3's transparency view on first visit -- it isn't rendered at
                page-build time to avoid ballooning the initial WebSocket payload for a view most
                sessions never open."""
                restore_tab_name = tab_restore._props.get("name", "")
                if e.value == restore_tab_name and state.current_mapping:
                    if map_json_input is not None and not map_json_input.value.strip():
                        state.restore_mapping = dict(state.current_mapping)
                        map_json_input.value = json.dumps(state.current_mapping, indent=2, ensure_ascii=False)
                        ui.notify(f"Mapping mit {len(state.current_mapping)} Einträgen aus aktueller Analyse vorgeladen.", type="info", icon="auto_awesome")

                transparency_tab_name = tab_transparency._props.get("name", "")
                if e.value == transparency_tab_name and not _transparency_loaded:
                    _transparency_loaded.append(True)
                    load_transparency_view()

            tabs.on_value_change(on_tab_change)


            with ui.tab_panels(tabs, value=tab_anonymize).classes("w-full p-4"):
                # TAB 1: Anonymize
                with ui.tab_panel(tab_anonymize):
                    ui.label("Stufe 1: Dokument laden & Text-Eingabe").classes("text-base font-bold text-slate-800 mb-1")
                    
                    # 100% Unified Modern Dropzone (Click anywhere to pick, or Drag & Drop anywhere)
                    with ui.card().classes(
                        "w-full py-7 px-4 bg-slate-50 hover:bg-slate-100 border-2 border-dashed border-blue-400 hover:border-blue-600 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer shadow-none transition-all duration-150 mb-2 select-none"
                    ) as dropzone_card:
                        drop_card_id = dropzone_card.id
                        dropzone_card.props(f'id="custom_dropzone_{drop_card_id}"')

                        with ui.row().classes("items-center gap-2 mb-1 pointer-events-none"):
                            ui.icon("cloud_upload", size="lg").classes("text-blue-600")
                            ui.label("Datei hier ablegen oder zum Auswählen klicken").classes("text-base font-bold text-slate-800")
                        ui.label("Word (.docx), PDF, Text (.txt, .md), CSV, JSON • Liest auch geöffnete Dateien fehlerfrei").classes("text-xs text-slate-500 pointer-events-none")

                        # Click opens Explorer
                        dropzone_card.on("click", open_native_file_dialog)

                        # Drop handler via HTTP streaming / path / base64
                        async def on_file_dropped(e):
                            data = e.args
                            logging.info(f"on_file_dropped received: {type(data)} -> {data}")
                            temp_paths = None
                            try:
                                raw_bytes, filename, temp_paths = extract_upload_payload(data, UPLOAD_DIR)
                                if not check_mutation_allowed():
                                    return
                                await extract_and_load_file_bytes(raw_bytes, filename)
                            except Exception as ex:
                                err_msg = f"{type(ex).__name__}: {str(ex)}"
                                logging.error(f"Drop error: {err_msg}", exc_info=True)
                                ui.notify(f"Fehler beim Laden: {err_msg}", type="negative", timeout=15000)
                            finally:
                                if temp_paths:
                                    cleanup_upload_paths(*temp_paths)

                        ui.on("file_dropped", on_file_dropped)
                        state.register_mutating_element(dropzone_card, "workspace")

                        # Hook HTML5 drag events with HTTP streaming upload
                        ui.add_body_html(f"""
                        <script>
                        (function() {{
                            function setupDropzone() {{
                                const cardId = {drop_card_id};
                                const el = document.getElementById('custom_dropzone_' + cardId);
                                if (!el || el._dropAttached) return;
                                el._dropAttached = true;

                                window.addEventListener('dragover', function(e) {{ e.preventDefault(); }}, false);
                                window.addEventListener('drop', function(e) {{ e.preventDefault(); }}, false);

                                el.addEventListener('dragenter', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    el.style.borderColor = '#1d4ed8';
                                    el.style.backgroundColor = '#dbeafe';
                                }}, false);

                                el.addEventListener('dragover', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    el.style.borderColor = '#1d4ed8';
                                    el.style.backgroundColor = '#dbeafe';
                                }}, false);

                                el.addEventListener('dragleave', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    el.style.borderColor = '#60a5fa';
                                    el.style.backgroundColor = '#f8fafc';
                                }}, false);

                                el.addEventListener('drop', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    el.style.borderColor = '#60a5fa';
                                    el.style.backgroundColor = '#f8fafc';

                                    const files = e.dataTransfer.files;
                                    if (!files || files.length === 0) return;
                                    const file = files[0];

                                    const formData = new FormData();
                                    formData.append('file', file);
                                    fetch('/api/upload', {{
                                        method: 'POST',
                                        body: formData
                                    }})
                                    .then(res => {{
                                        if (!res.ok) throw new Error('HTTP ' + res.status);
                                        return res.json();
                                    }})
                                    .then(data => {{
                                        emitEvent('file_dropped', {{ file_id: data.file_id, name: data.filename }});
                                    }})
                                    .catch(err => {{
                                        console.error('Upload failed:', err);
                                        const reader = new FileReader();
                                        reader.onload = function(evt) {{
                                            const b64 = evt.target.result.split(',')[1];
                                            emitEvent('file_dropped', {{ name: file.name, base64: b64 }});
                                        }};
                                        reader.readAsDataURL(file);
                                    }});
                                }}, false);
                            }}
                            if (document.readyState === 'loading') {{
                                document.addEventListener('DOMContentLoaded', setupDropzone);
                            }} else {{
                                setupDropzone();
                            }}
                            setInterval(setupDropzone, 500);
                        }})();
                        </script>
                        """)

                    # Info badge explaining digital text & image/OCR scope
                    with ui.row().classes("w-full items-center gap-1.5 px-3 py-2 bg-blue-50/70 border border-blue-200/80 rounded-lg text-xs text-slate-700 mb-2"):
                        ui.icon("info", size="xs").classes("text-blue-600 shrink-0")
                        ui.label("Hinweis: Extrahiert digitalen Text, Formatierungen, Listen & Tabellen aus Word (.docx), PDF, Text (.txt, .md), CSV und JSON. Bei PDFs werden auch Vektordiagramme, Formularfelder und bestehende PDF-Textschichten erfasst (daher oft mehr Text als in Word). Reine Screenshots/Bilder ohne Textschicht erfordern OCR (für eine spätere Phase geplant).").classes("flex-1 text-[11px] leading-relaxed")

                    # Extraction settings row with detailed rationale tooltip
                    async def on_extraction_opt_change():
                        if state.last_raw_bytes and state.filename:
                            await extract_and_load_file_bytes(state.last_raw_bytes, state.filename)

                    with ui.row().classes("w-full items-center justify-between gap-3 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg mb-3 flex-wrap text-xs text-slate-700"):
                        with ui.row().classes("items-center gap-1.5"):
                            ui.icon("tune", size="xs").classes("text-slate-500")
                            ui.label("Erweiterte Extraktions-Optionen:").classes("font-semibold text-slate-700")

                        with ui.row().classes("items-center gap-4 flex-wrap"):
                            async def on_toggle_headers_footers(e):
                                if not check_mutation_allowed():
                                    return
                                state.include_headers_footers = bool(e.value)
                                await on_extraction_opt_change()

                            async def on_toggle_picture_text(e):
                                if not check_mutation_allowed():
                                    return
                                state.extract_picture_text = bool(e.value)
                                await on_extraction_opt_change()

                            hf_checkbox = ui.checkbox(
                                "Kopf- und Fußzeilen einbeziehen",
                                value=state.include_headers_footers,
                                on_change=on_toggle_headers_footers,
                            ).props("dense size=sm").tooltip("Laufende Kopf- und Fußzeilen (z. B. Seitenzahlen, Dokumenttitel) mit einlesen. Standard: Aus.")
                            state.register_mutating_element(hf_checkbox, "workspace")

                            with ui.row().classes("items-center gap-1"):
                                pt_checkbox = ui.checkbox(
                                    "Text aus PDF-Bildern & Grafiken extrahieren",
                                    value=state.extract_picture_text,
                                    on_change=on_toggle_picture_text,
                                ).props("dense size=sm").tooltip("Liest Textboxen in Diagrammen & Vektorgrafiken aus.")
                                state.register_mutating_element(pt_checkbox, "workspace")
                                with ui.button(icon="help_outline").props("flat round dense size=xs color=grey").classes("p-0 min-h-0 min-w-0"):
                                    with ui.tooltip().classes("bg-slate-900 text-white text-xs max-w-md p-2.5 leading-relaxed shadow-lg rounded-lg"):
                                        ui.markdown(
                                            "**Warum teilweise ausschalten?**\n\n"
                                            "Lineare PDF-Parser lesen Text zeilenweise von links nach rechts über die gesamte Seite. "
                                            "Bei mehrspaltigen **2D-Organigrammen, Stammbäumen oder Flussdiagrammen** werden dadurch Boxen quer durcheinandergewürfelt und Vor- bzw. Nachnamen verschiedener Personen vertauscht. "
                                            "Deaktivieren Sie diese Option, um solche Diagramme sauber zu überspringen."
                                        )

                    # Live extraction progress card (for large PDF/Docx files)
                    with ui.card().classes("w-full p-3 bg-blue-50 border border-blue-200 rounded-xl mb-3 flex-col gap-1") as extraction_progress_card:
                        extraction_progress_card.set_visibility(False)
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm", color="primary")
                            extraction_progress_label = ui.label("Lese Datei ein...").classes("text-xs font-semibold text-slate-700")
                        extraction_progress_bar = ui.linear_progress(value=0.0, show_value=False).props("color=primary stripe rounded instant-feedback").classes("w-full")

                    # Document info badge when loaded (with remove button)
                    with ui.row().classes("w-full mb-2 items-center gap-2") as file_badge_card:
                        file_badge_card.set_visibility(False)
                        with ui.row().classes("items-center gap-2 bg-blue-100 border border-blue-300 rounded-lg px-3 py-1 text-xs text-blue-950"):
                            ui.icon("description", size="xs").classes("text-blue-700")
                            file_badge_label = ui.label("").classes("font-bold font-mono")
                            ui.button(icon="close", on_click=reset_workspace).props("flat round dense size=xs color=negative").classes("p-0 min-h-0 min-w-0 ml-1").tooltip("Geladenes Dokument entfernen")

                    with ui.expansion("Originaltext ansehen / direkt bearbeiten (Markdown)", icon="edit_note", value=True).classes("w-full mb-2"):
                        def on_raw_text_change(e):
                            if not check_mutation_allowed():
                                return
                            state.raw_text = e.value or ""
                            state.document_revision += 1
                            state.llm_triage_results.clear()
                            state.llm_triage_snapshot = ""
                            state.llm_staged_selections.clear()
                            if llm_panel_holder is not None:
                                build_llm_panel()
                            has_content = bool(state.raw_text and state.raw_text.strip())
                            if analyze_btn is not None:
                                analyze_btn.set_enabled(has_content)
                            if reset_btn is not None:
                                reset_btn.set_visibility(has_content)

                        raw_text_area = ui.textarea(
                            value=state.raw_text,
                            placeholder="Text hier eingeben oder Dokument oben hineinziehen...",
                            on_change=on_raw_text_change,
                        ).props("outlined rows=6").classes("w-full font-mono text-sm")

                    # Interactive LLM Setup & Preloading Panel (Top-Down Workflow step before Analysis)
                    llm_setup_holder = ui.column().classes("w-full mb-2")
                    build_llm_setup_panel()

                    # Re-analysis Warning Banner (appears when entities or settings changed after initial analysis)
                    with ui.card().classes("w-full p-2.5 bg-amber-50 border border-amber-300 rounded-lg mb-2") as reanalysis_warning_card:
                        reanalysis_warning_card.set_visibility(False)
                        with ui.row().classes("w-full items-center gap-2 text-amber-900 text-xs"):
                            ui.icon("warning", size="sm").classes("text-amber-600")
                            ui.label("Die Erkennungs-Einstellungen wurden geändert. Deaktivierte Treffer sind bereits aus der Vorschau entfernt; klicken Sie auf „Text / Dokument analysieren“, um neue automatische Treffer zu übernehmen.").classes("font-medium")

                    # Prominent Analysis & Workspace Reset Buttons in Action Row
                    with ui.row().classes("w-full items-center justify-between mt-1 mb-4 gap-3 flex-wrap"):
                        analyze_btn = ui.button(
                            "🔍 Text / Dokument analysieren",
                            icon="psychology",
                            color="primary",
                            on_click=run_analysis,
                        ).props("unelevated").classes("px-4 py-2 font-bold")
                        analyze_btn.set_enabled(bool(state.raw_text and state.raw_text.strip()))

                        reset_btn = ui.button(
                            "🗑️ Workspace zurücksetzen",
                            icon="delete_sweep",
                            color="slate",
                            on_click=reset_workspace,
                        ).props("outline dense").tooltip("Workspace leeren (Text, Dokument, Tabelle und Vorschau)")
                        reset_btn.set_visibility(bool(state.raw_text and state.raw_text.strip()))

                    progress_holder = ui.column().classes("w-full")

                    ui.separator().classes("my-3")

                    # Step 2: Manual Entity Marking (Instant Document-Specific Search)
                    ui.label("Stufe 2: Review-Tabelle & Manuelles Markieren").classes("text-base font-bold text-slate-800 mb-1")
                    
                    with ui.card().classes("w-full p-3 bg-blue-50/70 border border-blue-200 rounded-lg mb-3"):
                        with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                            with ui.row().classes("items-center gap-2 flex-grow flex-wrap"):
                                ui.icon("person_add", size="sm").classes("text-blue-700")
                                ui.label("Fehlenden Begriff / Namen im Dokument markieren:").classes("text-xs font-bold text-slate-800")
                                manual_input = ui.input(placeholder="z. B. Remo").props("dense outlined bg-white").classes("w-44 text-xs")
                                manual_type = ui.select(options=AVAILABLE_ENTITIES, value="PERSON").props("dense outlined bg-white").classes("w-36 text-xs")
                                save_perm_check = ui.checkbox("Dauerhaft in Begriffsliste speichern", value=False).props("dense").classes("text-xs text-slate-600")
                                state.register_mutating_element(manual_input, "workspace")
                                state.register_mutating_element(manual_type, "workspace")
                                state.register_mutating_element(save_perm_check, "workspace")

                                async def add_manual_entity():
                                    if not check_mutation_allowed():
                                        return
                                    term = manual_input.value.strip()
                                    if not term:
                                        ui.notify("Bitte einen Begriff eingeben.", type="warning")
                                        return
                                    if not state.raw_text or not state.raw_text.strip():
                                        ui.notify("Kein Text im Workspace vorhanden.", type="warning")
                                        return
                                    if state.entity_modes.get(manual_type.value, ENTITY_MODE_OFF) == ENTITY_MODE_OFF:
                                        ui.notify(
                                            f"{manual_type.value} ist vollständig ausgeschaltet. Bitte zuerst 'Nur Glossar & manuell' oder 'Alle Quellen' wählen.",
                                            type="warning",
                                        )
                                        return

                                    # Search whole-word matches in current document text
                                    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                                    matches = list(pattern.finditer(state.raw_text))
                                    if not matches:
                                        ui.notify(f"Begriff '{term}' wurde im aktuellen Dokumenttext nicht gefunden.", type="warning")
                                        return

                                    # Collect existing accepted spans
                                    existing_spans = []
                                    for g in state.entity_groups:
                                        if g.text_key != term.lower():
                                            for occ in g.occurrences:
                                                existing_spans.append((occ.start, occ.end))

                                    # Add occurrences that don't overlap with longer existing entities
                                    new_occurrences = []
                                    for m in matches:
                                        start, end = m.start(), m.end()
                                        overlaps_existing = any(not (end <= s or start >= e) for s, e in existing_spans)
                                        if not overlaps_existing:
                                            ctx_html = extract_context_snippet(state.raw_text, start, end)
                                            fingerprint = compute_context_fingerprint(state.raw_text, start, end)
                                            new_occurrences.append(
                                                EntityOccurrence(
                                                    start=start,
                                                    end=end,
                                                    score=1.0,
                                                    context_html=ctx_html,
                                                    needs_review=False,
                                                    source="manual",
                                                    method="manual",
                                                    occ_id=uuid.uuid4().hex,
                                                    context_fingerprint=fingerprint,
                                                )
                                            )

                                    if not new_occurrences:
                                        ui.notify(f"Alle Fundstellen von '{term}' sind bereits Teil von längeren Namen (z. B. Vollname).", type="info")
                                        return

                                    # Find or create EntityGroup
                                    existing_group = next((g for g in state.entity_groups if g.group_id == term.lower()), None)
                                    if existing_group:
                                        existing_group.occurrences = new_occurrences
                                        existing_group.entity_type = manual_type.value
                                        existing_group.enabled = True
                                    else:
                                        new_g = EntityGroup(
                                            original_text=matches[0].group(0),
                                            entity_type=manual_type.value,
                                            group_id=term.lower(),
                                        )
                                        new_g.occurrences = new_occurrences
                                        state.entity_groups.append(new_g)

                                    # Only persist permanently if checkbox is checked
                                    if save_perm_check.value:
                                        lines = [l.strip() for l in state.glossary_text.splitlines() if l.strip()]
                                        new_line = f"{term}: {manual_type.value}"
                                        if new_line not in lines:
                                            state.glossary_text += f"\n{new_line}"
                                            render_glossary_list_ui()
                                            save_current_config(state)
                                            ui.notify(f"'{term}' dauerhaft in der Begriffsliste gespeichert.", type="info")

                                    from local_anonymizer.anonymizer import compute_smart_link_proposals
                                    compute_smart_link_proposals(state.entity_groups)
                                    compute_reactive_preview(state)
                                    build_review_table()
                                    refresh_preview_and_exports()

                                    ui.notify(f"'{term}' ({len(new_occurrences)} Treffer) im aktuellen Dokument erfasst.", type="positive", icon="check")
                                    manual_input.value = ""

                                add_manual_btn = ui.button("➕ Hinzufügen", icon="add", color="positive", on_click=add_manual_entity).props("unelevated dense size=sm")
                                state.register_mutating_element(add_manual_btn, "workspace")

                    # LLM Proposal Panel (Review-Assistenz)
                    llm_panel_holder = ui.column().classes("w-full mb-3")
                    build_llm_panel()

                    table_holder = ui.column().classes("w-full mb-4")

                    ui.separator().classes("my-4")

                    ui.label("Stufe 3: Vorschau & Export").classes("text-base font-bold text-slate-800 mb-1")
                    preview_holder = ui.column().classes("w-full mb-2")
                    export_holder = ui.column().classes("w-full")

                # TAB 2: Restore
                with ui.tab_panel(tab_restore):
                    ui.label("De-Anonymisierung / Wiederherstellung").classes("text-base font-bold text-slate-800 mb-1")
                    ui.label("Laden Sie die vom Cloud-LLM beantwortete Datei (.docx, .md, .txt) oder fügen Sie den Text ein:").classes("text-sm text-slate-600 mb-4")

                    with ui.row().classes("w-full gap-4"):
                        # Column 1: Anonymized LLM Response
                        with ui.column().classes("flex-1"):
                            ui.label("1. LLM-Antwort (Text oder Dokument):").classes("font-semibold text-xs text-slate-700 mb-1")

                            def load_restore_text(text: str, filename: str):
                                if not check_mutation_allowed():
                                    return
                                state.restore_anon_text = text
                                if restore_anon_input is not None:
                                    restore_anon_input.value = text
                                ui.notify(f"'{filename}' geladen ({len(text)} Zeichen).", type="positive")

                            def open_restore_file_dialog():
                                if not check_mutation_allowed():
                                    return
                                try:
                                    import tkinter as tk
                                    from tkinter import filedialog
                                    root = tk.Tk()
                                    root.withdraw()
                                    root.attributes("-topmost", True)
                                    filepath = filedialog.askopenfilename(
                                        title="Anonymisierte LLM-Antwort auswählen",
                                        filetypes=[
                                            ("Dokumente (*.docx, *.pdf, *.txt, *.md)", "*.docx;*.pdf;*.txt;*.md"),
                                            ("Alle Dateien (*.*)", "*.*"),
                                        ]
                                    )
                                    root.destroy()
                                    if filepath:
                                        p = Path(filepath)
                                        raw_b = safe_read_bytes(p)
                                        text = read_document_from_bytes(raw_b, p.name)
                                        load_restore_text(text, p.name)
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {str(ex)}", type="negative")

                            async def on_restore_file_dropped(e):
                                data = e.args
                                temp_paths = None
                                try:
                                    raw_bytes, filename, temp_paths = extract_upload_payload(data, UPLOAD_DIR)
                                    if not check_mutation_allowed():
                                        return
                                    text = await asyncio.to_thread(read_document_from_bytes, raw_bytes, filename)
                                    load_restore_text(text, filename)
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {type(ex).__name__}: {str(ex)}", type="negative", timeout=15000)
                                finally:
                                    if temp_paths:
                                        cleanup_upload_paths(*temp_paths)

                            with ui.card().classes(
                                "w-full py-4 px-3 bg-slate-50 hover:bg-slate-100 border-2 border-dashed border-blue-300 hover:border-blue-500 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer shadow-none transition-all duration-150 mb-2 select-none"
                            ) as restore_drop_card:
                                restore_drop_id = restore_drop_card.id
                                restore_drop_card.props(f'id="restore_dropzone_{restore_drop_id}"')
                                with ui.row().classes("items-center gap-2 pointer-events-none"):
                                    ui.icon("upload_file", size="sm").classes("text-blue-500")
                                    ui.label("Ablegen oder klicken").classes("text-sm font-semibold text-slate-700")
                                ui.label(".docx · .pdf · .txt · .md").classes("text-[11px] text-slate-400 pointer-events-none")
                                restore_drop_card.on("click", open_restore_file_dialog)
                                ui.on("restore_file_dropped", on_restore_file_dropped)
                                state.register_mutating_element(restore_drop_card, "workspace")

                            ui.add_body_html(f"""
                            <script>
                            (function() {{
                                function setupRestoreDropzone() {{
                                    const el = document.getElementById('restore_dropzone_{restore_drop_id}');
                                    if (!el || el._dropAttached) return;
                                    el._dropAttached = true;
                                    el.addEventListener('dragover', function(e) {{ e.preventDefault(); e.stopPropagation(); el.style.borderColor='#2563eb'; el.style.backgroundColor='#dbeafe'; }}, false);
                                    el.addEventListener('dragleave', function(e) {{ e.preventDefault(); e.stopPropagation(); el.style.borderColor=''; el.style.backgroundColor=''; }}, false);
                                    el.addEventListener('drop', function(e) {{
                                        e.preventDefault(); e.stopPropagation();
                                        el.style.borderColor=''; el.style.backgroundColor='';
                                        const files = e.dataTransfer.files;
                                        if (!files || !files.length) return;
                                        const file = files[0];
                                        const formData = new FormData();
                                        formData.append('file', file);
                                        fetch('/api/upload', {{
                                            method: 'POST',
                                            body: formData
                                        }})
                                        .then(res => {{
                                            if (!res.ok) throw new Error('HTTP ' + res.status);
                                            return res.json();
                                        }})
                                        .then(data => {{
                                            emitEvent('restore_file_dropped', {{ file_id: data.file_id, name: data.filename }});
                                        }})
                                        .catch(err => {{
                                            console.error('Upload failed:', err);
                                            const reader = new FileReader();
                                            reader.onload = function(evt) {{ emitEvent('restore_file_dropped', {{ name: file.name, base64: evt.target.result.split(',')[1] }}); }};
                                            reader.readAsDataURL(file);
                                        }});
                                    }}, false);
                                }}
                                if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setupRestoreDropzone);
                                else setupRestoreDropzone();
                                setInterval(setupRestoreDropzone, 500);
                            }})();
                            </script>
                            """)

                            def on_anon_change(e):
                                if not check_mutation_allowed():
                                    return
                                state.restore_anon_text = e.value
                            restore_anon_input = ui.textarea(
                                placeholder="[PERSON_1_STUDENT_VOLLNAME] arbeitet an [ORGANIZATION_1_HOCHSCHULE]...",
                                on_change=on_anon_change,
                            ).props("outlined rows=6").classes("w-full font-mono text-xs")

                        # Column 2: Mapping File (auto-preloaded from current analysis)
                        with ui.column().classes("flex-1"):
                            ui.label("2. Mapping-Tabelle (.json):").classes("font-semibold text-xs text-slate-700 mb-1")

                            # Pre-populate from current analysis mapping if available
                            initial_map_val = ""
                            if state.current_mapping:
                                state.restore_mapping = dict(state.current_mapping)
                                initial_map_val = json.dumps(state.current_mapping, indent=2, ensure_ascii=False)
                            elif state.restore_mapping:
                                initial_map_val = json.dumps(state.restore_mapping, indent=2, ensure_ascii=False)

                            def load_mapping_data(mapping_data: dict):
                                if not check_mutation_allowed():
                                    return
                                state.restore_mapping = mapping_data
                                if map_json_input is not None:
                                    map_json_input.value = json.dumps(mapping_data, indent=2, ensure_ascii=False)
                                ui.notify(f"Mapping-Tabelle geladen ({len(mapping_data)} Einträge).", type="positive")

                            def open_mapping_file_dialog():
                                if not check_mutation_allowed():
                                    return
                                try:
                                    import tkinter as tk
                                    from tkinter import filedialog
                                    root = tk.Tk()
                                    root.withdraw()
                                    root.attributes("-topmost", True)
                                    filepath = filedialog.askopenfilename(
                                        title="Mapping-Datei (.json) auswählen",
                                        filetypes=[
                                            ("JSON-Dateien (*.json)", "*.json"),
                                            ("Alle Dateien (*.*)", "*.*"),
                                        ]
                                    )
                                    root.destroy()
                                    if filepath:
                                        p = Path(filepath)
                                        raw_b = safe_read_bytes(p)
                                        load_mapping_data(json.loads(raw_b.decode("utf-8")))
                                except Exception as ex:
                                    ui.notify(f"Ungültige JSON-Mapping-Datei: {str(ex)}", type="negative")

                            async def on_mapping_file_dropped(e):
                                data = e.args
                                temp_paths = None
                                try:
                                    raw_bytes, filename, temp_paths = extract_upload_payload(data, UPLOAD_DIR)
                                    if not check_mutation_allowed():
                                        return
                                    load_mapping_data(json.loads(raw_bytes.decode("utf-8")))
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {str(ex)}", type="negative", timeout=15000)
                                finally:
                                    if temp_paths:
                                        cleanup_upload_paths(*temp_paths)

                            with ui.card().classes(
                                "w-full py-4 px-3 bg-slate-50 hover:bg-slate-100 border-2 border-dashed border-blue-300 hover:border-blue-500 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer shadow-none transition-all duration-150 mb-2 select-none"
                            ) as mapping_drop_card:
                                mapping_drop_id = mapping_drop_card.id
                                mapping_drop_card.props(f'id="mapping_dropzone_{mapping_drop_id}"')
                                with ui.row().classes("items-center gap-2 pointer-events-none"):
                                    ui.icon("data_object", size="sm").classes("text-blue-500")
                                    ui.label("Ablegen oder klicken").classes("text-sm font-semibold text-slate-700")
                                if initial_map_val:
                                    ui.label(f"✅ {len(state.restore_mapping)} Einträge aus aktueller Analyse vorgeladen").classes("text-[11px] text-teal-600 font-semibold pointer-events-none")
                                else:
                                    ui.label("mapping.json ablegen oder klicken").classes("text-[11px] text-slate-400 pointer-events-none")
                                mapping_drop_card.on("click", open_mapping_file_dialog)
                                ui.on("mapping_file_dropped", on_mapping_file_dropped)
                                state.register_mutating_element(mapping_drop_card, "workspace")

                            ui.add_body_html(f"""
                            <script>
                            (function() {{
                                function setupMappingDropzone() {{
                                    const el = document.getElementById('mapping_dropzone_{mapping_drop_id}');
                                    if (!el || el._dropAttached) return;
                                    el._dropAttached = true;
                                    el.addEventListener('dragover', function(e) {{ e.preventDefault(); e.stopPropagation(); el.style.borderColor='#2563eb'; el.style.backgroundColor='#dbeafe'; }}, false);
                                    el.addEventListener('dragleave', function(e) {{ e.preventDefault(); e.stopPropagation(); el.style.borderColor=''; el.style.backgroundColor=''; }}, false);
                                    el.addEventListener('drop', function(e) {{
                                        e.preventDefault(); e.stopPropagation();
                                        el.style.borderColor=''; el.style.backgroundColor='';
                                        const files = e.dataTransfer.files;
                                        if (!files || !files.length) return;
                                        const file = files[0];
                                        const formData = new FormData();
                                        formData.append('file', file);
                                        fetch('/api/upload', {{
                                            method: 'POST',
                                            body: formData
                                        }})
                                        .then(res => {{
                                            if (!res.ok) throw new Error('HTTP ' + res.status);
                                            return res.json();
                                        }})
                                        .then(data => {{
                                            emitEvent('mapping_file_dropped', {{ file_id: data.file_id, name: data.filename }});
                                        }})
                                        .catch(err => {{
                                            console.error('Upload failed:', err);
                                            const reader = new FileReader();
                                            reader.onload = function(evt) {{ emitEvent('mapping_file_dropped', {{ name: file.name, base64: evt.target.result.split(',')[1] }}); }};
                                            reader.readAsDataURL(file);
                                        }});
                                    }}, false);
                                }}
                                if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setupMappingDropzone);
                                else setupMappingDropzone();
                                setInterval(setupMappingDropzone, 500);
                            }})();
                            </script>
                            """)

                            def on_map_text_change(e):
                                if not check_mutation_allowed():
                                    return
                                try:
                                    if e.value.strip():
                                        state.restore_mapping = json.loads(e.value)
                                except Exception:
                                    pass

                            map_json_input = ui.textarea(
                                value=initial_map_val,
                                placeholder='{\n  "[PERSON_1_STUDENT_VOLLNAME]": "Julia Meier"\n}',
                                on_change=on_map_text_change,
                            ).props("outlined rows=6").classes("w-full font-mono text-xs")

                    def run_restore():
                        if not check_mutation_allowed():
                            return
                        if not state.restore_anon_text:
                            ui.notify("Bitte anonymisierten Text einfügen oder Datei hochladen.", type="warning")
                            return
                        if not state.restore_mapping:
                            ui.notify("Bitte Mapping-Tabelle laden oder eingeben.", type="warning")
                            return

                        from local_anonymizer.anonymizer import LocalAnonymizer
                        restored = LocalAnonymizer.de_anonymize(state.restore_anon_text, state.restore_mapping)
                        state.restored_text = restored
                        restored_preview.value = restored
                        ui.notify("Dokument erfolgreich wiederhergestellt!", type="positive")

                    ui.button("Dokument Wiederherstellen", icon="restore", on_click=run_restore, color="primary").classes("mt-4").props("unelevated")

                    ui.separator().classes("my-4")

                    ui.label("Wiederhergestelltes Originaldokument:").classes("font-semibold text-slate-700 mb-1")
                    restored_preview = ui.textarea().props("readonly rows=10").classes("w-full font-mono text-sm bg-slate-50 border rounded p-2")

                    with ui.row().classes("gap-3 mt-2 flex-wrap items-center"):
                        async def copy_restored_clipboard():
                            await ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(state.restored_text)});')
                            ui.notify("Wiederhergestellter Text in Zwischenablage kopiert!", type="positive", icon="content_copy")

                        ui.button(
                            "📋 In Zwischenablage kopieren",
                            icon="content_copy",
                            color="secondary",
                            on_click=copy_restored_clipboard,
                        ).props("unelevated")

                        # Word (.docx) Export with real styles
                        def save_restored_docx():
                            if not state.restored_text:
                                ui.notify("Kein Text vorhanden zum Speichern.", type="warning")
                                return
                            raw_docx = save_markdown_to_docx_bytes(state.restored_text)
                            path = native_save_file("restored_document.docx", raw_docx, "Als Word-Dokument speichern")
                            if path:
                                ui.notify(f"Word-Dokument gespeichert: {Path(path).name}", type="positive", icon="check")
                            else:
                                ui.download(raw_docx, filename="restored_document.docx")

                        ui.button(
                            "💾 Als Word speichern (.docx)",
                            icon="description",
                            color="primary",
                            on_click=save_restored_docx,
                        ).props("unelevated").tooltip("Exportiert mit echten Word-Formatvorlagen (Heading 1-4, Listen & Tabellen)")

                        def save_restored_txt():
                            path = native_save_file("restored_document.txt", state.restored_text, "Wiederhergestellten Text speichern (.txt)")
                            if path:
                                ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                            else:
                                ui.download(state.restored_text.encode("utf-8"), filename="restored_document.txt")

                        ui.button(
                            "💾 Als Text speichern (.txt)",
                            icon="save",
                            color="slate",
                            on_click=save_restored_txt,
                        ).props("outline")

                        def save_restored_md():
                            path = native_save_file("restored_document.md", state.restored_text, "Als Markdown speichern (.md)")
                            if path:
                                ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                            else:
                                ui.download(state.restored_text.encode("utf-8"), filename="restored_document.md")

                        ui.button(
                            "💾 Als Markdown speichern (.md)",
                            icon="edit_note",
                            color="slate",
                            on_click=save_restored_md,
                        ).props("outline")

                # TAB 3: Transparency -- how detection actually works
                with ui.tab_panel(tab_transparency):
                    ui.label("Wie werden Entitäten erkannt?").classes("text-base font-bold text-slate-800 mb-1")
                    ui.label(
                        "Pro Kategorie: ob sie aktuell aktiv ist, und über welche Mechanismen sie erkannt wird -- "
                        "Zero-Shot-Prompts via GLiNER, das spezialisierte EU-PII-Klassifikationsmodell "
                        "(selektiv für Personen, Orte, IDs und Gesundheitsdaten), reguläre Ausdrücke (z. B. AHV, UID, IBAN), "
                        "externe Bibliotheken (z. B. Google phonenumbers) oder deine eigene Begriffsliste. "
                        "Reine Anzeige, hier wird nichts verändert."
                    ).classes("text-xs text-slate-500 mb-1")
                    ui.label(
                        "Hat eine Kategorie mehrere Quellen (z. B. GLiNER-Prompt + EU-PII-Modell + Regex), arbeiten sie "
                        "ergänzend zusammen: Alle Treffer werden gesammelt. Bei Überlappungen gewinnt die Hierarchie "
                        "Begriffsliste (3) > Deterministisch / Bibliothek / Regex (2) > Lokale KI-Modelle (1). "
                        "Innerhalb der KI-Modelle gewinnt der längere Trefferspan oder der höhere Konfidenzwert."
                    ).classes("text-xs text-slate-400 italic mb-3")

                    ui.button("🔄 Aktualisieren", on_click=lambda: load_transparency_view()).props("outline dense size=sm").classes("mb-2")
                    transparency_container = ui.column().classes("w-full gap-1")
                    with transparency_container:
                        ui.label("Wird beim ersten Öffnen dieses Tabs geladen …").classes("text-xs text-slate-400")

                    def load_transparency_view():
                        transparency_container.clear()
                        with transparency_container:
                            ui.spinner(size="lg").classes("mx-auto my-6")
                            ui.label("Lade Erkennungslogik...").classes("text-xs text-slate-400 text-center w-full")

                        def fetch_overview():
                            with _model_lock:
                                anon = get_synced_cached_anonymizer(state)
                                return anon.get_entity_source_overview()

                        async def load_and_render():
                            try:
                                overview = await asyncio.to_thread(fetch_overview)
                            except Exception as ex:
                                transparency_container.clear()
                                with transparency_container:
                                    ui.label(f"Fehler beim Laden: {ex}").classes("text-xs text-rose-600")
                                return
                            transparency_container.clear()
                            with transparency_container:
                                render_entity_source_overview(overview)

                        asyncio.create_task(load_and_render())

    async def check_readiness_expiry():
        if state.llm_ready_info is not None:
            if state.llm_ready_expires_at > 0.0 and time.time() >= state.llm_ready_expires_at:
                state.invalidate_llm_ready()
                build_llm_setup_panel()
                build_llm_panel()
                return

            if state.llm_setup_state == "idle" and not state.is_llm_running and state.llm_provider_type == "ollama" and verify_ollama_model_running is not None:
                ready_info_before = state.llm_ready_info
                bound_url_before = state.llm_ready_bound_url
                bound_model_before = state.llm_ready_bound_model
                provider_type_before = state.llm_provider_type
                url_snap = state.config.llm_base_url.strip()
                model_snap = state.config.llm_model_name.strip()

                try:
                    active_info = await verify_ollama_model_running(url_snap, model_snap)
                    # Verify state did not drift during await
                    if (
                        state.llm_ready_info is ready_info_before
                        and state.llm_ready_bound_url == bound_url_before
                        and state.llm_ready_bound_model == bound_model_before
                        and state.llm_provider_type == provider_type_before
                    ):
                        if active_info is None:
                            state.invalidate_llm_ready()
                            build_llm_setup_panel()
                            build_llm_panel()
                        elif active_info.expires_at:
                            new_exp = parse_iso_expiry(active_info.expires_at) if parse_iso_expiry else 0.0
                            if new_exp > 0.0:
                                state.llm_ready_expires_at = new_exp
                                state.llm_ready_info = active_info
                except Exception:
                    pass

    ui.timer(2.0, check_readiness_expiry)


def main():
    parser = argparse.ArgumentParser(description="Privacy-First Local Anonymizer GUI")
    parser.add_argument("--native", action="store_true", default=True, help="Launch as native desktop window (default: True)")
    parser.add_argument("--browser", action="store_true", help="Launch in default web browser instead of native window")
    parser.add_argument("--port", type=int, default=8080, help="Port to run GUI on (default: 8080)")
    args, _ = parser.parse_known_args()

    use_native = args.native and not args.browser

    # Configure native window dimensions
    app.native.window_args["width"] = 1250
    app.native.window_args["height"] = 850

    logging.info(f"Starting application (native={use_native}, port={args.port})...")
    ui.run(
        title="Privacy-First Local Anonymizer",
        native=use_native,
        port=args.port,
        reload=False,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    if os.environ.get("LOCAL_ANONYMIZER_PDF_WORKER") != "1":
        main()
