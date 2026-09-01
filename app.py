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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from nicegui import app, ui

from local_anonymizer.anonymizer import REVIEW_SCORE_THRESHOLD, clean_tag
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

# Optional LLM Triage Layer (Phase 6A)
try:
    from local_anonymizer.llm import (
        TriageItem,
        TriageEnvelope,
        validate_batch_response,
        LocalApiProvider,
        prepare_triage_batches,
        ApplyCommand,
        ApplyService,
    )
    from local_anonymizer.llm.apply_service import compute_triage_snapshot
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    compute_triage_snapshot = None  # type: ignore

# Silence presidio analyzer language mismatch warnings
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# Shared temp upload directory for large file Drag & Drop (cross-process, zero WebSocket limits)
UPLOAD_DIR = CONFIG_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit


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
        logging.error(f"api_upload failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


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
                base_grp.enabled = prev.enabled
                base_grp.surface_tag = prev.surface_tag
                base_grp.parent_group_id = prev.parent_group_id
            else:
                base_grp = EntityGroup(original_text=norm, entity_type=res.entity_type, group_id=key)
            base_groups_dict[key] = base_grp
        base_groups_dict[key].occurrences.append(occ)
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


def reset_app_state(st: "AppState") -> None:
    """Reset all document-specific state in AppState cleanly."""
    if getattr(st, "llm_active_task", None) and not st.llm_active_task.done():
        st.llm_active_task.cancel()
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
    if getattr(st, "llm_provider", None) is not None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(st.close_llm_provider())
        except Exception:
            pass


async def reset_app_state_async(st: "AppState") -> None:
    """Asynchronously cancel active LLM tasks, close provider session, and reset state."""
    if getattr(st, "llm_active_task", None) and not st.llm_active_task.done():
        st.llm_active_task.cancel()
        try:
            await st.llm_active_task
        except (asyncio.CancelledError, Exception):
            pass
        st.llm_active_task = None
    await st.close_llm_provider()
    reset_app_state(st)


def load_document_into_state(
    state: "AppState",
    text: str,
    filename: str,
    raw_bytes: Optional[bytes] = None,
) -> None:
    """
    Load extracted document text and optional raw bytes into AppState.
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
        self.mutating_ui_elements: List[Any] = []

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
            return json.loads(text)
        except Exception:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            parts = line.split(":", 1)
            glossary[parts[0].strip()] = parts[1].strip().upper()
        elif "=" in line:
            parts = line.split("=", 1)
            glossary[parts[0].strip()] = parts[1].strip().upper()
    return glossary


def parse_ignore_terms(text: str) -> List[str]:
    """Parse ignore terms separated by comma or newlines."""
    terms = []
    for line in text.replace(",", "\n").splitlines():
        t = line.strip()
        if t:
            terms.append(t)
    return terms


def build_anonymizer(app_state: Optional[AppState] = None):
    """Build LocalAnonymizer instance with specified state settings or loaded config."""
    from local_anonymizer.anonymizer import LocalAnonymizer

    if app_state is not None:
        glossary = parse_glossary(app_state.glossary_text)
        ignore_terms = parse_ignore_terms(app_state.ignore_terms_text)
        general_entities, glossary_entities = get_recognizer_entities(app_state.entity_modes)
        return LocalAnonymizer(
            language="de",
            glossary=glossary,
            ignore_terms=ignore_terms,
            enabled_entities=general_entities,
            enabled_glossary_entities=glossary_entities,
            gliner_model=app_state.gliner_model_name,
            gliner_threshold=app_state.gliner_threshold,
            enable_eupii=app_state.enable_eupii,
            eupii_threshold=app_state.eupii_threshold,
            eupii_model=app_state.eupii_model_name,
            entity_modes=app_state.entity_modes,
        )
    else:
        cfg = AppConfig.load()
        entity_modes = resolve_entity_modes(cfg)
        general_entities, glossary_entities = get_recognizer_entities(entity_modes)
        return LocalAnonymizer(
            language="de",
            glossary=parse_glossary(cfg.glossary),
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


def sync_cached_anonymizer_settings(anon, app_state: "AppState") -> None:
    """
    Push the current UI-configured settings (entity modes, threshold, ignore terms, glossary)
    onto an already-built LocalAnonymizer instance, so a cached instance stays in sync with
    sidebar edits instead of only reflecting the settings it was first constructed with.

    Single source of truth for this sync -- previously duplicated inline, which is how the
    glossary update silently wrote to a dead `.terms` attribute instead of the real `.glossary`
    for a while without anyone noticing.
    """
    general_entities, glossary_entities = get_recognizer_entities(app_state.entity_modes)
    anon.enabled_entities = general_entities
    anon.enabled_glossary_entities = glossary_entities
    anon.gliner_recognizer.threshold = app_state.gliner_threshold
    anon.set_eupii_enabled(app_state.enable_eupii, app_state.eupii_threshold)
    anon.set_entity_modes(app_state.entity_modes)
    anon.set_ignore_terms(parse_ignore_terms(app_state.ignore_terms_text))
    new_glossary = parse_glossary(app_state.glossary_text)
    anon.set_glossary(new_glossary)


def get_synced_cached_anonymizer(app_state: "AppState"):
    """Return the shared cached LocalAnonymizer, building it if needed and syncing it to the
    current UI settings. Must be called while holding `_model_lock`."""
    global _cached_anonymizer
    if _cached_anonymizer is None:
        _cached_anonymizer = build_anonymizer(app_state)
    else:
        sync_cached_anonymizer_settings(_cached_anonymizer, app_state)
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


# Supported entity labels
AVAILABLE_ENTITIES = sorted([
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "IBAN_CODE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "ID_NUMBER",
    "FINANCIAL_DATA",
    "HEALTH_DATA",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "URL",
    "USERNAME",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "ADDRESS",
    "AHV_NUMBER",
    "UID_NUMBER",
    "IT_SYSTEM",
    "ROLE",
])

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
    st.config.format_mode = st.format_mode
    st.config.entity_modes = dict(st.entity_modes)
    st.config.active_entities = [
        entity for entity, mode in st.entity_modes.items()
        if mode in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
    ]
    st.config.gliner_model_name = st.gliner_model_name
    st.config.gliner_threshold = st.gliner_threshold
    st.config.enable_eupii = st.enable_eupii
    st.config.eupii_threshold = st.eupii_threshold
    st.config.eupii_model_name = st.eupii_model_name
    st.config.ignore_terms = st.ignore_terms_text
    st.config.glossary = st.glossary_text
    st.config.export_format = st.export_format
    st.config.save()


async def ensure_models_downloaded_with_dialog(state: AppState) -> bool:
    """
    Ensure all AI models required by the current configuration are available in the local cache.
    If a model needs to be downloaded for the first time, prompts the user with an explicit
    confirmation dialog detailing download size, cache location, and offline privacy guarantees.
    Returns True if models are ready to use, False if cancelled or download failed.
    """
    needs_gliner = any(m == ENTITY_MODE_ALL for m in state.entity_modes.values())
    needs_eupii = state.enable_eupii and any(
        state.entity_modes.get(e) in (ENTITY_MODE_ALL, ENTITY_MODE_EXPLICIT_EUPII)
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
                    anon = get_synced_cached_anonymizer(state)
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


def extract_context_snippet(raw_text: str, start: int, end: int, window: int = 40) -> str:
    """Extract contextual snippet around entity with highlighted keyword."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(raw_text), end + window)

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
def create_ui():
    state = AppState()
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
    llm_settings_container = None
    llm_panel_holder = None
    manual_input = None
    manual_type = None
    save_perm_check = None
    add_manual_btn = None
    restore_anon_input = None
    map_json_input = None
    restored_preview = None

    def load_content_into_workspace(text: str, filename: str, raw_bytes: Optional[bytes] = None):
        """Unified workspace loader."""
        if not check_mutation_allowed():
            return
        load_document_into_state(state, text, filename, raw_bytes=raw_bytes)
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
            load_content_into_workspace(text, filename, raw_bytes=raw_bytes)
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

    def reset_workspace():
        """Reset raw text, filename, and analysis table."""
        if not check_mutation_allowed():
            return
        reset_app_state(state)
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
        ignore_container.clear()
        with ignore_container:
            raw_ignores = parse_ignore_terms(state.ignore_terms_text)
            unique_ignores = sorted(list({t.strip() for t in raw_ignores if t.strip()}), key=lambda s: s.lower())

            with ui.row().classes("w-full items-center gap-1 mb-2"):
                new_ig_input = ui.input(placeholder="Neuer Begriff...").props("dense outlined bg-white").classes("flex-grow text-xs")
                def add_ig():
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
                ui.button(icon="add", on_click=add_ig, color="primary").props("dense flat size=sm")

            with ui.column().classes("w-full max-h-48 overflow-y-auto gap-1 pr-1"):
                if not unique_ignores:
                    ui.label("Keine Begriffe auf der Ignore-Liste.").classes("text-[11px] text-slate-400 italic")
                else:
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        for term in unique_ignores:
                            def make_remove_ig(t):
                                def on_remove():
                                    curr = [x for x in parse_ignore_terms(state.ignore_terms_text) if x.lower() != t.lower()]
                                    state.ignore_terms_text = ", ".join(curr)
                                    save_current_config(state)
                                    render_ignore_list_ui()
                                    refresh_preview_and_exports()
                                    ui.notify(f"'{t}' aus Ignore-Liste entfernt.", type="info")
                                return on_remove

                            with ui.row().classes("items-center gap-1 bg-slate-100 border border-slate-300 rounded px-2 py-0.5 shadow-none"):
                                ui.label(term).classes("font-mono font-semibold text-xs text-slate-900")
                                ui.button(icon="close", on_click=make_remove_ig(term)).props("flat round dense size=xs color=negative").classes("p-0 min-h-0 min-w-0 ml-0.5 hover:bg-red-100")

    def render_glossary_list_ui():
        if not glossary_container:
            return
        glossary_container.clear()
        with glossary_container:
            glossary_dict = parse_glossary(state.glossary_text)
            sorted_keys = sorted(glossary_dict.keys(), key=lambda s: s.lower())

            with ui.row().classes("w-full items-center gap-1 mb-2 flex-wrap"):
                new_g_term = ui.input(placeholder="Begriff...").props("dense outlined bg-white").classes("flex-grow text-xs")
                new_g_type = ui.select(options=AVAILABLE_ENTITIES, value="ORGANIZATION").props("dense outlined bg-white").classes("w-28 text-xs")
                def add_g():
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
                ui.button(icon="add", on_click=add_g, color="positive").props("dense flat size=sm")

            with ui.column().classes("w-full max-h-48 overflow-y-auto gap-1 pr-1"):
                if not sorted_keys:
                    ui.label("Keine eigenen Begriffe in der Begriffsliste.").classes("text-[11px] text-slate-400 italic")
                else:
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        for term in sorted_keys:
                            ent_type = glossary_dict[term]
                            def make_remove_g(t):
                                def on_remove():
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
                                ui.button(icon="close", on_click=make_remove_g(term)).props("flat round dense size=xs color=negative").classes("p-0 min-h-0 min-w-0 ml-0.5 hover:bg-red-100")

    def render_llm_settings_ui():
        if llm_settings_container is None:
            return
        llm_settings_container.clear()
        with llm_settings_container:
            if not LLM_AVAILABLE:
                ui.label("Das Extra `[llm]` ist nicht installiert. Installieren Sie es mit:").classes("text-[11px] text-amber-800 font-semibold mb-1")
                ui.code("pip install local-anonymizer[llm]").classes("text-[10px] w-full mb-2")
                return

            def on_llm_toggle(e):
                if not check_mutation_allowed():
                    return
                state.config.llm_enabled = bool(e.value)
                save_current_config(state)
                render_llm_settings_ui()
                build_llm_panel()
                build_review_table()

            ui.switch("LLM-Assistenz aktivieren", value=state.config.llm_enabled, on_change=on_llm_toggle).props("dense").classes("text-xs mb-2 font-semibold")

            if state.config.llm_enabled:
                ui.label("API-Endpunkt (OpenAI-kompatibel, z. B. Ollama/LM Studio):").classes("text-[11px] text-slate-600 font-medium")

                def on_base_url_change(e):
                    if not check_mutation_allowed():
                        return
                    state.config.llm_base_url = (e.value or "").strip()
                    save_current_config(state)
                    if state.llm_provider:
                        asyncio.create_task(state.close_llm_provider())
                    build_llm_panel()

                ui.input(
                    value=state.config.llm_base_url,
                    placeholder="http://127.0.0.1:11434/v1",
                    on_change=on_base_url_change,
                ).props("dense outlined bg-white").classes("w-full text-xs mb-2")

                ui.label("Modellname (z. B. phi4:latest, qwen2.5:3b):").classes("text-[11px] text-slate-600 font-medium")

                def on_model_name_change(e):
                    if not check_mutation_allowed():
                        return
                    state.config.llm_model_name = (e.value or "").strip()
                    save_current_config(state)
                    if state.llm_provider:
                        asyncio.create_task(state.close_llm_provider())
                    build_llm_panel()

                ui.input(
                    value=state.config.llm_model_name,
                    placeholder="phi4:latest",
                    on_change=on_model_name_change,
                ).props("dense outlined bg-white").classes("w-full text-xs mb-2")

                def on_auto_review_toggle(e):
                    if not check_mutation_allowed():
                        return
                    state.config.llm_auto_review = bool(e.value)
                    save_current_config(state)

                ui.checkbox(
                    "LLM-Review direkt an die Textanalyse anschließen",
                    value=state.config.llm_auto_review,
                    on_change=on_auto_review_toggle,
                ).props("dense").classes("text-xs text-slate-700 mb-2").tooltip(
                    "Wenn aktiviert, startet nach der lokalen Erkennung automatisch die LLM-Triage der Fundstellen."
                )

    def refresh_preview_and_exports():
        if not preview_holder or not export_holder:
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
        if not check_mutation_allowed():
            return
        if not state.raw_text or not state.raw_text.strip():
            ui.notify("Bitte laden Sie zuerst ein Dokument hoch oder fügen Sie Text ein.", type="warning")
            return

        # Ensure required AI models are confirmed and downloaded before starting analysis
        ready = await ensure_models_downloaded_with_dialog(state)
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

        try:
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

            def do_analysis(text):
                with _model_lock:
                    anon = get_synced_cached_anonymizer(state)
                return anon.analyze(text)

            loop = asyncio.get_running_loop()
            analysis_task = asyncio.create_task(asyncio.to_thread(do_analysis, state.raw_text))

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
            total_occurrences = sum(g.count for g in state.entity_groups)
            build_llm_panel()
            build_review_table()
            refresh_preview_and_exports()

            # Mark all complete (green)
            update_step_ui(5, 1.0, f"Abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen)")
            ui.notify(f"Analyse abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen).", type="positive")

            # Check if LLM review should immediately chain into execution (Decision 1 from Handoff 1316)
            if state.config.llm_enabled and state.config.llm_auto_review and LLM_AVAILABLE and state.entity_groups:
                await asyncio.sleep(0.5)
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
            if analyze_btn:
                analyze_btn.props(remove="loading")
            if progress_holder:
                progress_holder.clear()

    def launch_llm_triage(triggered_from_analysis: bool = False) -> Optional[asyncio.Task]:
        """Central launcher for LLM triage runs, creating exactly one tracked task in state.llm_active_task."""
        if state.is_llm_running or (state.llm_active_task and not state.llm_active_task.done()):
            if not triggered_from_analysis:
                ui.notify("Eine LLM-Prüfung läuft bereits.", type="info")
            return state.llm_active_task

        state.llm_active_task = asyncio.create_task(run_llm_triage(triggered_from_analysis=triggered_from_analysis))
        return state.llm_active_task

    def check_mutation_allowed() -> bool:
        """Central guard against mutating state while LLM triage is running."""
        if state.is_llm_running:
            ui.notify("Aktion während laufender LLM-Prüfung gesperrt.", type="warning")
            return False
        return True

    def set_mutating_controls_disabled(disabled: bool):
        """Disable/enable all mutating UI controls during active LLM inference."""
        if analyze_btn:
            analyze_btn.set_enabled(not disabled and bool(state.raw_text and state.raw_text.strip()))
        if reset_btn:
            reset_btn.set_enabled(not disabled)
        if raw_text_area:
            if disabled:
                raw_text_area.props("readonly")
            else:
                raw_text_area.props(remove="readonly")
        if add_manual_btn:
            add_manual_btn.set_enabled(not disabled)
        if manual_input:
            manual_input.set_enabled(not disabled)
        if manual_type:
            manual_type.set_enabled(not disabled)
        for elem in state.mutating_ui_elements:
            try:
                elem.set_enabled(not disabled)
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
                                ui.badge("Ignorieren", color="grey-8").props("dense")
                            elif imp["action"] == "recategorize":
                                ui.badge(f"{imp['old_type']} ➔ {imp['new_type']}", color="amber-8").props("dense")
                            elif imp["action"] == "keep":
                                ui.badge("Bestätigt", color="emerald-7").props("dense outline")

                            if imp.get("new_role") != imp.get("old_role"):
                                ui.badge(f"Rolle: {imp['new_role']}", color="blue-7").props("dense")

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

        border_cls = "border-primary ring-1 ring-primary/40 bg-white" if is_staged else "border-slate-200 bg-white"

        with ui.card().props(f'id="llm_card_{item.occ_id}"').classes(f"w-full p-2.5 my-1 border rounded shadow-none {border_cls}"):
            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                with ui.row().classes("items-center gap-2 flex-1 min-w-[280px]"):
                    stage_chk = ui.checkbox(value=is_staged, on_change=on_stage_toggle).props("dense").tooltip("Für Sammelübernahme vormerken")
                    state.mutating_ui_elements.append(stage_chk)
                    ui.label(grp.original_text).classes("font-mono font-bold text-xs text-slate-900")
                    ui.badge(grp.entity_type, color="slate").props("dense outline")
                    if grp.role:
                        ui.badge(f"Rolle: {grp.role}", color="teal").props("dense outline")
                    if occ.context_html:
                        ui.html(occ.context_html).classes("text-xs text-slate-700 ml-2 flex-1")

                with ui.row().classes("items-center gap-2"):
                    if item.action == "discard":
                        ui.badge("Ignorieren", color="grey-8").props("dense")
                    elif item.action == "recategorize":
                        ui.badge(f"➔ {item.new_entity_type}", color="amber-8").props("dense")
                    elif item.action == "keep":
                        ui.badge("Bestätigt", color="emerald-7").props("dense outline")

                    if getattr(item, "descriptor_suggestion", None):
                        ui.badge(f"Rolle: {item.descriptor_suggestion}", color="blue-7").props("dense")

                    conf_color = "positive" if item.confidence == "high" else ("warning" if item.confidence == "medium" else "grey")
                    ui.badge(f"{item.confidence}", color=conf_color).props("dense outline").tooltip(f"Modell-Konfidenz: {item.confidence}")

                    def single_apply():
                        open_apply_dialog([item.occ_id])

                    single_btn = ui.button("Übernehmen", icon="check", color="positive", on_click=single_apply).props("outline dense size=xs")
                    state.mutating_ui_elements.append(single_btn)

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

    def build_llm_panel():
        if not llm_panel_holder:
            return
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
                            ui.label("Aktivieren Sie die Option in der Konfiguration (Seitenleiste), um Fundstellen automatisch durch ein lokales LLM prüfen zu lassen.").classes("text-slate-500")
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
                            is_start_enabled = has_entities and has_model and not state.is_llm_running
                            triage_start_btn.set_enabled(is_start_enabled)
                            if not has_entities:
                                triage_start_btn.tooltip("Führen Sie zuerst eine Textanalyse durch, um Fundstellen zu ermitteln.")
                            elif not has_model:
                                triage_start_btn.tooltip("Geben Sie in der Seitenleiste einen Modellnamen an (z. B. phi4:latest).")

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
                        with ui.row().classes("w-full items-center gap-2 p-2 bg-amber-100/70 border border-amber-300 rounded text-amber-950 text-xs my-2"):
                            ui.icon("warning", size="xs").classes("text-amber-700")
                            ui.label(f"Einige Batches konnten nicht verarbeitet werden ({len(state.llm_unprocessed_occ_ids)} Fundstellen ungeprüft). Sammelübernahme ist gesperrt; Einzelübernahmen sind möglich.").classes("font-medium")

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

                            ui.button("Alle vormerken", on_click=stage_all).props("flat dense size=xs color=slate")
                            ui.button("Auswahl leeren", on_click=unstage_all).props("flat dense size=xs color=slate")

                            selected_count = len(state.llm_staged_selections)
                            apply_bulk_btn = ui.button(
                                f"Ausgewählte Änderungen übernehmen ({selected_count})",
                                icon="done_all",
                                color="positive",
                                on_click=lambda: open_apply_dialog(list(state.llm_staged_selections)),
                            ).props("unelevated dense size=sm")

                            if selected_count == 0 or state.llm_partial_failure:
                                apply_bulk_btn.disable()
                                if state.llm_partial_failure:
                                    apply_bulk_btn.tooltip("Sammelübernahme bei unvollständigem Gesamtlauf gesperrt. Bitte Einzelübernahmen nutzen.")

                    render_proposal_categories(keep_items, recat_items, discard_items)

    async def run_llm_triage(triggered_from_analysis: bool = False):
        if state.is_llm_running:
            ui.notify("Eine LLM-Prüfung läuft bereits.", type="info")
            return

        if not LLM_AVAILABLE:
            ui.notify("LLM-Paket nicht verfügbar. Bitte `pip install local-anonymizer[llm]` installieren.", type="warning")
            return

        if not state.config.llm_enabled:
            if not triggered_from_analysis:
                ui.notify("Lokale LLM-Review-Assistenz ist deaktiviert. Bitte in den Einstellungen aktivieren.", type="info")
            return

        if not state.config.llm_model_name or not state.config.llm_model_name.strip():
            ui.notify("Bitte geben Sie einen Modellnamen in den LLM-Einstellungen an (z. B. phi4:latest).", type="warning")
            return

        if not state.raw_text or not state.entity_groups:
            if not triggered_from_analysis:
                ui.notify("Keine analysierten Fundstellen vorhanden.", type="info")
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
                ui.notify("Keine aktiven Fundstellen zum Prüfen gefunden.", type="info")
            return

        snapshot_hash = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups)
        batches = prepare_triage_batches(candidates, state.document_revision, snapshot_hash)

        try:
            if state.llm_provider is None:
                state.llm_provider = LocalApiProvider(
                    base_url=state.config.llm_base_url,
                    model_name=state.config.llm_model_name,
                )
        except Exception as e:
            ui.notify(f"Fehler bei LLM-Initialisierung: {e}", type="negative")
            return

        state.is_llm_running = True
        state.llm_partial_failure = False
        state.llm_unprocessed_occ_ids.clear()
        state.llm_triage_results.clear()
        state.llm_staged_selections.clear()
        state.llm_triage_snapshot = snapshot_hash

        set_mutating_controls_disabled(True)
        build_llm_panel()
        build_review_table()

        ui.notify(f"Starte LLM-Triage ({len(candidates)} Fundstellen in {len(batches)} Batches)...", type="info")

        try:
            for batch_idx, batch in enumerate(batches):
                current_snap = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups)
                if current_snap != snapshot_hash or state.document_revision != batch.document_revision:
                    logging.info("Dokument- oder Reviewzustand während LLM-Lauf geändert -> Lauf abgebrochen.")
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
                    post_snap = compute_triage_snapshot(state.raw_text, state.analysis_revision, state.entity_groups)
                    if post_snap != snapshot_hash or state.document_revision != batch.document_revision:
                        logging.info("Dokument- oder Reviewzustand nach Inferenz geändert -> Batch verworfen.")
                        state.llm_partial_failure = True
                        for rem_b in batches[batch_idx:]:
                            state.llm_unprocessed_occ_ids.update(rem_b.occ_id_set)
                        break

                    envelope = TriageEnvelope.model_validate_json(raw_json)
                    validate_batch_response(
                        envelope,
                        batch.occ_id_set,
                        state.document_revision,
                        snapshot_hash,
                        expected_request_id=batch.request_id,
                    )

                    for item in envelope.items:
                        state.llm_triage_results[item.occ_id] = item

                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    logging.warning(f"Batch {batch.batch_index}/{batch.total_batches} fehlgeschlagen: {type(ex).__name__}")
                    state.llm_partial_failure = True
                    state.llm_unprocessed_occ_ids.update(batch.occ_id_set)

                build_llm_panel()
                await asyncio.sleep(0.01)

            if state.llm_partial_failure:
                ui.notify(
                    f"LLM-Triage unvollständig ({len(state.llm_triage_results)} geprüft, {len(state.llm_unprocessed_occ_ids)} ungeprüft).",
                    type="warning",
                )
            else:
                ui.notify(f"LLM-Triage abgeschlossen ({len(state.llm_triage_results)} Fundstellen geprüft).", type="positive")

        except asyncio.CancelledError:
            ui.notify("LLM-Triage abgebrochen.", type="info")
            state.llm_triage_results.clear()
            state.llm_triage_snapshot = ""
            state.llm_staged_selections.clear()
        except Exception as e:
            ui.notify(f"Fehler bei LLM-Triage: {e}", type="negative")
        finally:
            state.is_llm_running = False
            state.llm_active_task = None
            set_mutating_controls_disabled(False)
            build_llm_panel()
            build_review_table()

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
        state.mutating_ui_elements.clear()
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
                    state.mutating_ui_elements.append(sort_sel)
                    if state.is_llm_running:
                        sort_sel.disable()

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
                    state.mutating_ui_elements.extend([sel_all_btn, desel_all_btn])
                    if state.is_llm_running:
                        sel_all_btn.disable()
                        desel_all_btn.disable()

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
                                            state.mutating_ui_elements.append(master_check)
                                            if not is_category_enabled(state, master.entity_type) or state.is_llm_running:
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
                                            state.mutating_ui_elements.append(m_sel)
                                            if state.is_llm_running:
                                                m_sel.disable()

                                        # 3. Role Input Field (In-place badge update WITHOUT losing focus!)
                                        with ui.row().classes("items-center gap-1"):
                                            def make_role_change(grp, m_badge, c_badges):
                                                def on_change(e):
                                                    if not check_mutation_allowed():
                                                        return
                                                    grp.role = (e.value or "").strip()
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
                                            state.mutating_ui_elements.append(m_role_inp)
                                            if state.is_llm_running:
                                                m_role_inp.disable()

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
                                                    state.mutating_ui_elements.append(sugg_btn)
                                                    if state.is_llm_running:
                                                        sugg_btn.disable()

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
                                                    state.mutating_ui_elements.append(link_sel)
                                                    if state.is_llm_running:
                                                        link_sel.disable()

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
                                            state.mutating_ui_elements.extend([ign_btn, gloss_btn, manual_button])
                                            if not is_category_enabled(state, master.entity_type) or state.is_llm_running:
                                                manual_button.disable()
                                            if state.is_llm_running:
                                                ign_btn.disable()
                                                gloss_btn.disable()

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
                                                    state.mutating_ui_elements.append(rev_btn)
                                                    if state.is_llm_running:
                                                        rev_btn.disable()
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
                                                    state.mutating_ui_elements.append(split_btn)
                                                    if state.is_llm_running:
                                                        split_btn.disable()

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
                                                state.mutating_ui_elements.append(child_check)
                                                if not is_category_enabled(state, child.entity_type) or state.is_llm_running:
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
                                                state.mutating_ui_elements.append(surf_sel)
                                                if state.is_llm_running:
                                                    surf_sel.disable()

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
                                                state.mutating_ui_elements.append(unlink_btn)
                                                if state.is_llm_running:
                                                    unlink_btn.disable()

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
                                                        state.mutating_ui_elements.append(rev_child_btn)
                                                        if state.is_llm_running:
                                                            rev_child_btn.disable()
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
                                                        state.mutating_ui_elements.append(split_child_btn)
                                                        if state.is_llm_running:
                                                            split_child_btn.disable()

    # --- Main Layout ---
    with ui.row().classes("w-full no-wrap p-4 gap-6"):
        # Sidebar: Configuration
        with ui.card().classes("w-80 p-4 shrink-0 bg-slate-50 border shadow-sm"):
            ui.label("⚙️ Konfiguration").classes("text-base font-bold text-slate-800 mb-3")

            # Format Mode Selector
            ui.label("Platzhalter-Format:").classes("text-xs font-semibold text-slate-700 mb-1")
            def on_mode_change(e):
                state.format_mode = e.value
                save_current_config(state)
                refresh_preview_and_exports()
                build_review_table()

            ui.radio(
                {
                    "numbered": "Modus 1: [TYP_NR] (Klassisch)",
                    "numbered_role": "Modus 2: [TYP_NR_ROLLE] (Empfohlen)",
                    "role_only": "Modus 3: [TYP_ROLLE] (Kompakt)",
                },
                value=state.format_mode,
                on_change=on_mode_change,
            ).props("dense").classes("text-xs mb-3")

            # Export Format Choice (.txt vs .md)
            ui.separator().classes("my-2")
            ui.label("Standard Export-Format:").classes("text-xs font-semibold text-slate-700 mb-1")
            def on_export_fmt_change(e):
                state.export_format = e.value
                save_current_config(state)
                refresh_preview_and_exports()

            ui.radio(
                {
                    "txt": ".txt (Reiner Text)",
                    "md": ".md (Markdown-Formatierung)",
                },
                value=state.export_format,
                on_change=on_export_fmt_change,
            ).props("dense inline").classes("text-xs mb-3")

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

            ui.separator().classes("my-2")

            ui.label("Erkennungs-Schwellenwert (GLiNER):").classes("text-xs font-semibold text-slate-700")
            def on_thresh_change(e):
                state.gliner_threshold = e.value
                save_current_config(state)
                if state.entity_groups and reanalysis_warning_card is not None:
                    reanalysis_warning_card.set_visibility(True)
            thresh_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.gliner_threshold, on_change=on_thresh_change)
            ui.label().bind_text_from(thresh_slider, "value", lambda v: f"GLiNER Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

            ui.separator().classes("my-2")

            # EU-PII Multilingual Model Toggle & Threshold
            ui.label("EU-PII Multilingual Modell:").classes("text-xs font-semibold text-slate-700")
            with ui.row().classes("w-full items-center justify-between gap-1 mb-1"):
                eupii_spinner = ui.spinner(size="xs", color="primary")
                eupii_spinner.set_visibility(False)

                async def on_eupii_toggle(e):
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

            def on_eupii_thresh_change(e):
                state.eupii_threshold = e.value
                save_current_config(state)
                with _model_lock:
                    if _cached_anonymizer is not None:
                        _cached_anonymizer.set_eupii_enabled(state.enable_eupii, state.eupii_threshold)
                if state.entity_groups and reanalysis_warning_card is not None:
                    reanalysis_warning_card.set_visibility(True)

            eupii_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.eupii_threshold, on_change=on_eupii_thresh_change)
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

            # Interactive LLM Configuration
            with ui.expansion("🤖 Lokale LLM-Review-Assistenz", icon="psychology").classes("w-full text-xs mt-1"):
                llm_settings_container = ui.column().classes("w-full")
                render_llm_settings_ui()

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
                            filename = data.get("name", "dokument")
                            filepath = data.get("path", "")
                            file_id = data.get("file_id", "")
                            bin_path = UPLOAD_DIR / f"{file_id}.bin" if file_id else None
                            meta_path = UPLOAD_DIR / f"{file_id}.json" if file_id else None
                            try:
                                if bin_path and bin_path.exists():
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
                                    exists = bin_path.exists() if bin_path else False
                                    raise ValueError(f"Keine Dateidaten empfangen (file_id={file_id}, exists={exists}, keys={list(data.keys()) if isinstance(data, dict) else data})")

                                await extract_and_load_file_bytes(raw_bytes, filename)
                            except Exception as ex:
                                err_msg = f"{type(ex).__name__}: {str(ex)}"
                                logging.error(f"Drop error: {err_msg}", exc_info=True)
                                ui.notify(f"Fehler beim Laden: {err_msg}", type="negative", timeout=15000)
                            finally:
                                if bin_path:
                                    try:
                                        bin_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                if meta_path:
                                    try:
                                        meta_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass

                        ui.on("file_dropped", on_file_dropped)

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
                                state.include_headers_footers = bool(e.value)
                                await on_extraction_opt_change()

                            async def on_toggle_picture_text(e):
                                state.extract_picture_text = bool(e.value)
                                await on_extraction_opt_change()

                            ui.checkbox(
                                "Kopf- und Fußzeilen einbeziehen",
                                value=state.include_headers_footers,
                                on_change=on_toggle_headers_footers,
                            ).props("dense size=sm").tooltip("Laufende Kopf- und Fußzeilen (z. B. Seitenzahlen, Dokumenttitel) mit einlesen. Standard: Aus.")

                            with ui.row().classes("items-center gap-1"):
                                ui.checkbox(
                                    "Text aus PDF-Bildern & Grafiken extrahieren",
                                    value=state.extract_picture_text,
                                    on_change=on_toggle_picture_text,
                                ).props("dense size=sm").tooltip("Liest Textboxen in Diagrammen & Vektorgrafiken aus.")
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
                                state.restore_anon_text = text
                                if restore_anon_input is not None:
                                    restore_anon_input.value = text
                                ui.notify(f"'{filename}' geladen ({len(text)} Zeichen).", type="positive")

                            def open_restore_file_dialog():
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
                                filename = data.get("name", "dokument")
                                filepath = data.get("path", "")
                                file_id = data.get("file_id", "")
                                bin_path = UPLOAD_DIR / f"{file_id}.bin" if file_id else None
                                meta_path = UPLOAD_DIR / f"{file_id}.json" if file_id else None
                                try:
                                    if bin_path and bin_path.exists():
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
                                        exists = bin_path.exists() if bin_path else False
                                        raise ValueError(f"Keine Dateidaten empfangen (file_id={file_id}, exists={exists}, keys={list(data.keys()) if isinstance(data, dict) else data})")
                                    text = await asyncio.to_thread(read_document_from_bytes, raw_bytes, filename)
                                    load_restore_text(text, filename)
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {type(ex).__name__}: {str(ex)}", type="negative", timeout=15000)
                                finally:
                                    if bin_path:
                                        try:
                                            bin_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass
                                    if meta_path:
                                        try:
                                            meta_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass

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
                                state.restore_mapping = mapping_data
                                if map_json_input is not None:
                                    map_json_input.value = json.dumps(mapping_data, indent=2, ensure_ascii=False)
                                ui.notify(f"Mapping-Tabelle geladen ({len(mapping_data)} Einträge).", type="positive")

                            def open_mapping_file_dialog():
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
                                filename = data.get("name", "mapping.json")
                                filepath = data.get("path", "")
                                file_id = data.get("file_id", "")
                                bin_path = UPLOAD_DIR / f"{file_id}.bin" if file_id else None
                                meta_path = UPLOAD_DIR / f"{file_id}.json" if file_id else None
                                try:
                                    if bin_path and bin_path.exists():
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
                                        exists = bin_path.exists() if bin_path else False
                                        raise ValueError(f"Keine Dateidaten empfangen (file_id={file_id}, exists={exists}, keys={list(data.keys()) if isinstance(data, dict) else data})")
                                    load_mapping_data(json.loads(raw_bytes.decode("utf-8")))
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {str(ex)}", type="negative", timeout=15000)
                                finally:
                                    if bin_path:
                                        try:
                                            bin_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass
                                    if meta_path:
                                        try:
                                            meta_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass

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
