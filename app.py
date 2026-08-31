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
from local_anonymizer.config import (
    AppConfig,
    CONFIG_DIR,
    ENTITY_MODE_ALL,
    ENTITY_MODE_EXPLICIT_ONLY,
    ENTITY_MODE_OFF,
    LOG_FILE,
)
from local_anonymizer.extractors import (
    UnsupportedFileFormatError,
    create_docx_from_markdown,
    extract_text_from_txt_bytes,
    read_document_from_bytes,
    safe_read_bytes,
    save_markdown_to_docx_bytes,
    strip_html_markup,
)

# Silence presidio analyzer language mismatch warnings
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# Shared temp upload directory for large file Drag & Drop (cross-process, zero WebSocket limits)
UPLOAD_DIR = CONFIG_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit


def cleanup_temp_uploads():
    """Clean up any stale uploaded temporary files from previous sessions."""
    try:
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.glob("*"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


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


# --- Data Models for Grouped Review ---
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


class EntityGroup:
    def __init__(self, original_text: str, entity_type: str):
        self.original_text: str = original_text
        self.entity_type: str = entity_type
        self.enabled: bool = True
        self.role: str = ""
        self.parent_group_text: Optional[str] = None
        self.surface_tag: str = ""
        self.placeholder: str = ""
        self.suggested_parent: Optional[str] = None
        self.suggested_tag: Optional[str] = None
        self.suggested_candidates: List[str] = []
        self.occurrences: List[EntityOccurrence] = []

    @property
    def key(self) -> str:
        return self.original_text.strip().lower()

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


# --- App State ---
class AppState:
    def __init__(self):
        self.config: AppConfig = AppConfig.load()
        self.filename: str = ""
        self.raw_text: str = ""
        self.entity_groups: List[EntityGroup] = []
        self.entity_modes: Dict[str, str] = resolve_entity_modes(self.config)
        # Legacy compatibility for config files and code paths that still expose active_entities.
        self.active_entities: List[str] = [
            entity for entity, mode in self.entity_modes.items() if mode == ENTITY_MODE_ALL
        ]
        self.format_mode: str = self.config.format_mode  # "numbered", "numbered_role", "role_only"
        self.export_format: str = self.config.export_format  # "txt", "md"
        self.gliner_threshold: float = self.config.gliner_threshold
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
            gliner_threshold=app_state.gliner_threshold,
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
            gliner_threshold=cfg.gliner_threshold,
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
        with ui.card().classes("w-full p-2 " + ("bg-white" if active else "bg-slate-50 opacity-60")):
            with ui.row().classes("items-center gap-2"):
                ui.icon(
                    "check_circle" if active else "radio_button_unchecked",
                    color="positive" if active else "grey",
                ).classes("text-sm")
                ui.label(row["category"]).classes("text-sm font-mono font-bold text-slate-800")
                mode = row.get("mode")
                if mode == ENTITY_MODE_EXPLICIT_ONLY:
                    ui.badge("nur explizite Einträge", color="teal").props("dense")
                elif mode == "automatic_only":
                    ui.badge("nur automatische Erkennung", color="purple").props("dense")
                elif not active:
                    ui.badge("inaktiv", color="grey-5").props("dense")

            with ui.column().classes("w-full gap-1 mt-1 pl-6"):
                for src in row["sources"]:
                    kind_label, kind_color = _SOURCE_KIND_LABELS.get(src["kind"], (src["kind"], "grey"))
                    with ui.row().classes("items-start gap-2 flex-wrap"):
                        ui.badge(kind_label, color=kind_color).props("dense outline")
                        if src["kind"] == "prompt":
                            ui.label(", ".join(f'"{p}"' for p in src["prompts"])).classes(
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
                            ui.label(f'externe Bibliothek ({src["recognizer"]})').classes("text-xs text-slate-600")


def _warmup_background_thread():
    """Worker to initialize GLiNER and Presidio in the background without blocking the UI."""
    global _cached_anonymizer, _model_ready
    try:
        logging.info("Background warming up GLiNER & Presidio AI models...")
        anon = build_anonymizer()
        with _model_lock:
            _cached_anonymizer = anon
            _model_ready = True
        logging.info("GLiNER & Presidio AI models ready.")
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

ENTITY_MODE_OPTIONS: Dict[str, str] = {
    ENTITY_MODE_OFF: "Aus – nichts anonymisieren",
    ENTITY_MODE_EXPLICIT_ONLY: "Nur Glossar & manuell",
    ENTITY_MODE_ALL: "Alle Quellen",
}

ENTITY_MODE_COLORS: Dict[str, str] = {
    ENTITY_MODE_OFF: "bg-red-100 text-red-900 border-red-300",
    ENTITY_MODE_EXPLICIT_ONLY: "bg-orange-100 text-orange-900 border-orange-300",
    ENTITY_MODE_ALL: "bg-green-100 text-green-900 border-green-300",
}


def entity_mode_classes(mode: str) -> str:
    """Return stable base and state color classes for one category mode selector."""
    return f"w-44 text-xs {ENTITY_MODE_COLORS.get(mode, ENTITY_MODE_COLORS[ENTITY_MODE_OFF])}"

RECOGNIZER_METHODS: Dict[str, str] = {
    "GLiNERRecognizer": "ai",
    "AddressPatternRecognizer": "regex",
    "AHVNumberRecognizer": "regex",
    "UIDNumberRecognizer": "regex",
}

METHOD_DISPLAY: Dict[str, Tuple[str, str, str]] = {
    "ai": ("🤖 KI", "blue", "Durch den lokalen KI-Erkenner gefunden"),
    "regex": ("🔤 Regex", "purple", "Durch einen regulären Ausdruck gefunden"),
    "library": ("📚 Bibliothek", "grey-7", "Durch eine lokale Bibliotheks-Erkennung gefunden"),
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
    return {
        entity: saved_modes.get(
            entity,
            ENTITY_MODE_ALL if entity in legacy_active else ENTITY_MODE_OFF,
        )
        for entity in AVAILABLE_ENTITIES
    }


def get_recognizer_entities(entity_modes: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Translate UI modes into general-recognizer and glossary category filters."""
    general_entities = [
        entity for entity in AVAILABLE_ENTITIES
        if entity_modes.get(entity, ENTITY_MODE_OFF) == ENTITY_MODE_ALL
    ]
    glossary_entities = [
        entity for entity in AVAILABLE_ENTITIES
        if entity_modes.get(entity, ENTITY_MODE_OFF) in (ENTITY_MODE_EXPLICIT_ONLY, ENTITY_MODE_ALL)
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
    return list(group.occurrences)


# Start background warmup only after all configuration helpers are defined.
threading.Thread(target=_warmup_background_thread, daemon=True).start()


def save_current_config(st: AppState):
    """Save user modifications back to persistent config."""
    st.config.format_mode = st.format_mode
    st.config.entity_modes = dict(st.entity_modes)
    st.config.active_entities = [
        entity for entity, mode in st.entity_modes.items() if mode == ENTITY_MODE_ALL
    ]
    st.config.gliner_threshold = st.gliner_threshold
    st.config.ignore_terms = st.ignore_terms_text
    st.config.glossary = st.glossary_text
    st.config.export_format = st.export_format
    st.config.save()


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
        is_child = g.parent_group_text and g.parent_group_text.strip().lower() != g.key
        if not is_child:
            count = category_counters.get(g.entity_type, 0) + 1
            category_counters[g.entity_type] = count
            group_info[g.key] = {
                "id": count,
                "role": g.role,
                "surface_tag": g.surface_tag,
            }

    # Pass 2: Linked children
    for g in active_groups:
        is_child = g.parent_group_text and g.parent_group_text.strip().lower() != g.key
        if is_child:
            parent_key = g.parent_group_text.strip().lower()
            if parent_key in group_info:
                p_id = group_info[parent_key]["id"]
                p_role = group_info[parent_key]["role"] or g.role
            else:
                count = category_counters.get(g.entity_type, 0) + 1
                category_counters[g.entity_type] = count
                p_id = count
                p_role = g.role

            group_info[g.key] = {
                "id": p_id,
                "role": p_role,
                "surface_tag": g.surface_tag or "VORNAME",
            }

    # 2. Check collisions for Mode 3 (role_only)
    role_type_groups: Dict[Tuple[str, str], Set[int]] = {}
    for g in active_groups:
        info = group_info.get(g.key, {})
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

        info = group_info.get(g.key, {"id": 1, "role": "", "surface_tag": ""})
        ent_id = info["id"]
        role_str = clean_tag(info["role"])
        tag_str = clean_tag(info["surface_tag"])
        suffix_tag = f"_{tag_str}" if tag_str else ""
        pair = (g.entity_type, role_str)
        is_colliding = pair in st.colliding_roles

        if st.format_mode == "role_only" and role_str and not is_colliding:
            placeholder = f"[{g.entity_type}_{role_str}{suffix_tag}]"
        elif (st.format_mode in ("numbered_role", "role_only") or is_colliding) and role_str:
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
    restore_anon_input = None
    map_json_input = None
    restored_preview = None

    def load_content_into_workspace(text: str, filename: str):
        """Unified workspace loader."""
        state.filename = filename
        state.raw_text = text
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
        ui.notify(f"Datei '{filename}' geladen ({len(text)} Zeichen).", type="positive")

    async def extract_and_load_file_bytes(raw_bytes: bytes, filename: str):
        """Asynchronously extract structured text from document bytes with live UI progress."""
        state.last_raw_bytes = raw_bytes
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
            load_content_into_workspace(text, filename)
        except Exception as ex:
            err_msg = f"{type(ex).__name__}: {str(ex)}"
            logging.error(f"File extraction error: {err_msg}", exc_info=True)
            ui.notify(f"Fehler beim Einlesen von '{filename}': {err_msg}", type="negative", timeout=15000)
        finally:
            if extraction_progress_card is not None:
                extraction_progress_card.set_visibility(False)

    async def open_native_file_dialog():
        """Open native OS file picker with full Win32 lock-sharing support."""
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
        state.filename = ""
        state.raw_text = ""
        state.last_raw_bytes = None
        state.entity_groups = []
        state.current_mapping = {}
        state.current_anon_text = ""
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
        if not state.raw_text or not state.raw_text.strip():
            ui.notify("Bitte laden Sie zuerst ein Dokument hoch oder fügen Sie Text ein.", type="warning")
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

            groups_dict: Dict[str, EntityGroup] = {}
            for res in results:
                orig = state.raw_text[res.start:res.end]
                norm = orig.strip()
                key = norm.lower()
                needs_rev = res.score < REVIEW_SCORE_THRESHOLD
                ctx_html = extract_context_snippet(state.raw_text, res.start, res.end)

                source, method = classify_recognition_result(res)
                occ = EntityOccurrence(
                    start=res.start,
                    end=res.end,
                    score=res.score,
                    context_html=ctx_html,
                    needs_review=needs_rev,
                    source=source,
                    method=method,
                )
                if key not in groups_dict:
                    groups_dict[key] = EntityGroup(original_text=norm, entity_type=res.entity_type)
                groups_dict[key].occurrences.append(occ)

            state.entity_groups = list(groups_dict.values())

            # Step 4: Smart-Linking proposals
            update_step_ui(3, 0.90, "Namensbezüge & Rollen-Kandidaten...")
            await asyncio.sleep(0.05)

            from local_anonymizer.anonymizer import compute_smart_link_proposals
            compute_smart_link_proposals(state.entity_groups)

            # Step 5: Reactive Preview & Mapping
            update_step_ui(4, 0.98, "Generiere Vorschau & Mapping...")
            await asyncio.sleep(0.05)

            compute_reactive_preview(state)
            total_occurrences = sum(g.count for g in state.entity_groups)
            build_review_table()
            refresh_preview_and_exports()

            # Mark all complete (green)
            update_step_ui(5, 1.0, f"Abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen)")
            ui.notify(f"Analyse abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen).", type="positive")

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

    def get_sorted_groups() -> List[EntityGroup]:
        """Return entity groups sorted according to user selection."""
        if state.sort_by == "Alphabetisch (A–Z)":
            return sorted(state.entity_groups, key=lambda g: g.original_text.lower())
        elif state.sort_by == "Erstes Auftreten im Text":
            return sorted(state.entity_groups, key=lambda g: g.first_start)
        elif state.sort_by == "Häufigkeit (meiste Treffer zuerst)":
            return sorted(state.entity_groups, key=lambda g: (-g.count, g.first_start))
        elif state.sort_by == "Entitätstyp (PERSON, ORG, ...)":
            return sorted(state.entity_groups, key=lambda g: (g.entity_type, g.original_text.lower()))
        elif state.sort_by == "⚠️ Review-Bedarf zuerst":
            return sorted(state.entity_groups, key=lambda g: (not g.needs_review, -g.count, g.first_start))
        else:
            return sorted(state.entity_groups, key=lambda g: g.original_text.lower())

    def build_review_table():
        if not table_holder:
            return
        table_holder.clear()
        with table_holder:
            if not state.entity_groups:
                ui.label("Keine zu anonymisierenden Entitäten im Text erkannt.").classes("text-slate-500 italic p-4")
                return

            total_hits = sum(g.count for g in state.entity_groups)
            unique_count = len(state.entity_groups)

            # Precompute all entity names once in O(n) for linking dropdowns
            all_entity_names = sorted(
                list({g.original_text.strip() for g in state.entity_groups if g.original_text.strip()}),
                key=lambda s: s.lower(),
            )

            ui.label(f"Erkannte Entitäten ({unique_count} Begriffe, {total_hits} Fundstellen gesamt)").classes("text-lg font-bold text-slate-800 mb-1")
            ui.label("Vergeben Sie optionale Rollen oder übernehmen Sie Verknüpfungsvorschläge. Jede Zeile zeigt den zugeordneten Platzhalter und klappt Fundstellen auf:").classes("text-sm text-slate-600 mb-3")

            # Toolbar: Sorting & Bulk actions
            with ui.row().classes("w-full items-center justify-between bg-slate-100 p-2.5 rounded-lg border mb-3 flex-wrap gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sort", size="sm").classes("text-slate-600")
                    ui.label("Sortierung:").classes("text-xs font-semibold text-slate-700")

                    def on_sort_change(e):
                        state.sort_by = e.value
                        build_review_table()

                    ui.select(
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

                with ui.row().classes("items-center gap-2"):
                    def select_all():
                        for g in state.entity_groups:
                            g.enabled = True
                        refresh_preview_and_exports()
                        build_review_table()

                    def deselect_all():
                        for g in state.entity_groups:
                            g.enabled = False
                        refresh_preview_and_exports()
                        build_review_table()

                    ui.button("Alle aktivieren", icon="select_all", on_click=select_all, color="slate").props("outline dense size=sm")
                    ui.button("Alle abwählen", icon="deselect", on_click=deselect_all, color="slate").props("outline dense size=sm")

            sorted_groups = get_sorted_groups()
            from local_anonymizer.anonymizer import build_entity_tree
            tree = build_entity_tree(sorted_groups)

            # Render Tree Nodes (Roots & Indented Children)
            for node_idx, root_node in enumerate(tree):
                master: EntityGroup = root_node.item
                has_children = len(root_node.children) > 0
                row_bg = "bg-amber-50" if master.needs_review else ("bg-white" if node_idx % 2 == 0 else "bg-slate-50")

                with ui.column().classes("w-full mb-2"):
                    child_badge_refs: List[Tuple[EntityGroup, Any]] = []

                    # Master Row
                    with ui.expansion().classes(f"w-full border rounded {row_bg}") as exp:
                        with exp.add_slot("header"):
                            with ui.row().classes("w-full items-center justify-between gap-3 pr-2 flex-wrap"):
                                # 1. Checkbox + Name + Count + Assigned Placeholder Badge
                                with ui.row().classes("items-center gap-2 min-w-[280px]"):
                                    def make_group_check(grp):
                                        def on_change(e):
                                            grp.enabled = e.value
                                            refresh_preview_and_exports()
                                            build_review_table()
                                        return on_change

                                    master_check = ui.checkbox(value=master.enabled, on_change=make_group_check(master)).props("dense")
                                    if not is_category_enabled(state, master.entity_type):
                                        master_check.disable()
                                    ui.label(master.original_text).classes("font-mono text-sm font-bold text-slate-800")
                                    ui.badge(f"{master.count}x", color="primary" if master.count > 1 else "grey-6").props("dense")

                                    # Master Placeholder Badge
                                    master_badge = ui.badge(master.placeholder, color="blue-9").props("outline dense").classes("text-xs font-mono font-bold")

                                    if has_children:
                                        ui.badge(f"🔗 {len(root_node.children)} verknüpft", color="teal").props("dense").tooltip("Hauptperson mit verknüpften Schreibweisen")

                                # 2. Type Selector
                                with ui.row().classes("items-center gap-1"):
                                    def make_group_select(grp):
                                        def on_change(e):
                                            grp.entity_type = e.value
                                            refresh_preview_and_exports()
                                            build_review_table()
                                        return on_change

                                    ui.select(
                                        options=AVAILABLE_ENTITIES,
                                        value=master.entity_type,
                                        on_change=make_group_select(master),
                                    ).props("dense outlined bg-white").classes("w-36 text-xs")

                                # 3. Role Input Field (In-place badge update WITHOUT losing focus!)
                                with ui.row().classes("items-center gap-1"):
                                    def make_role_change(grp, m_badge, c_badges):
                                        def on_change(e):
                                            grp.role = (e.value or "").strip()
                                            compute_reactive_preview(state)
                                            m_badge.set_text(grp.placeholder)
                                            for c_grp, c_badge in c_badges:
                                                c_badge.set_text(c_grp.placeholder)
                                            refresh_preview_and_exports()
                                        return on_change

                                    ui.input(
                                        label="Rolle (optional)",
                                        value=master.role,
                                        placeholder="z.B. Student",
                                        on_change=make_role_change(master, master_badge, child_badge_refs),
                                    ).props("dense outlined bg-white debounce=300").classes("w-32 text-xs")

                                # 4. Interactive Smart-Linking Proposal (Confirmation Model) or Link Dropdown
                                if not has_children:
                                    if master.suggested_parent and not master.parent_group_text:
                                        with ui.row().classes("items-center gap-1"):
                                            def make_apply_suggestion(grp, p_target, tag):
                                                def on_click():
                                                    grp.parent_group_text = p_target
                                                    grp.surface_tag = tag
                                                    grp.suggested_parent = None
                                                    grp.suggested_tag = None
                                                    grp.suggested_candidates = []
                                                    p_match = next((x for x in state.entity_groups if x.original_text == p_target), None)
                                                    if p_match and not p_match.surface_tag:
                                                        p_match.surface_tag = "VOLLNAME"
                                                    ui.notify(f"'{grp.original_text}' mit '{p_target}' ({tag}) verknüpft.", type="positive", icon="link")
                                                    refresh_preview_and_exports()
                                                    build_review_table()
                                                return on_click

                                            tag_label = SURFACE_TAG_OPTIONS.get(master.suggested_tag, master.suggested_tag or "Vorname")
                                            ui.button(
                                                f"💡 Mit '{master.suggested_parent}' verknüpfen",
                                                icon="auto_awesome",
                                                color="teal-8",
                                                on_click=make_apply_suggestion(master, master.suggested_parent, master.suggested_tag or "VORNAME"),
                                            ).props("outline dense size=sm").tooltip(f"Vorschlag übernehmen: Als {tag_label} verknüpfen")

                                    elif master.suggested_candidates:
                                        ui.badge(f"💡 {len(master.suggested_candidates)} Namenskandidaten", color="amber-8").props("outline dense").tooltip("Mehrere passende Personen gefunden. Bitte im Dropdown rechts auswählen.")

                                    # Manual Link Dropdown: All other recognized entities in the workspace
                                    all_other_names = [n for n in all_entity_names if n.lower() != master.key]
                                    if all_other_names:
                                        with ui.row().classes("items-center gap-1"):
                                            def make_link_to_master(grp):
                                                def on_change(e):
                                                    val = (e.value or "").strip()
                                                    if val and val != "Eigenständig":
                                                        p_match = next((x for x in state.entity_groups if x.original_text.lower() == val.lower()), None)
                                                        if p_match:
                                                            grp.parent_group_text = p_match.original_text
                                                            if not grp.surface_tag:
                                                                grp.surface_tag = "VORNAME"
                                                            grp.suggested_parent = None
                                                            grp.suggested_candidates = []
                                                            if not p_match.surface_tag:
                                                                p_match.surface_tag = "VOLLNAME"
                                                            ui.notify(f"'{grp.original_text}' verknüpft mit '{p_match.original_text}'.", type="info")
                                                            refresh_preview_and_exports()
                                                            build_review_table()
                                                        else:
                                                            ui.notify(f"Ungültige Verknüpfung: '{val}' existiert nicht.", type="warning")
                                                return on_change

                                            ui.select(
                                                options=["Eigenständig"] + all_other_names,
                                                value="Eigenständig",
                                                label="Verknüpfen mit:",
                                                with_input=True,
                                                on_change=make_link_to_master(master),
                                            ).props('dense outlined bg-white use-input clearable options-dense menu-props="{ maxHeight: \'300px\' }"').classes("min-w-[200px] max-w-[320px] text-xs").tooltip("Zielperson / Bezug auswählen oder durch Tippen filtern")

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
                                            current_ignores = parse_ignore_terms(state.ignore_terms_text)
                                            if term not in current_ignores:
                                                current_ignores.append(term)
                                                state.ignore_terms_text = ", ".join(current_ignores)
                                                render_ignore_list_ui()
                                                save_current_config(state)
                                            grp.enabled = False
                                            ui.notify(f"'{term}' zur Ignore-Liste hinzugefügt.", type="info")
                                            refresh_preview_and_exports()
                                            build_review_table()
                                        return on_click

                                    def make_add_to_glossary(term, grp):
                                        def on_click():
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
                                            # Make the action effective immediately in the current
                                            # review, even before the next full analysis.
                                            for occurrence in grp.occurrences:
                                                occurrence.source = "glossary"
                                                occurrence.method = "glossary_direct"
                                                occurrence.method_detail = "direct"
                                            grp.enabled = True
                                            save_current_config(state)
                                            render_glossary_list_ui()
                                            compute_reactive_preview(state)
                                            refresh_preview_and_exports()
                                            build_review_table()
                                            ui.notify(f"'{term}' als {grp.entity_type} zur Begriffsliste hinzugefügt.", type="positive", icon="library_add")
                                        return on_click

                                    def make_mark_manual(term, grp):
                                        def on_click():
                                            for occurrence in grp.occurrences:
                                                occurrence.source = "manual"
                                                occurrence.method = "manual"
                                                occurrence.method_detail = "manual"
                                                occurrence.needs_review = False
                                            grp.enabled = True
                                            compute_reactive_preview(state)
                                            refresh_preview_and_exports()
                                            build_review_table()
                                            ui.notify(f"'{term}' nur für diesen Durchlauf manuell markiert.", type="info", icon="edit_note")
                                        return on_click

                                    ui.button(
                                        icon="block",
                                        on_click=make_group_ignore(master.original_text, master),
                                    ).props("flat round dense size=sm color=grey-8").tooltip("Begriff ignorieren und zur Ignore-Liste hinzufügen")
                                    ui.button(
                                        icon="library_add",
                                        on_click=make_add_to_glossary(master.original_text, master),
                                    ).props("flat round dense size=sm color=teal-8").tooltip("Diesen Begriff dauerhaft mit diesem Entitätstyp zum Glossar hinzufügen")
                                    manual_button = ui.button(
                                        icon="edit_note",
                                        on_click=make_mark_manual(master.original_text, master),
                                    ).props("flat round dense size=sm color=orange-8").tooltip("Diesen Begriff nur für den aktuellen Durchlauf als manuell markiert behandeln")
                                    if not is_category_enabled(state, master.entity_type):
                                        manual_button.disable()

                        # Expanded content: Context occurrences
                        with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                            ui.label(f"Fundstellen im Text ({master.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                            for occ_idx, occ in enumerate(master.occurrences, start=1):
                                with ui.row().classes("items-center gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                    ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                    ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                    method_label, method_color, method_tooltip = method_display(occ.method)
                                    ui.badge(method_label, color=method_color).props("dense outline").tooltip(method_tooltip)
                                    ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

                    # Render Indented Linked Children
                    for child_node in root_node.children:
                        child: EntityGroup = child_node.item
                        with ui.expansion().classes("w-full ml-6 my-1 bg-teal-50/50 border-l-4 border-teal-500 border rounded shadow-none") as child_exp:
                            with child_exp.add_slot("header"):
                                with ui.row().classes("w-full items-center justify-between gap-3 pr-2 flex-wrap"):
                                    with ui.row().classes("items-center gap-2 min-w-[260px]"):
                                        ui.label("↳").classes("text-teal-700 font-bold text-base")
                                        child_check = ui.checkbox(value=child.enabled, on_change=make_group_check(child)).props("dense")
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
                                                c_grp.surface_tag = e.value
                                                refresh_preview_and_exports()
                                                build_review_table()
                                            return on_change

                                        ui.select(
                                            options=SURFACE_TAG_OPTIONS,
                                            value=child.surface_tag or "VORNAME",
                                            label="Schreibweise:",
                                            on_change=make_surface_change(child),
                                        ).props("dense outlined bg-white").classes("w-44 text-xs")

                                    # Explicit UNLINK Action
                                    with ui.row().classes("items-center gap-2"):
                                        def make_unlink_action(c_grp):
                                            def on_click():
                                                c_grp.parent_group_text = None
                                                c_grp.surface_tag = ""
                                                ui.notify(f"'{c_grp.original_text}' getrennt und als eigenständige Entität gesetzt.", type="info")
                                                refresh_preview_and_exports()
                                                build_review_table()
                                            return on_click

                                        ui.button(
                                            "✕ Trennen",
                                            icon="link_off",
                                            color="negative",
                                            on_click=make_unlink_action(child),
                                        ).props("flat dense size=sm").tooltip("Verknüpfung aufheben und als separate Entität führen")

                            # Child Occurrences
                            with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                                ui.label(f"Fundstellen von '{child.original_text}' ({child.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                            for occ_idx, occ in enumerate(child.occurrences, start=1):
                                with ui.row().classes("items-center gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                    ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                    ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                    method_label, method_color, method_tooltip = method_display(occ.method)
                                    ui.badge(method_label, color=method_color).props("dense outline").tooltip(method_tooltip)
                                    ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

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
                                entity for entity, mode in state.entity_modes.items()
                                if mode == ENTITY_MODE_ALL
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
                        options=ENTITY_MODE_OPTIONS,
                        value=state.entity_modes.get(ent, ENTITY_MODE_OFF),
                        on_change=mode_change,
                    ).props("dense outlined options-dense").classes(
                        entity_mode_classes(state.entity_modes.get(ent, ENTITY_MODE_OFF))
                    )
                    selector_ref.append(mode_select)

            ui.separator().classes("my-2")

            ui.label("Erkennungs-Schwellenwert:").classes("text-xs font-semibold text-slate-700")
            def on_thresh_change(e):
                state.gliner_threshold = e.value
                save_current_config(state)
                if state.entity_groups and reanalysis_warning_card is not None:
                    reanalysis_warning_card.set_visibility(True)
            thresh_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.gliner_threshold, on_change=on_thresh_change)
            ui.label().bind_text_from(thresh_slider, "value", lambda v: f"Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

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
                            state.raw_text = e.value or ""
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
                                        if g.key != term.lower():
                                            for occ in g.occurrences:
                                                existing_spans.append((occ.start, occ.end))

                                    # Add occurrences that don't overlap with longer existing entities
                                    new_occurrences = []
                                    for m in matches:
                                        start, end = m.start(), m.end()
                                        overlaps_existing = any(not (end <= s or start >= e) for s, e in existing_spans)
                                        if not overlaps_existing:
                                            ctx_html = extract_context_snippet(state.raw_text, start, end)
                                            new_occurrences.append(
                                                EntityOccurrence(
                                                    start=start,
                                                    end=end,
                                                    score=1.0,
                                                    context_html=ctx_html,
                                                    needs_review=False,
                                                    source="manual",
                                                    method="manual",
                                                )
                                            )

                                    if not new_occurrences:
                                        ui.notify(f"Alle Fundstellen von '{term}' sind bereits Teil von längeren Namen (z. B. Vollname).", type="info")
                                        return

                                    # Find or create EntityGroup
                                    existing_group = next((g for g in state.entity_groups if g.key == term.lower()), None)
                                    if existing_group:
                                        existing_group.occurrences = new_occurrences
                                        existing_group.entity_type = manual_type.value
                                        existing_group.enabled = True
                                    else:
                                        new_g = EntityGroup(original_text=matches[0].group(0), entity_type=manual_type.value)
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

                                ui.button("➕ Hinzufügen", icon="add", color="positive", on_click=add_manual_entity).props("unelevated dense size=sm")

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
                        "Pro Kategorie: ob sie aktuell aktiv ist, und über welchen Mechanismus sie erkannt wird -- "
                        "ein KI-Prompt an das Zero-Shot-Modell, ein regulärer Ausdruck, eine externe Bibliothek, "
                        "oder deine eigene Begriffsliste. Reine Anzeige, hier wird nichts verändert."
                    ).classes("text-xs text-slate-500 mb-1")
                    ui.label(
                        "Hat eine Kategorie mehrere Quellen (z. B. KI-Prompt + Regex), arbeiten sie ergänzend "
                        "zusammen statt sich zu widersprechen: alle Kandidaten werden gesammelt, und bei "
                        "überlappenden Funden gewinnt der längere Fund -- bei gleicher Länge dein Glossar vor "
                        "allen anderen Quellen, sonst der höhere Konfidenzwert."
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
    main()
