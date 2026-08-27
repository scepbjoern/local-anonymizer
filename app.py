"""
Privacy-First Local Anonymizer - NiceGUI Interactive Review Application.
Runs completely locally and offline with in-memory processing.
"""

import argparse
import asyncio
import html
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nicegui import app, ui

from local_anonymizer.anonymizer import (
    AnonymizationResult,
    DetectedEntity,
    EntityTreeNode,
    LocalAnonymizer,
    build_entity_tree,
    clean_tag,
)
from local_anonymizer.config import AppConfig, LOG_FILE
from local_anonymizer.extractors import (
    UnsupportedFileFormatError,
    create_docx_from_markdown,
    extract_text_from_txt_bytes,
    read_document_from_bytes,
    save_markdown_to_docx_bytes,
)


# --- Data Models for Grouped Review ---
@dataclass
class EntityOccurrence:
    start: int
    end: int
    score: float
    context_html: str
    needs_review: bool


class EntityGroup:
    def __init__(self, original_text: str, entity_type: str):
        self.original_text: str = original_text
        self.entity_type: str = entity_type
        self.enabled: bool = True
        self.role: str = ""
        self.parent_group_text: Optional[str] = None
        self.surface_tag: str = ""
        self.placeholder: str = ""
        self.occurrences: List[EntityOccurrence] = []

    @property
    def key(self) -> str:
        return self.original_text.strip().lower()

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def max_score(self) -> float:
        return max((occ.score for occ in self.occurrences), default=0.0)

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
        self.anonymizer: Optional[LocalAnonymizer] = None
        self.entity_groups: List[EntityGroup] = []
        self.active_entities: List[str] = list(self.config.active_entities)
        self.format_mode: str = self.config.format_mode  # "numbered", "numbered_role", "role_only"
        self.export_format: str = self.config.export_format  # "txt", "md"
        self.gliner_threshold: float = self.config.gliner_threshold
        self.sort_by: str = "Erstes Auftreten im Text"
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


state = AppState()

# Supported entity labels
AVAILABLE_ENTITIES = [
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "IBAN_CODE",
    "CREDIT_CARD",
    "ID_NUMBER",
    "FINANCIAL_DATA",
    "HEALTH_DATA",
    "IP_ADDRESS",
]

SURFACE_TAG_OPTIONS = [
    ("VOLLNAME", "Vollname (z. B. Julia Meier)"),
    ("VORNAME", "Vorname (z. B. Julia)"),
    ("NACHNAME", "Nachname (z. B. Meier)"),
    ("ANREDE", "Anrede / Titel (z. B. Frau Meier)"),
    ("KURZFORM", "Kurzform / Kürzel (z. B. JM)"),
]


def save_current_config():
    """Save user modifications back to persistent config."""
    state.config.format_mode = state.format_mode
    state.config.active_entities = state.active_entities
    state.config.gliner_threshold = state.gliner_threshold
    state.config.ignore_terms = state.ignore_terms_text
    state.config.glossary = state.glossary_text
    state.config.export_format = state.export_format
    state.config.save()


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


def build_anonymizer() -> LocalAnonymizer:
    """Build LocalAnonymizer instance with current state settings."""
    glossary = parse_glossary(state.glossary_text)
    ignore_terms = parse_ignore_terms(state.ignore_terms_text)
    return LocalAnonymizer(
        language="de",
        glossary=glossary,
        ignore_terms=ignore_terms,
        enabled_entities=state.active_entities if state.active_entities else None,
        gliner_threshold=state.gliner_threshold,
    )


def compute_reactive_preview() -> Tuple[str, Dict[str, str], Dict]:
    """
    Recalculate placeholder substitution based on current active groups, roles, format mode, and entity links.
    """
    if not state.raw_text:
        return "", {}, {}

    active_groups = [g for g in state.entity_groups if g.enabled]
    if not active_groups:
        state.current_mapping = {}
        state.current_anon_text = state.raw_text
        state.current_report = {"source_file": state.filename, "entity_count": 0, "mapping": {}, "entities": []}
        return state.raw_text, {}, state.current_report

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

    state.colliding_roles = {pair for pair, ids in role_type_groups.items() if len(ids) > 1}

    # 3. Generate placeholders
    mapping: Dict[str, str] = {}
    final_entities_report = []
    flat_occurrences = []

    for g in state.entity_groups:
        if not g.enabled:
            g.placeholder = "(ignoriert / inaktiv)"
            continue

        info = group_info.get(g.key, {"id": 1, "role": "", "surface_tag": ""})
        ent_id = info["id"]
        role_str = clean_tag(info["role"])
        tag_str = clean_tag(info["surface_tag"])
        suffix_tag = f"_{tag_str}" if tag_str else ""
        pair = (g.entity_type, role_str)
        is_colliding = pair in state.colliding_roles

        if state.format_mode == "role_only" and role_str and not is_colliding:
            placeholder = f"[{g.entity_type}_{role_str}{suffix_tag}]"
        elif (state.format_mode in ("numbered_role", "role_only") or is_colliding) and role_str:
            placeholder = f"[{g.entity_type}_{ent_id}_{role_str}{suffix_tag}]"
        else:
            # Modus 1 (Numbered)
            placeholder = f"[{g.entity_type}_{ent_id}{suffix_tag}]"

        g.placeholder = placeholder
        mapping[placeholder] = g.original_text

        for occ in g.occurrences:
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
    chars = list(state.raw_text)
    for item in sorted_for_sub:
        start, end = item["start"], item["end"]
        chars[start:end] = list(item["placeholder"])

    anonymized_text = "".join(chars)

    audit_report = {
        "source_file": state.filename,
        "format_mode": state.format_mode,
        "entity_count": len(final_entities_report),
        "unique_entities_count": len(mapping),
        "mapping": mapping,
        "entities": final_entities_report,
    }

    state.current_mapping = mapping
    state.current_report = audit_report
    state.current_anon_text = anonymized_text

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


# --- UI Building ---
def create_ui():
    ui.colors(primary="#1976D2", secondary="#26A69A", accent="#9C27B0", positive="#2E7D32", warning="#F57C00", negative="#C62828")

    # Header
    with ui.header().classes("bg-slate-800 text-white p-4 shadow-md flex items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("lock", size="md").classes("text-teal-400")
            ui.label("Privacy-First Local Anonymizer").classes("text-xl font-bold")
            ui.badge("100% Lokal & Offline", color="teal").props("outline")
        ui.label("Review & Korrektur Workspace").classes("text-sm text-slate-300")

    # Reactive UI container holders
    preview_holder = None
    table_holder = None
    export_holder = None
    raw_text_area = None
    restore_anon_input = None
    map_json_input = None
    restored_preview = None

    def refresh_preview_and_exports():
        if not preview_holder or not export_holder:
            return
        anon_text, mapping, audit_report = compute_reactive_preview()
        stem = Path(state.filename).stem or "dokument"
        ext = state.export_format

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

    async def run_analysis():
        if not state.raw_text:
            ui.notify("Bitte laden Sie zuerst ein Dokument hoch oder fügen Sie Text ein.", type="warning")
            return

        if table_holder:
            table_holder.clear()
            with table_holder:
                with ui.row().classes("items-center gap-3 p-4 bg-blue-50 rounded border border-blue-200"):
                    ui.spinner(size="md", color="primary")
                    ui.label("Dokument wird lokal analysiert (NER, Markdown, Genitiv-Erkennung)...").classes("text-slate-700 text-sm font-medium")

        try:
            anonymizer = build_anonymizer()
            results = await asyncio.to_thread(anonymizer.analyze, state.raw_text)

            groups_dict: Dict[str, EntityGroup] = {}
            for res in results:
                orig = state.raw_text[res.start:res.end]
                norm = orig.strip()
                key = norm.lower()
                needs_rev = 0.70 <= res.score < 0.85
                ctx_html = extract_context_snippet(state.raw_text, res.start, res.end)

                occ = EntityOccurrence(
                    start=res.start,
                    end=res.end,
                    score=res.score,
                    context_html=ctx_html,
                    needs_review=needs_rev,
                )
                if key not in groups_dict:
                    groups_dict[key] = EntityGroup(original_text=norm, entity_type=res.entity_type)
                groups_dict[key].occurrences.append(occ)

            state.entity_groups = list(groups_dict.values())

            # Automatic Smart Linking Detection
            for g in state.entity_groups:
                g_words = g.original_text.split()
                if len(g_words) == 1:
                    for other in state.entity_groups:
                        if other.key != g.key and other.entity_type == g.entity_type:
                            other_words = other.original_text.split()
                            if len(other_words) > 1:
                                if g.original_text.lower() == other_words[0].lower():
                                    g.parent_group_text = other.original_text
                                    g.surface_tag = "VORNAME"
                                    if not other.surface_tag:
                                        other.surface_tag = "VOLLNAME"
                                    break
                                elif g.original_text.lower() == other_words[-1].lower():
                                    g.parent_group_text = other.original_text
                                    g.surface_tag = "NACHNAME"
                                    if not other.surface_tag:
                                        other.surface_tag = "VOLLNAME"
                                    break

            # Compute initial placeholders so badges exist immediately
            compute_reactive_preview()

            total_occurrences = sum(g.count for g in state.entity_groups)
            ui.notify(f"Analyse abgeschlossen: {len(state.entity_groups)} Begriffe ({total_occurrences} Fundstellen).", type="positive")
            build_review_table()
            refresh_preview_and_exports()

        except ValueError as ve:
            if table_holder:
                table_holder.clear()
            ui.notify(f"Verarbeitungsfehler: {str(ve)}", type="negative", close_button=True, timeout=10000)
        except Exception as e:
            if table_holder:
                table_holder.clear()
            logging.error(f"Analysis error: {e}", exc_info=True)
            ui.notify(f"Unerwarteter Fehler bei der Analyse: {str(e)}", type="negative", close_button=True)

    async def handle_upload(e):
        try:
            if hasattr(e, "file"):
                data = await e.file.read()
                filename = e.file.name
            elif hasattr(e, "content"):
                data = e.content.read()
                filename = e.name
            else:
                raise ValueError("Unbekanntes Upload-Event-Format")

            state.filename = filename
            state.raw_text = read_document_from_bytes(data, filename)
            if raw_text_area is not None:
                raw_text_area.value = state.raw_text
            ui.notify(f"Datei '{filename}' erfolgreich geladen ({len(state.raw_text)} Zeichen).", type="positive")
            await run_analysis()
        except UnsupportedFileFormatError as fe:
            ui.notify(f"Nicht unterstütztes Format: {str(fe)}", type="negative")
        except ValueError as ve:
            ui.notify(f"Fehler beim Lesen des Dokuments: {str(ve)}", type="negative", timeout=10000)
        except Exception as ex:
            logging.error(f"Upload error: {ex}", exc_info=True)
            ui.notify(f"Fehler beim Upload: {str(ex)}", type="negative")

    def get_sorted_groups() -> List[EntityGroup]:
        """Return entity groups sorted according to user selection."""
        if state.sort_by == "Häufigkeit (meiste Treffer zuerst)":
            return sorted(state.entity_groups, key=lambda g: (-g.count, g.first_start))
        elif state.sort_by == "Entitätstyp (PERSON, ORG, ...)":
            return sorted(state.entity_groups, key=lambda g: (g.entity_type, g.original_text.lower()))
        elif state.sort_by == "Alphabetisch (A–Z)":
            return sorted(state.entity_groups, key=lambda g: g.original_text.lower())
        elif state.sort_by == "⚠️ Review-Bedarf zuerst":
            return sorted(state.entity_groups, key=lambda g: (not g.needs_review, -g.count, g.first_start))
        else:
            return sorted(state.entity_groups, key=lambda g: g.first_start)

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

            ui.label(f"Erkannte Entitäten ({unique_count} Begriffe, {total_hits} Fundstellen gesamt)").classes("text-lg font-bold text-slate-800 mb-1")
            ui.label("Vergeben Sie optionale Rollen oder verknüpfen Sie Schreibweisen. Jede Zeile zeigt den zugeordneten Platzhalter und klappt Fundstellen auf:").classes("text-sm text-slate-600 mb-3")

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
                            "Erstes Auftreten im Text",
                            "Häufigkeit (meiste Treffer zuerst)",
                            "Entitätstyp (PERSON, ORG, ...)",
                            "Alphabetisch (A–Z)",
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
            tree = build_entity_tree(sorted_groups)

            # Render Tree Nodes (Roots & Indented Children)
            for node_idx, root_node in enumerate(tree):
                master: EntityGroup = root_node.item
                has_children = len(root_node.children) > 0
                row_bg = "bg-amber-50" if master.needs_review else ("bg-white" if node_idx % 2 == 0 else "bg-slate-50")

                with ui.column().classes("w-full mb-2"):
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

                                    ui.checkbox(value=master.enabled, on_change=make_group_check(master)).props("dense")
                                    ui.label(master.original_text).classes("font-mono text-sm font-bold text-slate-800")
                                    ui.badge(f"{master.count}x", color="primary" if master.count > 1 else "grey-6").props("dense")

                                    # Assigned Placeholder Badge
                                    ui.badge(master.placeholder, color="blue-9").props("outline dense").classes("text-xs font-mono font-bold")

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

                                # 3. Role Input Field
                                with ui.row().classes("items-center gap-1"):
                                    def make_role_change(grp):
                                        def on_change(e):
                                            grp.role = e.value.strip()
                                            refresh_preview_and_exports()
                                            build_review_table()
                                        return on_change

                                    ui.input(
                                        label="Rolle (optional)",
                                        value=master.role,
                                        placeholder="z.B. Student",
                                        on_change=make_role_change(master),
                                    ).props("dense outlined bg-white").classes("w-32 text-xs")

                                # 4. Link to another master (if not already master with children)
                                if not has_children:
                                    other_masters = [
                                        g.original_text for g in state.entity_groups
                                        if g.key != master.key and g.entity_type == master.entity_type and not g.parent_group_text
                                    ]
                                    if other_masters:
                                        with ui.row().classes("items-center gap-1"):
                                            def make_link_to_master(grp):
                                                def on_change(e):
                                                    if e.value and e.value != "Eigenständig":
                                                        grp.parent_group_text = e.value
                                                        if not grp.surface_tag:
                                                            grp.surface_tag = "VORNAME"
                                                        ui.notify(f"'{grp.original_text}' verknüpft mit '{e.value}'.", type="info")
                                                        refresh_preview_and_exports()
                                                        build_review_table()
                                                return on_change

                                            ui.select(
                                                options=["Eigenständig"] + other_masters,
                                                value="Eigenständig",
                                                label="Verknüpfen mit:",
                                                on_change=make_link_to_master(master),
                                            ).props("dense outlined bg-white").classes("w-36 text-xs")

                                # 5. Score & Action
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(f"{master.max_score:.2f}").classes("text-xs font-mono text-slate-600")
                                    if master.needs_review:
                                        ui.badge("⚠️ Review", color="warning").props("dense")
                                    else:
                                        ui.badge("✓ Sicher", color="positive").props("dense outline")

                                    def make_group_ignore(term, grp):
                                        def on_click():
                                            current_ignores = parse_ignore_terms(state.ignore_terms_text)
                                            if term not in current_ignores:
                                                state.ignore_terms_text += f", {term}"
                                                ignore_input.value = state.ignore_terms_text
                                                save_current_config()
                                            grp.enabled = False
                                            ui.notify(f"'{term}' zur Ignore-Liste hinzugefügt.", type="info")
                                            refresh_preview_and_exports()
                                            build_review_table()
                                        return on_click

                                    ui.button(
                                        "Ignorieren",
                                        icon="block",
                                        on_click=make_group_ignore(master.original_text, master),
                                    ).props("flat dense size=sm color=grey-8")

                        # Expanded content: Context occurrences
                        with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                            ui.label(f"Fundstellen im Text ({master.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                            for occ_idx, occ in enumerate(master.occurrences, start=1):
                                with ui.row().classes("items-center gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                    ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                    ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                    ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

                    # Render Indented Linked Children
                    for child_node in root_node.children:
                        child: EntityGroup = child_node.item
                        with ui.expansion().classes("w-full ml-6 my-1 bg-teal-50/50 border-l-4 border-teal-500 border rounded shadow-none") as child_exp:
                            with child_exp.add_slot("header"):
                                with ui.row().classes("w-full items-center justify-between gap-3 pr-2 flex-wrap"):
                                    with ui.row().classes("items-center gap-2 min-w-[260px]"):
                                        ui.label("↳").classes("text-teal-700 font-bold text-base")
                                        ui.checkbox(value=child.enabled, on_change=make_group_check(child)).props("dense")
                                        ui.label(child.original_text).classes("font-mono text-sm font-bold text-teal-900")
                                        ui.badge(f"{child.count}x", color="teal-6").props("dense")
                                        ui.badge(child.placeholder, color="teal-9").props("outline dense").classes("text-xs font-mono font-bold")

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
                                        ).props("flat dense size=sm").tooltip("Verknüpfung aufheben und als separate Entität mit eigener Nummer führen")

                            # Child Occurrences
                            with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                                ui.label(f"Fundstellen von '{child.original_text}' ({child.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                                for occ_idx, occ in enumerate(child.occurrences, start=1):
                                    with ui.row().classes("items-center gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                        ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                        ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                        ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

    # --- Main Layout ---
    with ui.row().classes("w-full no-wrap p-4 gap-6"):
        # Sidebar: Configuration
        with ui.card().classes("w-80 p-4 shrink-0 bg-slate-50 border shadow-sm"):
            ui.label("⚙️ Konfiguration").classes("text-base font-bold text-slate-800 mb-3")

            # Format Mode Selector (Phase 3 Feature)
            ui.label("Platzhalter-Format:").classes("text-xs font-semibold text-slate-700 mb-1")
            def on_mode_change(e):
                state.format_mode = e.value
                save_current_config()
                refresh_preview_and_exports()
                build_review_table()

            ui.radio(
                {
                    "numbered": "Modus 1: [TYP_NR] (Standard)",
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
                save_current_config()
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

            ui.label("Zu anonymisierende Entitäten:").classes("text-xs font-semibold text-slate-700 mb-1")
            for ent in AVAILABLE_ENTITIES:
                def make_ent_toggle(e_name):
                    async def toggle(val):
                        if val.value and e_name not in state.active_entities:
                            state.active_entities.append(e_name)
                        elif not val.value and e_name in state.active_entities:
                            state.active_entities.remove(e_name)
                        save_current_config()
                        if state.raw_text:
                            await run_analysis()
                    return toggle

                ui.checkbox(
                    ent,
                    value=ent in state.active_entities,
                    on_change=make_ent_toggle(ent),
                ).classes("text-xs my-0")

            ui.separator().classes("my-2")

            ui.label("Erkennungs-Schwellenwert:").classes("text-xs font-semibold text-slate-700")
            def on_thresh_change(e):
                state.gliner_threshold = e.value
                save_current_config()
            thresh_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.gliner_threshold, on_change=on_thresh_change)
            ui.label().bind_text_from(thresh_slider, "value", lambda v: f"Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

            ui.separator().classes("my-2")

            with ui.expansion("Ignore-Liste (Nicht ersetzen)", icon="visibility_off").classes("w-full text-xs"):
                def on_ignore_change(e):
                    state.ignore_terms_text = e.value
                    save_current_config()
                ignore_input = ui.textarea(
                    value=state.ignore_terms_text,
                    on_change=on_ignore_change,
                ).props("outlined dense rows=4").classes("w-full font-mono text-xs")

            with ui.expansion("Fuzzy-Glossar (Eigene Begriffe)", icon="library_books").classes("w-full text-xs mt-1"):
                def on_glossary_change(e):
                    state.glossary_text = e.value
                    save_current_config()
                ui.textarea(
                    value=state.glossary_text,
                    on_change=on_glossary_change,
                ).props("outlined dense rows=4").classes("w-full font-mono text-xs")

            ui.button("Neu analysieren", icon="refresh", on_click=run_analysis, color="primary").classes("w-full mt-4").props("unelevated")

        # Main Workspace
        with ui.column().classes("flex-grow"):
            with ui.tabs().classes("w-full border-b") as tabs:
                tab_anonymize = ui.tab("🔒 Anonymisieren & Review")
                tab_restore = ui.tab("🔄 Wiederherstellen (De-Anonymize)")

            with ui.tab_panels(tabs, value=tab_anonymize).classes("w-full p-4"):
                # TAB 1: Anonymize
                with ui.tab_panel(tab_anonymize):
                    ui.label("Stufe 1: Dokument laden & Text-Eingabe").classes("text-base font-bold text-slate-800 mb-1")
                    
                    # Direct Document Upload in Main Workspace
                    with ui.card().classes("w-full p-3 bg-slate-50 border border-dashed border-slate-300 rounded-lg mb-3"):
                        with ui.row().classes("w-full items-center justify-between gap-4 flex-wrap"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("cloud_upload", size="md").classes("text-primary")
                                with ui.column().classes("gap-0"):
                                    ui.label("Dokument hier hochladen (.pdf, .docx, .txt, .md, .csv, .json):").classes("text-xs font-bold text-slate-700")
                                    ui.label("Struktur (Überschriften, Listen & Tabellen) wird automatisch als Markdown extrahiert").classes("text-[11px] text-slate-500")

                            ui.upload(
                                on_upload=handle_upload,
                                auto_upload=True,
                                max_files=1,
                            ).props("accept='.txt,.md,.docx,.pdf,.json,.csv' outlined dense flat").classes("w-80")

                    with ui.expansion("Originaltext ansehen / direkt bearbeiten (Markdown)", icon="edit_note").classes("w-full mb-4"):
                        def on_raw_text_change(e):
                            state.raw_text = e.value
                        raw_text_area = ui.textarea(
                            value=state.raw_text,
                            placeholder="Text hier eingeben oder Dokument oben hochladen...",
                            on_change=on_raw_text_change,
                        ).props("outlined rows=6").classes("w-full font-mono text-sm")

                    ui.separator().classes("my-4")

                    # Step 2: Manual Entity Marking (Bug 3b - High Priority)
                    ui.label("Stufe 2: Review-Tabelle & Manuelles Markieren").classes("text-base font-bold text-slate-800 mb-1")
                    
                    with ui.card().classes("w-full p-3 bg-blue-50/70 border border-blue-200 rounded-lg mb-3"):
                        with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                            with ui.row().classes("items-center gap-2 flex-grow"):
                                ui.icon("person_add", size="sm").classes("text-blue-700")
                                ui.label("Fehlenden Begriff / Namen manuell erfassen:").classes("text-xs font-bold text-slate-800")
                                manual_input = ui.input(placeholder="z. B. Remo").props("dense outlined bg-white").classes("w-44 text-xs")
                                manual_type = ui.select(options=AVAILABLE_ENTITIES, value="PERSON").props("dense outlined bg-white").classes("w-36 text-xs")

                                async def add_manual_entity():
                                    term = manual_input.value.strip()
                                    if not term:
                                        ui.notify("Bitte einen Begriff eingeben.", type="warning")
                                        return
                                    current_lines = [l.strip() for l in state.glossary_text.splitlines() if l.strip()]
                                    new_line = f"{term}: {manual_type.value}"
                                    if new_line not in current_lines:
                                        state.glossary_text += f"\n{new_line}"
                                        save_current_config()
                                    ui.notify(f"Begriff '{term}' als {manual_type.value} erfasst.", type="positive", icon="check")
                                    manual_input.value = ""
                                    await run_analysis()

                                ui.button("➕ Hinzufügen & Analysieren", icon="add", color="positive", on_click=add_manual_entity).props("unelevated dense size=sm")

                    table_holder = ui.column().classes("w-full mb-4")

                    ui.separator().classes("my-4")

                    ui.label("Stufe 3: Vorschau & Export").classes("text-base font-bold text-slate-800 mb-1")
                    preview_holder = ui.column().classes("w-full mb-2")
                    export_holder = ui.column().classes("w-full")

                # TAB 2: Restore
                with ui.tab_panel(tab_restore):
                    # Auto-prepopulate mapping from Tab 1 if available
                    def on_tab_restore_opened():
                        if not state.restore_mapping and state.current_mapping:
                            state.restore_mapping = dict(state.current_mapping)
                            if map_json_input is not None:
                                map_json_input.value = json.dumps(state.restore_mapping, indent=2, ensure_ascii=False)
                            ui.notify("Mapping aus aktuellem Anonymisierungs-Review automatisch übernommen.", type="info")

                    ui.label("De-Anonymisierung / Wiederherstellung").classes("text-base font-bold text-slate-800 mb-1")
                    ui.label("Laden Sie die vom Cloud-LLM beantwortete Datei (.docx, .md, .txt) oder fügen Sie den Text ein:").classes("text-sm text-slate-600 mb-4")

                    with ui.row().classes("w-full gap-4"):
                        # Column 1: Anonymized LLM Response
                        with ui.column().classes("flex-1"):
                            ui.label("1. LLM-Antwort (Text oder Dokument):").classes("font-semibold text-xs text-slate-700")

                            async def handle_restore_file_upload(e):
                                try:
                                    if hasattr(e, "file"):
                                        data = await e.file.read()
                                        filename = e.file.name
                                    elif hasattr(e, "content"):
                                        data = e.content.read()
                                        filename = e.name
                                    else:
                                        raise ValueError("Unbekanntes Upload-Format")
                                    text = read_document_from_bytes(data, filename)
                                    state.restore_anon_text = text
                                    restore_anon_input.value = text
                                    ui.notify(f"LLM-Antwort '{filename}' geladen ({len(text)} Zeichen).", type="positive")
                                except Exception as ex:
                                    ui.notify(f"Fehler beim Laden: {str(ex)}", type="negative")

                            ui.upload(
                                on_upload=handle_restore_file_upload,
                                auto_upload=True,
                                max_files=1,
                            ).props("accept='.txt,.md,.docx,.pdf' outlined dense flat").classes("w-full mb-2")

                            def on_anon_change(e):
                                state.restore_anon_text = e.value
                            restore_anon_input = ui.textarea(
                                placeholder="[PERSON_1_STUDENT_VOLLNAME] arbeitet an [ORGANIZATION_1_HOCHSCHULE]...",
                                on_change=on_anon_change,
                            ).props("outlined rows=7").classes("w-full font-mono text-xs")

                        # Column 2: Mapping File
                        with ui.column().classes("flex-1"):
                            ui.label("2. Mapping-Tabelle (.json):").classes("font-semibold text-xs text-slate-700")

                            async def on_map_upload(e):
                                try:
                                    if hasattr(e, "file"):
                                        data = await e.file.read()
                                    elif hasattr(e, "content"):
                                        data = e.content.read()
                                    else:
                                        raise ValueError("Unbekanntes Upload-Format")
                                    mapping_data = json.loads(data.decode("utf-8"))
                                    state.restore_mapping = mapping_data
                                    map_json_input.value = json.dumps(mapping_data, indent=2, ensure_ascii=False)
                                    ui.notify("Mapping-Tabelle geladen.", type="positive")
                                except Exception as ex:
                                    ui.notify(f"Ungültige JSON-Mapping-Datei: {str(ex)}", type="negative")

                            ui.upload(on_upload=on_map_upload, auto_upload=True, max_files=1).props("accept='.json' outlined dense flat").classes("w-full mb-2")

                            def on_map_text_change(e):
                                try:
                                    if e.value.strip():
                                        state.restore_mapping = json.loads(e.value)
                                except Exception:
                                    pass
                            
                            initial_map_val = json.dumps(state.restore_mapping or state.current_mapping, indent=2, ensure_ascii=False) if (state.restore_mapping or state.current_mapping) else ""
                            if not state.restore_mapping and state.current_mapping:
                                state.restore_mapping = dict(state.current_mapping)

                            map_json_input = ui.textarea(
                                value=initial_map_val,
                                placeholder='{\n  "[PERSON_1_STUDENT_VOLLNAME]": "Julia Meier"\n}',
                                on_change=on_map_text_change,
                            ).props("outlined rows=7").classes("w-full font-mono text-xs")

                    def run_restore():
                        if not state.restore_anon_text:
                            ui.notify("Bitte anonymisierten Text einfügen oder Datei hochladen.", type="warning")
                            return
                        if not state.restore_mapping:
                            ui.notify("Bitte Mapping-Tabelle laden oder eingeben.", type="warning")
                            return

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


def main():
    parser = argparse.ArgumentParser(description="Privacy-First Local Anonymizer GUI")
    parser.add_argument("--native", action="store_true", default=True, help="Launch as native desktop window (default: True)")
    parser.add_argument("--browser", action="store_true", help="Launch in default web browser instead of native window")
    parser.add_argument("--port", type=int, default=8080, help="Port to run GUI on (default: 8080)")
    args, _ = parser.parse_known_args()

    use_native = args.native and not args.browser

    create_ui()

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
