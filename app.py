"""
Privacy-First Local Anonymizer - NiceGUI Interactive Review Application.
Runs completely locally with in-memory processing.
"""

import argparse
import asyncio
import html
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from nicegui import app, ui

from local_anonymizer.anonymizer import AnonymizationResult, DetectedEntity, LocalAnonymizer
from local_anonymizer.extractors import UnsupportedFileFormatError, read_document_from_bytes


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
        self.occurrences: List[EntityOccurrence] = []

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
        self.filename: str = ""
        self.raw_text: str = ""
        self.anonymizer: Optional[LocalAnonymizer] = None
        self.entity_groups: List[EntityGroup] = []
        self.active_entities: List[str] = ["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"]
        self.gliner_threshold: float = 0.55
        self.sort_by: str = "Erstes Auftreten im Text"
        self.ignore_terms_text: str = "CAS, DAS, MAS, BSc, MSc, PhD, MBA, Studierende, Studierenden, Dozent, Dozenten, Lehrperson, Berater, Aufgabensteller"
        self.glossary_text: str = "ZHAW: ORGANIZATION\nHWZ: ORGANIZATION\nUZH: ORGANIZATION\nETH: ORGANIZATION"
        
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
    Recalculate placeholder substitution based on current active checkbox and type selections.
    """
    if not state.raw_text:
        return "", {}, {}

    # Flatten active occurrences from enabled entity groups
    active_occurrences = []
    for group in state.entity_groups:
        if group.enabled:
            for occ in group.occurrences:
                active_occurrences.append({
                    "original": group.original_text,
                    "type": group.entity_type,
                    "start": occ.start,
                    "end": occ.end,
                    "score": occ.score,
                    "needs_review": occ.needs_review,
                })

    # Category counters and placeholder generation
    category_counters: Dict[str, int] = {}
    value_to_placeholder: Dict[Tuple[str, str], str] = {}
    mapping: Dict[str, str] = {}
    final_entities_report = []

    for item in active_occurrences:
        orig = item["original"]
        etype = item["type"]
        norm_key = (orig.strip().lower(), etype)

        if norm_key not in value_to_placeholder:
            count = category_counters.get(etype, 0) + 1
            category_counters[etype] = count
            placeholder = f"[{etype}_{count}]"
            value_to_placeholder[norm_key] = placeholder
            mapping[placeholder] = orig
        else:
            placeholder = value_to_placeholder[norm_key]

        item["placeholder"] = placeholder
        final_entities_report.append({
            "type": etype,
            "original": orig,
            "placeholder": placeholder,
            "score": round(item["score"], 3),
            "needs_review": item["needs_review"],
        })

    # Sort accepted items by start descending to replace in reverse character order
    sorted_for_sub = sorted(active_occurrences, key=lambda x: x["start"], reverse=True)
    chars = list(state.raw_text)
    for item in sorted_for_sub:
        start, end = item["start"], item["end"]
        chars[start:end] = list(item["placeholder"])

    anonymized_text = "".join(chars)

    audit_report = {
        "source_file": state.filename,
        "entity_count": len(final_entities_report),
        "unique_entities_count": len(mapping),
        "mapping": mapping,
        "entities": final_entities_report,
    }

    return anonymized_text, mapping, audit_report


def native_save_file(default_filename: str, content: str, title: str = "Datei speichern") -> Optional[str]:
    """Open native OS save dialog on Windows/Desktop to save a file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_filename,
            defaultextension=Path(default_filename).suffix,
            filetypes=[("Dateien", f"*{Path(default_filename).suffix}"), ("Alle Dateien", "*.*")],
        )
        root.destroy()
        if filepath:
            Path(filepath).write_text(content, encoding="utf-8")
            return filepath
    except Exception as e:
        print(f"Native save dialog error: {e}")
    return None


def native_export_folder(stem: str, anon_text: str, mapping: dict, report: dict) -> Optional[str]:
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
            (out_dir / f"{stem}_anonymized.txt").write_text(anon_text, encoding="utf-8")
            (out_dir / f"{stem}_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(out_dir)
    except Exception as e:
        print(f"Native folder export error: {e}")
    return None


# --- UI Building ---
def create_ui():
    ui.colors(primary="#1976D2", secondary="#26A69A", accent="#9C27B0", positive="#2E7D32", warning="#F57C00", negative="#C62828")

    # Header
    with ui.header().classes("bg-slate-800 text-white p-4 shadow-md flex items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("lock", size="md").classes("text-teal-400")
            ui.label("Privacy-First Local Anonymizer").classes("text-xl font-bold")
            ui.badge("100% Lokal & In-Memory", color="teal").props("outline")
        ui.label("Review & Korrektur Workspace").classes("text-sm text-slate-300")

    # Reactive UI container holders
    preview_holder = None
    table_holder = None
    export_holder = None
    raw_text_area = None

    def refresh_preview_and_exports():
        if not preview_holder or not export_holder:
            return
        anon_text, mapping, audit_report = compute_reactive_preview()
        stem = Path(state.filename).stem or "dokument"

        preview_holder.clear()
        with preview_holder:
            ui.label("Anonymisierte Vorschau:").classes("font-semibold text-slate-700 mb-1")
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
                    path = native_save_file(f"{stem}_anonymized.txt", anon_text, "Anonymisierten Text speichern")
                    if path:
                        ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                    else:
                        ui.download(anon_text.encode("utf-8"), filename=f"{stem}_anonymized.txt")

                ui.button(
                    "💾 Text speichern (.txt)",
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
                    out_path = native_export_folder(stem, anon_text, mapping, audit_report)
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
                    ui.label("Dokument wird lokal analysiert (NER & Entitätserkennung)...").classes("text-slate-700 text-sm font-medium")

        try:
            anonymizer = build_anonymizer()
            # Run CPU-intensive NER in a background worker thread
            results = await asyncio.to_thread(anonymizer.analyze, state.raw_text)

            # Group detected entities by canonical term
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

            total_occurrences = sum(g.count for g in state.entity_groups)
            ui.notify(f"Analyse abgeschlossen: {len(state.entity_groups)} einzigartige Begriffe ({total_occurrences} Fundstellen).", type="positive")
            build_review_table()
            refresh_preview_and_exports()

        except ValueError as ve:
            if table_holder:
                table_holder.clear()
            ui.notify(f"Verarbeitungsfehler: {str(ve)}", type="negative", close_button=True, timeout=10000)
        except Exception as e:
            if table_holder:
                table_holder.clear()
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
            # Default: Erstes Auftreten im Text
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
            ui.label("Prüfen Sie gefundene Begriffe gebündelt. Ein Klick auf die Zeile klappt alle Fundstellen im Kontext auf:").classes("text-sm text-slate-600 mb-3")

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

            # Grouped Entity Rows (Accordion for Context Snippets)
            for idx, group in enumerate(sorted_groups):
                row_bg = "bg-amber-50" if group.needs_review else ("bg-white" if idx % 2 == 0 else "bg-slate-50")
                
                with ui.expansion().classes(f"w-full border rounded mb-1 {row_bg}") as exp:
                    with exp.add_slot("header"):
                        with ui.row().classes("w-full items-center justify-between gap-3 pr-2"):
                            # Left part: Checkbox + Name + Count Badge
                            with ui.row().classes("items-center gap-2 flex-1 min-w-[200px]"):
                                def make_group_check(grp):
                                    def on_change(e):
                                        grp.enabled = e.value
                                        refresh_preview_and_exports()
                                    return on_change

                                ui.checkbox(value=group.enabled, on_change=make_group_check(group)).props("dense")
                                ui.label(group.original_text).classes("font-mono text-sm font-bold text-slate-800")
                                ui.badge(f"{group.count}x", color="primary" if group.count > 1 else "grey-6").props("dense")

                            # Middle: Type Selector
                            with ui.row().classes("items-center gap-2"):
                                def make_group_select(grp):
                                    def on_change(e):
                                        grp.entity_type = e.value
                                        refresh_preview_and_exports()
                                    return on_change

                                ui.select(
                                    options=AVAILABLE_ENTITIES,
                                    value=group.entity_type,
                                    on_change=make_group_select(group),
                                ).props("dense outlined bg-white").classes("w-44 text-xs")

                            # Right part: Score & Review Badge
                            with ui.row().classes("items-center gap-3"):
                                ui.label(f"Score: {group.max_score:.2f}").classes("text-xs font-mono text-slate-600")
                                if group.needs_review:
                                    ui.badge("⚠️ Review", color="warning").props("dense")
                                else:
                                    ui.badge("✓ Sicher", color="positive").props("dense outline")

                                # Action: Ignore Term
                                def make_group_ignore(term, grp):
                                    def on_click():
                                        current_ignores = parse_ignore_terms(state.ignore_terms_text)
                                        if term not in current_ignores:
                                            state.ignore_terms_text += f", {term}"
                                            ignore_input.value = state.ignore_terms_text
                                        grp.enabled = False
                                        ui.notify(f"'{term}' zur Ignore-Liste hinzugefügt.", type="info")
                                        refresh_preview_and_exports()
                                        build_review_table()
                                    return on_click

                                ui.button(
                                    "Ignorieren",
                                    icon="block",
                                    on_click=make_group_ignore(group.original_text, group),
                                ).props("flat dense size=sm color=grey-8")

                    # Expanded slot: Occurrences in context
                    with ui.column().classes("p-3 bg-white border-t gap-2 w-full"):
                        ui.label(f"Fundstellen im Text ({group.count} Vorkommen):").classes("text-xs font-bold text-slate-700")
                        for occ_idx, occ in enumerate(group.occurrences, start=1):
                            with ui.row().classes("items-center gap-2 p-1.5 bg-slate-50 rounded border text-xs w-full"):
                                ui.label(f"#{occ_idx}").classes("font-bold text-slate-500 w-6")
                                ui.html(occ.context_html).classes("flex-1 text-slate-800")
                                ui.label(f"Score: {occ.score:.2f}").classes("text-slate-400 font-mono text-[10px]")

    # --- Layout ---
    with ui.row().classes("w-full no-wrap p-4 gap-6"):
        # Sidebar
        with ui.card().classes("w-80 p-4 shrink-0 bg-slate-50 border shadow-sm"):
            ui.label("⚙️ Konfiguration").classes("text-base font-bold text-slate-800 mb-3")

            ui.label("Dokument laden:").classes("text-xs font-semibold text-slate-700")
            ui.upload(
                on_upload=handle_upload,
                auto_upload=True,
                max_files=1,
            ).props("accept='.txt,.md,.docx,.pdf,.json,.csv' outlined dense flat").classes("w-full mb-4")

            ui.separator().classes("my-2")

            ui.label("Zu anonymisierende Entitäten:").classes("text-xs font-semibold text-slate-700 mb-1")
            for ent in ["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "DATE_TIME", "IBAN_CODE"]:
                def make_ent_toggle(e_name):
                    async def toggle(val):
                        if val.value and e_name not in state.active_entities:
                            state.active_entities.append(e_name)
                        elif not val.value and e_name in state.active_entities:
                            state.active_entities.remove(e_name)
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
            thresh_slider = ui.slider(min=0.20, max=0.95, step=0.05, value=state.gliner_threshold, on_change=on_thresh_change)
            ui.label().bind_text_from(thresh_slider, "value", lambda v: f"Schwellenwert: {v:.2f}").classes("text-xs text-slate-500 mb-2")

            ui.separator().classes("my-2")

            with ui.expansion("Ignore-Liste (Nicht ersetzen)", icon="visibility_off").classes("w-full text-xs"):
                def on_ignore_change(e):
                    state.ignore_terms_text = e.value
                ignore_input = ui.textarea(
                    value=state.ignore_terms_text,
                    on_change=on_ignore_change,
                ).props("outlined dense rows=4").classes("w-full font-mono text-xs")

            with ui.expansion("Fuzzy-Glossar (Eigene Begriffe)", icon="library_books").classes("w-full text-xs mt-1"):
                def on_glossary_change(e):
                    state.glossary_text = e.value
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
                    ui.label("Stufe 1: Dokument & Text-Eingabe").classes("text-base font-bold text-slate-800 mb-1")
                    
                    with ui.expansion("Originaltext anzeigen / direkt bearbeiten", icon="edit_note").classes("w-full mb-4"):
                        def on_raw_text_change(e):
                            state.raw_text = e.value
                        raw_text_area = ui.textarea(
                            value=state.raw_text,
                            placeholder="Text hier eingeben oder Dokument über die Sidebar hochladen...",
                            on_change=on_raw_text_change,
                        ).props("outlined rows=6").classes("w-full font-mono text-sm")

                    ui.separator().classes("my-4")

                    ui.label("Stufe 2: Interaktive Review-Tabelle").classes("text-base font-bold text-slate-800 mb-1")
                    table_holder = ui.column().classes("w-full mb-4")

                    ui.separator().classes("my-4")

                    ui.label("Stufe 3: Vorschau & Export").classes("text-base font-bold text-slate-800 mb-1")
                    preview_holder = ui.column().classes("w-full mb-2")
                    export_holder = ui.column().classes("w-full")

                # TAB 2: Restore
                with ui.tab_panel(tab_restore):
                    ui.label("De-Anonymisierung / Wiederherstellung").classes("text-base font-bold text-slate-800 mb-2")
                    ui.label("Fügen Sie den vom externen LLM beantworteten Text und die lokale Mapping-Tabelle ein, um das Originaldokument wiederherzustellen:").classes("text-sm text-slate-600 mb-4")

                    with ui.row().classes("w-full gap-4"):
                        with ui.column().classes("flex-1"):
                            ui.label("1. Anonymisierter Text (oder LLM-Antwort):").classes("font-semibold text-xs text-slate-700")
                            def on_anon_change(e):
                                state.restore_anon_text = e.value
                            restore_anon_input = ui.textarea(
                                placeholder="[PERSON_1] arbeitet an [ORGANIZATION_1]...",
                                on_change=on_anon_change,
                            ).props("outlined rows=8").classes("w-full font-mono text-xs")

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

                            ui.upload(on_upload=on_map_upload, auto_upload=True, max_files=1).props("accept='.json' outlined dense").classes("w-full mb-2")

                            def on_map_text_change(e):
                                try:
                                    if e.value.strip():
                                        state.restore_mapping = json.loads(e.value)
                                except Exception:
                                    pass
                            map_json_input = ui.textarea(
                                placeholder='{\n  "[PERSON_1]": "Max Mustermann"\n}',
                                on_change=on_map_text_change,
                            ).props("outlined rows=4").classes("w-full font-mono text-xs")

                    def run_restore():
                        if not state.restore_anon_text:
                            ui.notify("Bitte anonymisierten Text einfügen.", type="warning")
                            return
                        if not state.restore_mapping:
                            ui.notify("Bitte Mapping-Tabelle laden oder eingeben.", type="warning")
                            return

                        restored = LocalAnonymizer.de_anonymize(state.restore_anon_text, state.restore_mapping)
                        state.restored_text = restored
                        restored_preview.value = restored
                        ui.notify("Dokument erfolgreich wiederhergestellt!", type="positive")

                    ui.button("Wiederherstellen", icon="restore", on_click=run_restore, color="primary").classes("mt-4").props("unelevated")

                    ui.separator().classes("my-4")

                    ui.label("Wiederhergestelltes Originaldokument:").classes("font-semibold text-slate-700 mb-1")
                    restored_preview = ui.textarea().props("readonly rows=10").classes("w-full font-mono text-sm bg-slate-50 border rounded p-2")

                    with ui.row().classes("gap-3 mt-2"):
                        async def copy_restored_clipboard():
                            await ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(state.restored_text)});')
                            ui.notify("Wiederhergestellter Text in Zwischenablage kopiert!", type="positive", icon="content_copy")

                        ui.button(
                            "📋 In Zwischenablage kopieren",
                            icon="content_copy",
                            color="secondary",
                            on_click=copy_restored_clipboard,
                        ).props("unelevated")

                        def save_restored():
                            path = native_save_file("restored_document.txt", state.restored_text, "Wiederhergestellten Text speichern")
                            if path:
                                ui.notify(f"Gespeichert: {Path(path).name}", type="positive", icon="check")
                            else:
                                ui.download(state.restored_text.encode("utf-8"), filename="restored_document.txt")

                        ui.button(
                            "💾 Wiederhergestellten Text speichern",
                            icon="download",
                            color="primary",
                            on_click=save_restored,
                        ).props("unelevated")


def main():
    parser = argparse.ArgumentParser(description="Privacy-First Local Anonymizer GUI")
    parser.add_argument("--native", action="store_true", default=True, help="Launch as native desktop window (default: True)")
    parser.add_argument("--browser", action="store_true", help="Launch in default web browser instead of native window")
    parser.add_argument("--port", type=int, default=8080, help="Port to run GUI on (default: 8080)")
    args, _ = parser.parse_known_args()

    use_native = args.native and not args.browser

    create_ui()

    ui.run(
        title="Privacy-First Local Anonymizer",
        native=use_native,
        port=args.port,
        reload=False,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
