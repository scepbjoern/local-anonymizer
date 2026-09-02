"""Model catalog management and evaluation lookup (Phase 6A.1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from local_anonymizer.llm.schema import CatalogModelEntry, CatalogSchema, validate_model_name

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


class CatalogError(Exception):
    """Raised when the static catalog cannot be loaded or is invalid."""
    pass


_CACHED_CATALOG: Optional[CatalogSchema] = None


def load_catalog(catalog_path: Optional[Path] = None, force_reload: bool = False) -> CatalogSchema:
    """
    Load and validate the bundled static model catalog JSON.
    Caches the parsed schema in-memory. Raises CatalogError on parse or validation failure.
    """
    global _CACHED_CATALOG
    if _CACHED_CATALOG is not None and not force_reload and catalog_path is None:
        return _CACHED_CATALOG

    target_path = catalog_path or DEFAULT_CATALOG_PATH
    if not target_path.exists():
        raise CatalogError(f"Modellkatalog-Datei nicht gefunden: {target_path}")

    try:
        raw_text = target_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        catalog = CatalogSchema.model_validate(data)
        if catalog_path is None:
            _CACHED_CATALOG = catalog
        return catalog
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modellkatalogs: {e}")
        raise CatalogError(f"Ungültiger oder beschädigter Modellkatalog ({type(e).__name__})") from e


def find_catalog_entry(model_name: str, catalog: Optional[CatalogSchema] = None) -> Optional[CatalogModelEntry]:
    """
    Look up a model in the catalog.
    Enforces exact tag match priority first, then checks explicit verified aliases.
    Returns None if the model is not listed in the catalog.
    """
    if not model_name:
        return None

    try:
        clean_name = validate_model_name(model_name)
    except ValueError:
        return None

    if catalog is None:
        try:
            catalog = load_catalog()
        except CatalogError:
            return None

    clean_lower = clean_name.lower()

    # Pass 1: Exact tag match on canonical_name or tested_tag
    for m in catalog.models:
        if m.canonical_name.lower() == clean_lower or m.tested_tag.lower() == clean_lower:
            return m

    # Pass 2: Match against verified aliases
    for m in catalog.models:
        if any(alias.lower() == clean_lower for alias in m.aliases):
            return m

    return None


def get_model_suitability_badge(
    model_name: str,
    phase: str = "phase_6a_triage",
    catalog: Optional[CatalogSchema] = None,
) -> Tuple[str, str, str]:
    """
    Return (display_label, color, tooltip_text) for a model name.
    If not found in catalog, returns ('Nicht evaluiert', 'grey-7', 'Unbekanntes Modell').
    """
    entry = find_catalog_entry(model_name, catalog=catalog)
    if entry is None:
        return ("Nicht evaluiert", "grey-7", "Dieses Modell wurde noch nicht offiziell für die Anonymisierung evaluiert.")

    eval_phase = getattr(entry, phase, None)
    if eval_phase is None or eval_phase.status == "untested":
        return ("Nicht evaluiert", "grey-7", eval_phase.reason if eval_phase else "Keine Testergebnisse vorhanden.")
    elif eval_phase.status == "recommended":
        return ("Empfohlen", "positive", f"Referenzmodell: {eval_phase.reason}")
    elif eval_phase.status == "suitable":
        return ("Geeignet", "primary", f"Getestet: {eval_phase.reason}")
    elif eval_phase.status == "limited":
        return ("Eingeschränkt", "warning", f"Hinweis: {eval_phase.reason}")
    elif eval_phase.status == "not_recommended":
        return ("Nicht empfohlen", "negative", f"Warnung: {eval_phase.reason}")

    return ("Nicht evaluiert", "grey-7", "Unbekannter Status.")
