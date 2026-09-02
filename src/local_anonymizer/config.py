import json
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

CONFIG_DIR = Path.home() / ".local-anonymizer"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_FILE = CONFIG_DIR / "app.log"

ENTITY_MODE_OFF = "off"
ENTITY_MODE_EXPLICIT_ONLY = "explicit_only"
ENTITY_MODE_EXPLICIT_EUPII = "explicit_eupii"
ENTITY_MODE_ALL = "all"
VALID_ENTITY_MODES = {
    ENTITY_MODE_OFF,
    ENTITY_MODE_EXPLICIT_ONLY,
    ENTITY_MODE_EXPLICIT_EUPII,
    ENTITY_MODE_ALL,
}

# Set up global logging fallback with rotation (1MB max, 2 backups, WARNING level)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Only add handler if not already present
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    file_handler = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=1_000_000,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Silence harmless language mismatch warnings from presidio default recognizer loader
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


DEFAULT_GLINER_MODEL_NAME = "urchade/gliner_multi_pii-v1"
DEFAULT_EUPII_MODEL_NAME = "bardsai/eu-pii-anonimization-multilang"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LLM_MODEL_NAME = "qwen3:8b"
DEFAULT_LLM_PROVIDER_TYPE = "ollama"


class AppConfig:
    def __init__(self):
        self.format_mode: str = "numbered_role"
        self.active_entities: List[str] = [
            "PERSON",
            "ORGANIZATION",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "LOCATION",
        ]
        # New source-aware setting. An empty mapping keeps older config files backward-compatible;
        # the UI derives the initial modes from active_entities when no mapping exists yet.
        self.entity_modes: Dict[str, str] = {}
        self.gliner_model_name: str = DEFAULT_GLINER_MODEL_NAME
        self.gliner_threshold: float = 0.55
        self.enable_eupii: bool = True
        self.eupii_threshold: float = 0.50
        self.eupii_model_name: str = DEFAULT_EUPII_MODEL_NAME
        self.ignore_terms: str = (
            "CAS, DAS, MAS, BSc, MSc, PhD, MBA, Studierende, Studierenden, "
            "Dozent, Dozenten, Lehrperson, Berater, Aufgabensteller"
        )
        self.glossary: str = (
            "ZHAW: ORGANIZATION\nHWZ: ORGANIZATION\nUZH: ORGANIZATION\nETH: ORGANIZATION"
        )
        self.export_format: str = "txt"
        self.llm_enabled: bool = False
        self.llm_base_url: str = DEFAULT_LLM_BASE_URL
        self.llm_model_name: str = DEFAULT_LLM_MODEL_NAME
        self.llm_provider_type: str = DEFAULT_LLM_PROVIDER_TYPE
        self.llm_auto_review: bool = True

    def to_dict(self) -> dict:
        return {
            "format_mode": self.format_mode,
            "active_entities": self.active_entities,
            "entity_modes": self.entity_modes,
            "gliner_model_name": self.gliner_model_name,
            "gliner_threshold": self.gliner_threshold,
            "enable_eupii": self.enable_eupii,
            "eupii_threshold": self.eupii_threshold,
            "eupii_model_name": self.eupii_model_name,
            "ignore_terms": self.ignore_terms,
            "glossary": self.glossary,
            "export_format": self.export_format,
            "llm_enabled": self.llm_enabled,
            "llm_base_url": self.llm_base_url,
            "llm_model_name": self.llm_model_name,
            "llm_provider_type": self.llm_provider_type,
            "llm_auto_review": self.llm_auto_review,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        config = cls()
        config.format_mode = data.get("format_mode", config.format_mode)
        config.active_entities = data.get("active_entities", config.active_entities)
        raw_modes = data.get("entity_modes", {})
        if isinstance(raw_modes, dict):
            config.entity_modes = {
                str(entity): str(mode)
                for entity, mode in raw_modes.items()
                if str(mode) in VALID_ENTITY_MODES
            }
        config.gliner_model_name = str(data.get("gliner_model_name", config.gliner_model_name))
        config.gliner_threshold = data.get("gliner_threshold", config.gliner_threshold)
        config.enable_eupii = bool(data.get("enable_eupii", config.enable_eupii))
        config.eupii_threshold = float(data.get("eupii_threshold", config.eupii_threshold))
        config.eupii_model_name = str(data.get("eupii_model_name", config.eupii_model_name))
        config.ignore_terms = data.get("ignore_terms", config.ignore_terms)
        config.glossary = data.get("glossary", config.glossary)
        config.export_format = data.get("export_format", config.export_format)
        config.llm_enabled = bool(data.get("llm_enabled", False))
        config.llm_base_url = str(data.get("llm_base_url", DEFAULT_LLM_BASE_URL)).strip()
        raw_model_name = str(data.get("llm_model_name", DEFAULT_LLM_MODEL_NAME)).strip()
        if ":cloud" in raw_model_name.lower():
            raw_model_name = DEFAULT_LLM_MODEL_NAME
        config.llm_model_name = raw_model_name
        raw_provider_type = str(
            data.get("llm_provider_type", DEFAULT_LLM_PROVIDER_TYPE)
        ).strip().lower()
        config.llm_provider_type = (
            raw_provider_type
            if raw_provider_type in ("ollama", "generic")
            else DEFAULT_LLM_PROVIDER_TYPE
        )
        config.llm_auto_review = bool(data.get("llm_auto_review", True))
        return config

    def save(self) -> bool:
        """
        Save global settings to manifest.json and active project profile.
        Falls back to logging an error if ProfileStore is unavailable.
        """
        try:
            from local_anonymizer.profiles import ProfileStore, normalize_term_key, ScopedTerm
            from local_anonymizer.anonymizer import AVAILABLE_ENTITIES

            store = ProfileStore(CONFIG_DIR)
            store.initialize_or_migrate()
            manifest = store.load_manifest()
            manifest_revision = int(manifest["revision"])
            manifest.update(
                {
                    "format_mode": self.format_mode,
                    "gliner_model_name": self.gliner_model_name,
                    "gliner_threshold": self.gliner_threshold,
                    "enable_eupii": self.enable_eupii,
                    "eupii_threshold": self.eupii_threshold,
                    "eupii_model_name": self.eupii_model_name,
                    "export_format": self.export_format,
                    "llm_enabled": self.llm_enabled,
                    "llm_base_url": self.llm_base_url,
                    "llm_model_name": self.llm_model_name,
                    "llm_provider_type": self.llm_provider_type,
                    "llm_auto_review": self.llm_auto_review,
                }
            )
            store.save_manifest(manifest, expected_revision=manifest_revision)

            active_id = manifest.get("active_project_id")
            if active_id:
                try:
                    proj = store.load_project_profile(active_id)
                    if self.entity_modes:
                        proj.entity_modes = dict(self.entity_modes)

                    # Parse legacy "Term: TYPE" or "Term: TYPE | role" glossary string
                    if isinstance(self.glossary, str):
                        g_terms: dict = {}
                        for line in self.glossary.splitlines():
                            line_str = line.strip()
                            if not line_str or line_str.startswith("#"):
                                continue
                            separator = ":" if ":" in line_str else "=" if "=" in line_str else None
                            if separator is None:
                                logger.warning("Ignoring malformed glossary entry during config save: %s", line_str)
                                continue
                            parts = line_str.split(separator, 1)
                            term_text = parts[0].strip()
                            type_part = parts[1].strip()
                            role_text = None
                            if "|" in type_part:
                                tp, rp = type_part.split("|", 1)
                                ent_t = tp.strip().upper()
                                role_text = rp.strip()
                            else:
                                ent_t = type_part.upper()
                            if ent_t not in AVAILABLE_ENTITIES:
                                logger.warning("Ignoring glossary entry with unknown entity type: %s", line_str)
                                continue
                            k = normalize_term_key(term_text)
                            if k:
                                g_terms[k] = ScopedTerm(
                                    term=term_text, term_key=k, entity_type=ent_t, role=role_text
                                )
                        proj.glossary_terms = g_terms

                    # Parse legacy comma/newline-separated ignore terms
                    if isinstance(self.ignore_terms, str):
                        i_terms: dict = {}
                        for it in re.split(r"[\n,]+", self.ignore_terms):
                            it_text = it.strip()
                            if it_text:
                                k = normalize_term_key(it_text)
                                if k:
                                    i_terms[k] = ScopedTerm(term=it_text, term_key=k)
                        proj.ignore_terms = i_terms

                    store.save_project_profile(proj, expected_revision=proj.revision)
                except Exception as ex:
                    logger.warning(f"Could not save active project profile during AppConfig.save: {ex}")
                    raise
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False

    @classmethod
    def load(cls) -> "AppConfig":
        """
        Load configuration from manifest.json and active project profile.
        Returns defaults if ProfileStore fails (e.g. first-run, broken store).
        """
        try:
            from local_anonymizer.profiles import ProfileStore

            store = ProfileStore(CONFIG_DIR)
            store.initialize_or_migrate()
            manifest = store.load_manifest()

            config = cls()
            config.format_mode = str(manifest.get("format_mode", config.format_mode))
            config.gliner_model_name = str(
                manifest.get("gliner_model_name", config.gliner_model_name)
            )
            config.gliner_threshold = float(
                manifest.get("gliner_threshold", config.gliner_threshold)
            )
            config.enable_eupii = bool(manifest.get("enable_eupii", config.enable_eupii))
            config.eupii_threshold = float(
                manifest.get("eupii_threshold", config.eupii_threshold)
            )
            config.eupii_model_name = str(
                manifest.get("eupii_model_name", config.eupii_model_name)
            )
            config.export_format = str(manifest.get("export_format", config.export_format))
            config.llm_enabled = bool(manifest.get("llm_enabled", config.llm_enabled))
            config.llm_base_url = str(manifest.get("llm_base_url", config.llm_base_url))
            raw_llm_model = str(manifest.get("llm_model_name", config.llm_model_name))
            if ":cloud" in raw_llm_model.lower():
                raw_llm_model = DEFAULT_LLM_MODEL_NAME
            config.llm_model_name = raw_llm_model
            raw_provider = str(
                manifest.get("llm_provider_type", config.llm_provider_type)
            ).strip().lower()
            config.llm_provider_type = (
                raw_provider
                if raw_provider in ("ollama", "generic")
                else DEFAULT_LLM_PROVIDER_TYPE
            )
            config.llm_auto_review = bool(manifest.get("llm_auto_review", config.llm_auto_review))

            active_id = manifest.get("active_project_id")
            if active_id:
                try:
                    proj = store.load_project_profile(active_id)
                    config.entity_modes = dict(proj.entity_modes)

                    # Reconstruct glossary string from ScopedTerms
                    glossary_lines = []
                    for t in proj.glossary_terms.values():
                        if t.role:
                            glossary_lines.append(f"{t.term}: {t.entity_type} | {t.role}")
                        else:
                            glossary_lines.append(f"{t.term}: {t.entity_type}")
                    config.glossary = "\n".join(glossary_lines)

                    # Reconstruct ignore_terms string from ScopedTerms
                    config.ignore_terms = ", ".join(
                        t.term for t in proj.ignore_terms.values()
                    )
                except Exception as ex:
                    logger.warning(f"Could not load active project {active_id}: {ex}")

            return config
        except Exception as e:
            logging.error(f"Failed to load config via ProfileStore, using defaults: {e}")
            return cls()
