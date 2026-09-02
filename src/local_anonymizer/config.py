import json
import logging
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

# Silence harmless language mismatch warnings from presidio's default recognizer loader
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
        self.ignore_terms: str = "CAS, DAS, MAS, BSc, MSc, PhD, MBA, Studierende, Studierenden, Dozent, Dozenten, Lehrperson, Berater, Aufgabensteller"
        self.glossary: str = "ZHAW: ORGANIZATION\nHWZ: ORGANIZATION\nUZH: ORGANIZATION\nETH: ORGANIZATION"
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
        raw_provider_type = str(data.get("llm_provider_type", DEFAULT_LLM_PROVIDER_TYPE)).strip().lower()
        config.llm_provider_type = raw_provider_type if raw_provider_type in ("ollama", "generic") else DEFAULT_LLM_PROVIDER_TYPE
        config.llm_auto_review = bool(data.get("llm_auto_review", True))
        return config

    def save(self):
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                logging.error(f"Failed to load config, using defaults: {e}")
                return cls()
        return cls()
