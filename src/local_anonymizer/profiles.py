"""
Phase 5b.1: Project Profiles, Scopes, Category Templates, and Persistence Store.

Provides:
- ScopeLevel enum (APP_DEFAULT, SYSTEM, PROJECT, DOCUMENT)
- ScopedTerm model with strict validation and term_key normalization
- CategoryTemplate model with 5 immutable built-in presets covering all 23 entities
- SystemProfile, ProjectProfile, and DocumentProfileOverlay models
- EffectiveConfig immutable snapshot with canonical SHA-256 snapshot_hash
- ScopeResolutionEngine resolving App-Defaults -> System -> Project -> Document with overrides
- ProfileStore with platform file-locking, CAS revision checking, and atomic writes
- Resilient transactional v1 -> v2 migration with journaling and fault recovery
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import portalocker

from local_anonymizer.anonymizer import AVAILABLE_ENTITIES, DEFAULT_IGNORE_TERMS

logger = logging.getLogger(__name__)

# Canonical Entity Modes
ENTITY_MODE_ALL = "all"
ENTITY_MODE_EXPLICIT_EUPII = "explicit_eupii"
ENTITY_MODE_EXPLICIT_ONLY = "explicit_only"
ENTITY_MODE_OFF = "off"

VALID_ENTITY_MODES: Set[str] = {
    ENTITY_MODE_ALL,
    ENTITY_MODE_EXPLICIT_EUPII,
    ENTITY_MODE_EXPLICIT_ONLY,
    ENTITY_MODE_OFF,
}

# Regex for illegal control characters (line breaks are intentionally allowed
# in fields whose contract permits multiline text; all other C0 controls and
# DEL remain rejected).
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Built-in Template Identifiers
TEMPLATE_STANDARD_FULL = "builtin-standard-full"
TEMPLATE_CH_STANDARD = "builtin-ch-standard"
TEMPLATE_MEDICAL = "builtin-medical"
TEMPLATE_BUSINESS_FINANCE = "builtin-business-finance"
TEMPLATE_MINIMAL_IDS = "builtin-minimal-ids"

BUILTIN_TEMPLATE_IDS: Set[str] = {
    TEMPLATE_STANDARD_FULL,
    TEMPLATE_CH_STANDARD,
    TEMPLATE_MEDICAL,
    TEMPLATE_BUSINESS_FINANCE,
    TEMPLATE_MINIMAL_IDS,
}

MANIFEST_FIELDS: Set[str] = {
    "schema_version", "revision", "active_project_id", "default_project_id", "warning_acknowledged_version",
    "format_mode", "gliner_model_name", "gliner_threshold", "enable_eupii",
    "eupii_model_name", "eupii_threshold", "export_format", "llm_enabled",
    "llm_base_url", "llm_model_name", "llm_provider_type", "llm_auto_review",
    "migration_complete", "updated_at",
}


def normalize_term_key(term: str) -> str:
    """Canonical normalization for dictionary term keys: NFKC, strip, casefold."""
    if not term:
        return ""
    return unicodedata.normalize("NFKC", term).strip().casefold()


def validate_id_string(id_str: str, allow_builtins: bool = False) -> str:
    """
    Validate that id_str is a canonical lowercase UUIDv4 or allowed built-in ID.
    Returns canonical validated ID string.
    """
    if not isinstance(id_str, str):
        raise ValueError(f"ID must be a string, got {type(id_str)}")
    id_clean = id_str.strip()
    if allow_builtins and id_clean in BUILTIN_TEMPLATE_IDS:
        return id_clean
    try:
        val = uuid.UUID(id_clean)
        canonical = str(val)
        if canonical != id_clean.lower() or val.version != 4:
            raise ValueError(f"ID is not a canonical UUIDv4: '{id_str}'")
        return canonical
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"Invalid UUIDv4: '{id_str}'") from e


def validate_clean_string(val: str, field_name: str, max_length: int, allow_empty: bool = False) -> str:
    """Validate string length, empty constraints, and absence of illegal control characters."""
    if not isinstance(val, str):
        raise ValueError(f"{field_name} must be a string, got {type(val)}")
    trimmed = val.strip()
    if not trimmed and not allow_empty:
        raise ValueError(f"{field_name} must not be empty.")
    if len(val) > max_length:
        raise ValueError(f"{field_name} exceeds max length of {max_length} characters (length={len(val)}).")
    if CONTROL_CHAR_RE.search(val):
        raise ValueError(f"{field_name} contains illegal control characters.")
    return val


def validate_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the typed v2 manifest and reject fields unknown to this release."""
    if not isinstance(data, dict):
        raise ValueError("Manifest data must be a dict")
    unknown = set(data) - MANIFEST_FIELDS
    if unknown:
        raise ValueError(f"Unknown fields in Manifest: {unknown}")
    schema_version = data.get("schema_version", 2)
    revision = data.get("revision", 1)
    if schema_version != 2 or isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"Unsupported manifest schema_version: {schema_version}")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError(f"Manifest revision must be a positive integer: {revision}")

    active_project_id = data.get("active_project_id")
    if active_project_id is not None:
        data["active_project_id"] = validate_id_string(active_project_id)
    default_project_id = data.get("default_project_id")
    if default_project_id is not None:
        data["default_project_id"] = validate_id_string(default_project_id)
    warning_version = data.get("warning_acknowledged_version", 0)
    if isinstance(warning_version, bool) or not isinstance(warning_version, int) or warning_version < 0:
        raise ValueError("Manifest.warning_acknowledged_version must be a non-negative integer")

    bool_fields = ("enable_eupii", "llm_enabled", "llm_auto_review", "migration_complete")
    for field_name in bool_fields:
        if field_name in data and not isinstance(data[field_name], bool):
            raise ValueError(f"Manifest.{field_name} must be boolean")
    text_fields = (
        "format_mode", "gliner_model_name", "eupii_model_name", "export_format",
        "llm_base_url", "llm_model_name", "llm_provider_type",
    )
    for field_name in text_fields:
        if field_name in data:
            validate_clean_string(data[field_name], f"Manifest.{field_name}", 1000, allow_empty=True)
    numeric_fields = ("gliner_threshold", "eupii_threshold", "updated_at")
    for field_name in numeric_fields:
        if field_name in data and (isinstance(data[field_name], bool) or not isinstance(data[field_name], (int, float))):
            raise ValueError(f"Manifest.{field_name} must be numeric")
    return data


class ScopeLevel(str, Enum):
    APP_DEFAULT = "app_default"
    SYSTEM = "system"
    PROJECT = "project"
    DOCUMENT = "document"


@dataclass
class ScopedTerm:
    term: str
    term_key: str
    entity_type: Optional[str] = None
    role: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def validate(self, available_entities: Set[str], is_ignore_term: bool = False) -> None:
        validate_clean_string(self.term, "term", max_length=200, allow_empty=False)
        expected_key = normalize_term_key(self.term)
        if self.term_key != expected_key:
            raise ValueError(f"Term key mismatch: expected '{expected_key}', got '{self.term_key}'.")
        if is_ignore_term:
            if self.entity_type is not None:
                raise ValueError("Ignore term must not specify an entity_type.")
            if self.role is not None:
                raise ValueError("Ignore term must not specify a role.")
        else:
            if not self.entity_type or self.entity_type not in available_entities:
                raise ValueError(f"Glossary term requires valid entity_type from AVAILABLE_ENTITIES: '{self.entity_type}'")
            if self.role is not None and self.role.strip():
                validate_clean_string(self.role, "role", max_length=100, allow_empty=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "term_key": self.term_key,
            "entity_type": self.entity_type,
            "role": self.role,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], available_entities: Set[str], is_ignore_term: bool = False) -> "ScopedTerm":
        if not isinstance(data, dict):
            raise ValueError(f"ScopedTerm data must be a dict, got {type(data)}")
        allowed_keys = {"term", "term_key", "entity_type", "role", "created_at"}
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown fields in ScopedTerm: {unknown}")

        raw_term = data.get("term", "")
        if not isinstance(raw_term, str):
            raise ValueError("ScopedTerm.term must be a string")
        expected_key = normalize_term_key(raw_term)
        term_key = data.get("term_key", expected_key)
        if not isinstance(term_key, str):
            raise ValueError("ScopedTerm.term_key must be a string")

        ent_type = data.get("entity_type")
        if ent_type is not None and not isinstance(ent_type, str):
            raise ValueError("ScopedTerm.entity_type must be a string or null")
        ent_type_str = ent_type

        role_val = data.get("role")
        if role_val is not None and not isinstance(role_val, str):
            raise ValueError("ScopedTerm.role must be a string or null")
        role_str = role_val

        created_at = data.get("created_at", time.time())
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            raise ValueError("ScopedTerm.created_at must be numeric")

        instance = cls(
            term=raw_term,
            term_key=term_key,
            entity_type=ent_type_str,
            role=role_str,
            created_at=created_at,
        )
        instance.validate(available_entities, is_ignore_term=is_ignore_term)
        return instance


@dataclass
class CategoryTemplate:
    template_id: str
    name: str
    description: str = ""
    is_builtin: bool = False
    entity_modes: Dict[str, str] = field(default_factory=dict)
    schema_version: int = 2
    revision: int = 1

    def validate(self, available_entities: Set[str]) -> None:
        self.template_id = validate_id_string(self.template_id, allow_builtins=True)
        validate_clean_string(self.name, "name", max_length=100, allow_empty=False)
        validate_clean_string(self.description, "description", max_length=500, allow_empty=True)
        if self.schema_version != 2:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError(f"Revision must be a positive integer, got: {self.revision}")
        for ent, mode in self.entity_modes.items():
            if ent not in available_entities:
                raise ValueError(f"Invalid entity in template: '{ent}'")
            if mode not in VALID_ENTITY_MODES:
                raise ValueError(f"Invalid entity mode in template: '{mode}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "is_builtin": self.is_builtin,
            "entity_modes": dict(self.entity_modes),
            "schema_version": self.schema_version,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], available_entities: Set[str]) -> "CategoryTemplate":
        if not isinstance(data, dict):
            raise ValueError(f"CategoryTemplate data must be a dict, got {type(data)}")
        allowed_keys = {
            "template_id",
            "name",
            "description",
            "is_builtin",
            "entity_modes",
            "schema_version",
            "revision",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown fields in CategoryTemplate: {unknown}")

        template_id = data.get("template_id", "")
        name = data.get("name", "")
        description = data.get("description", "")
        if not all(isinstance(value, str) for value in (template_id, name, description)):
            raise ValueError("CategoryTemplate textual fields must be strings")
        loaded_builtin = data.get("is_builtin", False)
        if not isinstance(loaded_builtin, bool):
            raise ValueError("CategoryTemplate.is_builtin must be boolean")
        # Built-in protection: rely strictly on canonical ID matching BUILTIN_TEMPLATE_IDS
        is_builtin = template_id in BUILTIN_TEMPLATE_IDS
        raw_modes = data.get("entity_modes", {})
        if not isinstance(raw_modes, dict):
            raise ValueError("CategoryTemplate.entity_modes must be a dictionary")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw_modes.items()):
            raise ValueError("CategoryTemplate.entity_modes must map strings to strings")
        entity_modes = dict(raw_modes)
        schema_version = data.get("schema_version", 2)
        revision = data.get("revision", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("CategoryTemplate.schema_version must be an integer")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError("CategoryTemplate.revision must be a positive integer")

        instance = cls(
            template_id=template_id,
            name=name,
            description=description,
            is_builtin=is_builtin,
            entity_modes=entity_modes,
            schema_version=schema_version,
            revision=revision,
        )
        instance.validate(available_entities)
        return instance


def get_builtin_templates(available_entities: Set[str]) -> Dict[str, CategoryTemplate]:
    """Generate the 5 immutable built-in templates covering all 23 available entities."""
    # 1. Standard (Vollständig) - All 23 entities set to 'all'
    standard_modes = {e: ENTITY_MODE_ALL for e in available_entities}

    # 2. Schweizer Kontext (DSG-orientiert, unverbindlich)
    ch_standard_modes = {
        "PERSON": ENTITY_MODE_ALL,
        "ORGANIZATION": ENTITY_MODE_ALL,
        "EMAIL_ADDRESS": ENTITY_MODE_ALL,
        "PHONE_NUMBER": ENTITY_MODE_ALL,
        "LOCATION": ENTITY_MODE_ALL,
        "DATE_TIME": ENTITY_MODE_ALL,
        "IBAN_CODE": ENTITY_MODE_ALL,
        "CREDIT_CARD": ENTITY_MODE_ALL,
        "BANK_ACCOUNT": ENTITY_MODE_ALL,
        "ID_NUMBER": ENTITY_MODE_ALL,
        "FINANCIAL_DATA": ENTITY_MODE_ALL,
        "HEALTH_DATA": ENTITY_MODE_EXPLICIT_EUPII,
        "IP_ADDRESS": ENTITY_MODE_EXPLICIT_ONLY,
        "MAC_ADDRESS": ENTITY_MODE_EXPLICIT_ONLY,
        "URL": ENTITY_MODE_ALL,
        "USERNAME": ENTITY_MODE_ALL,
        "CRYPTO": ENTITY_MODE_EXPLICIT_ONLY,
        "MEDICAL_LICENSE": ENTITY_MODE_ALL,
        "ADDRESS": ENTITY_MODE_ALL,
        "AHV_NUMBER": ENTITY_MODE_ALL,
        "UID_NUMBER": ENTITY_MODE_ALL,
        "IT_SYSTEM": ENTITY_MODE_OFF,
        "ROLE": ENTITY_MODE_EXPLICIT_ONLY,
    }

    # 3. Medizin & Gesundheit
    medical_modes = {
        "PERSON": ENTITY_MODE_ALL,
        "ORGANIZATION": ENTITY_MODE_ALL,
        "EMAIL_ADDRESS": ENTITY_MODE_ALL,
        "PHONE_NUMBER": ENTITY_MODE_ALL,
        "LOCATION": ENTITY_MODE_ALL,
        "DATE_TIME": ENTITY_MODE_ALL,
        "IBAN_CODE": ENTITY_MODE_EXPLICIT_ONLY,
        "CREDIT_CARD": ENTITY_MODE_OFF,
        "BANK_ACCOUNT": ENTITY_MODE_EXPLICIT_ONLY,
        "ID_NUMBER": ENTITY_MODE_ALL,
        "FINANCIAL_DATA": ENTITY_MODE_EXPLICIT_ONLY,
        "HEALTH_DATA": ENTITY_MODE_ALL,
        "IP_ADDRESS": ENTITY_MODE_OFF,
        "MAC_ADDRESS": ENTITY_MODE_OFF,
        "URL": ENTITY_MODE_OFF,
        "USERNAME": ENTITY_MODE_OFF,
        "CRYPTO": ENTITY_MODE_OFF,
        "MEDICAL_LICENSE": ENTITY_MODE_ALL,
        "ADDRESS": ENTITY_MODE_ALL,
        "AHV_NUMBER": ENTITY_MODE_ALL,
        "UID_NUMBER": ENTITY_MODE_EXPLICIT_ONLY,
        "IT_SYSTEM": ENTITY_MODE_OFF,
        "ROLE": ENTITY_MODE_EXPLICIT_ONLY,
    }

    # 4. Geschäftskorrespondenz & Finanzen
    business_modes = {
        "PERSON": ENTITY_MODE_ALL,
        "ORGANIZATION": ENTITY_MODE_ALL,
        "EMAIL_ADDRESS": ENTITY_MODE_ALL,
        "PHONE_NUMBER": ENTITY_MODE_ALL,
        "LOCATION": ENTITY_MODE_ALL,
        "DATE_TIME": ENTITY_MODE_ALL,
        "IBAN_CODE": ENTITY_MODE_ALL,
        "CREDIT_CARD": ENTITY_MODE_ALL,
        "BANK_ACCOUNT": ENTITY_MODE_ALL,
        "ID_NUMBER": ENTITY_MODE_ALL,
        "FINANCIAL_DATA": ENTITY_MODE_ALL,
        "HEALTH_DATA": ENTITY_MODE_OFF,
        "IP_ADDRESS": ENTITY_MODE_EXPLICIT_ONLY,
        "MAC_ADDRESS": ENTITY_MODE_EXPLICIT_ONLY,
        "URL": ENTITY_MODE_ALL,
        "USERNAME": ENTITY_MODE_ALL,
        "CRYPTO": ENTITY_MODE_ALL,
        "MEDICAL_LICENSE": ENTITY_MODE_OFF,
        "ADDRESS": ENTITY_MODE_ALL,
        "AHV_NUMBER": ENTITY_MODE_ALL,
        "UID_NUMBER": ENTITY_MODE_ALL,
        "IT_SYSTEM": ENTITY_MODE_EXPLICIT_ONLY,
        "ROLE": ENTITY_MODE_EXPLICIT_ONLY,
    }

    # 5. Minimal (Direkte IDs)
    minimal_modes = {
        "PERSON": ENTITY_MODE_ALL,
        "ORGANIZATION": ENTITY_MODE_OFF,
        "EMAIL_ADDRESS": ENTITY_MODE_ALL,
        "PHONE_NUMBER": ENTITY_MODE_ALL,
        "LOCATION": ENTITY_MODE_OFF,
        "DATE_TIME": ENTITY_MODE_OFF,
        "IBAN_CODE": ENTITY_MODE_ALL,
        "CREDIT_CARD": ENTITY_MODE_ALL,
        "BANK_ACCOUNT": ENTITY_MODE_ALL,
        "ID_NUMBER": ENTITY_MODE_ALL,
        "FINANCIAL_DATA": ENTITY_MODE_OFF,
        "HEALTH_DATA": ENTITY_MODE_OFF,
        "IP_ADDRESS": ENTITY_MODE_OFF,
        "MAC_ADDRESS": ENTITY_MODE_OFF,
        "URL": ENTITY_MODE_OFF,
        "USERNAME": ENTITY_MODE_ALL,
        "CRYPTO": ENTITY_MODE_OFF,
        "MEDICAL_LICENSE": ENTITY_MODE_OFF,
        "ADDRESS": ENTITY_MODE_EXPLICIT_ONLY,
        "AHV_NUMBER": ENTITY_MODE_ALL,
        "UID_NUMBER": ENTITY_MODE_ALL,
        "IT_SYSTEM": ENTITY_MODE_OFF,
        "ROLE": ENTITY_MODE_OFF,
    }

    templates = {
        TEMPLATE_STANDARD_FULL: CategoryTemplate(
            template_id=TEMPLATE_STANDARD_FULL,
            name="Standard (Vollständig)",
            description="Aktiviert alle Erkennungsquellen für alle 23 unterstützten Kategorien.",
            is_builtin=True,
            entity_modes=standard_modes,
        ),
        TEMPLATE_CH_STANDARD: CategoryTemplate(
            template_id=TEMPLATE_CH_STANDARD,
            name="Schweizer Kontext (DSG-orientiert, unverbindlich)",
            description="Fokus auf typische Schweizer Kennungen (AHV, UID, Adressen, Bankdaten, Personen).",
            is_builtin=True,
            entity_modes=ch_standard_modes,
        ),
        TEMPLATE_MEDICAL: CategoryTemplate(
            template_id=TEMPLATE_MEDICAL,
            name="Medizin & Gesundheit",
            description="Fokus auf Patientendaten, Diagnosen, AHV/Versicherungsnummern und Arztbezeichnungen.",
            is_builtin=True,
            entity_modes=medical_modes,
        ),
        TEMPLATE_BUSINESS_FINANCE: CategoryTemplate(
            template_id=TEMPLATE_BUSINESS_FINANCE,
            name="Geschäftskorrespondenz & Finanzen",
            description="Fokus auf Verträge, Unternehmensnamen, IBAN, Kreditkarten, UID und Finanzdaten.",
            is_builtin=True,
            entity_modes=business_modes,
        ),
        TEMPLATE_MINIMAL_IDS: CategoryTemplate(
            template_id=TEMPLATE_MINIMAL_IDS,
            name="Minimal (Direkte Identifikatoren)",
            description="Schützt nur direkte Identifikatoren (Namen, E-Mail, Telefon, Ausweise, Konten).",
            is_builtin=True,
            entity_modes=minimal_modes,
        ),
    }

    for t in templates.values():
        t.validate(available_entities)
    return templates


@dataclass
class SystemProfile:
    glossary_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    ignore_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    schema_version: int = 2
    revision: int = 1
    updated_at: float = field(default_factory=time.time)

    def validate(self, available_entities: Set[str]) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Unsupported schema_version in SystemProfile: {self.schema_version}")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError(f"Revision must be a positive integer, got: {self.revision}")
        for k, term in self.glossary_terms.items():
            if k != term.term_key:
                raise ValueError(f"Key '{k}' does not match glossary term_key '{term.term_key}'")
            term.validate(available_entities, is_ignore_term=False)
        for k, term in self.ignore_terms.items():
            if k != term.term_key:
                raise ValueError(f"Key '{k}' does not match ignore term_key '{term.term_key}'")
            term.validate(available_entities, is_ignore_term=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "glossary_terms": {k: t.to_dict() for k, t in self.glossary_terms.items()},
            "ignore_terms": {k: t.to_dict() for k, t in self.ignore_terms.items()},
            "schema_version": self.schema_version,
            "revision": self.revision,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], available_entities: Set[str]) -> "SystemProfile":
        if not isinstance(data, dict):
            raise ValueError(f"SystemProfile data must be a dict, got {type(data)}")
        allowed_keys = {"glossary_terms", "ignore_terms", "schema_version", "revision", "updated_at"}
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown fields in SystemProfile: {unknown}")

        schema_version = data.get("schema_version", 2)
        revision = data.get("revision", 1)
        updated_at = data.get("updated_at", time.time())
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("SystemProfile.schema_version must be an integer")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError("SystemProfile.revision must be a positive integer")
        if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
            raise ValueError("SystemProfile.updated_at must be numeric")

        raw_glossary = data.get("glossary_terms", {})
        raw_ignore = data.get("ignore_terms", {})
        if not isinstance(raw_glossary, dict) or not isinstance(raw_ignore, dict):
            raise ValueError("SystemProfile term collections must be dictionaries")

        glossary_terms: Dict[str, ScopedTerm] = {}
        for k, v in raw_glossary.items():
            if not isinstance(k, str):
                raise ValueError("SystemProfile glossary keys must be strings")
            glossary_terms[k] = ScopedTerm.from_dict(v, available_entities, is_ignore_term=False)

        ignore_terms: Dict[str, ScopedTerm] = {}
        for k, v in raw_ignore.items():
            if not isinstance(k, str):
                raise ValueError("SystemProfile ignore keys must be strings")
            ignore_terms[k] = ScopedTerm.from_dict(v, available_entities, is_ignore_term=True)

        instance = cls(
            glossary_terms=glossary_terms,
            ignore_terms=ignore_terms,
            schema_version=schema_version,
            revision=revision,
            updated_at=updated_at,
        )
        instance.validate(available_entities)
        return instance


@dataclass
class ProjectProfile:
    project_id: str
    project_name: str
    description: str = ""
    is_default: bool = False
    template_id: Optional[str] = None
    entity_modes: Dict[str, str] = field(default_factory=dict)
    glossary_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    ignore_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    disabled_inherited_glossary: Set[str] = field(default_factory=set)
    disabled_inherited_ignore: Set[str] = field(default_factory=set)
    schema_version: int = 2
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def validate(self, available_entities: Set[str]) -> None:
        self.project_id = validate_id_string(self.project_id)
        validate_clean_string(self.project_name, "project_name", max_length=100, allow_empty=False)
        validate_clean_string(self.description, "description", max_length=500, allow_empty=True)
        if self.template_id:
            self.template_id = validate_id_string(self.template_id, allow_builtins=True)
        if self.schema_version != 2:
            raise ValueError(f"Unsupported schema_version in ProjectProfile: {self.schema_version}")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError(f"Revision must be a positive integer, got: {self.revision}")
        for ent, mode in self.entity_modes.items():
            if ent not in available_entities:
                raise ValueError(f"Invalid entity in project: '{ent}'")
            if mode not in VALID_ENTITY_MODES:
                raise ValueError(f"Invalid entity mode in project: '{mode}'")
        for k, term in self.glossary_terms.items():
            if k != term.term_key:
                raise ValueError(f"Key '{k}' does not match glossary term_key '{term.term_key}'")
            term.validate(available_entities, is_ignore_term=False)
        for k, term in self.ignore_terms.items():
            if k != term.term_key:
                raise ValueError(f"Key '{k}' does not match ignore term_key '{term.term_key}'")
            term.validate(available_entities, is_ignore_term=True)
        for key in (*self.disabled_inherited_glossary, *self.disabled_inherited_ignore):
            if not isinstance(key, str) or key != normalize_term_key(key):
                raise ValueError("Disabled inherited keys must be normalized strings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "description": self.description,
            "is_default": self.is_default,
            "template_id": self.template_id,
            "entity_modes": dict(self.entity_modes),
            "glossary_terms": {k: t.to_dict() for k, t in self.glossary_terms.items()},
            "ignore_terms": {k: t.to_dict() for k, t in self.ignore_terms.items()},
            "disabled_inherited_glossary": sorted(list(self.disabled_inherited_glossary)),
            "disabled_inherited_ignore": sorted(list(self.disabled_inherited_ignore)),
            "schema_version": self.schema_version,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], available_entities: Set[str]) -> "ProjectProfile":
        if not isinstance(data, dict):
            raise ValueError(f"ProjectProfile data must be a dict, got {type(data)}")
        allowed_keys = {
            "project_id",
            "project_name",
            "description",
            "is_default",
            "template_id",
            "entity_modes",
            "glossary_terms",
            "ignore_terms",
            "disabled_inherited_glossary",
            "disabled_inherited_ignore",
            "schema_version",
            "revision",
            "created_at",
            "updated_at",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown fields in ProjectProfile: {unknown}")

        project_id = data.get("project_id", "")
        project_name = data.get("project_name", "")
        description = data.get("description", "")
        is_default = data.get("is_default", False)
        if not all(isinstance(value, str) for value in (project_id, project_name, description)):
            raise ValueError("ProjectProfile textual fields must be strings")
        if not isinstance(is_default, bool):
            raise ValueError("ProjectProfile.is_default must be boolean")
        template_id = data.get("template_id")
        if template_id is not None and not isinstance(template_id, str):
            raise ValueError("ProjectProfile.template_id must be a string or null")
        template_id_str = template_id

        raw_modes = data.get("entity_modes", {})
        raw_glossary = data.get("glossary_terms", {})
        raw_ignore = data.get("ignore_terms", {})
        raw_disabled_glossary = data.get("disabled_inherited_glossary", [])
        raw_disabled_ignore = data.get("disabled_inherited_ignore", [])
        if not all(isinstance(value, dict) for value in (raw_modes, raw_glossary, raw_ignore)):
            raise ValueError("ProjectProfile maps must be dictionaries")
        if not isinstance(raw_disabled_glossary, list) or not isinstance(raw_disabled_ignore, list):
            raise ValueError("ProjectProfile disabled inherited fields must be lists")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw_modes.items()):
            raise ValueError("ProjectProfile.entity_modes must map strings to strings")
        if any(not isinstance(k, str) for k in (*raw_glossary.keys(), *raw_ignore.keys())):
            raise ValueError("ProjectProfile term keys must be strings")
        if any(not isinstance(value, str) for value in (*raw_disabled_glossary, *raw_disabled_ignore)):
            raise ValueError("ProjectProfile disabled inherited keys must be strings")
        entity_modes = dict(raw_modes)

        glossary_terms: Dict[str, ScopedTerm] = {}
        for k, v in raw_glossary.items():
            glossary_terms[k] = ScopedTerm.from_dict(v, available_entities, is_ignore_term=False)

        ignore_terms: Dict[str, ScopedTerm] = {}
        for k, v in raw_ignore.items():
            ignore_terms[k] = ScopedTerm.from_dict(v, available_entities, is_ignore_term=True)

        disabled_glossary = set(raw_disabled_glossary)
        disabled_ignore = set(raw_disabled_ignore)

        schema_version = data.get("schema_version", 2)
        revision = data.get("revision", 1)
        created_at = data.get("created_at", time.time())
        updated_at = data.get("updated_at", time.time())
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("ProjectProfile.schema_version must be an integer")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError("ProjectProfile.revision must be a positive integer")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (created_at, updated_at)):
            raise ValueError("ProjectProfile timestamps must be numeric")

        instance = cls(
            project_id=project_id,
            project_name=project_name,
            description=description,
            is_default=is_default,
            template_id=template_id_str,
            entity_modes=entity_modes,
            glossary_terms=glossary_terms,
            ignore_terms=ignore_terms,
            disabled_inherited_glossary=disabled_glossary,
            disabled_inherited_ignore=disabled_ignore,
            schema_version=schema_version,
            revision=revision,
            created_at=created_at,
            updated_at=updated_at,
        )
        instance.validate(available_entities)
        return instance


@dataclass
class DocumentProfileOverlay:
    """Pure in-memory transient overlay attached to AppState."""
    entity_modes: Dict[str, str] = field(default_factory=dict)
    glossary_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    ignore_terms: Dict[str, ScopedTerm] = field(default_factory=dict)
    disabled_inherited_glossary: Set[str] = field(default_factory=set)
    disabled_inherited_ignore: Set[str] = field(default_factory=set)
    dirty: bool = False
    overlay_revision: int = 1

    def validate(self, available_entities: Set[str]) -> None:
        if not isinstance(self.dirty, bool):
            raise ValueError("DocumentProfileOverlay.dirty must be boolean")
        if isinstance(self.overlay_revision, bool) or not isinstance(self.overlay_revision, int) or self.overlay_revision < 1:
            raise ValueError("DocumentProfileOverlay.overlay_revision must be a positive integer")
        if not isinstance(self.entity_modes, dict) or not isinstance(self.glossary_terms, dict) or not isinstance(self.ignore_terms, dict):
            raise ValueError("DocumentProfileOverlay maps must be dictionaries")
        for ent, mode in self.entity_modes.items():
            if not isinstance(ent, str) or not isinstance(mode, str):
                raise ValueError("DocumentProfileOverlay.entity_modes must map strings to strings")
            if ent not in available_entities or mode not in VALID_ENTITY_MODES:
                raise ValueError(f"Invalid overlay entity mode: {ent}={mode}")
        for key, term in self.glossary_terms.items():
            if not isinstance(key, str) or key != term.term_key:
                raise ValueError("DocumentProfileOverlay glossary keys must match term_key")
            term.validate(available_entities, is_ignore_term=False)
        for key, term in self.ignore_terms.items():
            if not isinstance(key, str) or key != term.term_key:
                raise ValueError("DocumentProfileOverlay ignore keys must match term_key")
            term.validate(available_entities, is_ignore_term=True)
        for key in (*self.disabled_inherited_glossary, *self.disabled_inherited_ignore):
            if not isinstance(key, str) or key != normalize_term_key(key):
                raise ValueError("DocumentProfileOverlay disabled keys must be normalized strings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_modes": dict(self.entity_modes),
            "glossary_terms": {k: term.to_dict() for k, term in self.glossary_terms.items()},
            "ignore_terms": {k: term.to_dict() for k, term in self.ignore_terms.items()},
            "disabled_inherited_glossary": sorted(self.disabled_inherited_glossary),
            "disabled_inherited_ignore": sorted(self.disabled_inherited_ignore),
            "dirty": self.dirty,
            "overlay_revision": self.overlay_revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], available_entities: Set[str]) -> "DocumentProfileOverlay":
        if not isinstance(data, dict):
            raise ValueError("DocumentProfileOverlay data must be a dict")
        allowed = {
            "entity_modes", "glossary_terms", "ignore_terms",
            "disabled_inherited_glossary", "disabled_inherited_ignore",
            "dirty", "overlay_revision",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in DocumentProfileOverlay: {unknown}")
        raw_modes = data.get("entity_modes", {})
        raw_glossary = data.get("glossary_terms", {})
        raw_ignore = data.get("ignore_terms", {})
        raw_disabled_glossary = data.get("disabled_inherited_glossary", [])
        raw_disabled_ignore = data.get("disabled_inherited_ignore", [])
        if not all(isinstance(value, dict) for value in (raw_modes, raw_glossary, raw_ignore)):
            raise ValueError("DocumentProfileOverlay maps must be dictionaries")
        if not isinstance(raw_disabled_glossary, list) or not isinstance(raw_disabled_ignore, list):
            raise ValueError("DocumentProfileOverlay disabled fields must be lists")
        overlay = cls(
            entity_modes=dict(raw_modes),
            glossary_terms={k: ScopedTerm.from_dict(v, available_entities, False) for k, v in raw_glossary.items()},
            ignore_terms={k: ScopedTerm.from_dict(v, available_entities, True) for k, v in raw_ignore.items()},
            disabled_inherited_glossary=set(raw_disabled_glossary),
            disabled_inherited_ignore=set(raw_disabled_ignore),
            dirty=data.get("dirty", False),
            overlay_revision=data.get("overlay_revision", 1),
        )
        overlay.validate(available_entities)
        return overlay


class EffectiveConfig:
    """
    Immutable computed configuration snapshot consumed by LocalAnonymizer.
    Guarantees true immutability via MappingProxyType, Tuple, and __setattr__ protection.
    """
    __slots__ = (
        "_project_id",
        "_project_name",
        "_entity_modes",
        "_glossary",
        "_glossary_roles",
        "_glossary_provenance",
        "_ignore_terms",
        "_ignore_provenance",
        "_conflict_terms",
        "_system_revision",
        "_project_revision",
        "_overlay_revision",
        "_config_revision",
        "_snapshot_hash",
    )

    def __init__(
        self,
        project_id: str,
        project_name: str,
        entity_modes: Dict[str, str],
        glossary: Dict[str, str],
        glossary_roles: Dict[str, str],
        glossary_provenance: Dict[str, Tuple[ScopeLevel, str]],
        ignore_terms: List[str],
        ignore_provenance: Dict[str, Tuple[ScopeLevel, str]],
        conflict_terms: Optional[Dict[str, str]] = None,
        system_revision: int = 1,
        project_revision: int = 1,
        overlay_revision: int = 1,
    ):
        object.__setattr__(self, "_project_id", project_id)
        object.__setattr__(self, "_project_name", project_name)
        object.__setattr__(self, "_entity_modes", MappingProxyType(dict(entity_modes)))
        object.__setattr__(self, "_glossary", MappingProxyType(dict(glossary)))
        object.__setattr__(self, "_glossary_roles", MappingProxyType(dict(glossary_roles)))
        object.__setattr__(self, "_glossary_provenance", MappingProxyType(dict(glossary_provenance)))
        object.__setattr__(self, "_ignore_terms", tuple(sorted(list(ignore_terms))))
        object.__setattr__(self, "_ignore_provenance", MappingProxyType(dict(ignore_provenance)))
        object.__setattr__(self, "_conflict_terms", MappingProxyType(dict(conflict_terms or {})))
        object.__setattr__(self, "_system_revision", system_revision)
        object.__setattr__(self, "_project_revision", project_revision)
        object.__setattr__(self, "_overlay_revision", overlay_revision)

        # Keep the three revision dimensions structured; arithmetic packing
        # would make distinct revision triples collide once a component grows.
        object.__setattr__(self, "_config_revision", (system_revision, project_revision, overlay_revision))

        # Canonical deterministic SHA-256 hash over all behavior-affecting parameters
        hash_payload = {
            "project_id": project_id,
            "system_revision": system_revision,
            "project_revision": project_revision,
            "overlay_revision": overlay_revision,
            "entity_modes": dict(self._entity_modes),
            "glossary": dict(self._glossary),
            "glossary_roles": dict(self._glossary_roles),
            "ignore_terms": list(self._ignore_terms),
        }
        canonical_json = json.dumps(hash_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        object.__setattr__(self, "_snapshot_hash", hashlib.sha256(canonical_json.encode("utf-8")).hexdigest())

    def __setattr__(self, name, value):
        raise TypeError(f"EffectiveConfig is immutable. Cannot set attribute '{name}'.")

    def __delattr__(self, name):
        raise TypeError(f"EffectiveConfig is immutable. Cannot delete attribute '{name}'.")

    @property
    def project_id(self) -> str: return self._project_id
    @property
    def project_name(self) -> str: return self._project_name
    @property
    def entity_modes(self) -> Mapping[str, str]: return self._entity_modes
    @property
    def glossary(self) -> Mapping[str, str]: return self._glossary
    @property
    def glossary_roles(self) -> Mapping[str, str]: return self._glossary_roles
    @property
    def glossary_provenance(self) -> Mapping[str, Tuple[ScopeLevel, str]]: return self._glossary_provenance
    @property
    def ignore_terms(self) -> Tuple[str, ...]: return self._ignore_terms
    @property
    def ignore_provenance(self) -> Mapping[str, Tuple[ScopeLevel, str]]: return self._ignore_provenance
    @property
    def conflict_terms(self) -> Mapping[str, str]: return self._conflict_terms
    @property
    def system_revision(self) -> int: return self._system_revision
    @property
    def project_revision(self) -> int: return self._project_revision
    @property
    def overlay_revision(self) -> int: return self._overlay_revision
    @property
    def config_revision(self) -> Tuple[int, int, int]: return self._config_revision
    @property
    def snapshot_hash(self) -> str: return self._snapshot_hash

    def to_anonymizer_kwargs(self) -> Dict[str, Any]:
        """Adapter: extract kwargs for LocalAnonymizer."""
        return {
            "entity_modes": dict(self._entity_modes),
            "glossary": dict(self._glossary),
            "glossary_roles": dict(self._glossary_roles),
            "ignore_terms": list(self._ignore_terms),
        }


class ScopeResolutionEngine:
    """
    Resolves configuration across:
    Level 0: App-Defaults (DEFAULT_IGNORE_TERMS, default modes)
    Level 1: System Profile (system_profile.json)
    Level 2: Project Profile (projects/<project_id>.json)
    Level 3: Document Profile Overlay (in-memory)
    """

    @classmethod
    def resolve(
        cls,
        system_profile: SystemProfile,
        project_profile: ProjectProfile,
        document_overlay: Optional[DocumentProfileOverlay] = None,
        available_entities: Optional[Set[str]] = None,
    ) -> EffectiveConfig:
        if available_entities is None:
            available_entities = set(AVAILABLE_ENTITIES)

        doc = document_overlay or DocumentProfileOverlay()

        # 1. Resolve Entity Modes (Default 'all' -> Project entity_modes -> Doc entity_modes)
        resolved_modes: Dict[str, str] = {e: ENTITY_MODE_ALL for e in available_entities}
        for ent, mode in project_profile.entity_modes.items():
            if ent in available_entities and mode in VALID_ENTITY_MODES:
                resolved_modes[ent] = mode
        for ent, mode in doc.entity_modes.items():
            if ent in available_entities and mode in VALID_ENTITY_MODES:
                resolved_modes[ent] = mode

        # 2. Scope resolution for Glossary & Ignore lists
        # Track: active_action[key] -> ('glossary'|'ignore', level, term_obj, detail_label)
        active_actions: Dict[str, Tuple[str, ScopeLevel, ScopedTerm, str]] = {}
        conflict_terms: Dict[str, str] = {}

        # Layer 0: App Defaults (DEFAULT_IGNORE_TERMS)
        for term_str in DEFAULT_IGNORE_TERMS:
            k = normalize_term_key(term_str)
            if k:
                term_obj = ScopedTerm(term=term_str, term_key=k)
                active_actions[k] = ("ignore", ScopeLevel.APP_DEFAULT, term_obj, "App-Default")

        # Layer 1: System Profile
        # 1a. System Ignore
        for k, term in system_profile.ignore_terms.items():
            active_actions[k] = ("ignore", ScopeLevel.SYSTEM, term, "System")
        # 1b. System Glossary
        for k, term in system_profile.glossary_terms.items():
            if k in system_profile.ignore_terms:
                # Intra-scope conflict on system level: privacy-conservative fallback (glossary wins)
                conflict_terms[k] = f"System conflict on '{term.term}': glossary overrides ignore"
            active_actions[k] = ("glossary", ScopeLevel.SYSTEM, term, "System")

        # Layer 2: Project Profile overrides
        # 2a. Apply disabled inherited items
        for k in project_profile.disabled_inherited_glossary:
            if k in active_actions and active_actions[k][0] == "glossary" and active_actions[k][1] == ScopeLevel.SYSTEM:
                del active_actions[k]
        for k in project_profile.disabled_inherited_ignore:
            if k in active_actions and active_actions[k][0] == "ignore" and active_actions[k][1] == ScopeLevel.SYSTEM:
                del active_actions[k]

        # 2b. Project Ignore
        proj_label = f"Projekt: {project_profile.project_name}"
        for k, term in project_profile.ignore_terms.items():
            active_actions[k] = ("ignore", ScopeLevel.PROJECT, term, proj_label)

        # 2c. Project Glossary (overrides ignore if higher or intra-scope with privacy-conservative rule)
        for k, term in project_profile.glossary_terms.items():
            if k in project_profile.ignore_terms:
                conflict_terms[k] = f"Project conflict on '{term.term}': glossary overrides ignore"
            active_actions[k] = ("glossary", ScopeLevel.PROJECT, term, proj_label)

        # Layer 3: Document Overlay overrides
        for k in doc.disabled_inherited_glossary:
            if k in active_actions and active_actions[k][0] == "glossary":
                del active_actions[k]
        for k in doc.disabled_inherited_ignore:
            if k in active_actions and active_actions[k][0] == "ignore":
                del active_actions[k]

        for k, term in doc.ignore_terms.items():
            active_actions[k] = ("ignore", ScopeLevel.DOCUMENT, term, "Dokument")
        for k, term in doc.glossary_terms.items():
            if k in doc.ignore_terms:
                conflict_terms[k] = f"Document conflict on '{term.term}': glossary overrides ignore"
            active_actions[k] = ("glossary", ScopeLevel.DOCUMENT, term, "Dokument")

        # Assemble final mappings
        glossary_map: Dict[str, str] = {}
        glossary_roles: Dict[str, str] = {}
        glossary_provenance: Dict[str, Tuple[ScopeLevel, str]] = {}
        ignore_terms_list: List[str] = []
        ignore_provenance: Dict[str, Tuple[ScopeLevel, str]] = {}

        for k, (action, scope, term_obj, label) in active_actions.items():
            if action == "glossary" and term_obj.entity_type:
                glossary_map[term_obj.term] = term_obj.entity_type
                if term_obj.role:
                    glossary_roles[term_obj.term] = term_obj.role
                glossary_provenance[k] = (scope, label)
            elif action == "ignore":
                ignore_terms_list.append(term_obj.term)
                ignore_provenance[k] = (scope, label)

        return EffectiveConfig(
            project_id=project_profile.project_id,
            project_name=project_profile.project_name,
            entity_modes=resolved_modes,
            glossary=glossary_map,
            glossary_roles=glossary_roles,
            glossary_provenance=glossary_provenance,
            ignore_terms=ignore_terms_list,
            ignore_provenance=ignore_provenance,
            conflict_terms=conflict_terms,
            system_revision=system_profile.revision,
            project_revision=project_profile.revision,
            overlay_revision=doc.overlay_revision,
        )


class RevisionConflictError(Exception):
    """Raised when on-disk revision does not match expected memory revision (CAS failure)."""
    pass


class ProfileStore:
    """
    Transactional, file-locked on-disk persistence manager for ~/.local-anonymizer/.
    Handles Manifest, SystemProfile, ProjectProfiles, and CustomTemplates.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or (Path.home() / ".local-anonymizer")
        self.lock_file = self.root_dir / "store.lock"
        self.manifest_file = self.root_dir / "manifest.json"
        self.system_file = self.root_dir / "system_profile.json"
        self.projects_dir = self.root_dir / "projects"
        self.templates_dir = self.root_dir / "templates"
        self.backups_dir = self.root_dir / "backups"
        self.journal_file = self.root_dir / "migration.journal"
        self.available_entities = set(AVAILABLE_ENTITIES)
        self._thread_lock = threading.RLock()

        # Ensure directories exist
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _file_lock(self):
        """Platform-safe file lock on ~/.local-anonymizer/store.lock exclusively."""
        # Blocking mode serializes writers across processes; passing timeout
        # together with LOCK_EX is ignored by portalocker on Windows.
        return portalocker.Lock(str(self.lock_file), flags=portalocker.LOCK_EX)

    def _atomic_write_json(self, target_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomatically using a unique temp file and os.replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{target_path.name}.tmp.{uuid.uuid4().hex[:8]}"
        tmp_path = target_path.parent / tmp_name
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, target_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _load_manifest_unlocked(self) -> Dict[str, Any]:
        return validate_manifest(json.loads(self.manifest_file.read_text(encoding="utf-8")))

    def _load_project_profile_unlocked(self, project_id: str) -> ProjectProfile:
        proj_file = self.projects_dir / f"{project_id}.json"
        if not proj_file.exists():
            raise FileNotFoundError(f"Project profile not found: {project_id}")
        data = json.loads(proj_file.read_text(encoding="utf-8"))
        return ProjectProfile.from_dict(data, self.available_entities)

    def _list_projects_unlocked(self) -> List[ProjectProfile]:
        results: List[ProjectProfile] = []
        for f in self.projects_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(ProjectProfile.from_dict(data, self.available_entities))
            except Exception as ex:
                logger.warning(f"Could not load project file {f.name}: {ex}")
        results.sort(key=lambda p: (not p.is_default, p.project_name.casefold()))
        return results

    def _load_template_unlocked(self, template_id: str) -> Optional[CategoryTemplate]:
        builtins = get_builtin_templates(self.available_entities)
        if template_id in builtins:
            return builtins[template_id]
        canon_id = validate_id_string(template_id, allow_builtins=True)
        tpl_file = self.templates_dir / f"{canon_id}.json"
        if not tpl_file.exists():
            return None
        data = json.loads(tpl_file.read_text(encoding="utf-8"))
        return CategoryTemplate.from_dict(data, self.available_entities)

    def _list_custom_templates_unlocked(self) -> List[CategoryTemplate]:
        results: List[CategoryTemplate] = []
        for f in self.templates_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(CategoryTemplate.from_dict(data, self.available_entities))
            except Exception as ex:
                logger.warning(f"Could not load template file {f.name}: {ex}")
        results.sort(key=lambda t: t.name.casefold())
        return results

    def _save_manifest_unlocked(
        self, manifest_data: Dict[str, Any], expected_revision: Optional[int] = None
    ) -> Dict[str, Any]:
        validate_manifest(manifest_data)
        current = self._load_manifest_unlocked()
        curr_rev = current.get("revision", 1)
        if expected_revision is not None and curr_rev != expected_revision:
            raise RevisionConflictError(
                f"Manifest revision conflict: on_disk={curr_rev}, expected={expected_revision}"
            )
        new_manifest = dict(manifest_data)
        new_manifest["revision"] = curr_rev + 1
        new_manifest["updated_at"] = time.time()
        validate_manifest(new_manifest)
        self._atomic_write_json(self.manifest_file, new_manifest)
        return new_manifest

    def _save_project_profile_unlocked(
        self, profile: ProjectProfile, expected_revision: Optional[int] = None
    ) -> ProjectProfile:
        canon_id = validate_id_string(profile.project_id)
        profile.validate(self.available_entities)
        norm_name = profile.project_name.strip().casefold()
        for other_p in self._list_projects_unlocked():
            if other_p.project_id != canon_id and other_p.project_name.strip().casefold() == norm_name:
                raise ValueError(f"A project with name '{profile.project_name}' already exists.")

        proj_file = self.projects_dir / f"{canon_id}.json"
        if proj_file.exists():
            curr_data = json.loads(proj_file.read_text(encoding="utf-8"))
            curr_rev = curr_data.get("revision", 1)
            if expected_revision is not None and curr_rev != expected_revision:
                raise RevisionConflictError(
                    f"Project revision conflict: on_disk={curr_rev}, expected={expected_revision}"
                )
            profile.revision = curr_rev + 1
        else:
            profile.revision = 1
        profile.updated_at = time.time()
        self._atomic_write_json(proj_file, profile.to_dict())
        return profile

    # --- Migration and Initialization ---

    def initialize_or_migrate(self, legacy_config_file: Optional[Path] = None) -> None:
        """Run journaled migration from legacy config.json if needed, or initialize fresh stores."""
        with self._thread_lock, self._file_lock():
            self._initialize_or_migrate_unlocked(legacy_config_file)

    def _initialize_or_migrate_unlocked(self, legacy_config_file: Optional[Path] = None) -> None:
        """Initialize or migrate while the caller owns the store lock."""
        # 1. Recover pending journal if present
        if self.journal_file.exists():
            self._resume_journal_migration()
            return

        # 2. If manifest exists and valid, store is already initialized
        if self.manifest_file.exists():
            try:
                manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
                if manifest.get("schema_version") == 2:
                    return
            except Exception:
                logger.warning("Manifest exists but could not be read cleanly; will inspect migration.")

        # 3. Check for legacy config.json
        legacy_path = legacy_config_file or (self.root_dir / "config.json")
        if legacy_path.exists():
            self._perform_migration(legacy_path)
        else:
            self._initialize_fresh_defaults()

    def _perform_migration(
        self, legacy_path: Path, existing_journal: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform or resume a journaled v1 -> v2 migration without changing its UUID."""
        journal_data = dict(existing_journal or {})
        source_path = legacy_path
        if source_path.exists():
            raw_bytes = source_path.read_bytes()
        else:
            source_hash = str(journal_data.get("source_hash", ""))
            backup_path = self.backups_dir / f"config.v1.backup.{source_hash[:12]}.json"
            if not source_hash or not backup_path.exists():
                raise FileNotFoundError(f"Legacy configuration and migration backup are missing: {source_path}")
            raw_bytes = backup_path.read_bytes()
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        if journal_data.get("source_hash") and journal_data["source_hash"] != source_hash:
            raise ValueError("Legacy configuration changed while migration was in progress")

        default_uuid = validate_id_string(
            journal_data.get("default_project_id") or str(uuid.uuid4())
        )
        journal_data.update({
            "source_hash": source_hash,
            "default_project_id": default_uuid,
            "legacy_file": str(source_path),
            "target_files": {
                "backup": str(self.backups_dir / f"config.v1.backup.{source_hash[:12]}.json"),
                "project": str(self.projects_dir / f"{default_uuid}.json"),
                "system": str(self.system_file),
                "manifest": str(self.manifest_file),
                "migrated_legacy": str(self.root_dir / "config.v1.migrated.json"),
            },
        })
        step = journal_data.get("step", "INITIALIZING")
        if step not in {"INITIALIZING", "BACKUP_CREATED", "PROJECT_WRITTEN", "SYSTEM_WRITTEN", "MANIFEST_WRITTEN", "LEGACY_RENAMED"}:
            raise ValueError(f"Unknown migration journal step: {step}")
        if not existing_journal:
            journal_data["step"] = "INITIALIZING"
            self._atomic_write_json(self.journal_file, journal_data)
            step = "INITIALIZING"

        legacy_data = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(legacy_data, dict):
            raise ValueError("Legacy configuration must contain a JSON object")

        def advance(next_step: str) -> None:
            nonlocal step
            journal_data["step"] = next_step
            self._atomic_write_json(self.journal_file, journal_data)
            step = next_step

        backup_path = self.backups_dir / f"config.v1.backup.{source_hash[:12]}.json"
        if step == "INITIALIZING":
            if not backup_path.exists():
                backup_path.write_bytes(raw_bytes)
            advance("BACKUP_CREATED")

        # Preserve the real legacy mode derivation: stored modes win; only the
        # four EU-PII categories use explicit_eupii when derived from active_entities.
        raw_modes = legacy_data.get("entity_modes", {})
        if not isinstance(raw_modes, dict):
            logger.warning("Legacy entity_modes is not a dictionary; deriving all modes from active_entities")
            raw_modes = {}
        active_values = legacy_data.get(
            "active_entities",
            ["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"],
        )
        if not isinstance(active_values, list):
            logger.warning("Legacy active_entities is not a list; treating it as empty")
            active_values = []
        active_list = {value for value in active_values if isinstance(value, str)}
        entity_modes: Dict[str, str] = {}
        eupii_candidates = {"PERSON", "LOCATION", "ID_NUMBER", "HEALTH_DATA"}
        for entity in self.available_entities:
            stored_mode = raw_modes.get(entity)
            if isinstance(stored_mode, str) and stored_mode in VALID_ENTITY_MODES:
                entity_modes[entity] = stored_mode
            else:
                if stored_mode is not None:
                    logger.warning("Ignoring invalid legacy entity mode for %s: %r", entity, stored_mode)
                entity_modes[entity] = (
                    ENTITY_MODE_EXPLICIT_EUPII if entity in eupii_candidates else ENTITY_MODE_ALL
                ) if entity in active_list else ENTITY_MODE_OFF

        glossary_terms: Dict[str, ScopedTerm] = {}
        raw_glossary = legacy_data.get("glossary", "")
        if isinstance(raw_glossary, str):
            for line in raw_glossary.splitlines():
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                separator = ":" if ":" in line_str else "=" if "=" in line_str else None
                if separator is None:
                    logger.warning("Ignoring malformed legacy glossary entry: %s", line_str)
                    continue
                term_text, type_part = (part.strip() for part in line_str.split(separator, 1))
                role_text = None
                if "|" in type_part:
                    type_part, role_text = (part.strip() for part in type_part.split("|", 1))
                entity_type = type_part.upper()
                if not term_text or entity_type not in self.available_entities:
                    logger.warning("Ignoring invalid legacy glossary entry: %s", line_str)
                    continue
                key = normalize_term_key(term_text)
                if key:
                    glossary_terms[key] = ScopedTerm(
                        term=term_text, term_key=key, entity_type=entity_type, role=role_text or None
                    )
        elif raw_glossary:
            logger.warning("Ignoring non-string legacy glossary value")

        ignore_terms: Dict[str, ScopedTerm] = {}
        raw_ignore = legacy_data.get("ignore_terms", "")
        if isinstance(raw_ignore, str):
            for item in re.split(r"[\n,]+", raw_ignore):
                term_text = item.strip()
                if term_text:
                    key = normalize_term_key(term_text)
                    if key:
                        ignore_terms[key] = ScopedTerm(term=term_text, term_key=key)
        elif raw_ignore:
            logger.warning("Ignoring non-string legacy ignore_terms value")

        project_file = self.projects_dir / f"{default_uuid}.json"
        if step == "BACKUP_CREATED":
            if project_file.exists():
                ProjectProfile.from_dict(json.loads(project_file.read_text(encoding="utf-8")), self.available_entities)
            else:
                default_project = ProjectProfile(
                    project_id=default_uuid,
                    project_name="Standard",
                    description="Automatisch migriertes Standard-Projekt.",
                    is_default=True,
                    entity_modes=entity_modes,
                    glossary_terms=glossary_terms,
                    ignore_terms=ignore_terms,
                    schema_version=2,
                    revision=1,
                )
                self._atomic_write_json(project_file, default_project.to_dict())
            advance("PROJECT_WRITTEN")

        if step == "PROJECT_WRITTEN":
            if self.system_file.exists():
                SystemProfile.from_dict(json.loads(self.system_file.read_text(encoding="utf-8")), self.available_entities)
            else:
                self._atomic_write_json(self.system_file, SystemProfile(schema_version=2, revision=1).to_dict())
            advance("SYSTEM_WRITTEN")

        if step == "SYSTEM_WRITTEN":
            if self.manifest_file.exists():
                validate_manifest(json.loads(self.manifest_file.read_text(encoding="utf-8")))
            else:
                manifest = {
                    "schema_version": 2,
                    "revision": 1,
                    "active_project_id": default_uuid,
                    "default_project_id": default_uuid,
                    "warning_acknowledged_version": 0,
                    "format_mode": legacy_data.get("format_mode", "numbered_role"),
                    "gliner_model_name": legacy_data.get("gliner_model_name", "urchade/gliner_multi_pii-v1"),
                    "gliner_threshold": float(legacy_data.get("gliner_threshold", 0.55)),
                    "enable_eupii": bool(legacy_data.get("enable_eupii", True)),
                    "eupii_model_name": legacy_data.get("eupii_model_name", "bardsai/eu-pii-anonimization-multilang"),
                    "eupii_threshold": float(legacy_data.get("eupii_threshold", 0.50)),
                    "export_format": legacy_data.get("export_format", "txt"),
                    "llm_enabled": bool(legacy_data.get("llm_enabled", False)),
                    "llm_base_url": legacy_data.get("llm_base_url", "http://127.0.0.1:11434/v1"),
                    "llm_model_name": legacy_data.get("llm_model_name", "qwen3:8b"),
                    "llm_provider_type": legacy_data.get("llm_provider_type", "ollama"),
                    "llm_auto_review": bool(legacy_data.get("llm_auto_review", True)),
                    "migration_complete": True,
                    "updated_at": time.time(),
                }
                validate_manifest(manifest)
                self._atomic_write_json(self.manifest_file, manifest)
            advance("MANIFEST_WRITTEN")

        if step == "MANIFEST_WRITTEN":
            migrated_legacy_path = self.root_dir / "config.v1.migrated.json"
            if source_path.exists() and source_path != migrated_legacy_path:
                if not migrated_legacy_path.exists():
                    os.replace(source_path, migrated_legacy_path)
            advance("LEGACY_RENAMED")

        if step == "LEGACY_RENAMED":
            self.journal_file.unlink(missing_ok=True)
            logger.info("Migration from v1 config.json completed successfully.")

    def _resume_journal_migration(self) -> None:
        """Resume interrupted migration from journal state."""
        journal_data = json.loads(self.journal_file.read_text(encoding="utf-8"))
        legacy_path = Path(journal_data.get("legacy_file", self.root_dir / "config.json"))
        self._perform_migration(legacy_path, journal_data)

    def _initialize_fresh_defaults(self) -> None:
        """Create fresh v2 stores when starting on a blank directory."""
        default_uuid = str(uuid.uuid4())
        default_project = ProjectProfile(
            project_id=default_uuid,
            project_name="Standard",
            description="Standard-Projekt.",
            is_default=True,
            entity_modes={e: ENTITY_MODE_ALL for e in self.available_entities},
            schema_version=2,
            revision=1,
        )
        self._atomic_write_json(self.projects_dir / f"{default_uuid}.json", default_project.to_dict())

        system_profile = SystemProfile(schema_version=2, revision=1)
        self._atomic_write_json(self.system_file, system_profile.to_dict())

        manifest = {
            "schema_version": 2,
            "revision": 1,
            "active_project_id": default_uuid,
            "default_project_id": default_uuid,
            "warning_acknowledged_version": 0,
            "format_mode": "numbered_role",
            "gliner_model_name": "urchade/gliner_multi_pii-v1",
            "gliner_threshold": 0.55,
            "enable_eupii": True,
            "eupii_model_name": "bardsai/eu-pii-anonimization-multilang",
            "eupii_threshold": 0.50,
            "export_format": "txt",
            "llm_enabled": False,
            "llm_base_url": "http://127.0.0.1:11434/v1",
            "llm_model_name": "qwen3:8b",
            "llm_provider_type": "ollama",
            "llm_auto_review": True,
            "migration_complete": True,
            "updated_at": time.time(),
        }
        self._atomic_write_json(self.manifest_file, manifest)

    # --- Store Operations ---

    def load_manifest(self) -> Dict[str, Any]:
        with self._thread_lock, self._file_lock():
            if not self.manifest_file.exists():
                self._initialize_or_migrate_unlocked()
            return self._load_manifest_unlocked()

    def save_manifest(self, manifest_data: Dict[str, Any], expected_revision: Optional[int] = None) -> Dict[str, Any]:
        with self._thread_lock, self._file_lock():
            return self._save_manifest_unlocked(manifest_data, expected_revision)

    def load_system_profile(self) -> SystemProfile:
        with self._thread_lock, self._file_lock():
            if not self.system_file.exists():
                self._initialize_or_migrate_unlocked()
            data = json.loads(self.system_file.read_text(encoding="utf-8"))
            return SystemProfile.from_dict(data, self.available_entities)

    def save_system_profile(self, profile: SystemProfile, expected_revision: Optional[int] = None) -> SystemProfile:
        with self._thread_lock, self._file_lock():
            profile.validate(self.available_entities)
            if self.system_file.exists():
                curr_data = json.loads(self.system_file.read_text(encoding="utf-8"))
                curr_rev = curr_data.get("revision", 1)
                if expected_revision is not None and curr_rev != expected_revision:
                    raise RevisionConflictError(
                        f"SystemProfile revision conflict: on_disk={curr_rev}, expected={expected_revision}"
                    )
                profile.revision = curr_rev + 1
            else:
                profile.revision = 1
            profile.updated_at = time.time()
            self._atomic_write_json(self.system_file, profile.to_dict())
            return profile

    def load_project_profile(self, project_id: str) -> ProjectProfile:
        canon_id = validate_id_string(project_id)
        with self._thread_lock, self._file_lock():
            return self._load_project_profile_unlocked(canon_id)

    def save_project_profile(self, profile: ProjectProfile, expected_revision: Optional[int] = None) -> ProjectProfile:
        with self._thread_lock, self._file_lock():
            return self._save_project_profile_unlocked(profile, expected_revision)

    def list_projects(self) -> List[ProjectProfile]:
        with self._thread_lock, self._file_lock():
            return self._list_projects_unlocked()

    def create_project(self, name: str, description: str = "", template_id: Optional[str] = None) -> ProjectProfile:
        with self._thread_lock, self._file_lock():
            validate_clean_string(name, "project_name", max_length=100, allow_empty=False)
            norm_name = name.strip().casefold()
            for existing in self._list_projects_unlocked():
                if existing.project_name.strip().casefold() == norm_name:
                    raise ValueError(f"Project with name '{name}' already exists.")

            new_id = str(uuid.uuid4())
            entity_modes: Dict[str, str] = {e: ENTITY_MODE_ALL for e in self.available_entities}
            if template_id:
                tpl = self._load_template_unlocked(template_id)
                if tpl:
                    for e, m in tpl.entity_modes.items():
                        if e in self.available_entities:
                            entity_modes[e] = m

            profile = ProjectProfile(
                project_id=new_id,
                project_name=name.strip(),
                description=description.strip(),
                is_default=False,
                template_id=template_id,
                entity_modes=entity_modes,
                schema_version=2,
                revision=1,
            )
            self._save_project_profile_unlocked(profile)
            return profile

    def apply_template_to_project(
        self, project_id: str, template_id: str, expected_revision: Optional[int] = None
    ) -> ProjectProfile:
        """Copy a template's modes into a project; existing snapshots are never linked dynamically."""
        canon_project_id = validate_id_string(project_id)
        with self._thread_lock, self._file_lock():
            profile = self._load_project_profile_unlocked(canon_project_id)
            template = self._load_template_unlocked(template_id)
            if template is None:
                raise FileNotFoundError(f"Template not found: {template_id}")
            profile.entity_modes = dict(template.entity_modes)
            profile.template_id = validate_id_string(template.template_id, allow_builtins=True)
            return self._save_project_profile_unlocked(profile, expected_revision)

    def delete_project(self, project_id: str) -> None:
        canon_id = validate_id_string(project_id)
        with self._thread_lock, self._file_lock():
            proj = self._load_project_profile_unlocked(canon_id)
            manifest = self._load_manifest_unlocked()
            stable_default_id = manifest.get("default_project_id")
            if canon_id == stable_default_id or proj.is_default:
                raise ValueError("The default project cannot be deleted.")
            all_projs = self._list_projects_unlocked()
            if len(all_projs) <= 1:
                raise ValueError("The last remaining project cannot be deleted.")

            if manifest.get("active_project_id") == canon_id:
                # Switch to default project
                default_p = next((p for p in all_projs if p.is_default), all_projs[0])
                manifest["active_project_id"] = default_p.project_id
                self._save_manifest_unlocked(manifest)

            proj_file = self.projects_dir / f"{canon_id}.json"
            proj_file.unlink(missing_ok=True)

    def load_template(self, template_id: str) -> Optional[CategoryTemplate]:
        with self._thread_lock, self._file_lock():
            return self._load_template_unlocked(template_id)

    def save_custom_template(self, template: CategoryTemplate, expected_revision: Optional[int] = None) -> CategoryTemplate:
        if template.template_id in BUILTIN_TEMPLATE_IDS:
            raise ValueError("Built-in templates cannot be modified or overwritten.")
        canon_id = validate_id_string(template.template_id)
        with self._thread_lock, self._file_lock():
            template.validate(self.available_entities)
            # Check duplicate template names
            norm_name = template.name.strip().casefold()
            for other in self._list_custom_templates_unlocked():
                if other.template_id != canon_id and other.name.strip().casefold() == norm_name:
                    raise ValueError(f"A template with name '{template.name}' already exists.")

            tpl_file = self.templates_dir / f"{canon_id}.json"
            if tpl_file.exists():
                curr_data = json.loads(tpl_file.read_text(encoding="utf-8"))
                curr_rev = curr_data.get("revision", 1)
                if expected_revision is not None and curr_rev != expected_revision:
                    raise RevisionConflictError(
                        f"Template revision conflict: on_disk={curr_rev}, expected={expected_revision}"
                    )
                template.revision = curr_rev + 1
            else:
                template.revision = 1
            self._atomic_write_json(tpl_file, template.to_dict())
            return template

    def list_custom_templates(self) -> List[CategoryTemplate]:
        with self._thread_lock, self._file_lock():
            return self._list_custom_templates_unlocked()

    def list_all_templates(self) -> List[CategoryTemplate]:
        builtins = list(get_builtin_templates(self.available_entities).values())
        customs = self.list_custom_templates()
        return builtins + customs

    def delete_custom_template(self, template_id: str) -> None:
        if template_id in BUILTIN_TEMPLATE_IDS:
            raise ValueError("Built-in templates cannot be deleted.")
        canon_id = validate_id_string(template_id)
        with self._thread_lock, self._file_lock():
            tpl_file = self.templates_dir / f"{canon_id}.json"
            tpl_file.unlink(missing_ok=True)

    def delete_migration_backups(self) -> int:
        """Delete legacy migration backups after user confirmation."""
        with self._thread_lock, self._file_lock():
            count = 0
            for f in self.backups_dir.glob("config.v1.backup.*.json"):
                f.unlink(missing_ok=True)
                count += 1
            migrated_legacy = self.root_dir / "config.v1.migrated.json"
            if migrated_legacy.exists():
                migrated_legacy.unlink(missing_ok=True)
                count += 1
            return count


class DirtyOverlayError(RuntimeError):
    """Raised when a document overlay would be lost without explicit discard."""


class ProfileController:
    """Single mutation boundary for durable profiles and volatile overlays."""

    def __init__(self, store: ProfileStore, active_project_id: Optional[str] = None):
        self.store = store
        self.available_entities = set(store.available_entities)
        self.manifest: Dict[str, Any] = {}
        self.system_profile: SystemProfile
        self.project_profile: ProjectProfile
        self.document_overlay = DocumentProfileOverlay()
        self.reload(active_project_id=active_project_id)

    @staticmethod
    def _clone_profile(profile: Any) -> Any:
        if isinstance(profile, SystemProfile):
            return SystemProfile.from_dict(profile.to_dict(), set(AVAILABLE_ENTITIES))
        if isinstance(profile, ProjectProfile):
            return ProjectProfile.from_dict(profile.to_dict(), set(AVAILABLE_ENTITIES))
        raise TypeError(f"Unsupported profile type: {type(profile)!r}")

    @staticmethod
    def _scope(scope: Any) -> ScopeLevel:
        if isinstance(scope, ScopeLevel):
            return scope
        try:
            return ScopeLevel(str(scope))
        except ValueError as ex:
            raise ValueError(f"Unknown profile scope: {scope!r}") from ex

    def reload(self, active_project_id: Optional[str] = None) -> None:
        self.manifest = self.store.load_manifest()
        self.system_profile = self.store.load_system_profile()
        project_id = active_project_id or self.manifest["active_project_id"]
        self.project_profile = self.store.load_project_profile(project_id)
        self.document_overlay = DocumentProfileOverlay()

    def effective_config(self) -> EffectiveConfig:
        return ScopeResolutionEngine.resolve(
            self.system_profile, self.project_profile, self.document_overlay, self.available_entities
        )

    def discard_overlay(self) -> DocumentProfileOverlay:
        previous_revision = self.document_overlay.overlay_revision
        self.document_overlay = DocumentProfileOverlay(overlay_revision=previous_revision + 1)
        return self.document_overlay

    def switch_project(self, project_id: str, discard_overlay: bool = False) -> ProjectProfile:
        if self.document_overlay.dirty and not discard_overlay:
            raise DirtyOverlayError("Document overlay contains unsaved changes")
        if discard_overlay:
            self.discard_overlay()
        project = self.store.load_project_profile(project_id)
        manifest = dict(self.manifest)
        expected_revision = int(manifest["revision"])
        manifest["active_project_id"] = project.project_id
        self.manifest = self.store.save_manifest(manifest, expected_revision=expected_revision)
        self.project_profile = project
        return project

    def _touch_overlay(self) -> None:
        self.document_overlay.dirty = True
        self.document_overlay.overlay_revision += 1

    def set_entity_mode(self, scope: Any, entity: str, mode: str) -> None:
        scope_level = self._scope(scope)
        if entity not in self.available_entities or mode not in VALID_ENTITY_MODES:
            raise ValueError(f"Invalid entity mode: {entity}={mode}")
        if scope_level == ScopeLevel.PROJECT:
            self.project_profile.entity_modes[entity] = mode
        elif scope_level == ScopeLevel.DOCUMENT:
            self.document_overlay.entity_modes[entity] = mode
            self._touch_overlay()
        else:
            raise ValueError("Entity modes are supported in project or document scope only")

    def _term_target(self, scope: ScopeLevel, glossary: bool) -> Dict[str, ScopedTerm]:
        if scope == ScopeLevel.SYSTEM:
            target = self.system_profile
        elif scope == ScopeLevel.PROJECT:
            target = self.project_profile
        elif scope == ScopeLevel.DOCUMENT:
            target = self.document_overlay
        else:
            raise ValueError("App-default terms are read-only")
        return getattr(target, "glossary_terms" if glossary else "ignore_terms")

    def upsert_glossary(self, scope: Any, term: str, entity_type: str, role: Optional[str] = None) -> ScopedTerm:
        scope_level = self._scope(scope)
        term_value = validate_clean_string(term, "term", 200, allow_empty=False).strip()
        entity_value = entity_type.strip().upper()
        if entity_value not in self.available_entities:
            raise ValueError(f"Invalid glossary entity type: {entity_type!r}")
        role_value = role.strip() if isinstance(role, str) and role.strip() else None
        item = ScopedTerm(term_value, normalize_term_key(term_value), entity_value, role_value)
        item.validate(self.available_entities, is_ignore_term=False)
        self._term_target(scope_level, glossary=True)[item.term_key] = item
        if scope_level == ScopeLevel.DOCUMENT:
            self._touch_overlay()
        return item

    def upsert_ignore(self, scope: Any, term: str) -> ScopedTerm:
        scope_level = self._scope(scope)
        term_value = validate_clean_string(term, "term", 200, allow_empty=False).strip()
        item = ScopedTerm(term_value, normalize_term_key(term_value))
        item.validate(self.available_entities, is_ignore_term=True)
        self._term_target(scope_level, glossary=False)[item.term_key] = item
        if scope_level == ScopeLevel.DOCUMENT:
            self._touch_overlay()
        return item

    def remove_term(self, scope: Any, term: str, glossary: bool) -> None:
        scope_level = self._scope(scope)
        self._term_target(scope_level, glossary=glossary).pop(normalize_term_key(term), None)
        if scope_level == ScopeLevel.DOCUMENT:
            self._touch_overlay()

    def disable_inherited(self, scope: Any, term: str, glossary: bool, disabled: bool = True) -> None:
        scope_level = self._scope(scope)
        if scope_level not in (ScopeLevel.PROJECT, ScopeLevel.DOCUMENT):
            raise ValueError("Inherited terms can only be disabled in project or document scope")
        target = self.project_profile if scope_level == ScopeLevel.PROJECT else self.document_overlay
        field_name = "disabled_inherited_glossary" if glossary else "disabled_inherited_ignore"
        values = getattr(target, field_name)
        key = normalize_term_key(term)
        (values.add if disabled else values.discard)(key)
        if scope_level == ScopeLevel.DOCUMENT:
            self._touch_overlay()

    def save_system(self, expected_revision: Optional[int] = None) -> SystemProfile:
        expected = self.system_profile.revision if expected_revision is None else expected_revision
        self.system_profile = self.store.save_system_profile(
            self._clone_profile(self.system_profile), expected_revision=expected
        )
        return self.system_profile

    def save_project(self, expected_revision: Optional[int] = None) -> ProjectProfile:
        expected = self.project_profile.revision if expected_revision is None else expected_revision
        self.project_profile = self.store.save_project_profile(
            self._clone_profile(self.project_profile), expected_revision=expected
        )
        return self.project_profile

    def acknowledge_warning(self, expected_revision: Optional[int] = None) -> Dict[str, Any]:
        expected = int(self.manifest["revision"] if expected_revision is None else expected_revision)
        manifest = dict(self.manifest)
        manifest["warning_acknowledged_version"] = 1
        self.manifest = self.store.save_manifest(manifest, expected_revision=expected)
        return self.manifest

    def warning_required(self) -> bool:
        return int(self.manifest.get("warning_acknowledged_version", 0)) < 1

    def list_all_templates(self) -> List[CategoryTemplate]:
        return self.store.list_all_templates()

    def save_custom_template(self, template: CategoryTemplate, expected_revision: Optional[int] = None) -> CategoryTemplate:
        return self.store.save_custom_template(template, expected_revision=expected_revision)

    def apply_template(self, template_id: str, expected_revision: Optional[int] = None) -> ProjectProfile:
        expected = self.project_profile.revision if expected_revision is None else expected_revision
        self.project_profile = self.store.apply_template_to_project(
            self.project_profile.project_id, template_id, expected_revision=expected
        )
        return self.project_profile

    def delete_custom_template(self, template_id: str) -> None:
        self.store.delete_custom_template(template_id)
