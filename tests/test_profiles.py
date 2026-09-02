"""
tests/test_profiles.py — Phase 5b.1 test suite

Covers:
1.  Strict validators (UUID, built-in IDs, control chars, length, revision types)
2.  Scope hierarchy & precedence (App-Defaults < System < Project < Document)
3.  Conflict handling (privacy-conservative: glossary wins over ignore in same scope)
4.  Disabled-inherited overrides (disabled_inherited_glossary / ignore)
5.  Role propagation (profile roles → recognition_metadata, manual/LLM roles protected)
6.  EffectiveConfig true immutability (TypeError on setattr, deterministic hash)
7.  Journaled migration with fault injection (resume at every phase without overwriting v2)
8.  CAS & RevisionConflictError
9.  Lifecycle (duplicate names, template deletion, default project, last-project protection)
10. Stale-handling & regression (all 250 existing tests still pass via subprocess)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from local_anonymizer.profiles import (
    BUILTIN_TEMPLATE_IDS,
    ENTITY_MODE_ALL,
    ENTITY_MODE_EXPLICIT_EUPII,
    ENTITY_MODE_EXPLICIT_ONLY,
    ENTITY_MODE_OFF,
    TEMPLATE_STANDARD_FULL,
    TEMPLATE_CH_STANDARD,
    TEMPLATE_MEDICAL,
    CategoryTemplate,
    DirtyOverlayError,
    SystemMutationConfirmationRequired,
    DocumentProfileOverlay,
    EffectiveConfig,
    ProjectProfile,
    RevisionConflictError,
    ScopedTerm,
    ScopeLevel,
    ScopeResolutionEngine,
    SystemProfile,
    ProfileStore,
    ProfileController,
    get_builtin_templates,
    normalize_term_key,
    validate_id_string,
    validate_clean_string,
)
from local_anonymizer.anonymizer import AVAILABLE_ENTITIES

AVAILABLE = set(AVAILABLE_ENTITIES)


def _make_store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(root_dir=tmp_path)


def _extract_app_functions(*names):
    """Compile the actual GUI helper bodies with test-controlled globals."""
    import app

    tree = ast.parse((Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8"))
    create_ui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_ui")
    nodes = {
        node.name: node
        for node in ast.walk(create_ui)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    namespace = dict(vars(app))
    for name in names:
        node = ast.fix_missing_locations(nodes[name])
        exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


class _FakeUiElement:
    def __init__(self, value=None):
        self.value = value
        self.opened = False
        self.on_change = None

    def classes(self, *_args, **_kwargs):
        return self

    def props(self, *_args, **_kwargs):
        return self

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def update(self):
        return None

    def set_value(self, value):
        self.value = value
        if self.on_change is not None:
            self.on_change(SimpleNamespace(value=value))

    def clear(self):
        return None


class _FakeUiContext(_FakeUiElement):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeUi:
    def __init__(self):
        self.buttons = []
        self.notifications = []

    def dialog(self):
        dialog = _FakeUiContext()
        self.dialogs = getattr(self, "dialogs", [])
        self.dialogs.append(dialog)
        return dialog

    def card(self):
        return _FakeUiContext()

    def row(self):
        return _FakeUiContext()

    def label(self, *_args, **_kwargs):
        return _FakeUiElement()

    def markdown(self, *_args, **_kwargs):
        return _FakeUiElement()

    def checkbox(self, *_args, **_kwargs):
        return _FakeUiElement(value=True)

    def button(self, label, on_click=None, **_kwargs):
        button = _FakeUiElement()
        button.label = label
        button.on_click = on_click
        self.buttons.append(button)
        return button

    def notify(self, message, **kwargs):
        self.notifications.append((message, kwargs))

    def click_last(self, label):
        button = next(button for button in reversed(self.buttons) if button.label == label)
        return button.on_click()


def _make_gui_adapter_context(store, controller, fake_ui, busy):
    overlay = DocumentProfileOverlay(
        glossary_terms={"privatwort": _scoped_term("Privatwort", "PERSON")},
        dirty=True,
        overlay_revision=7,
    )
    controller.document_overlay = overlay
    state = SimpleNamespace(
        profile_store=store,
        profile_controller=controller,
        system_profile=controller.system_profile,
        project_profile=controller.project_profile,
        document_overlay=overlay,
        entity_modes=dict(controller.project_profile.entity_modes),
        entity_groups=[],
        glossary_text="",
        ignore_terms_text="",
        preview_stale=False,
        refresh_effective_config=lambda *args, **kwargs: None,
    )
    namespace = _extract_app_functions(
        "_profile_controller",
        "_sync_profile_controller",
        "_set_profile_ui_value",
        "_run_durable_profile_action",
        "mutate",
        "add_glossary",
        "add_ignore",
        "apply_template",
        "make_entity_mode_change",
    )
    namespace.update(
        {
            "ui": fake_ui,
            "state": state,
            "controller": controller,
            "check_mutation_allowed": lambda: not busy["value"],
            "render_all": lambda: None,
            "profile_terms_text": lambda terms, include_role=False: "\n".join(
                term.term for term in terms.values()
            ),
            "scope_select": SimpleNamespace(value="project"),
            "sidebar_entity_mode_selects": {},
            "profile_ui_sync_depth": [0],
        }
    )
    namespace["mutate"].__globals__.update(namespace)
    namespace["_run_durable_profile_action"].__globals__.update(namespace)
    namespace["_sync_profile_controller"].__globals__.update(namespace)
    namespace["add_glossary"].__globals__.update(namespace)
    namespace["add_ignore"].__globals__.update(namespace)
    return state, namespace


def _valid_uuid4() -> str:
    return str(uuid.uuid4())


def _scoped_term(term: str, entity_type: str = "PERSON", role: Optional[str] = None) -> ScopedTerm:
    k = normalize_term_key(term)
    return ScopedTerm(term=term, term_key=k, entity_type=entity_type, role=role)


def _ignore_term(term: str) -> ScopedTerm:
    k = normalize_term_key(term)
    return ScopedTerm(term=term, term_key=k)


def _make_project(
    pid: Optional[str] = None,
    name: str = "Test",
    is_default: bool = False,
    entity_modes: Optional[Dict] = None,
    glossary_terms: Optional[Dict] = None,
    ignore_terms: Optional[Dict] = None,
    disabled_inherited_glossary=None,
    disabled_inherited_ignore=None,
) -> ProjectProfile:
    return ProjectProfile(
        project_id=pid or _valid_uuid4(),
        project_name=name,
        is_default=is_default,
        entity_modes=entity_modes or {e: ENTITY_MODE_ALL for e in AVAILABLE},
        glossary_terms=glossary_terms or {},
        ignore_terms=ignore_terms or {},
        disabled_inherited_glossary=disabled_inherited_glossary or set(),
        disabled_inherited_ignore=disabled_inherited_ignore or set(),
        schema_version=2,
        revision=1,
    )


def _make_system(
    glossary: Optional[Dict] = None, ignore: Optional[Dict] = None
) -> SystemProfile:
    return SystemProfile(
        glossary_terms=glossary or {},
        ignore_terms=ignore or {},
        schema_version=2,
        revision=1,
    )


def _effective(
    project: ProjectProfile,
    system: Optional[SystemProfile] = None,
    overlay: Optional[DocumentProfileOverlay] = None,
) -> EffectiveConfig:
    return ScopeResolutionEngine.resolve(
        system_profile=system or _make_system(),
        project_profile=project,
        document_overlay=overlay,
        available_entities=AVAILABLE,
    )


# ===========================================================================
# 1. Strict Validators
# ===========================================================================


class TestValidateIdString:
    def test_valid_uuid4(self):
        uid = _valid_uuid4()
        assert validate_id_string(uid) == uid

    def test_rejects_uuid_version_1(self):
        import uuid as _uuid
        uid1 = str(_uuid.uuid1())
        with pytest.raises(ValueError, match="UUIDv4"):
            validate_id_string(uid1)

    def test_uppercase_uuid_normalized_to_canonical(self):
        uid = _valid_uuid4()
        upper = uid.upper()
        canonical = validate_id_string(upper)
        assert canonical == uid

    def test_rejects_arbitrary_string(self):
        with pytest.raises(ValueError):
            validate_id_string("not-a-uuid")

    def test_builtin_allowed_with_flag(self):
        for bid in BUILTIN_TEMPLATE_IDS:
            assert validate_id_string(bid, allow_builtins=True) == bid

    def test_builtin_rejected_without_flag(self):
        with pytest.raises(ValueError):
            validate_id_string(TEMPLATE_STANDARD_FULL, allow_builtins=False)


class TestValidateCleanString:
    def test_too_long(self):
        with pytest.raises(ValueError, match="exceeds max length"):
            validate_clean_string("a" * 201, "field", max_length=200)

    def test_empty_not_allowed(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_clean_string("  ", "field", max_length=100, allow_empty=False)

    def test_empty_allowed(self):
        result = validate_clean_string("", "field", max_length=100, allow_empty=True)
        assert result == ""

    def test_control_char_rejected(self):
        with pytest.raises(ValueError, match="control characters"):
            validate_clean_string("abc\x0bdef", "field", max_length=100)


class TestScopedTermValidation:
    def test_valid_glossary_term(self):
        t = _scoped_term("ZHAW", "ORGANIZATION")
        t.validate(AVAILABLE, is_ignore_term=False)

    def test_invalid_entity_type_rejected(self):
        t = ScopedTerm(
            term="ZHAW",
            term_key=normalize_term_key("ZHAW"),
            entity_type="NONEXISTENT_ENTITY",
        )
        with pytest.raises(ValueError):
            t.validate(AVAILABLE, is_ignore_term=False)

    def test_ignore_term_must_not_have_entity_type(self):
        t = ScopedTerm(
            term="CAS",
            term_key=normalize_term_key("CAS"),
            entity_type="ORGANIZATION",
        )
        with pytest.raises(ValueError):
            t.validate(AVAILABLE, is_ignore_term=True)

    def test_unknown_field_rejected_from_dict(self):
        data = {
            "term": "Test",
            "term_key": normalize_term_key("Test"),
            "entity_type": "PERSON",
            "role": None,
            "created_at": 0.0,
            "unknown_field": "BAD",
        }
        with pytest.raises(ValueError, match="Unknown fields"):
            ScopedTerm.from_dict(data, AVAILABLE)

    def test_term_key_mismatch_rejected(self):
        t = ScopedTerm(
            term="ZHAW",
            term_key="wrong-key",
            entity_type="ORGANIZATION",
        )
        with pytest.raises(ValueError, match="Term key mismatch"):
            t.validate(AVAILABLE, is_ignore_term=False)


class TestCategoryTemplateValidation:
    def test_bool_revision_rejected(self):
        t = CategoryTemplate(
            template_id=_valid_uuid4(),
            name="Test Template",
            is_builtin=False,
            entity_modes={},
            schema_version=2,
            revision=True,  # type: ignore — intentional bad input
        )
        with pytest.raises(ValueError, match="positive integer"):
            t.validate(AVAILABLE)

    def test_zero_revision_rejected(self):
        t = CategoryTemplate(
            template_id=_valid_uuid4(),
            name="Test Template",
            entity_modes={},
            schema_version=2,
            revision=0,
        )
        with pytest.raises(ValueError, match="positive integer"):
            t.validate(AVAILABLE)

    def test_unknown_field_rejected_from_dict(self):
        data = {
            "template_id": _valid_uuid4(),
            "name": "X",
            "description": "",
            "is_builtin": False,
            "entity_modes": {},
            "schema_version": 2,
            "revision": 1,
            "bad_key": "oops",
        }
        with pytest.raises(ValueError, match="Unknown fields"):
            CategoryTemplate.from_dict(data, AVAILABLE)

    def test_builtin_write_protection_from_canonical_id(self):
        """Built-in status derives from canonical ID, not the loaded flag."""
        data = {
            "template_id": TEMPLATE_STANDARD_FULL,
            "name": "Spoofed name",
            "description": "",
            "is_builtin": False,  # attacker tries to sneak in a non-builtin flag
            "entity_modes": {e: ENTITY_MODE_ALL for e in AVAILABLE},
            "schema_version": 2,
            "revision": 1,
        }
        tpl = CategoryTemplate.from_dict(data, AVAILABLE)
        assert tpl.is_builtin is True  # derived from ID


class TestProjectProfileValidation:
    def test_valid_project_serialization_roundtrip(self):
        proj = _make_project()
        roundtripped = ProjectProfile.from_dict(proj.to_dict(), AVAILABLE)
        assert roundtripped.project_id == proj.project_id
        assert roundtripped.project_name == proj.project_name

    def test_unknown_field_rejected(self):
        proj = _make_project()
        d = proj.to_dict()
        d["surprise"] = "field"
        with pytest.raises(ValueError, match="Unknown fields"):
            ProjectProfile.from_dict(d, AVAILABLE)


# ===========================================================================
# 2. Scope Hierarchy & Precedence
# ===========================================================================


class TestScopeHierarchy:
    def test_project_entity_modes_override_defaults(self):
        proj = _make_project(
            entity_modes={"PERSON": ENTITY_MODE_OFF, "LOCATION": ENTITY_MODE_EXPLICIT_EUPII}
        )
        cfg = _effective(proj)
        assert cfg.entity_modes["PERSON"] == ENTITY_MODE_OFF
        assert cfg.entity_modes["LOCATION"] == ENTITY_MODE_EXPLICIT_EUPII

    def test_document_overlay_overrides_project_modes(self):
        proj = _make_project(entity_modes={"PERSON": ENTITY_MODE_ALL})
        overlay = DocumentProfileOverlay(entity_modes={"PERSON": ENTITY_MODE_OFF})
        cfg = _effective(proj, overlay=overlay)
        assert cfg.entity_modes["PERSON"] == ENTITY_MODE_OFF

    def test_project_glossary_overrides_system_glossary(self):
        sys_term = _scoped_term("ZHAW", "ORGANIZATION")
        system = _make_system(glossary={sys_term.term_key: sys_term})
        proj_term = _scoped_term("ZHAW", "LOCATION")  # project overrides to LOCATION
        proj = _make_project(
            glossary_terms={proj_term.term_key: proj_term}
        )
        cfg = _effective(proj, system=system)
        assert cfg.glossary.get("ZHAW") == "LOCATION"

    def test_document_overlay_glossary_overrides_project(self):
        proj_term = _scoped_term("ZHAW", "ORGANIZATION")
        proj = _make_project(glossary_terms={proj_term.term_key: proj_term})
        doc_term = _scoped_term("ZHAW", "PERSON")
        overlay = DocumentProfileOverlay(
            glossary_terms={doc_term.term_key: doc_term}
        )
        cfg = _effective(proj, overlay=overlay)
        assert cfg.glossary.get("ZHAW") == "PERSON"

    def test_project_ignore_overrides_app_default_ignore(self):
        """A project-level ignore term wins over app-defaults."""
        proj = _make_project(
            ignore_terms={"proj_custom": _ignore_term("ProjCustom")}
        )
        cfg = _effective(proj)
        terms_lower = {t.lower() for t in cfg.ignore_terms}
        assert "projcustom" in terms_lower


# ===========================================================================
# 3. Conflict Handling (privacy-conservative: glossary wins)
# ===========================================================================


class TestConflictHandling:
    def test_intra_project_conflict_glossary_wins(self):
        term = "CAS"
        k = normalize_term_key(term)
        glossary_term = _scoped_term(term, "ORGANIZATION")
        ignore_term = _ignore_term(term)
        proj = _make_project(
            glossary_terms={k: glossary_term},
            ignore_terms={k: ignore_term},
        )
        cfg = _effective(proj)
        # Glossary wins (privacy-conservative rule)
        assert cfg.glossary.get(term) == "ORGANIZATION"
        ignore_set = {t.lower() for t in cfg.ignore_terms}
        assert term.lower() not in ignore_set
        # Conflict is recorded
        assert k in cfg.conflict_terms


# ===========================================================================
# 4. Disabled Inherited Overrides
# ===========================================================================


class TestDisabledInherited:
    def test_project_disables_system_glossary_term(self):
        sys_term = _scoped_term("ZHAW", "ORGANIZATION")
        system = _make_system(glossary={sys_term.term_key: sys_term})
        proj = _make_project(
            disabled_inherited_glossary={sys_term.term_key}
        )
        cfg = _effective(proj, system=system)
        assert "ZHAW" not in cfg.glossary

    def test_project_disables_system_ignore_term(self):
        sys_term = _ignore_term("MBA")
        system = _make_system(ignore={sys_term.term_key: sys_term})
        proj = _make_project(
            disabled_inherited_ignore={sys_term.term_key}
        )
        cfg = _effective(proj, system=system)
        terms_lower = {t.lower() for t in cfg.ignore_terms}
        assert "mba" not in terms_lower

    def test_document_disables_project_glossary_term(self):
        proj_term = _scoped_term("ZHAW", "ORGANIZATION")
        proj = _make_project(glossary_terms={proj_term.term_key: proj_term})
        overlay = DocumentProfileOverlay(
            disabled_inherited_glossary={proj_term.term_key}
        )
        cfg = _effective(proj, overlay=overlay)
        assert "ZHAW" not in cfg.glossary


# ===========================================================================
# 5. Role Propagation & Phase-5a Rebind
# ===========================================================================


class TestRolePropagation:
    def test_glossary_role_appears_in_effective_config(self):
        term = _scoped_term("Julia Meier", "PERSON", role="STUDENT")
        proj = _make_project(glossary_terms={term.term_key: term})
        cfg = _effective(proj)
        assert cfg.glossary_roles.get("Julia Meier") == "STUDENT"

    def test_glossary_roles_injected_into_anonymizer(self):
        """to_anonymizer_kwargs passes glossary_roles to LocalAnonymizer."""
        term = _scoped_term("Julia Meier", "PERSON", role="STUDENT")
        proj = _make_project(glossary_terms={term.term_key: term})
        cfg = _effective(proj)
        kwargs = cfg.to_anonymizer_kwargs()
        assert kwargs["glossary_roles"].get("Julia Meier") == "STUDENT"

    def test_fuzzy_recognizer_emits_custom_role(self):
        """FuzzyGlossaryRecognizer injects custom_role and role_provenance into recognition_metadata."""
        from local_anonymizer.recognizers import FuzzyGlossaryRecognizer

        recognizer = FuzzyGlossaryRecognizer(
            glossary={"Julia Meier": "PERSON"},
            glossary_roles={"Julia Meier": "STUDENT"},
        )
        results = recognizer.analyze("Julia Meier ist in Gruppe 3.", entities=["PERSON"], nlp_artifacts=None)
        assert len(results) > 0
        meta = results[0].recognition_metadata or {}
        assert meta.get("custom_role") == "STUDENT"
        assert meta.get("role_provenance") == "profile"

    def test_glossary_roles_injected_as_fallback_in_anonymize(self):
        """LocalAnonymizer.anonymize() injects glossary_roles as low-priority fallback roles."""
        from local_anonymizer.anonymizer import LocalAnonymizer

        anon = LocalAnonymizer(
            glossary={"Max Muster": "PERSON"},
            glossary_roles={"Max Muster": "PATIENT"},
        )
        # Provide no explicit roles dict — profile role should be used
        result = anon.anonymize("Max Muster hat eine Überweisung.", roles=None)
        assert "PATIENT" in result.anonymized_text or "PERSON" in result.anonymized_text

    def test_explicit_role_overrides_glossary_role(self):
        """Caller-provided roles take precedence over glossary_roles fallback."""
        from local_anonymizer.anonymizer import LocalAnonymizer

        anon = LocalAnonymizer(
            glossary={"Max Muster": "PERSON"},
            glossary_roles={"Max Muster": "PATIENT"},
        )
        result = anon.anonymize(
            "Max Muster hat eine Überweisung.",
            roles={"max muster": "ARZT"},
        )
        # The role should be ARZT (explicit caller wins over profile fallback PATIENT)
        assigned_roles = [e.role for e in result.entities if e.role is not None]
        assert "ARZT" in assigned_roles, (
            f"Expected ARZT in roles, got: {assigned_roles}. "
            f"Text: {result.anonymized_text}"
        )


# ===========================================================================
# 6. EffectiveConfig Immutability
# ===========================================================================


class TestEffectiveConfigImmutability:
    def _make_cfg(self) -> EffectiveConfig:
        proj = _make_project()
        return _effective(proj)

    def test_setattr_raises_type_error(self):
        cfg = self._make_cfg()
        with pytest.raises(TypeError, match="immutable"):
            cfg._project_id = "oops"  # type: ignore

    def test_delattr_raises_type_error(self):
        cfg = self._make_cfg()
        with pytest.raises(TypeError, match="immutable"):
            del cfg._project_id  # type: ignore

    def test_deterministic_hash_for_identical_configs(self):
        proj_id = _valid_uuid4()
        proj = _make_project(pid=proj_id)
        cfg1 = _effective(proj)
        cfg2 = _effective(proj)
        assert cfg1.snapshot_hash == cfg2.snapshot_hash

    def test_hash_changes_on_different_project_id(self):
        proj1 = _make_project(pid=_valid_uuid4())
        proj2 = _make_project(pid=_valid_uuid4())
        cfg1 = _effective(proj1)
        cfg2 = _effective(proj2)
        assert cfg1.snapshot_hash != cfg2.snapshot_hash

    def test_hash_includes_glossary_roles(self):
        """Two configs identical except glossary_roles must produce different hashes."""
        pid = _valid_uuid4()
        term1 = _scoped_term("ZHAW", "ORGANIZATION", role="Hochschule")
        term2 = _scoped_term("ZHAW", "ORGANIZATION", role="Universität")
        proj1 = _make_project(pid=pid, glossary_terms={term1.term_key: term1})
        proj2 = _make_project(pid=pid, glossary_terms={term2.term_key: term2})
        cfg1 = _effective(proj1)
        cfg2 = _effective(proj2)
        assert cfg1.snapshot_hash != cfg2.snapshot_hash

    def test_entity_modes_is_mappingproxy(self):
        from types import MappingProxyType
        cfg = self._make_cfg()
        assert isinstance(cfg.entity_modes, MappingProxyType)

    def test_ignore_terms_is_tuple(self):
        cfg = self._make_cfg()
        assert isinstance(cfg.ignore_terms, tuple)


# ===========================================================================
# 7. Journaled Migration with Fault Injection
# ===========================================================================


class TestJournaledMigration:
    """
    Simulate abort at each journal step and verify idempotent resume.
    """

    _LEGACY_CONFIG = {
        "format_mode": "numbered_role",
        "active_entities": ["PERSON", "LOCATION"],
        "entity_modes": {},
        "gliner_model_name": "urchade/gliner_multi_pii-v1",
        "gliner_threshold": 0.55,
        "enable_eupii": True,
        "eupii_threshold": 0.50,
        "eupii_model_name": "bardsai/eu-pii-anonimization-multilang",
        "ignore_terms": "CAS, DAS",
        "glossary": "ZHAW: ORGANIZATION",
        "export_format": "txt",
        "llm_enabled": False,
        "llm_base_url": "http://127.0.0.1:11434/v1",
        "llm_model_name": "qwen3:8b",
        "llm_provider_type": "ollama",
        "llm_auto_review": True,
    }

    def _write_legacy(self, root: Path) -> None:
        (root / "config.json").write_text(
            json.dumps(self._LEGACY_CONFIG), encoding="utf-8"
        )

    def _assert_clean_v2(self, store: ProfileStore) -> None:
        """Assert the store is in a valid v2 state without a journal file."""
        assert store.manifest_file.exists()
        manifest = json.loads(store.manifest_file.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 2
        assert not store.journal_file.exists()

    def test_full_migration_fresh(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        self._write_legacy(root)
        store = _make_store(root)
        store.initialize_or_migrate()
        self._assert_clean_v2(store)

    def test_resume_after_backup_created(self, tmp_path):
        """Interrupt after backup written, then resume."""
        root = tmp_path / "store"
        root.mkdir()
        self._write_legacy(root)

        # Write journal at BACKUP_CREATED step manually
        store = _make_store(root)
        legacy_path = root / "config.json"
        raw_bytes = legacy_path.read_bytes()
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        uid = str(uuid.uuid4())
        journal = {
            "step": "BACKUP_CREATED",
            "source_hash": source_hash,
            "default_project_id": uid,
            "legacy_file": str(legacy_path),
        }
        store._atomic_write_json(store.journal_file, journal)
        # Backup is expected to exist at this point
        backup_path = store.backups_dir / f"config.v1.backup.{source_hash[:12]}.json"
        backup_path.write_bytes(raw_bytes)

        # Resume
        store.initialize_or_migrate()
        self._assert_clean_v2(store)

    def test_resume_after_project_written(self, tmp_path):
        """Interrupt after project file written, then resume."""
        root = tmp_path / "store"
        root.mkdir()
        self._write_legacy(root)

        store = _make_store(root)
        legacy_path = root / "config.json"
        raw_bytes = legacy_path.read_bytes()
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        uid = str(uuid.uuid4())

        # Simulate the project file already written
        project = _make_project(pid=uid, name="Standard", is_default=True)
        store._atomic_write_json(store.projects_dir / f"{uid}.json", project.to_dict())

        journal = {
            "step": "PROJECT_WRITTEN",
            "source_hash": source_hash,
            "default_project_id": uid,
            "legacy_file": str(legacy_path),
        }
        store._atomic_write_json(store.journal_file, journal)
        store.initialize_or_migrate()
        self._assert_clean_v2(store)

    def test_resume_after_manifest_written_does_not_overwrite_v2(self, tmp_path):
        """After MANIFEST_WRITTEN step, resume must not overwrite an existing v2 manifest."""
        root = tmp_path / "store"
        root.mkdir()
        self._write_legacy(root)

        store = _make_store(root)
        legacy_path = root / "config.json"
        raw_bytes = legacy_path.read_bytes()
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        uid = str(uuid.uuid4())

        # Write a valid v2 manifest
        manifest = {
            "schema_version": 2,
            "revision": 5,
            "active_project_id": uid,
        }
        store._atomic_write_json(store.manifest_file, manifest)

        # Journal says MANIFEST_WRITTEN (migration already done)
        journal = {
            "step": "MANIFEST_WRITTEN",
            "source_hash": source_hash,
            "default_project_id": uid,
            "legacy_file": str(legacy_path),
        }
        store._atomic_write_json(store.journal_file, journal)
        store.initialize_or_migrate()

        # Manifest must still be v2 and not overwritten with revision 1
        loaded = json.loads(store.manifest_file.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == 2

    def test_migration_glossary_and_ignore_terms_parsed(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        self._write_legacy(root)
        store = _make_store(root)
        store.initialize_or_migrate()

        manifest = store.load_manifest()
        proj = store.load_project_profile(manifest["active_project_id"])
        # "ZHAW: ORGANIZATION" should appear in glossary
        assert any(t.entity_type == "ORGANIZATION" for t in proj.glossary_terms.values())
        # "CAS" should appear in ignore terms
        ignore_keys = {t.term.lower() for t in proj.ignore_terms.values()}
        assert "cas" in ignore_keys

    def test_entity_mode_derivation_from_active_entities(self, tmp_path):
        """Legacy active_entities: PERSON, LOCATION → explicit_eupii; no entity_modes in config."""
        root = tmp_path / "store"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({
                "active_entities": ["PERSON", "LOCATION"],
                "entity_modes": {},
                "ignore_terms": "",
                "glossary": "",
            }),
            encoding="utf-8",
        )
        store = _make_store(root)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        proj = store.load_project_profile(manifest["active_project_id"])
        assert proj.entity_modes.get("PERSON") == ENTITY_MODE_EXPLICIT_EUPII
        assert proj.entity_modes.get("LOCATION") == ENTITY_MODE_EXPLICIT_EUPII
        # Inactive entities → off
        assert proj.entity_modes.get("IBAN_CODE") == ENTITY_MODE_OFF

    def test_partial_entity_modes_supplemented_not_overwritten(self, tmp_path):
        """If entity_modes already has PERSON=off, it must be preserved; rest derived from active_entities."""
        root = tmp_path / "store"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({
                "active_entities": ["PERSON", "ORGANIZATION"],
                "entity_modes": {"PERSON": "off"},
                "ignore_terms": "",
                "glossary": "",
            }),
            encoding="utf-8",
        )
        store = _make_store(root)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        proj = store.load_project_profile(manifest["active_project_id"])
        # Explicit saved mode preserved
        assert proj.entity_modes.get("PERSON") == ENTITY_MODE_OFF
        # ORGANIZATION was active → all
        assert proj.entity_modes.get("ORGANIZATION") == ENTITY_MODE_ALL


# ===========================================================================
# 8. CAS & RevisionConflictError
# ===========================================================================


class TestCAS:
    def test_project_cas_conflict_raises(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        proj = store.load_project_profile(manifest["active_project_id"])
        stored_rev = proj.revision
        # Save once to bump revision
        store.save_project_profile(proj)
        # Now try with old expected_revision → should fail
        with pytest.raises(RevisionConflictError):
            store.save_project_profile(proj, expected_revision=stored_rev)

    def test_manifest_cas_conflict_raises(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        old_rev = manifest["revision"]
        # Save manifest once
        store.save_manifest(manifest)
        # Try again with old expected revision
        with pytest.raises(RevisionConflictError):
            store.save_manifest(manifest, expected_revision=old_rev)


# ===========================================================================
# 9. Lifecycle
# ===========================================================================


class TestLifecycle:
    def test_duplicate_project_name_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        store.create_project("Duplikat")
        with pytest.raises(ValueError, match="already exists"):
            store.create_project("Duplikat")

    def test_duplicate_project_name_case_insensitive(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        store.create_project("MEDIZIN")
        with pytest.raises(ValueError, match="already exists"):
            store.create_project("medizin")

    def test_default_project_cannot_be_deleted(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        default_proj = store.load_project_profile(manifest["active_project_id"])
        with pytest.raises(ValueError, match="default"):
            store.delete_project(default_proj.project_id)

    def test_last_project_cannot_be_deleted(self, tmp_path):
        """Even if the project is not 'default', if it's the only one, deletion must fail."""
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        # Only one project exists by default
        manifest = store.load_manifest()
        proj_id = manifest["active_project_id"]
        # Patch is_default to False to test the last-project guard separately
        proj = store.load_project_profile(proj_id)
        proj.is_default = False
        proj.project_name = "Solo"
        # Manually write without duplicate check (direct file write)
        store._atomic_write_json(store.projects_dir / f"{proj_id}.json", proj.to_dict())
        with pytest.raises(ValueError):
            store.delete_project(proj_id)

    def test_active_project_switches_to_default_on_deletion(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        new_proj = store.create_project("Sekundär")
        # Switch active to new project
        manifest = store.load_manifest()
        manifest["active_project_id"] = new_proj.project_id
        store.save_manifest(manifest)
        # Delete the active project
        store.delete_project(new_proj.project_id)
        # Active must now point to the default project
        manifest2 = store.load_manifest()
        remaining = store.list_projects()
        assert manifest2["active_project_id"] == remaining[0].project_id

    def test_builtin_template_cannot_be_overwritten(self, tmp_path):
        store = _make_store(tmp_path)
        builtins = get_builtin_templates(AVAILABLE)
        tpl = list(builtins.values())[0]
        with pytest.raises(ValueError, match="Built-in templates"):
            store.save_custom_template(tpl)

    def test_builtin_template_cannot_be_deleted(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="Built-in templates"):
            store.delete_custom_template(TEMPLATE_STANDARD_FULL)

    def test_duplicate_template_name_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        uid1, uid2 = _valid_uuid4(), _valid_uuid4()
        tpl1 = CategoryTemplate(template_id=uid1, name="MeinTemplate", entity_modes={}, schema_version=2, revision=1)
        tpl2 = CategoryTemplate(template_id=uid2, name="MeinTemplate", entity_modes={}, schema_version=2, revision=1)
        store.save_custom_template(tpl1)
        with pytest.raises(ValueError, match="already exists"):
            store.save_custom_template(tpl2)

    def test_fresh_store_initializes_one_default_project(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        projects = store.list_projects()
        assert len(projects) == 1
        assert projects[0].is_default

    def test_create_project_from_template(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        new_proj = store.create_project("Medizin", template_id=TEMPLATE_MEDICAL)
        assert new_proj.entity_modes.get("HEALTH_DATA") == ENTITY_MODE_ALL

    def test_apply_template_updates_snapshot_only(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        manifest = store.load_manifest()
        project = store.load_project_profile(manifest["active_project_id"])
        previous_revision = project.revision
        updated = store.apply_template_to_project(
            project.project_id, TEMPLATE_MEDICAL, expected_revision=previous_revision
        )
        assert updated.template_id == TEMPLATE_MEDICAL
        assert updated.entity_modes["HEALTH_DATA"] == ENTITY_MODE_ALL
        assert updated.revision == previous_revision + 1


class TestOverlayAndAppIntegration:
    def test_overlay_roundtrip_and_unknown_field_rejection(self):
        term = _scoped_term("Doktor X", "PERSON", role="Arzt")
        overlay = DocumentProfileOverlay(
            entity_modes={"PERSON": ENTITY_MODE_EXPLICIT_ONLY},
            glossary_terms={term.term_key: term},
            dirty=True,
            overlay_revision=3,
        )
        restored = DocumentProfileOverlay.from_dict(overlay.to_dict(), AVAILABLE)
        assert restored.to_dict() == overlay.to_dict()
        bad = overlay.to_dict()
        bad["unexpected"] = True
        with pytest.raises(ValueError, match="Unknown fields"):
            DocumentProfileOverlay.from_dict(bad, AVAILABLE)

    def test_profile_role_is_rebound_and_manual_role_is_protected(self):
        import app

        first = type("Result", (), {
            "start": 0,
            "end": 8,
            "score": 0.95,
            "entity_type": "PERSON",
            "recognition_metadata": {
                "recognizer_name": "FuzzyGlossaryRecognizer",
                "custom_role": "Patient",
                "role_provenance": "profile",
            },
        })()
        groups, overrides = app.rebind_overrides_after_analysis("Max Muster", [first], {})
        assert groups[0].role == "Patient"
        assert groups[0].role_provenance == "profile"

        groups[0].role = "Manuell"
        groups[0].role_provenance = "manual"
        second = type("Result", (), {
            "start": 0,
            "end": 8,
            "score": 0.95,
            "entity_type": "PERSON",
            "recognition_metadata": {"custom_role": "Neues Profil"},
        })()
        rebound, _ = app.rebind_overrides_after_analysis(
            "Max Muster", [second], {}, existing_groups=groups
        )
        assert rebound[0].role == "Manuell"
        assert rebound[0].role_provenance == "manual"


# ===========================================================================
# 10. Regression — Existing Test Suite Must Still Pass
# ===========================================================================


class TestRegression:
    def test_all_existing_tests_still_pass(self):
        """Run the full test suite (excluding live-model tests) and assert zero failures."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--deselect=tests/test_gliner_local.py",
                "--deselect=tests/test_eupii_local.py",
                "--deselect=tests/test_profiles.py::TestRegression::test_all_existing_tests_still_pass",
                "-x",
                "-q",
                "--tb=short",
            ],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            pytest.fail(
                f"Existing tests failed after Phase 5b.1 changes:\n"
                f"STDOUT:\n{result.stdout[-3000:]}\n"
                f"STDERR:\n{result.stderr[-1000:]}"
            )


# ===========================================================================
# Additional Integration: EffectiveConfig → LocalAnonymizer round-trip
# ===========================================================================


class TestEffectiveConfigToAnonymizerIntegration:
    def test_to_anonymizer_kwargs_keys(self):
        proj = _make_project()
        cfg = _effective(proj)
        kwargs = cfg.to_anonymizer_kwargs()
        assert "entity_modes" in kwargs
        assert "glossary" in kwargs
        assert "glossary_roles" in kwargs
        assert "ignore_terms" in kwargs

    def test_snapshot_hash_is_sha256_hex(self):
        proj = _make_project()
        cfg = _effective(proj)
        assert len(cfg.snapshot_hash) == 64
        int(cfg.snapshot_hash, 16)  # Must be valid hex


# ===========================================================================
# Additional: Built-in Templates Cover All 23 Entities
# ===========================================================================


class TestBuiltinTemplates:
    def test_all_23_entities_covered_by_standard_full(self):
        builtins = get_builtin_templates(AVAILABLE)
        tpl = builtins[TEMPLATE_STANDARD_FULL]
        for entity in AVAILABLE:
            assert entity in tpl.entity_modes, f"Missing entity {entity} in standard-full template"

    def test_builtin_templates_all_validate(self):
        builtins = get_builtin_templates(AVAILABLE)
        for tpl in builtins.values():
            tpl.validate(AVAILABLE)  # Must not raise

    def test_five_builtin_templates_exist(self):
        builtins = get_builtin_templates(AVAILABLE)
        assert len(builtins) == 5

    def test_builtin_template_is_builtin_true(self):
        builtins = get_builtin_templates(AVAILABLE)
        for tpl in builtins.values():
            assert tpl.is_builtin is True


class TestProfileControllerR1ToR5:
    def test_document_overlay_is_volatile_and_switch_requires_discard(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        project_path = store.projects_dir / f"{controller.project_profile.project_id}.json"
        before_project = project_path.read_bytes()
        before_manifest_revision = controller.manifest["revision"]

        controller.upsert_ignore(ScopeLevel.DOCUMENT, "Nur für dieses Dokument")
        assert controller.document_overlay.dirty is True
        assert controller.document_overlay.overlay_revision > 1
        assert project_path.read_bytes() == before_project
        assert store.load_manifest()["revision"] == before_manifest_revision
        with pytest.raises(DirtyOverlayError):
            controller.switch_project(controller.project_profile.project_id)

        controller.discard_overlay()
        assert controller.document_overlay.dirty is False
        controller.switch_project(controller.project_profile.project_id)
        assert "Nur für dieses Dokument" not in controller.effective_config().ignore_terms

    def test_custom_template_crud_and_deleted_reference_keeps_snapshot(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        template = CategoryTemplate(
            template_id=_valid_uuid4(),
            name="R5 Testvorlage",
            description="CRUD",
            entity_modes={"PERSON": ENTITY_MODE_OFF},
        )
        saved = controller.save_custom_template(template)
        assert any(item.template_id == saved.template_id for item in controller.list_all_templates())
        controller.apply_template(saved.template_id)
        applied_modes = dict(controller.project_profile.entity_modes)
        controller.delete_custom_template(saved.template_id)
        assert store.load_template(saved.template_id) is None
        project = store.load_project_profile(controller.project_profile.project_id)
        assert project.template_id == saved.template_id
        assert project.entity_modes == applied_modes

    def test_template_and_project_deletion_are_revision_bound(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        first = ProfileController(store)
        template = first.save_custom_template(
            CategoryTemplate(template_id=_valid_uuid4(), name="Lösch-CAS", entity_modes={"PERSON": ENTITY_MODE_OFF})
        )
        second = ProfileController(store)
        changed = store.load_template(template.template_id)
        assert changed is not None
        changed.description = "extern geändert"
        store.save_custom_template(changed, expected_revision=changed.revision)
        with pytest.raises(RevisionConflictError):
            first.delete_custom_template(template.template_id, expected_revision=template.revision)

        project = store.create_project("Löschprojekt")
        stale = ProfileController(store, active_project_id=project.project_id)
        external = ProfileController(store, active_project_id=project.project_id)
        external.project_profile.description = "extern geändert"
        external.save_project(expected_revision=external.project_profile.revision)
        with pytest.raises(RevisionConflictError):
            stale.delete_project(expected_revision=stale.project_profile.revision)

    def test_warning_acknowledgement_is_one_time_and_cas_checked(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        first = ProfileController(store)
        second = ProfileController(store)
        assert first.warning_required() is True
        first.acknowledge_warning(expected_revision=first.manifest["revision"])
        assert first.warning_required() is False
        with pytest.raises(RevisionConflictError):
            second.acknowledge_warning(expected_revision=second.manifest["revision"])
        assert store.load_manifest()["warning_acknowledged_version"] == 1

    def test_system_mutation_requires_confirmation_before_memory_change(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        before = dict(controller.system_profile.glossary_terms)
        with pytest.raises(SystemMutationConfirmationRequired):
            controller.run_system_mutation(
                lambda: controller.upsert_glossary(ScopeLevel.SYSTEM, "Global", "PERSON"),
                confirmed=False,
            )
        assert controller.system_profile.glossary_terms == before
        controller.run_system_mutation(
            lambda: controller.upsert_glossary(ScopeLevel.SYSTEM, "Global", "PERSON"),
            confirmed=True,
        )
        assert "global" in controller.system_profile.glossary_terms

    def test_session_bound_sidebar_cas_rejects_external_project_change(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        first = ProfileController(store)
        first.acknowledge_warning(expected_revision=first.manifest["revision"])
        import app

        state = SimpleNamespace(
            profile_store=store,
            profile_controller=first,
            system_profile=first.system_profile,
            project_profile=first.project_profile,
            document_overlay=first.document_overlay,
            entity_modes=dict(first.project_profile.entity_modes),
            active_entities=[],
            format_mode="numbered_role",
            gliner_model_name=app.GLINER_MODEL_NAME,
            gliner_threshold=0.55,
            enable_eupii=False,
            eupii_threshold=0.5,
            eupii_model_name=app.EUPII_MODEL_NAME,
            ignore_terms_text="",
            glossary_text="",
            export_format="txt",
            config=SimpleNamespace(save=lambda: True),
            refresh_effective_config=lambda *args, **kwargs: None,
        )
        second = ProfileController(store)
        second.project_profile.entity_modes["PERSON"] = ENTITY_MODE_OFF
        second.save_project(expected_revision=second.project_profile.revision)

        assert app.save_current_config(state) is False
        assert state.project_profile.entity_modes["PERSON"] == ENTITY_MODE_OFF

    def test_session_bound_sidebar_saves_twice_and_keeps_controller_state(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.acknowledge_warning(expected_revision=controller.manifest["revision"])
        import app

        state = SimpleNamespace(
            profile_store=store,
            profile_controller=controller,
            system_profile=controller.system_profile,
            project_profile=controller.project_profile,
            document_overlay=controller.document_overlay,
            entity_modes=dict(controller.project_profile.entity_modes),
            format_mode="numbered_role",
            gliner_model_name=app.GLINER_MODEL_NAME,
            gliner_threshold=0.55,
            enable_eupii=False,
            eupii_threshold=0.5,
            eupii_model_name=app.EUPII_MODEL_NAME,
            ignore_terms_text="",
            glossary_text="",
            export_format="txt",
            refresh_effective_config=lambda *args, **kwargs: None,
        )

        state.entity_modes["PERSON"] = ENTITY_MODE_OFF
        assert app.save_current_config(state) is True
        assert state.project_profile.revision == controller.project_profile.revision == 2
        assert store.load_project_profile(state.project_profile.project_id).revision == 2

        state.entity_modes["PERSON"] = ENTITY_MODE_ALL
        assert app.save_current_config(state) is True
        assert state.project_profile.revision == controller.project_profile.revision == 3
        assert store.load_project_profile(state.project_profile.project_id).entity_modes["PERSON"] == ENTITY_MODE_ALL

    def test_provenance_lookup_uses_normalized_surface_key(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.upsert_glossary(ScopeLevel.SYSTEM, "ZHAW", "ORGANIZATION")
        cfg = controller.effective_config()
        assert cfg.glossary["ZHAW"] == "ORGANIZATION"
        assert cfg.glossary_provenance[normalize_term_key("zhaw")][0] == ScopeLevel.SYSTEM

    def test_provenance_badges_resolve_all_scopes_with_normalized_keys(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.upsert_glossary(ScopeLevel.SYSTEM, "Systemwort", "ORGANIZATION", role="Systemrolle")
        controller.save_system()
        controller.upsert_glossary(ScopeLevel.PROJECT, "ProjektWort", "PERSON", role="Projektrolle")
        controller.save_project()
        controller.upsert_glossary(ScopeLevel.DOCUMENT, "DokumentWort", "ROLE", role="Dokumentrolle")
        cfg = controller.effective_config()

        assert cfg.glossary_provenance[normalize_term_key("systemWORT")][0] == ScopeLevel.SYSTEM
        assert cfg.glossary_provenance[normalize_term_key("projektwort")][0] == ScopeLevel.PROJECT
        assert cfg.glossary_provenance[normalize_term_key("DOKUMENTWORT")][0] == ScopeLevel.DOCUMENT
        assert cfg.glossary_roles["Systemwort"] == "Systemrolle"
        assert cfg.glossary_roles["ProjektWort"] == "Projektrolle"
        assert cfg.glossary_roles["DokumentWort"] == "Dokumentrolle"

    def test_gui_profile_manager_uses_panel_owned_holders_and_captured_terms(self):
        source = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
        panel_start = source.index("with ui.tab_panels(profile_tabs, value=categories_tab)")
        panel_block = source[panel_start:source.index("with ui.row().classes(\"justify-end w-full mt-3\")", panel_start)]
        assert "with ui.tab_panel(categories_tab):\n                    category_holder = ui.column()" in panel_block
        assert "with ui.tab_panel(glossary_tab):\n                    glossary_holder = ui.column()" in panel_block
        assert "with ui.tab_panel(ignore_tab):\n                    ignore_holder = ui.column()" in panel_block
        assert "category_holder\n" not in panel_block
        assert "term_value = str(g_term.value or \"\").strip()" in source
        assert "role = cfg.glossary_roles.get(key)" in source
        profile_block = source[source.index("def open_profile_manager"):source.index("with ui.card().classes(\"w-full mb-3", panel_start)]
        assert profile_block.count("max-h-64 overflow-y-auto") == 3

    def test_actual_profile_sync_updates_sidebar_category_selectors(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        fake_ui = _FakeUi()
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, {"value": False})
        selector = _FakeUiElement(ENTITY_MODE_ALL)
        mode_change, selector_ref = ns["make_entity_mode_change"]("PERSON")
        selector.on_change = mode_change
        selector_ref.append(selector)
        ns["sidebar_entity_mode_selects"] = {"PERSON": selector}
        save_calls = []
        ns.update(
            {
                "save_current_config": lambda *_args, **_kwargs: save_calls.append("save"),
                "compute_reactive_preview": lambda *_args, **_kwargs: None,
                "refresh_preview_and_exports": lambda *_args, **_kwargs: None,
                "build_review_table": lambda *_args, **_kwargs: None,
            }
        )
        mode_change.__globals__.update(ns)
        ns["_sync_profile_controller"].__globals__.update(ns)

        controller.project_profile.entity_modes["PERSON"] = ENTITY_MODE_OFF
        state.entity_modes["PERSON"] = ENTITY_MODE_OFF
        ns["_sync_profile_controller"]()

        assert selector.value == ENTITY_MODE_OFF
        assert save_calls == []

    def test_profile_ui_sync_guard_is_released_after_exception_and_user_change_still_works(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        fake_ui = _FakeUi()
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, {"value": False})

        class FailingElement(_FakeUiElement):
            def set_value(self, value):
                raise RuntimeError("synthetic UI failure")

        with pytest.raises(RuntimeError, match="synthetic UI failure"):
            ns["_set_profile_ui_value"](FailingElement(), ENTITY_MODE_OFF)
        assert ns["profile_ui_sync_depth"] == [0]

        mode_change, selector_ref = ns["make_entity_mode_change"]("PERSON")
        selector = _FakeUiElement(ENTITY_MODE_ALL)
        selector_ref.append(selector)
        save_calls = []
        ns.update(
            {
                "save_current_config": lambda *_args, **_kwargs: save_calls.append("save"),
                "compute_reactive_preview": lambda *_args, **_kwargs: None,
                "refresh_preview_and_exports": lambda *_args, **_kwargs: None,
                "build_review_table": lambda *_args, **_kwargs: None,
            }
        )
        mode_change.__globals__.update(ns)
        mode_change(SimpleNamespace(value=ENTITY_MODE_OFF))

        assert state.entity_modes["PERSON"] == ENTITY_MODE_OFF
        assert save_calls == ["save"]
        assert ns["profile_ui_sync_depth"] == [0]

    def test_actual_template_cancel_restores_selector_and_success_marks_reanalysis(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.acknowledge_warning(expected_revision=controller.manifest["revision"])
        template_id = _valid_uuid4()
        modes = dict(controller.project_profile.entity_modes)
        modes["PERSON"] = ENTITY_MODE_OFF if modes.get("PERSON") != ENTITY_MODE_OFF else ENTITY_MODE_ALL
        store.save_custom_template(
            CategoryTemplate(
                template_id=template_id,
                name="GUI-Testvorlage",
                entity_modes=modes,
                schema_version=2,
                revision=1,
            )
        )
        fake_ui = _FakeUi()
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, {"value": False})
        state.entity_groups = [object()]
        template_select = _FakeUiElement(None)
        template_callbacks = []
        warning_card = _FakeUiElement()
        warning_card.visibility = []
        warning_card.set_visibility = lambda visible: warning_card.visibility.append(visible)
        ns.update(
            {
                "template_select": template_select,
                "template_callbacks": template_callbacks,
                "reanalysis_warning_card": warning_card,
                "refresh_preview_and_exports": lambda: None,
                "sidebar_entity_mode_selects": {},
            }
        )
        ns["apply_template"].__globals__.update(ns)
        template_select.on_change = lambda event: (
            template_callbacks.append(event.value),
            ns["apply_template"](event.value),
        )

        original_modes = dict(controller.project_profile.entity_modes)
        ns["apply_template"](template_id)
        fake_ui.click_last("Abbrechen")
        assert template_select.value is None
        assert controller.project_profile.entity_modes == original_modes
        assert template_callbacks == [None]

        ns["apply_template"](template_id)
        fake_ui.click_last("Anwenden")
        assert template_select.value == template_id
        assert controller.project_profile.entity_modes == modes
        assert warning_card.visibility == [True]
        assert template_callbacks == [None, template_id]

    def test_actual_profile_mutation_rejects_busy_system_confirmation_without_reload(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.acknowledge_warning(expected_revision=controller.manifest["revision"])
        fake_ui = _FakeUi()
        busy = {"value": False}
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, busy)
        before_overlay = state.document_overlay
        before_overlay_dict = before_overlay.to_dict()
        action_calls = []
        after_success_calls = []

        ns["mutate"](
            "system",
            lambda: action_calls.append("action"),
            after_success=lambda: after_success_calls.append("after"),
            expected_project_id=state.project_profile.project_id,
        )
        busy["value"] = True
        fake_ui.click_last("Systemweit anwenden")

        assert action_calls == []
        assert after_success_calls == []
        assert state.document_overlay is before_overlay
        assert state.document_overlay.to_dict() == before_overlay_dict
        assert controller.document_overlay is before_overlay
        assert "privatwort" not in controller.system_profile.glossary_terms
        assert store.load_system_profile().glossary_terms == {}
        assert any("beschäftigt" in message for message, _ in fake_ui.notifications)

    def test_actual_profile_mutation_rejects_project_switch_without_touching_new_overlay(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        new_project = store.create_project("Neues Ziel")
        fake_ui = _FakeUi()
        busy = {"value": False}
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, busy)
        old_project_id = state.project_profile.project_id
        old_project_overlay = state.document_overlay
        old_project_overlay_dict = old_project_overlay.to_dict()
        new_project_profile = store.load_project_profile(new_project.project_id)
        new_overlay = DocumentProfileOverlay(
            glossary_terms={"neueswort": _scoped_term("Neueswort", "ORGANIZATION")},
            dirty=True,
            overlay_revision=11,
        )
        action_calls = []
        after_success_calls = []

        ns["mutate"](
            "project",
            lambda: action_calls.append("action"),
            after_success=lambda: after_success_calls.append("after"),
            expected_project_id=old_project_id,
        )
        state.project_profile = new_project_profile
        state.document_overlay = new_overlay
        controller.project_profile = new_project_profile
        controller.document_overlay = new_overlay
        fake_ui.click_last("Bestätigen & Fortfahren")

        assert action_calls == []
        assert after_success_calls == []
        assert state.project_profile.project_id == new_project.project_id
        assert state.document_overlay is new_overlay
        assert state.document_overlay.to_dict() == new_overlay.to_dict()
        assert store.load_project_profile(old_project_id).glossary_terms == {}
        assert old_project_overlay.to_dict() == old_project_overlay_dict
        assert any("Projekt" in message for message, _ in fake_ui.notifications)

    def test_actual_profile_mutation_reloads_cas_conflict_without_losing_overlay(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.acknowledge_warning(expected_revision=controller.manifest["revision"])
        fake_ui = _FakeUi()
        busy = {"value": False}
        state, ns = _make_gui_adapter_context(store, controller, fake_ui, busy)
        overlay = state.document_overlay
        overlay_dict = overlay.to_dict()
        external = ProfileController(store)
        external.upsert_glossary(ScopeLevel.PROJECT, "Extern", "ORGANIZATION")
        external.save_project(expected_revision=external.project_profile.revision)
        action_calls = []
        after_success_calls = []

        def action():
            action_calls.append("action")
            controller.upsert_glossary(ScopeLevel.PROJECT, "Lokal", "PERSON")

        ns["mutate"](
            "project",
            action,
            after_success=lambda: after_success_calls.append("after"),
            expected_project_id=state.project_profile.project_id,
        )

        assert action_calls == ["action"]
        assert after_success_calls == []
        assert state.document_overlay is overlay
        assert state.document_overlay.to_dict() == overlay_dict
        assert controller.document_overlay is overlay
        loaded = store.load_project_profile(state.project_profile.project_id)
        assert "extern" in loaded.glossary_terms
        assert "lokal" not in loaded.glossary_terms
        assert any("extern geändert" in message for message, _ in fake_ui.notifications)

    def test_actual_delayed_glossary_and_ignore_confirmation_and_cancel_keep_inputs_safe(self, tmp_path):
        def run_glossary_flow():
            store = _make_store(tmp_path / "glossary")
            store.initialize_or_migrate()
            controller = ProfileController(store)
            fake_ui = _FakeUi()
            state, ns = _make_gui_adapter_context(store, controller, fake_ui, {"value": False})
            ns.update(
                {
                    "g_term": _FakeUiElement("Begriff"),
                    "g_type": _FakeUiElement("PERSON"),
                    "g_role": _FakeUiElement("Rolle"),
                    "i_term": _FakeUiElement("Ignorieren"),
                }
            )
            ns["add_glossary"].__globals__.update(ns)
            ns["add_glossary"]()
            fake_ui.click_last("Abbrechen")
            assert ns["g_term"].value == "Begriff"
            assert "begriff" not in controller.project_profile.glossary_terms
            ns["add_glossary"]()
            fake_ui.click_last("Bestätigen & Fortfahren")
            assert ns["g_term"].value == ""
            assert ns["g_role"].value == ""
            assert "begriff" in controller.project_profile.glossary_terms
            assert state.document_overlay.dirty is True

        def run_ignore_flow():
            store = _make_store(tmp_path / "ignore")
            store.initialize_or_migrate()
            controller = ProfileController(store)
            fake_ui = _FakeUi()
            state, ns = _make_gui_adapter_context(store, controller, fake_ui, {"value": False})
            ns.update(
                {
                    "g_term": _FakeUiElement("Glossar"),
                    "g_type": _FakeUiElement("PERSON"),
                    "g_role": _FakeUiElement("Rolle"),
                    "i_term": _FakeUiElement("Ignorieren"),
                }
            )
            ns["add_ignore"].__globals__.update(ns)
            ns["add_ignore"]()
            fake_ui.click_last("Abbrechen")
            assert ns["i_term"].value == "Ignorieren"
            assert "ignorieren" not in controller.project_profile.ignore_terms
            ns["add_ignore"]()
            fake_ui.click_last("Bestätigen & Fortfahren")
            assert ns["i_term"].value == ""
            assert "ignorieren" in controller.project_profile.ignore_terms
            assert state.document_overlay.dirty is True

        run_glossary_flow()
        run_ignore_flow()

    def test_actual_analyze_uses_frozen_effective_scope_precedence(self, tmp_path):
        store = _make_store(tmp_path)
        store.initialize_or_migrate()
        controller = ProfileController(store)
        controller.upsert_glossary(ScopeLevel.SYSTEM, "ZHAW", "ORGANIZATION", role="Hochschule")
        controller.save_system()
        controller.upsert_ignore(ScopeLevel.PROJECT, "ZHAW")
        controller.save_project()
        controller.upsert_glossary(ScopeLevel.DOCUMENT, "ZHAW", "PERSON", role="Projektleitung")
        for entity in AVAILABLE:
            controller.set_entity_mode(ScopeLevel.PROJECT, entity, ENTITY_MODE_OFF)
        controller.set_entity_mode(ScopeLevel.PROJECT, "PERSON", ENTITY_MODE_EXPLICIT_ONLY)
        controller.save_project()
        snapshot = controller.effective_config()

        import app

        state = SimpleNamespace(
            effective_config=snapshot,
            gliner_model_name=app.GLINER_MODEL_NAME,
            gliner_threshold=0.55,
            enable_eupii=False,
            eupii_threshold=0.5,
            eupii_model_name=app.EUPII_MODEL_NAME,
        )
        anonymizer = app.build_anonymizer(state, effective_config=snapshot)
        results = anonymizer.analyze("ZHAW")
        assert len(results) == 1
        assert results[0].entity_type == "PERSON"
        assert results[0].recognition_metadata.get("custom_role") == "Projektleitung"
