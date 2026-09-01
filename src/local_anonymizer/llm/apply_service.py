import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple


CANONICAL_APP_ENTITY_TYPES: Set[str] = {
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
}

ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "ORG": "ORGANIZATION",
    "PER": "PERSON",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}


def normalize_entity_type(raw_type: Optional[str]) -> Optional[str]:
    """
    Normalize model/provider entity types to canonical application types.
    Maps aliases like ORG -> ORGANIZATION, LOC -> LOCATION, PER -> PERSON.
    Returns canonical type string or None if unrecognized/unsupported.
    """
    if not raw_type or not isinstance(raw_type, str):
        return None
    cleaned = raw_type.strip().upper()
    if cleaned in CANONICAL_APP_ENTITY_TYPES:
        return cleaned
    if cleaned in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[cleaned]
    return None


def check_mutation_allowed(state: Any) -> bool:
    """Central guard against mutating state while LLM triage is running."""
    if getattr(state, "is_llm_running", False):
        return False
    return True


@dataclass
class ApplyCommand:
    occ_id: str
    action: Literal["keep", "recategorize", "discard"]
    new_entity_type: Optional[str] = None
    descriptor_suggestion: Optional[str] = None


def compute_triage_snapshot(
    raw_text: str,
    analysis_revision: int,
    entity_groups: List[Any],
) -> str:
    """
    Compute a deterministic SHA-256 snapshot hash for candidate triage state.
    Binds:
    1. SHA-256 of raw_text
    2. analysis_revision
    3. Canonically ordered candidate entries: (occ_id, start, end, entity_type, role, enabled)
    """
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else "empty"

    candidates = []
    for g in entity_groups:
        for occ in getattr(g, "occurrences", []):
            candidates.append((
                occ.occ_id,
                occ.start,
                occ.end,
                getattr(g, "entity_type", ""),
                getattr(g, "role", ""),
                bool(getattr(g, "enabled", True)),
            ))

    # Sort candidates canonically by occ_id
    candidates.sort(key=lambda c: c[0])

    entries_str = ";".join(
        f"{c[0]}:{c[1]}:{c[2]}:{c[3]}:{c[4]}:{c[5]}" for c in candidates
    )

    payload = f"{raw_hash}|{analysis_revision}|{entries_str}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApplyService:
    """
    Canonical two-phase service for executing validated LLM triage proposals on AppState.
    Guarantees atomic fail-safe mutation without partial state corruption.
    """

    @staticmethod
    def _find_occurrence_and_group(
        entity_groups: List[Any], occ_id: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Find the (EntityGroup, EntityOccurrence) pair for a given occ_id."""
        for g in entity_groups:
            for occ in getattr(g, "occurrences", []):
                if occ.occ_id == occ_id:
                    return g, occ
        return None, None

    @classmethod
    def prevalidate_and_preview_impact(
        cls,
        state: Any,
        expected_snapshot_hash: str,
        commands: List[ApplyCommand],
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Phase 1: Full validation against current snapshot and generation of impact preview.
        Returns: (is_valid, error_message, impact_list)
        """
        if not commands:
            return False, "Keine Befehle zur Übernahme ausgewählt.", []

        current_snapshot = compute_triage_snapshot(
            state.raw_text,
            state.analysis_revision,
            state.entity_groups,
        )
        if current_snapshot != expected_snapshot_hash:
            return False, "Triage-Snapshot ist veraltet (der Dokument- oder Review-Zustand wurde inzwischen verändert).", []

        impacts: List[Dict[str, Any]] = []
        seen_occ_ids: Set[str] = set()

        for cmd in commands:
            if cmd.occ_id in seen_occ_ids:
                return False, f"Doppelter Befehl für Fundstelle {cmd.occ_id}.", []
            seen_occ_ids.add(cmd.occ_id)

            norm_type: Optional[str] = None
            if cmd.action == "recategorize":
                norm_type = normalize_entity_type(cmd.new_entity_type)
                if not norm_type:
                    return (
                        False,
                        f"Ungültiger Entitätstyp '{cmd.new_entity_type}' für Fundstelle {cmd.occ_id} vorgeschlagen.",
                        [],
                    )

            grp, occ = cls._find_occurrence_and_group(state.entity_groups, cmd.occ_id)
            if grp is None or occ is None:
                return False, f"Fundstelle mit ID '{cmd.occ_id}' wurde im aktuellen Zustand nicht gefunden.", []

            will_split = len(grp.occurrences) > 1
            old_type = grp.entity_type
            new_type = norm_type if (cmd.action == "recategorize" and norm_type) else grp.entity_type
            old_role = grp.role
            new_role = cmd.descriptor_suggestion if cmd.descriptor_suggestion is not None else grp.role
            old_enabled = grp.enabled
            new_enabled = False if cmd.action == "discard" else True

            impacts.append({
                "occ_id": cmd.occ_id,
                "original_text": grp.original_text,
                "action": cmd.action,
                "old_type": old_type,
                "new_type": new_type,
                "old_role": old_role,
                "new_role": new_role,
                "old_enabled": old_enabled,
                "new_enabled": new_enabled,
                "will_split": will_split,
            })

        return True, "", impacts

    @classmethod
    def apply_mutations(
        cls,
        state: Any,
        expected_snapshot_hash: str,
        commands: List[ApplyCommand],
        split_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
        sync_fn: Optional[Callable[[Any, Any], None]] = None,
        preview_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Phase 2: Atomic execution of mutations after prevalidation.
        Guarantees that if any item fails, no mutations remain applied.
        """
        is_valid, err, _ = cls.prevalidate_and_preview_impact(state, expected_snapshot_hash, commands)
        if not is_valid:
            return False, err

        # Resolve helper functions from app if not provided
        if split_fn is None or sync_fn is None or preview_fn is None:
            try:
                import app as app_module
                split_fn = split_fn or getattr(app_module, "split_occurrence_to_new_group")
                sync_fn = sync_fn or getattr(app_module, "sync_group_overrides")
                preview_fn = preview_fn or getattr(app_module, "compute_reactive_preview")
            except Exception:
                pass

        # Create defensive deep transaction snapshot of state before mutations
        saved_groups = copy.deepcopy(getattr(state, "entity_groups", []))
        saved_overrides = copy.deepcopy(getattr(state, "occurrence_overrides", {}))
        saved_mapping = copy.deepcopy(getattr(state, "current_mapping", {}))
        saved_anon_text = getattr(state, "current_anon_text", "")
        saved_report = copy.deepcopy(getattr(state, "current_report", {}))
        saved_preview_rev = getattr(state, "preview_revision", 0)
        saved_llm_results = copy.deepcopy(getattr(state, "llm_triage_results", {}))
        saved_llm_snapshot = getattr(state, "llm_triage_snapshot", "")
        saved_llm_staged = copy.deepcopy(getattr(state, "llm_staged_selections", set()))

        try:
            # Execute mutations
            for cmd in commands:
                grp, occ = cls._find_occurrence_and_group(state.entity_groups, cmd.occ_id)
                if grp is None or occ is None:
                    raise RuntimeError(f"Fundstelle '{cmd.occ_id}' unerwartet nicht mehr auffindbar.")

                # If occurrence is part of a multi-occurrence group, split it into an isolated group
                if len(grp.occurrences) > 1 and split_fn is not None:
                    target_group = split_fn(state, grp, occ)
                else:
                    target_group = grp

                # Apply changes based on action
                if cmd.action == "discard":
                    target_group.enabled = False
                elif cmd.action == "recategorize":
                    norm_type = normalize_entity_type(cmd.new_entity_type)
                    if not norm_type:
                        raise ValueError(f"Ungültiger Entitätstyp '{cmd.new_entity_type}' für Fundstelle {cmd.occ_id}.")
                    target_group.entity_type = norm_type
                    if cmd.descriptor_suggestion is not None:
                        target_group.role = cmd.descriptor_suggestion
                    target_group.enabled = True
                elif cmd.action == "keep":
                    if cmd.descriptor_suggestion is not None:
                        target_group.role = cmd.descriptor_suggestion
                    target_group.enabled = True

                if sync_fn is not None:
                    sync_fn(state, target_group)

            # Invalidate / clear LLM proposals in state since candidates were mutated
            if hasattr(state, "llm_triage_results"):
                state.llm_triage_results.clear()
            if hasattr(state, "llm_triage_snapshot"):
                state.llm_triage_snapshot = ""
            if hasattr(state, "llm_staged_selections"):
                state.llm_staged_selections.clear()

            # Recalculate preview
            if preview_fn is not None:
                preview_fn(state)

            return True, f"{len(commands)} Änderungen erfolgreich übernommen."

        except Exception as exc:
            # Atomic rollback to exact initial state
            state.entity_groups = saved_groups
            state.occurrence_overrides = saved_overrides
            state.current_mapping = saved_mapping
            state.current_anon_text = saved_anon_text
            state.current_report = saved_report
            state.preview_revision = saved_preview_rev
            state.llm_triage_results = saved_llm_results
            state.llm_triage_snapshot = saved_llm_snapshot
            state.llm_staged_selections = saved_llm_staged
            return False, f"Transaktionsfehler beim Anwenden der Triage: {exc}"
