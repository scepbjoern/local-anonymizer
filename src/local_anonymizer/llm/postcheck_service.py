"""
Postcheck Service (Phase 6B - Ausgangskontrolle).
Implements unchanged-segment mapping, token budgeting, scope/category validation,
conflict checking, and atomic batch apply.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from local_anonymizer.llm.apply_service import normalize_entity_type
from local_anonymizer.llm.batching import estimate_tokens
from local_anonymizer.llm.postcheck_schema import (
    SYSTEM_POSTCHECK_PROMPT,
    USER_POSTCHECK_TEMPLATE,
    PostcheckEnvelope,
    PostcheckFindingItem,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app import (
        EntityGroup,
        EntityOccurrence,
        OccurrenceOverride,
    )

# Constants for Token Budget (Phase 6B)
MAX_POSTCHECK_TOTAL_BUDGET: int = 32000
POSTCHECK_RESPONSE_RESERVE: int = 4096
MAX_POSTCHECK_INPUT_TOKENS: int = MAX_POSTCHECK_TOTAL_BUDGET - POSTCHECK_RESPONSE_RESERVE  # 27904
POSTCHECK_CHAT_BUFFER: int = 100


def get_known_context_limit(
    model_name: str,
    catalog: Optional[Any] = None,
    catalog_path: Optional[Path] = None,
) -> Optional[int]:
    """
    Return known context limit in tokens if the model has a documented limit in the catalog,
    or None if the model is uncataloged or has no specified limit.
    Raises CatalogError on corrupted/invalid catalog data.
    """
    if not model_name or not model_name.strip():
        return None
    from local_anonymizer.llm.catalog import load_catalog, find_catalog_entry
    if catalog is None:
        catalog = load_catalog(catalog_path=catalog_path)
    entry = find_catalog_entry(model_name.strip(), catalog=catalog)
    if entry and getattr(entry, "context_limit", None):
        return int(entry.context_limit)
    return None


def calculate_postcheck_budget(anon_text: str, request_id: str = "test-req") -> Tuple[bool, int, int, str]:
    """
    Calculate estimated input tokens for postcheck (System-Prompt + User-Template + Text + Buffer).
    Returns: (is_within_budget, estimated_tokens, max_allowed_tokens, message)
    """
    user_prompt = USER_POSTCHECK_TEMPLATE.format(anon_text=anon_text, request_id=request_id)
    est_sys = estimate_tokens(SYSTEM_POSTCHECK_PROMPT)
    est_user = estimate_tokens(user_prompt)
    total_input = est_sys + est_user + POSTCHECK_CHAT_BUFFER

    if total_input > MAX_POSTCHECK_INPUT_TOKENS:
        msg = (
            f"Text zu gross für Ausgangskontrolle: Geschätzter Input von {total_input} Tokens "
            f"überschreitet das Limit von {MAX_POSTCHECK_INPUT_TOKENS} Tokens (Reserve: {POSTCHECK_RESPONSE_RESERVE} Tokens). "
            "Prüfung nicht durchgeführt."
        )
        return False, total_input, MAX_POSTCHECK_INPUT_TOKENS, msg

    return True, total_input, MAX_POSTCHECK_INPUT_TOKENS, ""


def compute_unchanged_segments(
    raw_text: str,
    active_occurrences: List[Dict[str, Any]],
) -> List[Tuple[int, int, int, int]]:
    """
    Compute intervals of text that remained completely unchanged between raw text and anonymized text.
    Each occurrence is expected to have 'start', 'end' (raw offsets) and 'placeholder' (replacement string).

    Returns a list of 4-tuples:
    (raw_start, raw_end, anon_start, anon_end)
    sorted by raw_start.
    """
    if not raw_text:
        return []

    if not active_occurrences:
        return [(0, len(raw_text), 0, len(raw_text))]

    # Sort occurrences by raw start offset
    sorted_occs = sorted(active_occurrences, key=lambda o: o["start"])

    segments: List[Tuple[int, int, int, int]] = []
    cur_raw = 0
    cur_anon = 0

    for occ in sorted_occs:
        o_raw_start = occ["start"]
        o_raw_end = occ["end"]
        placeholder = occ.get("placeholder", "")

        # Text before this placeholder
        if o_raw_start > cur_raw:
            chunk_len = o_raw_start - cur_raw
            segments.append((cur_raw, o_raw_start, cur_anon, cur_anon + chunk_len))
            cur_anon += chunk_len

        # Advance past the replacement
        cur_raw = o_raw_end
        cur_anon += len(placeholder)

    # Remaining text after the last placeholder
    if cur_raw < len(raw_text):
        chunk_len = len(raw_text) - cur_raw
        segments.append((cur_raw, len(raw_text), cur_anon, cur_anon + chunk_len))

    return segments


def map_output_slice_to_raw(
    output_start: int,
    output_end: int,
    found_text: str,
    current_anon_text: str,
    raw_text: str,
    unchanged_segments: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    """
    Map [output_start:output_end] from anonymized text back to raw text offsets [raw_start:raw_end].

    Strict requirements:
    1. 0 <= output_start < output_end <= len(current_anon_text)
    2. current_anon_text[output_start:output_end] == found_text
    3. The interval [output_start, output_end] must be FULLY contained within one unchanged segment.
    4. raw_text[raw_start:raw_end] == found_text

    Raises ValueError if mapping fails or overlaps any placeholder.
    """
    if output_start < 0 or output_end <= output_start or output_end > len(current_anon_text):
        raise ValueError(
            f"Ungültiger Zeichenbereich [{output_start}:{output_end}] für Anonymisierungstext (Länge {len(current_anon_text)})."
        )

    actual_slice = current_anon_text[output_start:output_end]
    if actual_slice != found_text:
        raise ValueError(
            f"Slice-Fehlschlag im Anonymisierungstext an [{output_start}:{output_end}]: Text stimmt nicht mit Fundstelle überein."
        )

    # Find the segment that completely encloses [output_start, output_end]
    matching_segment: Optional[Tuple[int, int, int, int]] = None
    for seg in unchanged_segments:
        raw_s, raw_e, anon_s, anon_e = seg
        if anon_s <= output_start and output_end <= anon_e:
            matching_segment = seg
            break

    if matching_segment is None:
        raise ValueError(
            f"Fundstelle an [{output_start}:{output_end}] liegt nicht vollständig in einem unveränderten Textsegment (überlappt evtl. Platzhalter)."
        )

    raw_s, raw_e, anon_s, anon_e = matching_segment
    offset_inside_segment = output_start - anon_s
    span_len = output_end - output_start

    raw_start = raw_s + offset_inside_segment
    raw_end = raw_start + span_len

    if raw_end > len(raw_text) or raw_text[raw_start:raw_end] != found_text:
        raise ValueError(
            f"Offset-Fehlschlag im Originaltext: Position [{raw_start}:{raw_end}] stimmt nicht mit Fundstelle überein."
        )

    return raw_start, raw_end


def validate_scope_and_category(
    entity_type: str,
    text: str,
    entity_modes: Dict[str, str],
    ignored_terms: Set[str],
    disabled_spans: Optional[List[Tuple[int, int]]] = None,
    raw_span: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str, str]:
    """
    Validate finding against active entity profile, ignore lists, and manually disabled occurrences.
    Returns: (is_allowed, normalized_type, error_reason)
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return False, "", f"Unbekannter oder unzulässiger Entitätstyp '{entity_type}'."

    cleaned_text = text.strip().lower()
    norm_ignores = {t.strip().lower() for t in ignored_terms if t.strip()}
    if cleaned_text in norm_ignores:
        return False, norm_type, "Begriff ist in der aktiven Ignorierliste geschützt."

    mode = entity_modes.get(norm_type, "all")
    if mode in ("off", "explicit_only"):
        return False, norm_type, f"Entitätstyp '{norm_type}' ist durch das Profil '{mode}' für neue Erkennungen gesperrt."

    if mode not in ("all", "explicit_eupii"):
        return False, norm_type, f"Nicht unterstützter Modus '{mode}' für Entitätstyp '{norm_type}'."

    if raw_span and disabled_spans:
        r_s, r_e = raw_span
        for ds, de in disabled_spans:
            if not (r_e <= ds or r_s >= de):
                return False, norm_type, "Fundstelle überlappt mit einer manuell deaktivierten Entität."

    return True, norm_type, ""


def check_batch_conflicts(
    findings: List[Dict[str, Any]],
    existing_spans: List[Tuple[int, int]],
) -> Tuple[bool, str]:
    """
    Check findings against each other and against existing spans for overlaps.
    Returns: (is_valid, error_message)
    """
    if not findings:
        return True, ""

    # Sort selected findings by raw_start
    sorted_f = sorted(findings, key=lambda f: f["raw_start"])

    # 1. Pairwise overlap check between selected findings
    for i in range(len(sorted_f) - 1):
        cur = sorted_f[i]
        nxt = sorted_f[i + 1]
        if cur["raw_end"] > nxt["raw_start"]:
            return False, (
                f"Konflikt zwischen ausgewählten Fundstellen: '{cur['text']}' [{cur['raw_start']}:{cur['raw_end']}] "
                f"überlappt mit '{nxt['text']}' [{nxt['raw_start']}:{nxt['raw_end']}]."
            )

    # 2. Overlap check against existing active entities
    for f in sorted_f:
        f_s, f_e = f["raw_start"], f["raw_end"]
        for e_s, e_e in existing_spans:
            if not (f_e <= e_s or f_s >= e_e):
                return False, (
                    f"Konflikt: Fundstelle '{f['text']}' [{f_s}:{f_e}] "
                    f"überlappt mit bestehender Entität [{e_s}:{e_e}]."
                )

    return True, ""


def atomic_apply_postcheck_findings(
    state: Any,
    findings_to_apply: List[Dict[str, Any]],
    expected_run_id: Optional[str] = None,
    preview_fn: Optional[Callable[[Any], Any]] = None,
    sync_fn: Optional[Callable[[Any, Any], None]] = None,
) -> Tuple[bool, str]:
    """
    Atomically apply selected postcheck findings into AppState.
    Guarantees all-or-nothing rollback on any validation or insertion error.
    """
    if not getattr(state, "is_postcheck_active", False):
        return False, "Ausgangskontrolle ist nicht aktiv."

    current_run_id = getattr(state, "postcheck_run_id", "")
    if expected_run_id is not None and current_run_id != expected_run_id:
        return False, f"Veralteter Lauf: Erwartet '{expected_run_id}', aktiv ist '{current_run_id}'."

    if not findings_to_apply:
        return False, "Keine Fundstellen zur Übernahme ausgewählt."

    # Pre-validate finding structure and raw text bounds
    raw_text = getattr(state, "raw_text", "")
    for f in findings_to_apply:
        r_s = f.get("raw_start")
        r_e = f.get("raw_end")
        txt = f.get("text")
        if r_s is None or r_e is None or txt is None or r_s < 0 or r_e > len(raw_text) or raw_text[r_s:r_e] != txt:
            return False, f"Fundstelle '{txt}' [{r_s}:{r_e}] ist ungültig oder passt nicht zum aktuellen Rohtext."

    # Collect existing active spans and disabled spans
    existing_active_spans: List[Tuple[int, int]] = []
    disabled_spans: List[Tuple[int, int]] = []

    for g in getattr(state, "entity_groups", []):
        if not g.enabled:
            for occ in g.occurrences:
                disabled_spans.append((occ.start, occ.end))
        else:
            for occ in g.occurrences:
                ov = getattr(state, "occurrence_overrides", {}).get(occ.occ_id)
                if ov and not ov.enabled:
                    disabled_spans.append((occ.start, occ.end))
                else:
                    existing_active_spans.append((occ.start, occ.end))

    # Reject if any finding overlaps disabled entities
    for f in findings_to_apply:
        f_s, f_e = f["raw_start"], f["raw_end"]
        for ds, de in disabled_spans:
            if not (f_e <= ds or f_s >= de):
                return False, f"Konflikt: Fundstelle '{f['text']}' [{f_s}:{f_e}] überlappt mit einer manuell deaktivierten Entität [{ds}:{de}]."

    # Validate pairwise and active overlaps
    valid, conflict_err = check_batch_conflicts(findings_to_apply, existing_active_spans)
    if not valid:
        return False, conflict_err

    # Create full transaction snapshot for rollback
    saved_groups = copy.deepcopy(getattr(state, "entity_groups", []))
    saved_overrides = copy.deepcopy(getattr(state, "occurrence_overrides", {}))
    saved_mapping = copy.deepcopy(getattr(state, "current_mapping", {}))
    saved_anon_text = getattr(state, "current_anon_text", "")
    saved_report = copy.deepcopy(getattr(state, "current_report", {}))
    saved_preview_rev = getattr(state, "preview_revision", 0)
    saved_findings = copy.deepcopy(getattr(state, "postcheck_findings", []))
    saved_selected_ids = copy.deepcopy(getattr(state, "postcheck_selected_ids", set()))
    saved_colliding_roles = copy.deepcopy(getattr(state, "colliding_roles", set()))
    saved_status_msg = getattr(state, "postcheck_status_msg", "")
    saved_run_id = getattr(state, "postcheck_run_id", "")
    saved_active = getattr(state, "is_postcheck_active", False)

    from app import (
        EntityGroup,
        EntityOccurrence,
        OccurrenceOverride,
        extract_context_snippet,
        compute_context_fingerprint,
    )

    try:
        for f in findings_to_apply:
            raw_s = f["raw_start"]
            raw_e = f["raw_end"]
            text = f["text"]
            ent_type = f["entity_type"]

            occ_id = f"occ_post_{uuid.uuid4().hex[:12]}"
            ctx_html = extract_context_snippet(raw_text, raw_s, raw_e) if raw_text else ""
            ctx_fingerprint = compute_context_fingerprint(raw_text, raw_s, raw_e) if raw_text else ""

            occ = EntityOccurrence(
                start=raw_s,
                end=raw_e,
                score=1.0,
                context_html=ctx_html,
                needs_review=False,
                source="llm_postcheck",
                method="llm_postcheck",
                occ_id=occ_id,
                context_fingerprint=ctx_fingerprint,
            )

            text_norm = text.strip().lower()

            # Find matching canonical base group:
            # Must match entity_type AND text_key
            # AND must be a base group (group_id == text_norm)
            # AND must be enabled
            # AND must not have a custom role or parent linking assigned
            matching_group = None
            for g in state.entity_groups:
                if (
                    g.entity_type == ent_type
                    and g.text_key == text_norm
                    and g.group_id == text_norm
                    and g.enabled is True
                    and not getattr(g, "role", "")
                    and not getattr(g, "parent_group_id", None)
                ):
                    matching_group = g
                    break

            if matching_group is not None:
                matching_group.occurrences.append(occ)
                matching_group.occurrences.sort(key=lambda o: o.start)
                if sync_fn is not None:
                    sync_fn(state, matching_group)
            else:
                existing_group_ids = {g.group_id for g in state.entity_groups}
                if text_norm not in existing_group_ids:
                    target_gid = text_norm
                else:
                    target_gid = f"split_{uuid.uuid4().hex[:8]}"

                new_group = EntityGroup(
                    original_text=text,
                    entity_type=ent_type,
                    group_id=target_gid,
                )
                new_group.occurrences.append(occ)
                new_group.enabled = True
                state.entity_groups.append(new_group)

                if target_gid != text_norm:
                    override = OccurrenceOverride(
                        target_group_id=target_gid,
                        context_fingerprint=ctx_fingerprint,
                        expected_original_text=text,
                        entity_type=ent_type,
                        role="",
                        enabled=True,
                    )
                    state.occurrence_overrides[occ_id] = override

                if sync_fn is not None:
                    sync_fn(state, new_group)

        # Invalidate remaining unselected postcheck findings
        if hasattr(state, "postcheck_findings"):
            state.postcheck_findings.clear()
        if hasattr(state, "postcheck_selected_ids"):
            state.postcheck_selected_ids.clear()

        # Update preview
        if preview_fn is not None:
            preview_fn(state)
        elif hasattr(state, "compute_reactive_preview"):
            state.compute_reactive_preview()

        # Unlock state
        state.is_postcheck_active = False
        state.postcheck_run_id = ""
        if hasattr(state, "set_all_mutating_elements_disabled"):
            state.set_all_mutating_elements_disabled(False)

        return True, f"{len(findings_to_apply)} Nachzügler erfolgreich übernommen."

    except Exception as exc:
        # Full atomic rollback of all touched state variables
        state.entity_groups = saved_groups
        state.occurrence_overrides = saved_overrides
        state.current_mapping = saved_mapping
        state.current_anon_text = saved_anon_text
        state.current_report = saved_report
        state.preview_revision = saved_preview_rev
        state.postcheck_findings = saved_findings
        state.postcheck_selected_ids = saved_selected_ids
        state.colliding_roles = saved_colliding_roles
        state.postcheck_status_msg = saved_status_msg
        state.postcheck_run_id = saved_run_id
        state.is_postcheck_active = saved_active
        return False, f"Fehler bei der Sammelübernahme (Rollback durchgeführt): {exc}"
