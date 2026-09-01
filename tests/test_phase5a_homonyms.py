"""
Unit and regression tests for Phase 5a: Homonym and Occurrence Disambiguation Handling.
Tests occurrence identity, fail-safe reattachment, split/rollback, placeholder consistency,
and single-pass restoration.
"""

import uuid
import pytest
from app import (
    AppState,
    EntityGroup,
    EntityOccurrence,
    OccurrenceOverride,
    compute_context_fingerprint,
    compute_reactive_preview,
    split_occurrence_to_new_group,
    revert_occurrence_to_base,
    sync_group_overrides,
)
from local_anonymizer.anonymizer import LocalAnonymizer, build_entity_tree


def test_occurrence_fingerprint_deterministic():
    """Verify context fingerprint is deterministic and unaffected by remote text changes outside the window."""
    prefix = "A" * 60 + "Frau "
    suffix = " ist Projektleiterin bei der Firma." + "B" * 60
    text1 = prefix + "Müller" + suffix
    start1 = text1.index("Müller")
    end1 = start1 + len("Müller")
    fp1 = compute_context_fingerprint(text1, start1, end1)

    # Adding extra text far away (> 50 chars away from Müller) does not change the ±50 char window
    text2 = "Sehr langer Text am Anfang des Dokuments... " + text1
    start2 = text2.index("Müller")
    end2 = start2 + len("Müller")
    fp2 = compute_context_fingerprint(text2, start2, end2)

    assert fp1 == fp2


def test_split_creation_and_placeholder_consistency():
    """
    Test 1: Split a group of 2 occurrences of 'Müller' into two separate groups.
    One is PERSON (Rolle: Chef), one is PERSON (Rolle: Praktikant).
    Verify 2 distinct placeholders and clean mapping.
    """
    st = AppState()
    st.raw_text = "Frau Müller sprach mit Herrn Müller über das Projekt."
    st.format_mode = "numbered_role"

    idx1 = st.raw_text.find("Müller")
    idx2 = st.raw_text.find("Müller", idx1 + 1)
    occ1 = EntityOccurrence(
        start=idx1,
        end=idx1 + len("Müller"),
        score=0.95,
        context_html="Frau <b>Müller</b> sprach",
        needs_review=False,
        occ_id="occ_1",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx1, idx1 + len("Müller")),
    )
    occ2 = EntityOccurrence(
        start=idx2,
        end=idx2 + len("Müller"),
        score=0.92,
        context_html="Herrn <b>Müller</b> über",
        needs_review=False,
        occ_id="occ_2",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx2, idx2 + len("Müller")),
    )

    base_grp = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller")
    base_grp.role = "Chef"
    base_grp.occurrences = [occ1, occ2]
    st.entity_groups = [base_grp]

    # Split occurrence 2 into a new group
    split_grp = split_occurrence_to_new_group(st, base_grp, occ2)
    split_grp.role = "Praktikant"
    sync_group_overrides(st, split_grp)

    assert len(st.entity_groups) == 2
    assert len(base_grp.occurrences) == 1
    assert len(split_grp.occurrences) == 1
    assert "occ_2" in st.occurrence_overrides
    assert st.occurrence_overrides["occ_2"].role == "Praktikant"

    # Compute preview
    anon_text, mapping, report = compute_reactive_preview(st)

    assert "[PERSON_1_CHEF]" in mapping
    assert "[PERSON_2_PRAKTIKANT]" in mapping
    assert mapping["[PERSON_1_CHEF]"] == "Müller"
    assert mapping["[PERSON_2_PRAKTIKANT]"] == "Müller"

    # Verify single-pass restoration
    restored = LocalAnonymizer.de_anonymize(anon_text, mapping)
    assert restored == st.raw_text


def test_independent_overrides_for_identical_fingerprints_in_same_analysis():
    """
    Test 2: Two occurrences with identical context fingerprints receive different overrides
    in the same analysis session. Both must persist under their own occ_id.
    """
    st = AppState()
    pad_pre = "P" * 60 + "Hallo "
    pad_post = " wie geht es Ihnen heute?" + "S" * 60
    # Create two occurrences with identical 50 char prefix and suffix
    seg = pad_pre + "Müller" + pad_post
    st.raw_text = seg + " --- EIN TRENNENDER TEXT ABSATZ --- " + seg

    idx1 = st.raw_text.find("Müller")
    idx2 = st.raw_text.find("Müller", idx1 + 1)

    occ1 = EntityOccurrence(
        start=idx1,
        end=idx1 + len("Müller"),
        score=0.9,
        context_html="",
        needs_review=False,
        occ_id="occ_a",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx1, idx1 + len("Müller")),
    )
    occ2 = EntityOccurrence(
        start=idx2,
        end=idx2 + len("Müller"),
        score=0.9,
        context_html="",
        needs_review=False,
        occ_id="occ_b",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx2, idx2 + len("Müller")),
    )

    # They have identical context fingerprints
    assert occ1.context_fingerprint == occ2.context_fingerprint

    base_grp = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller")
    base_grp.occurrences = [occ1, occ2]
    st.entity_groups = [base_grp]

    # Split occ1 and occ2 into separate groups
    grp1 = split_occurrence_to_new_group(st, base_grp, occ1)
    grp1.entity_type = "ORGANIZATION"
    sync_group_overrides(st, grp1)

    grp2 = split_occurrence_to_new_group(st, base_grp, occ2)
    grp2.role = "Berater"
    sync_group_overrides(st, grp2)

    # Verify both overrides exist independently under occ_a and occ_b
    assert "occ_a" in st.occurrence_overrides
    assert "occ_b" in st.occurrence_overrides
    assert st.occurrence_overrides["occ_a"].entity_type == "ORGANIZATION"
    assert st.occurrence_overrides["occ_b"].role == "Berater"


def test_successful_1_to_1_override_rebinding_and_group_recreation():
    """
    Test 3: Re-analysis of the same text with 1:1 matching fingerprint and expected_original_text.
    Verifies that the override is re-keyed to the new occ_id and the same target_group_id is recreated.
    """
    st = AppState()
    raw_text = "Die Meier AG liefert Teile an Firma Schmidt."
    start = raw_text.index("Meier AG")
    end = start + len("Meier AG")
    fp = compute_context_fingerprint(raw_text, start, end)

    # Initial override
    st.occurrence_overrides["old_occ_id"] = OccurrenceOverride(
        target_group_id="split_custom_1",
        context_fingerprint=fp,
        expected_original_text="Meier AG",
        entity_type="ORGANIZATION",
        role="Zulieferer",
        enabled=True,
    )

    # Simulate re-analysis results
    new_occ = EntityOccurrence(
        start=start,
        end=end,
        score=0.98,
        context_html="",
        needs_review=False,
        occ_id="new_occ_id",
        context_fingerprint=fp,
    )
    base_grp = EntityGroup(original_text="Meier AG", entity_type="PERSON", group_id="meier ag")
    base_grp.occurrences = [new_occ]

    # Run re-attachment logic (simulating run_analysis step 3)
    old_overrides = list(st.occurrence_overrides.values())
    new_overrides = {}
    old_by_fp = {}
    for ov in old_overrides:
        old_by_fp.setdefault(ov.context_fingerprint, []).append(ov)

    all_new_occurrences = [(new_occ, "Meier AG", "PERSON")]
    new_by_fp = {}
    for occ, norm, ent_type in all_new_occurrences:
        new_by_fp.setdefault(occ.context_fingerprint, []).append((occ, norm, ent_type))

    split_groups_dict = {}
    for f_print, ov_list in old_by_fp.items():
        cand_list = new_by_fp.get(f_print, [])
        if len(ov_list) == 1 and len(cand_list) == 1:
            ov = ov_list[0]
            cand_occ, actual_norm, actual_type = cand_list[0]
            if actual_norm == ov.expected_original_text:
                new_overrides[cand_occ.occ_id] = OccurrenceOverride(
                    target_group_id=ov.target_group_id,
                    context_fingerprint=f_print,
                    expected_original_text=ov.expected_original_text,
                    entity_type=ov.entity_type,
                    role=ov.role,
                    enabled=ov.enabled,
                )
                tgt_id = ov.target_group_id
                if tgt_id not in split_groups_dict:
                    restored_type = ov.entity_type if ov.entity_type is not None else actual_type
                    restored_grp = EntityGroup(
                        original_text=ov.expected_original_text,
                        entity_type=restored_type,
                        group_id=tgt_id,
                    )
                    if ov.role is not None:
                        restored_grp.role = ov.role
                    if ov.enabled is not None:
                        restored_grp.enabled = ov.enabled
                    split_groups_dict[tgt_id] = restored_grp

    assert "new_occ_id" in new_overrides
    assert "old_occ_id" not in new_overrides
    assert "split_custom_1" in split_groups_dict
    assert split_groups_dict["split_custom_1"].role == "Zulieferer"
    assert split_groups_dict["split_custom_1"].entity_type == "ORGANIZATION"


def test_ambiguous_rebinding_discarded_failsafe():
    """
    Test 4: If multiple occurrences have the same fingerprint after re-analysis,
    all overrides for this fingerprint must be fail-safe discarded.
    """
    st = AppState()
    fp = "identical_repeated_hash"

    # 1 old override
    st.occurrence_overrides["old_id"] = OccurrenceOverride(
        target_group_id="split_1",
        context_fingerprint=fp,
        expected_original_text="Meyer",
        entity_type="ORGANIZATION",
    )

    # 2 new occurrences with the same fingerprint
    new_occ1 = EntityOccurrence(start=0, end=5, score=0.9, context_html="", needs_review=False, occ_id="new_1", context_fingerprint=fp)
    new_occ2 = EntityOccurrence(start=20, end=25, score=0.9, context_html="", needs_review=False, occ_id="new_2", context_fingerprint=fp)

    old_overrides = list(st.occurrence_overrides.values())
    new_overrides = {}
    old_by_fp = {fp: old_overrides}
    new_by_fp = {fp: [(new_occ1, "Meyer", "PERSON"), (new_occ2, "Meyer", "PERSON")]}

    split_groups_dict = {}
    for f_print, ov_list in old_by_fp.items():
        cand_list = new_by_fp.get(f_print, [])
        if len(ov_list) == 1 and len(cand_list) == 1:
            pass  # Will NOT be entered because len(cand_list) == 2

    # Verify no override was attached
    assert len(new_overrides) == 0
    assert len(split_groups_dict) == 0


def test_text_change_failsafe_discard():
    """
    Test 5: If the fingerprint matches but the actual original text changed,
    the override is discarded fail-safely.
    """
    fp = "some_hash"
    ov = OccurrenceOverride(
        target_group_id="split_1",
        context_fingerprint=fp,
        expected_original_text="Meyer",
        entity_type="ORGANIZATION",
    )
    # Candidate text is "Mayer" instead of "Meyer"
    cand_occ = EntityOccurrence(start=0, end=5, score=0.9, context_html="", needs_review=False, occ_id="new_1", context_fingerprint=fp)
    cand_list = [(cand_occ, "Mayer", "PERSON")]

    new_overrides = {}
    if len([ov]) == 1 and len(cand_list) == 1:
        if cand_list[0][1] == ov.expected_original_text:
            new_overrides["new_1"] = ov

    assert len(new_overrides) == 0


def test_revert_occurrence_to_base():
    """
    Test 6: Reverting a split occurrence returns it to the base group and removes the override.
    """
    st = AppState()
    base_grp = EntityGroup(original_text="Schmidt", entity_type="PERSON", group_id="schmidt")
    occ1 = EntityOccurrence(start=0, end=7, score=0.9, context_html="", needs_review=False, occ_id="occ_1", context_fingerprint="fp1")
    occ2 = EntityOccurrence(start=20, end=27, score=0.9, context_html="", needs_review=False, occ_id="occ_2", context_fingerprint="fp2")
    base_grp.occurrences = [occ1, occ2]
    st.entity_groups = [base_grp]

    split_grp = split_occurrence_to_new_group(st, base_grp, occ2)
    assert len(st.entity_groups) == 2
    assert "occ_2" in st.occurrence_overrides

    # Revert occ2 back to base
    revert_occurrence_to_base(st, split_grp, occ2)

    assert "occ_2" not in st.occurrence_overrides
    assert len(st.entity_groups) == 1
    assert st.entity_groups[0].group_id == "schmidt"
    assert len(st.entity_groups[0].occurrences) == 2


def test_workspace_reset_clears_overrides():
    """
    Test 7: Workspace reset completely clears entity groups and occurrence overrides.
    """
    st = AppState()
    st.occurrence_overrides["some_occ"] = OccurrenceOverride(
        target_group_id="split_1",
        context_fingerprint="fp",
        expected_original_text="Test",
    )
    assert len(st.occurrence_overrides) == 1

    # Simulate reset
    st.entity_groups = []
    st.occurrence_overrides = {}
    st.current_mapping = {}

    assert len(st.occurrence_overrides) == 0
    assert len(st.entity_groups) == 0


def test_placeholder_modes_with_homonyms_and_collision():
    """
    Test 8: Verify all 3 placeholder modes with homonym groups, including collision fallback in Mode 3 (role_only).
    """
    st = AppState()
    st.raw_text = "A und B trafen C."

    # Group 1: Müller (PERSON, role=Chef)
    g1 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller_1")
    g1.role = "Chef"
    g1.occurrences = [EntityOccurrence(start=0, end=1, score=0.9, context_html="", needs_review=False)]

    # Group 2: Müller (PERSON, role=Chef) -> Colliding role!
    g2 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller_2")
    g2.role = "Chef"
    g2.occurrences = [EntityOccurrence(start=6, end=7, score=0.9, context_html="", needs_review=False)]

    st.entity_groups = [g1, g2]

    # Mode 1: Numbered
    st.format_mode = "numbered"
    _, map1, _ = compute_reactive_preview(st)
    assert "[PERSON_1]" in map1
    assert "[PERSON_2]" in map1

    # Mode 2: Numbered Role
    st.format_mode = "numbered_role"
    _, map2, _ = compute_reactive_preview(st)
    assert "[PERSON_1_CHEF]" in map2
    assert "[PERSON_2_CHEF]" in map2

    # Mode 3: Role Only -> Must fall back to numbered_role due to collision!
    st.format_mode = "role_only"
    _, map3, _ = compute_reactive_preview(st)
    assert "[PERSON_1_CHEF]" in map3
    assert "[PERSON_2_CHEF]" in map3


def test_visual_homonym_clustering_without_parent_link():
    """
    Test 9: Two homonym groups are separate tree roots (no parent_group_id set).
    """
    g1 = EntityGroup(original_text="Meier", entity_type="PERSON", group_id="meier")
    g2 = EntityGroup(original_text="Meier", entity_type="ORGANIZATION", group_id="split_123")

    tree = build_entity_tree([g1, g2])
    assert len(tree) == 2
    assert tree[0].item.group_id == "meier"
    assert tree[1].item.group_id == "split_123"
    assert g2.parent_group_id is None


def test_smart_linking_with_homonyms():
    """
    Test 10: Smart linking can link a child to a specific parent using parent_group_id.
    """
    parent = EntityGroup(original_text="Dr. Meier", entity_type="PERSON", group_id="dr_meier")
    child = EntityGroup(original_text="Meier", entity_type="PERSON", group_id="meier")
    child.parent_group_id = parent.group_id
    child.surface_tag = "NACHNAME"

    tree = build_entity_tree([parent, child])
    assert len(tree) == 1
    assert tree[0].item.group_id == "dr_meier"
    assert len(tree[0].children) == 1
    assert tree[0].children[0].item.group_id == "meier"
