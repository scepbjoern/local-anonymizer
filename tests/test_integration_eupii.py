import pytest
from local_anonymizer.recognizers import EUPiiRecognizer


@pytest.mark.integration
def test_eupii_live_model_integration_synthetic():
    """
    Live integration test for EUPiiRecognizer using the locally cached model.
    Runs only when explicitly selected with '-m integration'.
    Uses purely synthetic non-private test sentences.
    """
    rec = EUPiiRecognizer(model_name="bardsai/eu-pii-anonimization-multilang", threshold=0.50)
    rec.load()
    assert rec.model is not None
    assert rec.tokenizer is not None

    text = "Frau Dr. Julia Meier arbeitet am Universitätsspital in Bern. Diagnose: Akute Bronchitis."
    results = rec.analyze(text)

    assert len(results) > 0
    entity_types = {r.entity_type for r in results}
    assert "PERSON" in entity_types or "LOCATION" in entity_types or "HEALTH_DATA" in entity_types


@pytest.mark.integration
def test_eupii_live_model_medical_invoice_id_numbers():
    """
    Live integration test verifying that multi-digit synthetic Swiss insurance and invoice numbers
    (e.g. '1389039' and '1752347575') are recognized as complete, intact ID_NUMBER spans
    by EUPiiRecognizer without fragmentation into subtoken fragments (e.g. '13', '175', '2347575', '890').
    Also tests exact frequencies, exact span positions, negative cases (non-merged IDs),
    and full end-to-end anonymization & de-anonymization.
    """
    from local_anonymizer.anonymizer import LocalAnonymizer

    text = """VERSICHERTEN-NR. 1389039 Winterthur, 14.02.2026 LEISTUNGSABRECHNUNG 1752347575
Leistungsabrechnung 1752347575 Max Muster Versicherten-Nr. 1389039
Referenz-ID 112233 und Beleg-ID 445566"""

    anon = LocalAnonymizer(enable_eupii=True, eupii_threshold=0.50)
    results = anon.analyze(text)

    id_results = [r for r in results if r.entity_type == "ID_NUMBER"]

    # 1. Verify exact count and recognizer source for all ID results
    assert len(id_results) == 6, f"Expected exactly 6 ID_NUMBER results, got {len(id_results)}"
    for r in id_results:
        assert r.recognition_metadata.get("recognizer_name") == "EUPiiRecognizer"

    # 2. Verify exact span positions and matched texts
    expected_spans = [
        (17, 24, "1389039"),
        (68, 78, "1752347575"),
        (99, 109, "1752347575"),
        (134, 145, "Nr. 1389039"),
        (158, 164, "112233"),
        (178, 184, "445566"),
    ]
    actual_spans = [(r.start, r.end, text[r.start:r.end]) for r in id_results]
    assert actual_spans == expected_spans, f"Span mismatch! Expected: {expected_spans}, Actual: {actual_spans}"

    detected_ids = [text[r.start:r.end] for r in id_results]

    # 3. Verify exact frequencies
    assert detected_ids.count("1389039") == 1
    assert detected_ids.count("Nr. 1389039") == 1
    assert detected_ids.count("1752347575") == 2
    assert detected_ids.count("112233") == 1
    assert detected_ids.count("445566") == 1

    # 4. Verify subtoken fragments are NEVER detected as separate standalone IDs
    for frag in ["13", "175", "2347575", "890", "89039"]:
        assert frag not in detected_ids

    # 5. Negative case: Separate adjacent IDs are NOT erroneously merged across words/whitespace
    assert "112233 und Beleg-ID 445566" not in detected_ids
    assert "112233 445566" not in detected_ids

    # 6. End-to-end anonymization & restore verification
    anon_res = anon.anonymize(text)

    # Verify every number occurrence was fully replaced with a placeholder
    for raw_id in ["1389039", "1752347575", "112233", "445566"]:
        assert raw_id not in anon_res.anonymized_text

    # Verify byte-exact de-anonymization
    restored = LocalAnonymizer.de_anonymize(anon_res.anonymized_text, anon_res.mapping)
    assert restored == text
