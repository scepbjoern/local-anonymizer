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
