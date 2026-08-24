import pytest
from local_anonymizer.pipeline import AnonymizationPipeline
import json

def test_roundtrip_reversibility(tmp_path):
    pipeline = AnonymizationPipeline()
    in_file = tmp_path / "test.txt"
    in_file.write_text("Herr Prof. Dr. Max Mustermann von der ZHAW ruft an.", encoding="utf-8")
    
    out_dir = tmp_path / "out"
    pipeline.process_file(in_file, out_dir)
    
    anon_file = out_dir / "test_anonymized.txt"
    mapping_file = out_dir / "test_mapping.json"
    
    assert anon_file.exists()
    assert mapping_file.exists()
    
    restored_file = pipeline.restore_file(anon_file, mapping_file)
    restored_text = restored_file.read_text(encoding="utf-8")
    
    assert restored_text == "Herr Prof. Dr. Max Mustermann von der ZHAW ruft an."

def test_no_mapping_mode_audit(tmp_path):
    pipeline = AnonymizationPipeline()
    in_file = tmp_path / "test.txt"
    in_file.write_text("Max Mustermann", encoding="utf-8")
    
    out_dir = tmp_path / "out"
    pipeline.process_file(in_file, out_dir, no_mapping=True)
    
    anon_file = out_dir / "test_anonymized.txt"
    # Should be anonymized
    assert "[PERSON_1]" in anon_file.read_text(encoding="utf-8")
    
    # Mapping file should NOT exist
    mapping_file = out_dir / "test_mapping.json"
    assert not mapping_file.exists()
    
    # Report should not leak original terms
    report_file = out_dir / "test_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["entity_count"] > 0
    assert report["reversible"] == False
    for ent in report["entities"]:
        assert ent["original"] == "[REDACTED]"
