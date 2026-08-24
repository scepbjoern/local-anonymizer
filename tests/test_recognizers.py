import re
import pytest
from local_anonymizer.recognizers import is_sentence_boundary, GLiNERRecognizer, FuzzyGlossaryRecognizer

def test_abbreviation_aware_chunking():
    text = "Prof. Dr. Max wohnt in der Bahnhofstr. 12 am 14. Juli. Das ist super."
    # Find dots
    matches = list(re.finditer(r"\.", text))
    # Dot after Prof: should be False
    assert is_sentence_boundary(text, matches[0]) == False
    # Dot after Dr: should be False
    assert is_sentence_boundary(text, matches[1]) == False
    # Dot after Bahnhofstr: should be False
    assert is_sentence_boundary(text, matches[2]) == False
    # Dot after 14 (14. Juli): should be False
    assert is_sentence_boundary(text, matches[3]) == False
    # Dot after super: should be True
    assert is_sentence_boundary(text, matches[4]) == True

def test_fuzzy_glossary_typo_matching():
    recognizer = FuzzyGlossaryRecognizer(glossary={"ZHAW": "ORGANIZATION"}, high_confidence_threshold=80.0, review_threshold=70.0)
    results = recognizer.analyze("Die Studenten der ZHW sind hier.", entities=["ORGANIZATION"])
    # Should find ZHW as ORGANIZATION
    assert any(r.entity_type == "ORGANIZATION" and r.start == 18 and r.end == 21 for r in results)

def test_fuzzy_glossary_exact_match_priority():
    recognizer = FuzzyGlossaryRecognizer(glossary={"ZHAW": "ORGANIZATION"}, high_confidence_threshold=80.0, review_threshold=70.0)
    results = recognizer.analyze("Die ZHAW und die ZHW", entities=["ORGANIZATION"])
    # Both should be found, but ZHAW should have score 1.0
    zhaw_match = next(r for r in results if r.start == 4)
    assert zhaw_match.score == 1.0
