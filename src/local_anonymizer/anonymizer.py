"""Anonymization and de-anonymization engine using Presidio and local mappings."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine

from local_anonymizer.recognizers import FuzzyGlossaryRecognizer, GLiNERRecognizer

# Default terms that should never be scrubbed (common German academic, business, and role nouns)
DEFAULT_IGNORE_TERMS = [
    # Academic & educational degrees / programs
    "CAS", "DAS", "MAS", "BSc", "MSc", "PhD", "MBA", "EMBA", "Bachelor", "Master",
    # Generic roles (often falsely flagged as named persons by generic NER)
    "Studierende", "Studierenden", "Studierender", "Student", "Studenten", "Studentin", "Studentinnen",
    "Lehrperson", "Lehrpersonen", "Dozent", "Dozenten", "Dozentin", "Dozentinnen", "Dozierende", "Dozierenden",
    "Mitarbeiter", "Mitarbeitende", "Mitarbeitenden", "Mitarbeiterin", "Mitarbeiterinnen",
    "Kunde", "Kunden", "Kundin", "Kundinnen",
    "Patient", "Patienten", "Patientin", "Patientinnen",
    "Projektleiter", "Projektleiterin", "Consultant", "Berater", "Beraterin",
    "Aufgabensteller", "Aufgabenstellerin", "Experte", "Expertin", "Betreuer", "Betreuerin",
    # Generic entity descriptors and field labels
    "Unternehmen", "Unternehmens", "Firma", "Organisation", "Hochschule", "Universität",
    "Prüfung", "Prüfungen", "Vorlesung", "Modul", "Lehrgang", "Weiterbildung",
    "Telefon", "Tel", "Email", "E-Mail", "Mail", "Adresse", "Website", "Datum",
]


class BlankSpacyNlpEngine(SpacyNlpEngine):
    """Lightweight NLP engine using blank spaCy tokenizers (no heavy models needed)."""

    def __init__(self, languages: Sequence[str] = ("de", "en")):
        self.ner_model_configuration = NerModelConfiguration()
        self.nlp = {lang: spacy.blank(lang) for lang in languages}

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def get_supported_languages(self) -> List[str]:
        return list(self.nlp.keys())


@dataclass
class DetectedEntity:
    """Represents an entity detected in the text."""
    entity_type: str
    original_text: str
    placeholder: str
    start: int
    end: int
    score: float
    needs_review: bool = False


@dataclass
class AnonymizationResult:
    """Result of an anonymization process."""
    anonymized_text: str
    mapping: Dict[str, str]  # placeholder -> original_text
    entities: List[DetectedEntity] = field(default_factory=list)
    review_needed: List[DetectedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for export."""
        return {
            "mapping": self.mapping,
            "entity_count": len(self.entities),
            "review_count": len(self.review_needed),
            "entities": [
                {
                    "type": e.entity_type,
                    "original": e.original_text,
                    "placeholder": e.placeholder,
                    "score": round(e.score, 3),
                    "needs_review": e.needs_review,
                }
                for e in self.entities
            ],
        }


class LocalAnonymizer:
    """Core local anonymizer combining Presidio, GLiNER, Fuzzy Glossary, and Ignore lists."""

    def __init__(
        self,
        language: str = "de",
        glossary: Optional[Dict[str, str]] = None,
        ignore_terms: Optional[Sequence[str]] = None,
        custom_labels: Optional[Dict[str, str]] = None,
        gliner_model: str = "urchade/gliner_multi_pii-v1",
        gliner_threshold: float = 0.55,
        fuzzy_high_threshold: float = 90.0,
        fuzzy_review_threshold: float = 75.0,
        enabled_entities: Optional[Sequence[str]] = None,
    ):
        self.language = language
        self.glossary = glossary or {}
        
        # Combine user ignore terms with sensible default role/degree terms
        combined_ignore = set(DEFAULT_IGNORE_TERMS)
        if ignore_terms:
            combined_ignore.update(ignore_terms)
        self.ignore_terms = list(combined_ignore)

        self.enabled_entities = list(enabled_entities) if enabled_entities else None

        # Setup Presidio AnalyzerEngine with lightweight blank NLP engine
        nlp_engine = BlankSpacyNlpEngine(languages=["de", "en"])
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["de", "en"],
        )

        # Add custom recognizers
        self.gliner_recognizer = GLiNERRecognizer(
            model_name=gliner_model,
            custom_labels=custom_labels,
            threshold=gliner_threshold,
            supported_language=language,
        )
        self.fuzzy_recognizer = FuzzyGlossaryRecognizer(
            glossary=self.glossary,
            high_confidence_threshold=fuzzy_high_threshold,
            review_threshold=fuzzy_review_threshold,
            supported_language=language,
        )

        self.analyzer.registry.add_recognizer(self.gliner_recognizer)
        self.analyzer.registry.add_recognizer(self.fuzzy_recognizer)

    def add_glossary_term(self, term: str, entity_type: str = "PERSON") -> None:
        """Add a term to the fuzzy glossary dynamically."""
        self.fuzzy_recognizer.add_term(term, entity_type)

    def add_ignore_term(self, term: str) -> None:
        """Add a term to the ignore list."""
        if term not in self.ignore_terms:
            self.ignore_terms.append(term)

    def analyze(self, text: str) -> List[RecognizerResult]:
        """Analyze text using all configured recognizers with ignore filter and entity filters."""
        if not text:
            return []

        # Find spans of ignore terms to exclude false positives (e.g. "CAS", "Studierende", "Unternehmen")
        ignored_spans: List[Tuple[int, int]] = []
        for term in self.ignore_terms:
            if not term.strip():
                continue
            pattern = re.compile(r"\b" + re.escape(term.strip()) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                ignored_spans.append((match.start(), match.end()))

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.enabled_entities,
            score_threshold=0.50,
        )

        # 1. Filter out ignored spans
        filtered_results: List[RecognizerResult] = []
        for r in results:
            is_ignored = any(
                not (r.end <= ig_start or r.start >= ig_end)
                for ig_start, ig_end in ignored_spans
            )
            if not is_ignored:
                if self.enabled_entities is None or r.entity_type in self.enabled_entities:
                    filtered_results.append(r)

        # 2. Deduplicate / resolve overlapping results (prefer higher score, then longer span)
        filtered_results.sort(key=lambda r: (r.score, r.end - r.start), reverse=True)
        accepted: List[RecognizerResult] = []
        accepted_spans: List[Tuple[int, int]] = []

        for r in filtered_results:
            overlaps = any(
                not (r.end <= start or r.start >= end)
                for start, end in accepted_spans
            )
            if not overlaps:
                accepted.append(r)
                accepted_spans.append((r.start, r.end))

        # Sort by start position in text
        accepted.sort(key=lambda r: r.start)
        return accepted

    def anonymize(self, text: str) -> AnonymizationResult:
        """
        Anonymize text, producing a placeholder-substituted text and a local mapping.
        Placeholders follow the format [CATEGORY_N] (e.g. [PERSON_1], [ORGANIZATION_1]).
        Identical original texts for the same entity type receive the identical placeholder.
        """
        if not text:
            return AnonymizationResult(anonymized_text="", mapping={})

        recognizer_results = self.analyze(text)

        # Track category counters and entity-to-placeholder mappings
        category_counters: Dict[str, int] = {}
        # Mapping from (normalized_original_text, entity_type) -> placeholder
        value_to_placeholder: Dict[Tuple[str, str], str] = {}
        # Final mapping: placeholder -> original_text
        mapping: Dict[str, str] = {}

        detected_entities: List[DetectedEntity] = []
        review_needed: List[DetectedEntity] = []

        for res in recognizer_results:
            orig_val = text[res.start:res.end]
            norm_key = (orig_val.strip().lower(), res.entity_type)

            if norm_key not in value_to_placeholder:
                count = category_counters.get(res.entity_type, 0) + 1
                category_counters[res.entity_type] = count
                placeholder = f"[{res.entity_type}_{count}]"
                value_to_placeholder[norm_key] = placeholder
                mapping[placeholder] = orig_val
            else:
                placeholder = value_to_placeholder[norm_key]

            # Review flag: confidence between 0.70 and 0.85
            needs_review = 0.70 <= res.score < 0.85
            entity = DetectedEntity(
                entity_type=res.entity_type,
                original_text=orig_val,
                placeholder=placeholder,
                start=res.start,
                end=res.end,
                score=res.score,
                needs_review=needs_review,
            )
            detected_entities.append(entity)
            if needs_review:
                review_needed.append(entity)

        # Replace spans in reverse order (from end of text to start) to maintain indices
        anonymized_chars = list(text)
        for entity in reversed(detected_entities):
            anonymized_chars[entity.start:entity.end] = list(entity.placeholder)

        anonymized_text = "".join(anonymized_chars)

        return AnonymizationResult(
            anonymized_text=anonymized_text,
            mapping=mapping,
            entities=detected_entities,
            review_needed=review_needed,
        )

    @staticmethod
    def de_anonymize(anonymized_text: str, mapping: Dict[str, str]) -> str:
        """
        Restore original values in anonymized text using the mapping table.
        Sorts placeholders by length descending to prevent partial prefix collisions.
        """
        if not anonymized_text or not mapping:
            return anonymized_text

        result_text = anonymized_text
        # Sort placeholders by length descending
        sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)

        for placeholder in sorted_placeholders:
            original_value = mapping[placeholder]
            result_text = result_text.replace(placeholder, original_value)

        return result_text

    @staticmethod
    def save_mapping(mapping: Dict[str, str], file_path: Union[str, Path]) -> None:
        """Save placeholder mapping to a local JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
        """Load placeholder mapping from a JSON file."""
        path = Path(file_path)
        return json.loads(path.read_text(encoding="utf-8"))
