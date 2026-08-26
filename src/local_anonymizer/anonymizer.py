"""Anonymization and de-anonymization engine using Presidio and local mappings."""

import json
import re
import warnings
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
    "Name", "Namen", "Vorname", "Vornamen", "Nachname", "Nachnamen", "Rolle", "Titel", "Status",
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


def clean_tag(text: str) -> str:
    """Clean a role or surface tag to be uppercase alphanumeric with underscores."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip().upper())
    return cleaned.strip("_")


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
    role: Optional[str] = None
    surface_tag: Optional[str] = None


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
                    "role": e.role,
                    "surface_tag": e.surface_tag,
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
        self.gliner_threshold = gliner_threshold

        # Combine user ignore terms with sensible default role/degree terms
        combined_ignore = set(DEFAULT_IGNORE_TERMS)
        if ignore_terms:
            combined_ignore.update(ignore_terms)
        self.ignore_terms = list(combined_ignore)

        # Explicitly preserve empty list [] vs None (None = all entities, [] = no entities)
        self.enabled_entities = list(enabled_entities) if enabled_entities is not None else None

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

        # Validate enabled_entities against supported recognizer entities
        if self.enabled_entities is not None and len(self.enabled_entities) > 0:
            supported = set(self.analyzer.get_supported_entities(language=self.language))
            supported.update(self.gliner_recognizer.supported_entities)
            supported.update(self.fuzzy_recognizer.supported_entities)
            unknown = [e for e in self.enabled_entities if e not in supported]
            if unknown:
                warnings.warn(
                    f"Unknown entity type(s) configured in enabled_entities: {unknown}. "
                    f"Supported entities are: {sorted(supported)}",
                    UserWarning,
                    stacklevel=2,
                )

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

        # If enabled_entities is explicitly empty [], return no entities
        if self.enabled_entities is not None and len(self.enabled_entities) == 0:
            return []

        # Find spans of ignore terms to exclude false positives (e.g. "CAS", "Studierende", "Unternehmen")
        ignored_spans: List[Tuple[int, int]] = []
        for term in self.ignore_terms:
            if not term.strip():
                continue
            pattern = re.compile(r"\b" + re.escape(term.strip()) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                ignored_spans.append((match.start(), match.end()))

        # Run Presidio analysis without hardcoded score_threshold overriding recognizer thresholds
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.enabled_entities,
            score_threshold=None,
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

    def anonymize(
        self,
        text: str,
        format_mode: str = "numbered",
        roles: Optional[Dict[str, str]] = None,
        entity_links: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> AnonymizationResult:
        """
        Anonymize text, producing a placeholder-substituted text and a local mapping.

        Format Modes:
          - "numbered" (Modus 1): [TYPE_N] or [TYPE_N_TAG]
          - "numbered_role" (Modus 2): [TYPE_N_ROLE] or [TYPE_N_ROLE_TAG]
          - "role_only" (Modus 3): [TYPE_ROLE] (falls back to Modus 2 if multiple entities share (type, role))

        Parameters:
          - roles: dict of normalized_term -> role (e.g. {"julia": "STUDENT"})
          - entity_links: dict of child_term -> (parent_term, surface_tag)
            (e.g. {"julia": ("julia meier", "VORNAME"), "julia meier": ("", "VOLLNAME")})
        """
        if not text:
            return AnonymizationResult(anonymized_text="", mapping={})

        roles = {k.strip().lower(): v.strip() for k, v in (roles or {}).items() if v and v.strip()}
        entity_links = {
            k.strip().lower(): (v[0].strip().lower(), v[1].strip())
            for k, v in (entity_links or {}).items()
            if v and len(v) == 2
        }

        recognizer_results = self.analyze(text)

        # 1. Identify all unique entity terms and their types in order of first appearance
        # Key: (normalized_term, entity_type)
        unique_entities: List[Tuple[str, str]] = []
        for res in recognizer_results:
            orig = text[res.start:res.end].strip()
            key = (orig.lower(), res.entity_type)
            if key not in unique_entities:
                unique_entities.append(key)

        # 2. Assign entity IDs and roles (master vs linked)
        category_counters: Dict[str, int] = {}
        # entity_info: (norm_term, entity_type) -> {"id": int, "role": str, "surface_tag": str}
        entity_info: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # First pass: Process master entities (entities not linked to a different parent)
        for norm_term, etype in unique_entities:
            parent_info = entity_links.get(norm_term)
            is_child = parent_info and parent_info[0] and parent_info[0] != norm_term

            if not is_child:
                count = category_counters.get(etype, 0) + 1
                category_counters[etype] = count
                role = roles.get(norm_term, "")
                tag = parent_info[1] if parent_info else ""
                entity_info[(norm_term, etype)] = {
                    "id": count,
                    "role": role,
                    "surface_tag": tag,
                }

        # Second pass: Process linked child entities (inheriting parent's ID and role)
        for norm_term, etype in unique_entities:
            parent_info = entity_links.get(norm_term)
            is_child = parent_info and parent_info[0] and parent_info[0] != norm_term

            if is_child:
                parent_term, tag = parent_info
                parent_key = (parent_term, etype)
                if parent_key in entity_info:
                    p_id = entity_info[parent_key]["id"]
                    p_role = entity_info[parent_key]["role"] or roles.get(norm_term, "")
                else:
                    # Fallback if parent wasn't found as master
                    count = category_counters.get(etype, 0) + 1
                    category_counters[etype] = count
                    p_id = count
                    p_role = roles.get(norm_term, "")

                entity_info[(norm_term, etype)] = {
                    "id": p_id,
                    "role": p_role,
                    "surface_tag": tag or "VARIANT",
                }

        # 3. Collision check for Modus 3 ("role_only")
        # Count distinct master entities sharing the exact same (entity_type, role) pair
        role_type_groups: Dict[Tuple[str, str], Set[int]] = {}
        for (norm_term, etype), info in entity_info.items():
            if info["role"]:
                c_role = clean_tag(info["role"])
                if c_role:
                    pair = (etype, c_role)
                    role_type_groups.setdefault(pair, set()).add(info["id"])

        colliding_pairs: Set[Tuple[str, str]] = {
            pair for pair, ids in role_type_groups.items() if len(ids) > 1
        }

        if format_mode == "role_only" and colliding_pairs:
            for etype, c_role in colliding_pairs:
                warnings.warn(
                    f"Rollenkollision bei Entitätstyp '{etype}' mit Rolle '{c_role}': "
                    f"Mehrere Entitäten teilen dieselbe Rolle. Automatischer Fallback auf Modus 2 (nummeriert) für diese Rolle.",
                    UserWarning,
                    stacklevel=2,
                )

        # 4. Generate placeholders and mapping
        value_to_placeholder: Dict[Tuple[str, str], str] = {}
        mapping: Dict[str, str] = {}
        detected_entities: List[DetectedEntity] = []
        review_needed: List[DetectedEntity] = []

        for res in recognizer_results:
            orig_val = text[res.start:res.end]
            norm_term = orig_val.strip().lower()
            key = (norm_term, res.entity_type)

            info = entity_info.get(key, {"id": 1, "role": "", "surface_tag": ""})
            ent_id = info["id"]
            role_str = clean_tag(info["role"])
            tag_str = clean_tag(info["surface_tag"])

            if key not in value_to_placeholder:
                # Build placeholder according to format_mode & collision rules
                suffix_tag = f"_{tag_str}" if tag_str else ""
                pair = (res.entity_type, role_str)
                is_colliding = pair in colliding_pairs

                if format_mode == "role_only" and role_str and not is_colliding:
                    placeholder = f"[{res.entity_type}_{role_str}{suffix_tag}]"
                elif (format_mode in ("numbered_role", "role_only") or is_colliding) and role_str:
                    placeholder = f"[{res.entity_type}_{ent_id}_{role_str}{suffix_tag}]"
                else:
                    # Modus 1 (Numbered) or no role given
                    placeholder = f"[{res.entity_type}_{ent_id}{suffix_tag}]"

                value_to_placeholder[key] = placeholder
                mapping[placeholder] = orig_val
            else:
                placeholder = value_to_placeholder[key]

            needs_review = 0.70 <= res.score < 0.85
            entity = DetectedEntity(
                entity_type=res.entity_type,
                original_text=orig_val,
                placeholder=placeholder,
                start=res.start,
                end=res.end,
                score=res.score,
                needs_review=needs_review,
                role=info["role"] or None,
                surface_tag=info["surface_tag"] or None,
            )
            detected_entities.append(entity)
            if needs_review:
                review_needed.append(entity)

        # 5. Single-pass text substitution in reverse character order
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
        Uses a single-pass regex replacement to eliminate cascading substitutions.
        """
        if not anonymized_text or not mapping:
            return anonymized_text

        # Sort placeholder keys by length descending to match longest first
        sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in sorted_placeholders))

        return pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), anonymized_text)

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
