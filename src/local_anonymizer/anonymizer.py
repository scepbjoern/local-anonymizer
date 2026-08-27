"""Anonymization and de-anonymization engine using Presidio and local mappings."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

def _create_blank_spacy_engine(languages: Sequence[str] = ("de", "en")):
    """Lazy factory for lightweight NLP engine using blank spaCy tokenizers."""
    import spacy
    from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine

    class BlankSpacyNlpEngine(SpacyNlpEngine):
        def __init__(self, langs: Sequence[str] = ("de", "en")):
            self.ner_model_configuration = NerModelConfiguration()
            self.nlp = {lang: spacy.blank(lang) for lang in langs}

        def is_loaded(self) -> bool:
            return True

        def load(self) -> None:
            pass

        def get_supported_languages(self) -> List[str]:
            return list(self.nlp.keys())

    return BlankSpacyNlpEngine(languages)


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


def clean_tag(text: str) -> str:
    """Clean a role or surface tag to be uppercase alphanumeric with underscores."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip().upper())
    return cleaned.strip("_")


def trim_entity_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Trim leading/trailing HTML tags, whitespace, markdown syntax, and punctuation from entity bounds."""
    s, e = start, end
    tag_pattern = r"</?br\s*/?>|</?\w+\s*/?>|</?br|<|/?>|\s+|[*\-_`~,:;.!?()\[\]{}|]"
    while s < e:
        m = re.match(r"^(?:" + tag_pattern + r")+", text[s:e], flags=re.IGNORECASE)
        if m:
            s += m.end()
        else:
            break
    while e > s:
        m = re.search(r"(?:" + tag_pattern + r")+$", text[s:e], flags=re.IGNORECASE)
        if m:
            e -= (len(text[s:e]) - m.start())
        else:
            break
    return s, e


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
class EntityTreeNode:
    """Represents a node in a hierarchical entity tree (master and linked surface variants)."""
    item: Any
    children: List["EntityTreeNode"] = field(default_factory=list)


def build_entity_tree(
    items: List[Any],
    get_key: Optional[Callable[[Any], str]] = None,
    get_parent_key: Optional[Callable[[Any], Optional[str]]] = None,
) -> List[EntityTreeNode]:
    """
    Build a hierarchical tree of entities (roots/masters with linked children).
    
    Defaults to reading .key and .parent_group_text if callables are omitted.
    """
    if get_key is None:
        get_key = lambda x: getattr(x, "key", str(x).lower())
    if get_parent_key is None:
        get_parent_key = lambda x: (
            getattr(x, "parent_group_text", "").strip().lower()
            if getattr(x, "parent_group_text", None)
            else None
        )

    item_keys = {get_key(it): it for it in items}
    roots: List[EntityTreeNode] = []
    children_map: Dict[str, List[Any]] = {}

    for it in items:
        p_key = get_parent_key(it)
        it_key = get_key(it)
        if p_key and p_key != it_key and p_key in item_keys:
            children_map.setdefault(p_key, []).append(it)
        else:
            roots.append(EntityTreeNode(item=it))

    for root in roots:
        r_key = get_key(root.item)
        if r_key in children_map:
            root.children = [EntityTreeNode(item=c) for c in children_map[r_key]]

    return roots


HONORIFICS: Set[str] = {
    "frau", "herr", "herrn", "dr", "dr.", "prof", "prof.", "prof. dr.", "dozent", "dozentin",
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "sir", "madam"
}


def compute_smart_link_proposals(
    items: List[Any],
    get_text: Optional[Callable[[Any], str]] = None,
    get_entity_type: Optional[Callable[[Any], str]] = None,
    get_parent: Optional[Callable[[Any], Optional[str]]] = None,
) -> None:
    """
    Compute smart linking suggestions as interactive proposals (NO auto-commit!).
    Sets attributes .suggested_parent, .suggested_tag, and .suggested_candidates on each item.
    """
    if get_text is None:
        get_text = lambda x: getattr(x, "original_text", str(x))
    if get_entity_type is None:
        get_entity_type = lambda x: getattr(x, "entity_type", "")
    if get_parent is None:
        get_parent = lambda x: getattr(x, "parent_group_text", None)

    for g in items:
        g.suggested_parent = None
        g.suggested_tag = None
        g.suggested_candidates = []

        if get_parent(g):
            continue

        if get_entity_type(g) != "PERSON":
            continue

        orig_text = get_text(g)
        orig_lower = orig_text.lower().strip()
        words = orig_text.split()
        potential_candidates: List[Tuple[str, str]] = []

        # 1. Single word: Check genitive stem or first/last name
        if len(words) == 1:
            stem = re.sub(r"(s|'s|’s)$", "", orig_text, flags=re.IGNORECASE).strip()
            if stem and stem.lower() != orig_lower:
                for other in items:
                    if get_text(other).lower() != orig_lower and get_entity_type(other) == "PERSON" and not get_parent(other):
                        other_words = get_text(other).split()
                        if any(stem.lower() == w.lower() for w in other_words):
                            potential_candidates.append((get_text(other), "GENITIV"))

            for other in items:
                if get_text(other).lower() != orig_lower and get_entity_type(other) == "PERSON" and not get_parent(other):
                    other_words = get_text(other).split()
                    if len(other_words) > 1:
                        if orig_lower == other_words[0].lower():
                            potential_candidates.append((get_text(other), "VORNAME"))
                        elif orig_lower == other_words[-1].lower():
                            potential_candidates.append((get_text(other), "NACHNAME"))

        # 2. Multi-word: Check German/English honorifics (e.g. "Frau Meier", "Mr. Smith")
        elif len(words) >= 2 and words[0].lower().rstrip(".") in HONORIFICS:
            last_name = words[-1].lower()
            for other in items:
                if get_text(other).lower() != orig_lower and get_entity_type(other) == "PERSON" and not get_parent(other):
                    other_words = get_text(other).split()
                    if len(other_words) > 1 and other_words[0].lower().rstrip(".") not in HONORIFICS:
                        if last_name == other_words[-1].lower():
                            potential_candidates.append((get_text(other), "ANREDE"))

        unique_cand_dict: Dict[str, str] = {}
        for cand_name, tag in potential_candidates:
            unique_cand_dict.setdefault(cand_name, tag)

        if len(unique_cand_dict) == 1:
            cand_name, tag = list(unique_cand_dict.items())[0]
            g.suggested_parent = cand_name
            g.suggested_tag = tag
        elif len(unique_cand_dict) > 1:
            g.suggested_candidates = list(unique_cand_dict.keys())
            g.suggested_tag = list(unique_cand_dict.values())[0]


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
        from presidio_analyzer import AnalyzerEngine, RecognizerResult
        from local_anonymizer.recognizers import FuzzyGlossaryRecognizer, GLiNERRecognizer

        self._RecognizerResult = RecognizerResult
        nlp_engine = _create_blank_spacy_engine(languages=["de", "en"])
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

    def analyze(
        self,
        text: str,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[RecognizerResult]:
        """Analyze text using all configured recognizers with ignore filter, entity filters and progress reporting."""
        if not text:
            return []

        if on_progress:
            on_progress(0.10, "Ignore-Filterung und Vorverarbeitung...")

        # If enabled_entities is explicitly empty [], return no entities
        if self.enabled_entities is not None and len(self.enabled_entities) == 0:
            if on_progress:
                on_progress(1.00, "Keine Entitäten aktiviert.")
            return []

        # Find spans of ignore terms to exclude false positives (e.g. "CAS", "Studierende", "Unternehmen")
        ignored_spans: List[Tuple[int, int]] = []
        for term in self.ignore_terms:
            if not term.strip():
                continue
            pattern = re.compile(r"\b" + re.escape(term.strip()) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                ignored_spans.append((match.start(), match.end()))

        if on_progress:
            on_progress(0.25, "KI-Modell & Presidio Erkennung läuft...")

        # Run Presidio analysis without hardcoded score_threshold overriding recognizer thresholds
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.enabled_entities,
            score_threshold=None,
        )

        if on_progress:
            on_progress(0.65, "Filterung & Deduplizierung der Fundstellen...")

        # 1. Trim entity spans and filter out ignored spans
        filtered_results: List[RecognizerResult] = []
        for r in results:
            clean_s, clean_e = trim_entity_span(text, r.start, r.end)
            if clean_e <= clean_s or not text[clean_s:clean_e].strip():
                continue
            r.start = clean_s
            r.end = clean_e

            is_ignored = any(
                not (r.end <= ig_start or r.start >= ig_end)
                for ig_start, ig_end in ignored_spans
            )
            if not is_ignored:
                if self.enabled_entities is None or r.entity_type in self.enabled_entities:
                    filtered_results.append(r)

        # 2. Deduplicate / resolve overlapping results (prefer longer span, then higher score)
        filtered_results.sort(key=lambda r: (r.end - r.start, r.score), reverse=True)
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

        if on_progress:
            on_progress(0.80, "Genitiv-Erweiterung und Namensanalyse...")

        # 3. German/English genitive extension for recognized PERSON entities (e.g. "Julia" -> "Julias", "Julia's")
        person_results = [r for r in accepted if r.entity_type == "PERSON"]
        genitive_results: List[RecognizerResult] = []
        for r in person_results:
            p_text = text[r.start:r.end].strip()
            # Split full names into capitalized components (e.g. "Julia Meier" -> ["Julia", "Meier"])
            name_parts = [part for part in re.findall(r"\b[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ-]+\b", p_text) if len(part) >= 3]
            for part in name_parts:
                # Check if this part is in ignore terms (e.g. "Prof", "Dr")
                if any(part.lower() == ig.strip().lower() for ig in self.ignore_terms):
                    continue
                # Search for genitive: e.g. "Julias", "Julia's", "Julia’s"
                genitive_pattern = re.compile(r"\b" + re.escape(part) + r"(s|'s|’s)\b")
                for m in genitive_pattern.finditer(text):
                    m_start, m_end = m.start(), m.end()
                    # Check if already covered by accepted spans
                    is_covered = any(
                        not (m_end <= s or m_start >= e)
                        for s, e in accepted_spans
                    )
                    is_ignored = any(
                        not (m_end <= ig_s or m_start >= ig_e)
                        for ig_s, ig_e in ignored_spans
                    )
                    if not is_covered and not is_ignored:
                        genitive_results.append(
                            self._RecognizerResult(
                                entity_type="PERSON",
                                start=m_start,
                                end=m_end,
                                score=min(0.85, r.score),
                            )
                        )
                        accepted_spans.append((m_start, m_end))

        if genitive_results:
            accepted.extend(genitive_results)

        # Sort by start position in text
        accepted.sort(key=lambda r: r.start)

        if on_progress:
            on_progress(0.90, "Ergebnisse werden strukturiert...")

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
