"""Custom Presidio recognizers: GLiNER zero-shot PII recognizer and RapidFuzz fuzzy glossary recognizer."""

import os
import re
import warnings

# Disable noisy HuggingFace and transformers warnings for local CLI runs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")

try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

from typing import Dict, List, Optional, Sequence, Tuple
from presidio_analyzer import EntityRecognizer, RecognizerResult
from gliner import GLiNER
from rapidfuzz import fuzz


# Common abbreviations in German and English texts that should not trigger sentence boundaries
GERMAN_ABBREVIATIONS = {
    "dr", "prof", "hr", "fr", "frau", "herr", "nr", "st", "bzw", "etc",
    "z.b", "u.a", "d.h", "inkl", "ca", "vgl", "abs", "art", "jan", "feb", "mär",
    "apr", "jun", "jul", "aug", "sep", "okt", "nov", "dez", "univ", "ass", "dipl",
    "ing", "mag", "bsc", "msc", "phd", "co", "gmbh", "ag", "inc", "ltd"
}

GERMAN_MONTHS = {
    "januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
    "september", "oktober", "november", "dezember", "jan", "feb", "mär", "apr",
    "jun", "jul", "aug", "sep", "okt", "nov", "dez"
}


def is_sentence_boundary(text: str, match: re.Match) -> bool:
    """Check if punctuation match represents a true sentence boundary."""
    p_end = match.end()
    p_start = match.start()

    # Punctuation must be followed by whitespace, newline, or end of string
    if p_end < len(text) and not text[p_end].isspace():
        return False

    preceding = text[:p_start].rstrip()
    if not preceding:
        return False
    last_word = preceding.split()[-1].lower().rstrip(".!?")

    # Abbreviations or street suffixes (e.g. "Dr.", "Bahnhofstr.", "Nr.")
    if last_word in GERMAN_ABBREVIATIONS or last_word.endswith(("str", "str.")):
        return False

    # Ordinal numbers in dates (e.g. "14. Juli")
    if last_word.isdigit():
        after = text[p_end:].lstrip()
        if after:
            next_word = after.split()[0].lower().rstrip(",;.:")
            if next_word in GERMAN_MONTHS:
                return False

    # Lowercase continuation is rarely a new sentence
    after = text[p_end:].lstrip()
    if after and after[0].islower():
        return False

    return True


def split_paragraph_into_sentences(p_str: str) -> List[Tuple[int, int, str]]:
    """Split a single paragraph into sentence spans (start_rel, end_rel, sentence_text)."""
    boundaries: List[int] = []
    for m in re.finditer(r"[.!?]+", p_str):
        if is_sentence_boundary(p_str, m):
            boundaries.append(m.end())

    if not boundaries:
        return [(0, len(p_str), p_str)]

    sentences: List[Tuple[int, int, str]] = []
    curr = 0
    for b in boundaries:
        sentences.append((curr, b, p_str[curr:b]))
        curr = b
    if curr < len(p_str):
        sentences.append((curr, len(p_str), p_str[curr:]))
    return sentences


def chunk_text_with_offsets(text: str, max_chars: int = 800) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks of at most max_chars without breaking paragraphs, sentences, or entity spans.
    Returns list of (start_char_idx, end_char_idx, chunk_text).
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, len(text), text)]

    chunks: List[Tuple[int, int, str]] = []
    paragraphs = list(re.finditer(r"[^\r\n]+(?:\r?\n)?|\r?\n", text))

    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_pieces: List[str] = []

    for p in paragraphs:
        p_start, p_end = p.start(), p.end()
        p_str = p.group()

        if len(p_str) > max_chars:
            if current_pieces and current_start is not None and current_end is not None:
                chunks.append((current_start, current_end, "".join(current_pieces)))
                current_pieces = []
                current_start = None
                current_end = None

            # Sentence-aware chunking for long paragraphs
            sentences = split_paragraph_into_sentences(p_str)
            sub_start: Optional[int] = None
            sub_end: Optional[int] = None
            sub_pieces: List[str] = []

            for s_start_rel, s_end_rel, s_str in sentences:
                s_start = p_start + s_start_rel
                s_end = p_start + s_end_rel

                if sub_pieces and sub_start is not None and (s_end - sub_start) > max_chars:
                    chunks.append((sub_start, sub_end, "".join(sub_pieces)))  # type: ignore
                    sub_start = s_start
                    sub_end = s_end
                    sub_pieces = [s_str]
                else:
                    if sub_start is None:
                        sub_start = s_start
                    sub_end = s_end
                    sub_pieces.append(s_str)

            if sub_pieces and sub_start is not None and sub_end is not None:
                chunks.append((sub_start, sub_end, "".join(sub_pieces)))
        else:
            if current_pieces and current_start is not None and (p_end - current_start) > max_chars:
                chunks.append((current_start, current_end, "".join(current_pieces)))  # type: ignore
                current_start = p_start
                current_end = p_end
                current_pieces = [p_str]
            else:
                if current_start is None:
                    current_start = p_start
                current_end = p_end
                current_pieces.append(p_str)

    if current_pieces and current_start is not None and current_end is not None:
        chunks.append((current_start, current_end, "".join(current_pieces)))

    return chunks


class GLiNERRecognizer(EntityRecognizer):
    """Presidio EntityRecognizer that uses GLiNER for multilingual / zero-shot PII detection."""

    DEFAULT_LABEL_MAPPING = {
        # Persons
        "person": "PERSON",
        "person name": "PERSON",
        "name": "PERSON",
        "first name": "PERSON",
        "last name": "PERSON",
        # Organizations & Companies
        "organization": "ORGANIZATION",
        "company": "ORGANIZATION",
        "institution": "ORGANIZATION",
        "university": "ORGANIZATION",
        "school": "ORGANIZATION",
        "enterprise": "ORGANIZATION",
        # Contact
        "email": "EMAIL_ADDRESS",
        "email address": "EMAIL_ADDRESS",
        "phone number": "PHONE_NUMBER",
        "phone": "PHONE_NUMBER",
        "mobile phone": "PHONE_NUMBER",
        # Financial & IDs
        "iban": "IBAN_CODE",
        "credit card number": "CREDIT_CARD",
        "bank account": "BANK_ACCOUNT",
        "passport number": "ID_NUMBER",
        "id": "ID_NUMBER",
        "social security number": "ID_NUMBER",
        "ahv nummer": "ID_NUMBER",
        "tax number": "ID_NUMBER",
        # Locations & Addresses
        "address": "LOCATION",
        "street": "LOCATION",
        "location": "LOCATION",
        "city": "LOCATION",
        "country": "LOCATION",
        "postal code": "LOCATION",
        "zip code": "LOCATION",
        # Dates & Times
        "date": "DATE_TIME",
        "time": "DATE_TIME",
        "birth date": "DATE_TIME",
        # Digital Identifiers
        "ip address": "IP_ADDRESS",
        "username": "USERNAME",
        "url": "URL",
        # Health & Sensitive
        "medical condition": "HEALTH_DATA",
        "health data": "HEALTH_DATA",
        "salary": "FINANCIAL_DATA",
        "amount of money": "FINANCIAL_DATA",
    }

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi_pii-v1",
        label_mapping: Optional[Dict[str, str]] = None,
        custom_labels: Optional[Dict[str, str]] = None,
        threshold: float = 0.55,
        supported_language: str = "de",
        name: str = "GLiNERRecognizer",
    ):
        self.label_mapping = dict(label_mapping or self.DEFAULT_LABEL_MAPPING)
        if custom_labels:
            self.label_mapping.update(custom_labels)

        self.threshold = threshold
        self.model_name = model_name
        self.model: Optional[GLiNER] = None

        supported_entities = list(set(self.label_mapping.values()))
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    _MODEL_CACHE: Dict[str, GLiNER] = {}

    def load(self) -> None:
        """Load GLiNER model silently (locally if cached, otherwise download without noise)."""
        if self.model is None:
            if self.model_name in self._MODEL_CACHE:
                self.model = self._MODEL_CACHE[self.model_name]
                return

            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self.model = GLiNER.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                os.environ.pop("HF_HUB_OFFLINE", None)
                self.model = GLiNER.from_pretrained(self.model_name)
                os.environ["HF_HUB_OFFLINE"] = "1"

            self._MODEL_CACHE[self.model_name] = self.model

    def analyze(
        self,
        text: str,
        entities: Sequence[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """Analyze text with GLiNER using chunking to avoid token limit truncations."""
        if not text:
            return []

        if self.model is None:
            self.load()

        assert self.model is not None

        # Determine which GLiNER labels to query based on requested Presidio entities
        if entities:
            labels_to_query = [
                k for k, v in self.label_mapping.items() if v in entities
            ]
        else:
            labels_to_query = list(self.label_mapping.keys())

        if not labels_to_query:
            return []

        # Chunk the text to prevent the 384-token GLiNER truncation limit
        text_chunks = chunk_text_with_offsets(text, max_chars=700)
        results: List[RecognizerResult] = []

        for chunk_start, chunk_end, chunk_text in text_chunks:
            if not chunk_text.strip():
                continue

            predicted_entities = self.model.predict_entities(
                chunk_text,
                labels_to_query,
                threshold=self.threshold,
            )

            for pred in predicted_entities:
                gliner_label = pred["label"]
                presidio_entity = self.label_mapping.get(gliner_label, gliner_label.upper())

                # Filter if specific entities were requested
                if entities and presidio_entity not in entities:
                    continue

                global_start = chunk_start + pred["start"]
                global_end = chunk_start + pred["end"]

                results.append(
                    RecognizerResult(
                        entity_type=presidio_entity,
                        start=global_start,
                        end=global_end,
                        score=float(pred["score"]),
                    )
                )

        return results


class FuzzyGlossaryRecognizer(EntityRecognizer):
    """Presidio EntityRecognizer for company-specific terms with fuzzy matching via RapidFuzz."""

    def __init__(
        self,
        glossary: Optional[Dict[str, str]] = None,
        high_confidence_threshold: float = 90.0,
        review_threshold: float = 75.0,
        supported_language: str = "de",
        name: str = "FuzzyGlossaryRecognizer",
    ):
        """
        Initialize FuzzyGlossaryRecognizer.

        Args:
            glossary: Dictionary mapping canonical terms to entity types (e.g. {"ZHAW": "ORGANIZATION", "abcd": "PERSON"}).
            high_confidence_threshold: Similarity score for automatic replacement (>= 90% -> score 0.95).
            review_threshold: Similarity score for manual review flagging (>= 75% -> score 0.80).
            supported_language: Language code.
            name: Recognizer name.
        """
        self.glossary = glossary or {}
        self.high_confidence_threshold = high_confidence_threshold
        self.review_threshold = review_threshold

        supported_entities = list(set(self.glossary.values())) if self.glossary else ["ORGANIZATION", "PERSON", "CUSTOM_TERM"]
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    def add_term(self, term: str, entity_type: str = "ORGANIZATION") -> None:
        """Add or update a glossary term."""
        self.glossary[term] = entity_type
        if entity_type not in self.supported_entities:
            self.supported_entities.append(entity_type)

    def load(self) -> None:
        """Nothing to load for fuzzy glossary."""
        pass

    def analyze(
        self,
        text: str,
        entities: Sequence[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """Scan text with sliding windows and match against glossary terms."""
        if not text or not self.glossary:
            return []

        # Tokenize text into words with start and end character offsets
        word_matches = list(re.finditer(r"\b[\w.-]+\b", text))
        if not word_matches:
            return []

        # Determine max number of words in glossary terms
        max_glossary_words = max(len(term.split()) for term in self.glossary.keys())

        raw_candidates: List[Tuple[int, int, str, float, str]] = []  # (start, end, entity_type, score, term)

        num_words = len(word_matches)
        for i in range(num_words):
            for length in range(1, max_glossary_words + 1):
                if i + length > num_words:
                    break

                start_char = word_matches[i].start()
                end_char = word_matches[i + length - 1].end()
                span_text = text[start_char:end_char].strip()

                for canonical_term, entity_type in self.glossary.items():
                    if entities and entity_type not in entities:
                        continue

                    # Exact match (Case-insensitive or Case-sensitive)
                    if span_text.lower() == canonical_term.lower():
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 1.0, canonical_term)
                        )
                        continue

                    # Skip short single-character noise
                    if len(span_text) < 3 and len(canonical_term) < 3:
                        continue

                    ratio = fuzz.ratio(span_text.lower(), canonical_term.lower())
                    if ratio >= self.high_confidence_threshold:
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 0.95, canonical_term)
                        )
                    elif ratio >= self.review_threshold:
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 0.80, canonical_term)
                        )

        # Deduplicate overlapping candidates: sort by score desc, then span length desc
        raw_candidates.sort(key=lambda c: (c[3], c[1] - c[0]), reverse=True)
        accepted_spans: List[Tuple[int, int]] = []
        results: List[RecognizerResult] = []

        for start, end, entity_type, score, canonical_term in raw_candidates:
            # Check overlap with already accepted spans
            overlaps = any(
                not (end <= existing_start or start >= existing_end)
                for existing_start, existing_end in accepted_spans
            )
            if not overlaps:
                accepted_spans.append((start, end))
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        score=score,
                    )
                )

        return results
