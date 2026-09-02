import contextlib
import os
import re
import threading
import unicodedata
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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from gliner import GLiNER
from rapidfuzz import fuzz

_HF_HUB_LOCK = threading.Lock()


@contextlib.contextmanager
def set_huggingface_offline_mode(offline: bool):
    """
    Thread-safe context manager to synchronize HuggingFace offline mode across both
    os.environ and imported huggingface_hub runtime constants.
    """
    with _HF_HUB_LOCK:
        prev_env = os.environ.get("HF_HUB_OFFLINE")
        prev_const = None
        hf_constants = None
        try:
            import huggingface_hub.constants as hf_constants
            prev_const = getattr(hf_constants, "HF_HUB_OFFLINE", None)
        except ImportError:
            pass

        try:
            if offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                if hf_constants is not None:
                    hf_constants.HF_HUB_OFFLINE = True
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
                if hf_constants is not None:
                    hf_constants.HF_HUB_OFFLINE = False
            yield
        finally:
            if prev_env is not None:
                os.environ["HF_HUB_OFFLINE"] = prev_env
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

            if hf_constants is not None and prev_const is not None:
                hf_constants.HF_HUB_OFFLINE = prev_const


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


def get_optimal_device() -> str:
    """
    Detect the optimal compute device for PyTorch / GLiNER inference.
    Supports NVIDIA GPUs (cuda), Apple Silicon Macs (mps), and CPU fallback.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class GLiNERRecognizer(EntityRecognizer):
    """Presidio EntityRecognizer that uses GLiNER for multilingual / zero-shot PII detection."""

    DEFAULT_LABEL_MAPPING = {
        # Persons
        # These prompts intentionally describe a proper/named individual rather than the broad
        # concept of a person, which otherwise tends to match generic role nouns in German. The
        # broad prompt remains as a compatibility fallback because this model's multi-label
        # inference can otherwise miss real names in long, mixed documents.
        "person": "PERSON",
        "name": "PERSON",
        "person name": "PERSON",
        "first name": "PERSON",
        "last name": "PERSON",
        "person's proper name": "PERSON",
        "named person": "PERSON",
        "proper name": "PERSON",
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
        # Software and IT infrastructure (the glossary remains the authoritative source for
        # ambiguous product names such as SAP; these prompts are a fallback for unknown systems).
        "software": "IT_SYSTEM",
        "software system": "IT_SYSTEM",
        "IT system": "IT_SYSTEM",
        "application": "IT_SYSTEM",
        "database": "IT_SYSTEM",
        "platform": "IT_SYSTEM",
        # Roles are quasi-identifiers and therefore opt-in through the UI/configuration.
        "job title": "ROLE",
        "professional role": "ROLE",
        "position": "ROLE",
    }

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi_pii-v1",
        label_mapping: Optional[Dict[str, str]] = None,
        custom_labels: Optional[Dict[str, str]] = None,
        threshold: float = 0.55,
        supported_language: str = "de",
        name: str = "GLiNERRecognizer",
        entity_modes: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.entity_modes = entity_modes or {}
        self.label_mapping = dict(label_mapping or self.DEFAULT_LABEL_MAPPING)
        if custom_labels:
            self.label_mapping.update(custom_labels)

        self.model: Optional[GLiNER] = None

        supported_entities = list(set(self.label_mapping.values()))
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    _MODEL_CACHE: Dict[str, GLiNER] = {}

    def load(self) -> None:
        """Load GLiNER model silently and transfer to optimal compute device (CUDA / MPS / CPU)."""
        if self.model is None:
            if self.model_name in self._MODEL_CACHE:
                self.model = self._MODEL_CACHE[self.model_name]
                return

            with set_huggingface_offline_mode(True):
                try:
                    self.model = GLiNER.from_pretrained(self.model_name, local_files_only=True)
                except Exception:
                    self.model = None

            if self.model is None:
                with set_huggingface_offline_mode(False):
                    try:
                        self.model = GLiNER.from_pretrained(self.model_name)
                    except Exception as ex:
                        raise RuntimeError(
                            f"GLiNER Modell '{self.model_name}' (ca. 1.10 GB) konnte weder aus dem lokalen Cache noch online heruntergeladen werden: {ex}. "
                            f"Bitte überprüfe deine Internetverbindung oder verwende das Tool im reinen Regex- bzw. Glossar-Modus."
                        ) from ex

            device = get_optimal_device()
            if device != "cpu":
                try:
                    self.model.to(device)
                except Exception:
                    pass

            self._MODEL_CACHE[self.model_name] = self.model

    def analyze(
        self,
        text: str,
        entities: Sequence[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """Analyze text with GLiNER using batched multi-core/GPU chunk inference."""
        if not text:
            return []

        if self.model is None:
            self.load()

        assert self.model is not None

        # Determine which Presidio entities GLiNER is actually ALLOWED to return.
        # It must only return entities that have mode == "all".
        # (Mode "explicit_eupii" means EU-PII runs it, but GLiNER must NOT run it).
        requested_entities = set(entities) if entities else set(self.label_mapping.values())
        allowed_entities = set()
        for e in requested_entities:
            # Fall back to "all" if missing, to preserve behavior for config-less API usage
            mode = self.entity_modes.get(e, "all")
            if mode == "all":
                allowed_entities.add(e)

        if not allowed_entities:
            return []

        # ALWAYS query ALL mapped labels so GLiNER has the correct semantic "sinks".
        # This prevents Prompt-Label-Collapsing (e.g. where 'Exchange Online' is
        # misclassified as ORGANIZATION simply because IT_SYSTEM was turned off).
        labels_to_query = list(self.label_mapping.keys())

        # PERSON, IT_SYSTEM, and ROLE labels are deliberately queried in separate passes. They
        # use semantically focused prompts / are opt-in quasi-identifiers, while the established
        # PII labels should not lose confidence because these additional prompts compete for spans.
        person_labels = [
            label for label in labels_to_query if self.label_mapping.get(label) == "PERSON"
        ]
        legacy_person_labels = [
            label for label in ("person", "name", "person name", "first name", "last name")
            if label in person_labels
        ]
        refined_person_labels = [
            label for label in ("person's proper name", "named person", "proper name")
            if label in person_labels
        ]
        other_person_labels = [
            label for label in person_labels
            if label not in legacy_person_labels and label not in refined_person_labels
        ]
        it_system_labels = [
            label for label in labels_to_query if self.label_mapping.get(label) == "IT_SYSTEM"
        ]
        role_labels = [
            label for label in labels_to_query if self.label_mapping.get(label) == "ROLE"
        ]
        primary_labels = [
            label for label in labels_to_query
            if label not in person_labels and label not in it_system_labels and label not in role_labels
        ]

        # Chunk the text to prevent the 384-token GLiNER truncation limit
        text_chunks = chunk_text_with_offsets(text, max_chars=700)
        valid_chunks: List[Tuple[int, int, str]] = [
            (s, e, t) for s, e, t in text_chunks if t.strip()
        ]
        if not valid_chunks:
            return []

        chunk_texts = [c[2] for c in valid_chunks]
        results: List[RecognizerResult] = []

        def predict_for_labels(labels: List[str]) -> List[List[dict]]:
            if not labels:
                return []
            try:
                return self.model.inference(
                    chunk_texts, labels, threshold=self.threshold, batch_size=8
                )
            except Exception:
                return [
                    self.model.predict_entities(t, labels, threshold=self.threshold)
                    for t in chunk_texts
                ]

        # Base sinks are all non-PERSON labels. We always include them in every pass
        # so they can "absorb" things like 'Exchange Online' and prevent them from
        # collapsing into ORGANIZATION or PERSON.
        base_sinks = primary_labels + it_system_labels + role_labels

        passes = [
            # 1. Primary pass: queries all non-person labels. We keep all of them.
            (base_sinks, set(base_sinks)),
            # 2. Legacy person pass
            (legacy_person_labels + base_sinks, set(legacy_person_labels)),
            # 3. Refined person pass
            (refined_person_labels + base_sinks, set(refined_person_labels)),
            # 4. Other person pass
            (other_person_labels + base_sinks, set(other_person_labels)),
        ]

        for query_labels, keep_labels in passes:
            if not query_labels or not keep_labels:
                continue

            predicted_batches = predict_for_labels(query_labels)

            for (chunk_start, chunk_end, chunk_text), predicted_entities in zip(valid_chunks, predicted_batches):
                for pred in predicted_entities:
                    gliner_label = pred["label"]
                    if gliner_label not in keep_labels:
                        continue

                    presidio_entity = self.label_mapping.get(gliner_label, gliner_label.upper())

                    # Filter out entities that are not allowed (e.g. because they are off or EU-PII only)
                    if presidio_entity not in allowed_entities:
                        continue

                    global_start = chunk_start + pred["start"]
                    global_end = chunk_start + pred["end"]

                    results.append(
                        RecognizerResult(
                            entity_type=presidio_entity,
                            start=global_start,
                            end=global_end,
                            score=float(pred["score"]),
                            recognition_metadata={
                                "recognizer_name": self.name,
                                "detection_method": "ai",
                            },
                        )
                    )

        # GLiNER can occasionally return adjacent name components as separate PERSON spans
        # (e.g. "Julia" + "Meier") when several semantic prompts are active. Merge at most two
        # whitespace-separated components. The previous unbounded merge incorrectly joined
        # complete name lists such as "Benjamin Kägi Egzona Musliu" into one person. Punctuation
        # or a third component keeps identities separate. The weakest component score represents
        # the combined span conservatively.
        results.sort(key=lambda result: (result.start, result.end))
        merged_results: List[RecognizerResult] = []
        for result in results:
            previous = merged_results[-1] if merged_results else None
            previous_parts = text[previous.start:previous.end].split() if previous else []
            current_parts = text[result.start:result.end].split()
            if (
                previous is not None
                and result.entity_type == "PERSON"
                and previous.entity_type == "PERSON"
                and result.start >= merged_results[-1].end
                and not text[merged_results[-1].end:result.start].strip()
                and len(previous_parts) + len(current_parts) <= 2
            ):
                previous.end = result.end
                previous.score = min(previous.score, result.score)
            else:
                merged_results.append(result)

        return merged_results


class FuzzyGlossaryRecognizer(EntityRecognizer):
    """Presidio EntityRecognizer for company-specific terms with fuzzy matching via RapidFuzz."""

    def __init__(
        self,
        glossary: Optional[Dict[str, str]] = None,
        glossary_roles: Optional[Dict[str, str]] = None,
        high_confidence_threshold: float = 90.0,
        review_threshold: float = 75.0,
        supported_language: str = "de",
        name: str = "FuzzyGlossaryRecognizer",
    ):
        """
        Initialize FuzzyGlossaryRecognizer.

        Args:
            glossary: Dictionary mapping canonical terms to entity types (e.g. {"ZHAW": "ORGANIZATION", "abcd": "PERSON"}).
            glossary_roles: Dictionary mapping canonical terms to role tags (e.g. {"Max Muster": "PATIENT"}).
            high_confidence_threshold: Similarity score for automatic replacement (>= 90% -> score 0.95).
            review_threshold: Similarity score for manual review flagging (>= 75% -> score 0.80).
            supported_language: Language code.
            name: Recognizer name.
        """
        self.glossary = glossary or {}
        self.glossary_roles = glossary_roles or {}
        self.high_confidence_threshold = high_confidence_threshold
        self.review_threshold = review_threshold

        supported_entities = sorted(set(self.glossary.values()))
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    def add_term(self, term: str, entity_type: str = "ORGANIZATION", role: Optional[str] = None) -> None:
        """Add or update a glossary term."""
        self.glossary[term] = entity_type
        if role:
            self.glossary_roles[term] = role
        elif term in self.glossary_roles:
            del self.glossary_roles[term]
        if entity_type not in self.supported_entities:
            self.supported_entities.append(entity_type)

    def set_glossary(
        self,
        glossary: Optional[Dict[str, str]] = None,
        glossary_roles: Optional[Dict[str, str]] = None,
    ) -> None:
        """Replace the glossary and derive supported entity types from its configured values."""
        self.glossary = glossary or {}
        self.glossary_roles = glossary_roles or {}
        self.supported_entities = sorted(set(self.glossary.values()))

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

        # Find literal glossary terms first. This is intentionally independent from the word
        # tokenizer below so terms containing punctuation (e.g. "eClaims+", "C++", ".NET")
        # can still be classified as direct matches.
        raw_candidates: List[Tuple[int, int, str, float, str, str]] = []  # (..., term, match_kind)
        for canonical_term, entity_type in self.glossary.items():
            if entities and entity_type not in entities:
                continue
            term = canonical_term.strip()
            if not term:
                continue
            exact_pattern = re.compile(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                flags=re.IGNORECASE,
            )
            for match in exact_pattern.finditer(text):
                raw_candidates.append(
                    (match.start(), match.end(), entity_type, 1.0, canonical_term, "direct")
                )

        # Tokenize text into words with start and end character offsets for fuzzy matching.
        word_matches = list(re.finditer(r"\b[\w.-]+\b", text))

        # Determine max number of words in glossary terms
        max_glossary_words = max((len(term.split()) for term in self.glossary.keys()), default=0)

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

                    # Skip short single-character noise
                    if len(span_text) < 3 and len(canonical_term) < 3:
                        continue

                    # Literal matches were handled above. Normalize case and Unicode here so
                    # equivalent whitespace/Unicode forms still receive a direct label when
                    # they are represented by the tokenizer span.
                    normalized_span = " ".join(
                        unicodedata.normalize("NFKC", span_text).split()
                    ).casefold()
                    normalized_term = " ".join(
                        unicodedata.normalize("NFKC", canonical_term).split()
                    ).casefold()
                    if normalized_span == normalized_term:
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 1.0, canonical_term, "direct")
                        )
                        continue

                    ratio = fuzz.ratio(normalized_span, normalized_term)
                    if ratio >= self.high_confidence_threshold:
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 0.95, canonical_term, "fuzzy")
                        )
                    elif ratio >= self.review_threshold:
                        raw_candidates.append(
                            (start_char, end_char, entity_type, 0.80, canonical_term, "fuzzy")
                        )

        # Deduplicate overlapping candidates: sort by score desc, then span length desc
        raw_candidates.sort(key=lambda c: (c[3], c[1] - c[0]), reverse=True)
        accepted_spans: List[Tuple[int, int]] = []
        results: List[RecognizerResult] = []

        for start, end, entity_type, score, canonical_term, match_kind in raw_candidates:
            # Check overlap with already accepted spans
            overlaps = any(
                not (end <= existing_start or start >= existing_end)
                for existing_start, existing_end in accepted_spans
            )
            if not overlaps:
                accepted_spans.append((start, end))
                meta: Dict[str, Any] = {
                    "recognizer_name": self.name,
                    "glossary_match": match_kind,
                }
                if self.glossary_roles and canonical_term in self.glossary_roles:
                    meta["custom_role"] = self.glossary_roles[canonical_term]
                    meta["role_provenance"] = "profile"
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        score=score,
                        recognition_metadata=meta,
                    )
                )

        return results


CH_POSTAL_TOWN_REGEX = r"\b\d{4}\s+[A-ZÄÖÜ][\wäöüéèàßÄÖÜÉÈÀ-]+\b"
DE_POSTAL_TOWN_REGEX = r"\b\d{5}\s+[A-ZÄÖÜ][\wäöüßÄÖÜ-]+\b"
STREET_REGEX = (
    r"\b[A-ZÄÖÜ][\wäöüßÄÖÜéèàÉÈÀ-]*"
    r"(?:strasse|straße|str\.|weg|gasse|platz|allee|ring)\s+\d+[a-z]?\b"
)
FULL_CH_ADDRESS_REGEX = (
    rf"{STREET_REGEX}(?:\s*,\s*|\s+)\d{{4}}\s+"
    rf"[A-ZÄÖÜ][\wäöüéèàßÄÖÜÉÈÀ-]+\b"
)
FULL_DE_ADDRESS_REGEX = (
    rf"{STREET_REGEX}(?:\s*,\s*|\s+)\d{{5}}\s+"
    rf"[A-ZÄÖÜ][\wäöüßÄÖÜ-]+\b"
)


class AddressPatternRecognizer(PatternRecognizer):
    """Deterministic Swiss/German address recognizer with a conservative year guard."""

    def __init__(self, supported_language: str = "de", name: str = "AddressPatternRecognizer"):
        super().__init__(
            supported_entity="ADDRESS",
            name=name,
            supported_language=supported_language,
            patterns=[
                Pattern("full_ch_address", FULL_CH_ADDRESS_REGEX, 0.99),
                Pattern("full_de_address", FULL_DE_ADDRESS_REGEX, 0.99),
                Pattern("street_with_house_number", STREET_REGEX, 0.97),
                Pattern("ch_postal_code_and_town", CH_POSTAL_TOWN_REGEX, 0.95),
                Pattern("de_postal_code_and_town", DE_POSTAL_TOWN_REGEX, 0.95),
            ],
        )

    def validate_result(self, text: str) -> Optional[bool]:
        """Reject bare year-plus-town strings while keeping street-based addresses intact.

        A four-digit Swiss postal code and a year are structurally indistinguishable without
        more context. Bare values in the common 1900-2099 year range are therefore rejected;
        a complete street address still matches through its full-address pattern.
        """
        if re.fullmatch(CH_POSTAL_TOWN_REGEX, text, flags=re.IGNORECASE):
            postal_code = int(text[:4])
            if 1900 <= postal_code <= 2099:
                return False
        return True


class ChecksumPatternRecognizer(PatternRecognizer):
    """PatternRecognizer variant that keeps only matches passing a checksum validator."""

    def __init__(
        self,
        supported_entity: str,
        pattern: str,
        validator: Callable[[str], bool],
        name: str,
        supported_language: str = "de",
    ):
        self._checksum_validator = validator
        super().__init__(
            supported_entity=supported_entity,
            name=name,
            supported_language=supported_language,
            patterns=[Pattern(name.lower(), pattern, 0.99)],
        )

    def validate_result(self, text: str) -> Optional[bool]:
        return self._checksum_validator(text)


def is_valid_ahv_number(value: str) -> bool:
    """Validate a Swiss AHV number using its EAN-13-style check digit."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 13 or not digits.startswith("756"):
        return False

    weighted_sum = sum(
        int(digit) * (1 if index % 2 == 0 else 3)
        for index, digit in enumerate(digits[:12])
    )
    expected_check_digit = (10 - weighted_sum % 10) % 10
    return expected_check_digit == int(digits[-1])


def is_valid_uid_number(value: str) -> bool:
    """Validate a Swiss CHE/UID number using the official Modulo-11 algorithm."""
    if not re.fullmatch(r"CHE-\d{3}\.\d{3}\.\d{3}", value, flags=re.IGNORECASE):
        return False

    digits = re.sub(r"\D", "", value)
    weighted_sum = sum(
        int(digit) * weight
        for digit, weight in zip(digits[:8], (5, 4, 3, 2, 7, 6, 5, 4))
    )
    expected_check_digit = 11 - (weighted_sum % 11)
    if expected_check_digit == 10:
        return False
    if expected_check_digit == 11:
        expected_check_digit = 0
    return expected_check_digit == int(digits[-1])


class AHVNumberRecognizer(ChecksumPatternRecognizer):
    """Recognizer for dotted Swiss AHV/AVS numbers with checksum validation."""

    def __init__(self, supported_language: str = "de", name: str = "AHVNumberRecognizer"):
        super().__init__(
            supported_entity="AHV_NUMBER",
            pattern=r"\b756\.\d{4}\.\d{4}\.\d{2}\b",
            validator=is_valid_ahv_number,
            name=name,
            supported_language=supported_language,
        )


class UIDNumberRecognizer(ChecksumPatternRecognizer):
    """Recognizer for Swiss CHE/UID numbers with checksum validation."""

    def __init__(self, supported_language: str = "de", name: str = "UIDNumberRecognizer"):
        super().__init__(
            supported_entity="UID_NUMBER",
            pattern=r"\bCHE-\d{3}\.\d{3}\.\d{3}\b",
            validator=is_valid_uid_number,
            name=name,
            supported_language=supported_language,
        )


_EUPII_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_eupii_lock = threading.Lock()

GLINER_MODEL_NAME = "urchade/gliner_multi_pii-v1"
GLINER_MODEL_SIZE_MB = 1102  # ~1.10 GB download size
EUPII_MODEL_NAME = "bardsai/eu-pii-anonimization-multilang"
EUPII_MODEL_SIZE_MB = 1075  # ~1.07 GB download size (safetensors + tokenizer)


def is_model_cached(model_name: str, model_type: str = "transformers") -> bool:
    """
    Checks if model weights and configuration are available in the local cache or memory
    without downloading or redundantly allocating heavy model weights into memory.
    """
    # 1. In-memory cache check
    if model_type == "gliner":
        if model_name in GLiNERRecognizer._MODEL_CACHE:
            return True
    else:
        if model_name in _EUPII_MODEL_CACHE:
            return True

    # 2. Local directory path check
    if os.path.isdir(model_name):
        has_cfg = any(os.path.exists(os.path.join(model_name, f)) for f in ("config.json", "gliner_config.json"))
        has_weights = any(os.path.exists(os.path.join(model_name, f)) for f in ("model.safetensors", "pytorch_model.bin"))
        if has_cfg and has_weights:
            return True

    # 3. Fast HuggingFace Hub cache check without instantiating model
    try:
        from huggingface_hub import try_to_load_from_cache
        if model_type == "gliner":
            cfg = try_to_load_from_cache(model_name, "gliner_config.json") or try_to_load_from_cache(model_name, "config.json")
            weights = try_to_load_from_cache(model_name, "model.safetensors") or try_to_load_from_cache(model_name, "pytorch_model.bin")
            if isinstance(cfg, (str, os.PathLike)) and isinstance(weights, (str, os.PathLike)):
                return True
        else:
            cfg = try_to_load_from_cache(model_name, "config.json")
            weights = try_to_load_from_cache(model_name, "model.safetensors") or try_to_load_from_cache(model_name, "pytorch_model.bin")
            if isinstance(cfg, (str, os.PathLike)) and isinstance(weights, (str, os.PathLike)):
                return True
    except Exception:
        pass

    # 4. Fallback for test environments (e.g. mocked from_pretrained)
    with set_huggingface_offline_mode(True):
        if model_type == "gliner":
            try:
                from gliner import GLiNER
                if hasattr(GLiNER, "from_pretrained"):
                    GLiNER.from_pretrained(model_name, local_files_only=True)
                    return True
            except Exception:
                return False
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)
                return True
            except Exception:
                return False

    return False


EU_PII_CATEGORY_MAPPING: Dict[str, str] = {
    "PERSON_NAME": "PERSON",
    "PROPER_NAME": "PERSON",
    "PERSON_ALIAS": "PERSON",
    "LOCATION": "LOCATION",
    "POSTAL_ADDRESS": "LOCATION",
    "GEO_LOCATION": "LOCATION",
    "DOCUMENT_IDENTIFIER": "ID_NUMBER",
    "DOCUMENT_REFERENCE": "ID_NUMBER",
    "PERSON_IDENTIFIER": "ID_NUMBER",
    "BANK_ACCOUNT_IDENTIFIER": "ID_NUMBER",
    "ACCOUNT_IDENTIFIER": "ID_NUMBER",
    "ORGANIZATION_IDENTIFIER": "ID_NUMBER",
    "DEVICE_IDENTIFIER": "ID_NUMBER",
    "VEHICLE_IDENTIFIER": "ID_NUMBER",
    "HEALTH_DATA": "HEALTH_DATA",
}


class EUPiiRecognizer(EntityRecognizer):
    """
    Presidio EntityRecognizer wrapping the multilingual XLM-RoBERTa token classification model
    (bardsai/eu-pii-anonimization-multilang) with Fast Tokenizer overflow, sub-token span merging,
    whole-span score aggregation, and canonical deterministic span suppression.
    """

    def __init__(
        self,
        model_name: str = "bardsai/eu-pii-anonimization-multilang",
        threshold: float = 0.50,
        supported_language: str = "de",
        name: str = "EUPiiRecognizer",
        entity_modes: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.entity_modes = entity_modes or {}
        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.device = get_optimal_device()

        # Dedicated canonical recognizers for pre-computing protected deterministic spans
        from presidio_analyzer.predefined_recognizers import (
            DateRecognizer,
            EmailRecognizer,
            IbanRecognizer,
            PhoneRecognizer,
            UrlRecognizer,
        )

        self._ahv_rec = AHVNumberRecognizer(supported_language=supported_language)
        self._uid_rec = UIDNumberRecognizer(supported_language=supported_language)
        self._addr_rec = AddressPatternRecognizer(supported_language=supported_language)
        self._iban_rec = IbanRecognizer(supported_language=supported_language)
        self._email_rec = EmailRecognizer(supported_language=supported_language)
        self._phone_rec = PhoneRecognizer(
            supported_language=supported_language,
            supported_regions=["CH", "DE", "AT", "US", "GB", "FR", "IT"],
        )
        self._url_rec = UrlRecognizer(supported_language=supported_language)
        self._date_rec = DateRecognizer(supported_language=supported_language)

        supported_entities = sorted(list(set(EU_PII_CATEGORY_MAPPING.values())))
        self._is_initializing = True
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )
        self._is_initializing = False

    def load(self) -> None:
        """Load tokenizer and model with offline-first attempt and authorized first-run download fallback."""
        cache_key = f"{self.model_name}_{self.device}"
        if self._is_initializing:
            with _eupii_lock:
                if cache_key in _EUPII_MODEL_CACHE:
                    self.tokenizer, self.model = _EUPII_MODEL_CACHE[cache_key]
                    self.id2label = getattr(self.model.config, "id2label", {})
            return

        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        with _eupii_lock:
            if cache_key in _EUPII_MODEL_CACHE:
                self.tokenizer, self.model = _EUPII_MODEL_CACHE[cache_key]
                self.id2label = getattr(self.model.config, "id2label", {})
                return

            tokenizer = None
            model = None

            # 1. Offline-first attempt using local files only
            with set_huggingface_offline_mode(True):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    model = AutoModelForTokenClassification.from_pretrained(self.model_name, local_files_only=True)
                except Exception:
                    tokenizer = None
                    model = None

            # 2. Online download attempt on cache miss (authorized by explicit enable_eupii=True)
            if tokenizer is None or model is None:
                with set_huggingface_offline_mode(False):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                    except Exception as ex:
                        raise RuntimeError(
                            f"EU-PII Modell '{self.model_name}' (ca. 1.07 GB) konnte weder aus dem lokalen Cache noch online heruntergeladen werden: {ex}. "
                            f"Bitte überprüfe deine Internetverbindung oder deaktiviere das EU-PII-Modell in den Einstellungen."
                        ) from ex

            model.eval()
            if self.device != "cpu":
                try:
                    model.to(self.device)
                except Exception:
                    pass

            _EUPII_MODEL_CACHE[cache_key] = (tokenizer, model)
            self.tokenizer, self.model = tokenizer, model
            self.id2label = getattr(self.model.config, "id2label", {})

    def _get_protected_deterministic_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Runs canonical deterministic recognizers on the full text and collects all protected spans
        to prevent EU-PII from generating partial overlaps.
        """
        protected_spans: List[Tuple[int, int]] = []

        # 1. AHV
        for r in self._ahv_rec.analyze(text, entities=["ID_NUMBER"]):
            protected_spans.append((r.start, r.end))

        # 2. UID
        for r in self._uid_rec.analyze(text, entities=["ID_NUMBER"]):
            protected_spans.append((r.start, r.end))

        # 3. Address
        for r in self._addr_rec.analyze(text, entities=["LOCATION"]):
            protected_spans.append((r.start, r.end))

        # 4. IBAN
        for r in self._iban_rec.analyze(text, entities=["IBAN_CODE"]):
            protected_spans.append((r.start, r.end))

        # 5. Email
        for r in self._email_rec.analyze(text, entities=["EMAIL_ADDRESS"]):
            protected_spans.append((r.start, r.end))

        # 6. Phone
        for r in self._phone_rec.analyze(text, entities=["PHONE_NUMBER"]):
            protected_spans.append((r.start, r.end))

        # 7. URL
        for r in self._url_rec.analyze(text, entities=["URL"]):
            protected_spans.append((r.start, r.end))

        # 8. Date
        for r in self._date_rec.analyze(text, entities=["DATE_TIME"]):
            protected_spans.append((r.start, r.end))

        return protected_spans

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[Any] = None,
    ) -> List[RecognizerResult]:
        """
        Executes token classification over sliding windows with whole-span score aggregation,
        deduplication, and canonical overlap suppression.
        """
        if not text or not text.strip():
            return []

        requested_entities = set(entities) if entities else set(EU_PII_CATEGORY_MAPPING.values())
        allowed_entities = set()
        for e in requested_entities:
            mode = self.entity_modes.get(e, "all")
            if mode in ("all", "explicit_eupii"):
                allowed_entities.add(e)

        if not allowed_entities:
            return []

        self.load()
        assert self.model is not None and self.tokenizer is not None

        # 1. Pre-compute protected spans on full text
        protected_spans = self._get_protected_deterministic_spans(text)

        # 2. Fast Tokenizer Overflow encoding
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=64,
            max_length=384,
            truncation=True,
            padding=True,
        )
        offset_mapping = inputs.pop("offset_mapping")  # (num_windows, seq_len, 2)
        inputs.pop("overflow_to_sample_mapping", None)

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (num_windows, seq_len, num_classes)
            probs = torch.softmax(logits, dim=-1)

        raw_candidates: List[Dict[str, Any]] = []

        # Process each sliding window independently
        for window_idx in range(probs.shape[0]):
            w_probs = probs[window_idx]
            w_offsets = offset_mapping[window_idx]
            w_tokens = inputs["input_ids"][window_idx]

            current_canonical_type: Optional[str] = None
            current_start: Optional[int] = None
            current_end: Optional[int] = None
            current_scores: List[float] = []

            def flush_entity():
                nonlocal current_canonical_type, current_start, current_end, current_scores
                if current_canonical_type and current_start is not None and current_end is not None:
                    avg_score = sum(current_scores) / len(current_scores)
                    if avg_score >= self.threshold:
                        raw_candidates.append({
                            "type": current_canonical_type,
                            "start": current_start,
                            "end": current_end,
                            "score": avg_score,
                        })

            w_scores = probs[window_idx].max(dim=-1).values.tolist()
            w_ids = probs[window_idx].max(dim=-1).indices.tolist()
            w_offsets = offset_mapping[window_idx].tolist() if hasattr(offset_mapping[window_idx], "tolist") else list(offset_mapping[window_idx])

            for token_id, score, (start_char, end_char) in zip(w_ids, w_scores, w_offsets):
                if start_char == end_char:
                    flush_entity()
                    current_canonical_type = None
                    continue

                pred_label = self.id2label.get(token_id, "O")

                if pred_label == "O":
                    flush_entity()
                    current_canonical_type = None
                    continue

                bio_tag = pred_label[:1]
                raw_ent_type = pred_label[2:]
                canonical_type = EU_PII_CATEGORY_MAPPING.get(raw_ent_type, raw_ent_type)

                # Expand the current entity if:
                # 1) It is a direct sub-token continuation without intervening whitespace (e.g. BPE/wordpiece split of number or word)
                #    belonging to the same canonical category.
                # 2) It is an explicit I- tag of the same canonical category.
                is_subtoken = current_start is not None and current_end == start_char
                is_same_canonical = current_canonical_type == canonical_type

                if is_same_canonical and (bio_tag == "I" or is_subtoken):
                    current_end = end_char
                    current_scores.append(score)
                else:
                    flush_entity()
                    current_canonical_type = canonical_type
                    current_start = start_char
                    current_end = end_char
                    current_scores = [score]

            flush_entity()

        # 3. Deduplicate overlapping candidates across windows (keep highest score, then longest span)
        raw_candidates.sort(
            key=lambda c: (c["score"], c["end"] - c["start"]),
            reverse=True,
        )
        accepted_candidates: List[Dict[str, Any]] = []
        accepted_spans: List[Tuple[int, int]] = []

        for cand in raw_candidates:
            c_start = cand["start"]
            c_end = cand["end"]

            # Trim leading/trailing whitespace if any
            span_str = text[c_start:c_end]
            l_strip = len(span_str) - len(span_str.lstrip())
            r_strip = len(span_str) - len(span_str.rstrip())
            c_start += l_strip
            c_end -= r_strip

            if c_start >= c_end:
                continue

            # Reject sub-word fragments sliced inside unbroken alphanumeric words
            if c_start > 0 and text[c_start - 1].isalnum() and text[c_start].isalnum():
                continue
            if c_end < len(text) and text[c_end - 1].isalnum() and text[c_end].isalnum():
                continue

            overlaps_accepted = any(
                not (c_end <= a_start or c_start >= a_end)
                for a_start, a_end in accepted_spans
            )
            if not overlaps_accepted:
                accepted_spans.append((c_start, c_end))
                cand["start"] = c_start
                cand["end"] = c_end
                accepted_candidates.append(cand)

        # 4. Strict Overlap Rejection against protected deterministic spans
        results: List[RecognizerResult] = []
        for cand in accepted_candidates:
            c_start = cand["start"]
            c_end = cand["end"]
            overlaps_protected = any(
                not (c_end <= p_start or c_start >= p_end)
                for p_start, p_end in protected_spans
            )
            if overlaps_protected:
                continue

            std_type = cand["type"]
            if std_type not in allowed_entities:
                continue

            avg_score = round(cand["score"], 3)
            results.append(
                RecognizerResult(
                    entity_type=std_type,
                    start=c_start,
                    end=c_end,
                    score=avg_score,
                    recognition_metadata={
                        "recognizer_name": self.name,
                        "detection_method": "ai",
                        "method_detail": "bardsai/eu-pii (XLM-RoBERTa)",
                    },
                )
            )

        return results
