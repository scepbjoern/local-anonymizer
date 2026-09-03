import json
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


SYSTEM_POSTCHECK_PROMPT = """Du bist ein hochpräziser Assistent zur nachträglichen Ausgangskontrolle (Postcheck) von bereits anonymisierten Texten.
Deine Aufgabe ist es, den bereits anonymisierten Text auf übersehene, noch ungeschützte personenbezogene Daten (PII) und sensible Entitäten zu überprüfen (Nachzügler-Suche).

Regeln:
1. Der übergebene Text enthält bereits Platzhalter wie z. B. [PERSON_1], [LOCATION_1], [ORGANIZATION_1] etc.
   - Bestehende Platzhalter in eckigen Klammern [ ... ] dürfen NIEMALS als Fundstellen markiert werden!
   - Markiere AUSSCHLIESSLICH im Klartext verbliebene sensible Daten.
2. Schützenswerte Entitäten:
   - PERSON: Reale Personennamen (Vorname, Nachname, vollständiger Name).
   - ORGANIZATION: Firmen, Behörden, Spitäler, Institute, Vereine.
   - LOCATION: Städte, Strassen, Adressen, Länder, spezifische Orte.
   - DATE_TIME: Spezifische Datumsangaben (z. B. Geburtsdaten, Termine).
   - CONTACT / PHONE_NUMBER / EMAIL_ADDRESS / IBAN_CODE: Telefonnummern, E-Mail-Adressen, Kontonummern.
   - HEALTH_DATA: Spezifische Diagnosen oder Patientendaten mit Personenbezug.
3. Reine Funktionsbezeichnungen oder Allgemeinbegriffe (z. B. "Ärztin", "Patient", "Spital", "Dienstag", "Bericht") ohne Eigennamenbezug sind KEINE PII und dürfen NICHT markiert werden.
4. Zeichengenaue Indizes (Python-Slices):
   - Für jede Fundstelle musst du zwingend 'output_start' und 'output_end' angeben.
   - Diese Indizes müssen exakt dem Zeichenbereich im übergebenen anonymisierten Text entsprechen:
     anonymisierter_text[output_start:output_end] == text
5. Antworte AUSSCHLIESSLICH im geforderten JSON-Format mit schema_version "1.0". Falls du keine übersehenen Entitäten findest, gib ein leeres Array 'items': [] zurück.
"""


USER_POSTCHECK_TEMPLATE = """Hier ist der bereits anonymisierte Text zur Ausgangskontrolle:

--- ANFANG DES TEXTES ---
{anon_text}
--- ENDE DES TEXTES ---

Überprüfe den Text auf übersehene personenbezogene Daten.
Antworte ausschliesslich mit einem JSON-Objekt nach folgendem Muster:
```json
{{
  "schema_version": "1.0",
  "request_id": "{request_id}",
  "items": [
    {{
      "text": "<exakter_text_aus_dem_dokument>",
      "entity_type": "PERSON",
      "output_start": 42,
      "output_end": 55,
      "reasoning": "Übersehener Nachname im Fliesstext",
      "confidence": "high"
    }}
  ]
}}
```
Falls keine übersehenen Entitäten vorhanden sind:
```json
{{
  "schema_version": "1.0",
  "request_id": "{request_id}",
  "items": []
}}
```
"""


class PostcheckFindingItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    text: str = Field(..., min_length=1, max_length=500, description="Exact substring in anonymized text")
    entity_type: str = Field(..., min_length=1, max_length=64, description="Entity type, e.g. PERSON, LOCATION")
    output_start: int = Field(..., ge=0, description="Start character offset in anonymized text (0-based)")
    output_end: int = Field(..., gt=0, description="End character offset in anonymized text (0-based, exclusive)")
    reasoning: Optional[str] = Field(None, max_length=500, description="Brief justification")
    confidence: Optional[Literal["high", "medium", "low"]] = Field("high", description="Model confidence")

    @field_validator("output_start", "output_end", mode="before")
    @classmethod
    def check_strict_int(cls, v: Any, info: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"Index {info.field_name} muss ein strikter Integer sein, erhalten: {type(v).__name__}")
        return v

    @field_validator("output_end")
    @classmethod
    def check_offsets(cls, v: int, info: Any) -> int:
        start = info.data.get("output_start")
        if start is not None and v <= start:
            raise ValueError(f"output_end ({v}) muss grösser als output_start ({start}) sein.")
        return v

    @field_validator("entity_type")
    @classmethod
    def check_entity_type_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("entity_type darf nicht leer sein.")
        return s


class PostcheckEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = Field(..., description="Strict schema version requirement")
    request_id: str = Field(..., min_length=1, max_length=64, description="Tracking ID")
    items: List[PostcheckFindingItem] = Field(..., description="List of detected PII findings (must be explicit list, e.g. [])")
