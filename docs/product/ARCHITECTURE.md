# System Architecture – local-anonymizer

## 1. Übersicht & Pipeline-Architektur

`local-anonymizer` folgt einer modular aufgebauten, 100% lokalen Verarbeitungs-Pipeline ohne externe Netzwerkaufrufe.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          EINGABE-DOKUMENTE                             │
│                  (.pdf, .docx, .txt, .md, .csv, .json)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. STRUKTURIERTE EXTRAKTION                                            │
│    • PDF -> Markdown mit Überschriften & Tabellen (pymupdf4llm)        │
│    • DOCX -> Paragraphen & Tabellen (python-docx)                      │
│    • TXT/MD -> Multi-Encoding Fallback (utf-8-sig, cp1252, iso-8859)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. HYBRIDE ENTITÄTEN-ERKENNUNG (Presidio + GLiNER + RapidFuzz)         │
│    • Abkürzungs-bewusstes Satzgrenzen-Splitting (is_sentence_boundary) │
│    • Token-Overlap Chunking (<800 Zeichen)                             │
│    • GLiNER Zero-Shot Multi-PII Recognizer (mit Memory-Cache)          │
│    • RapidFuzz Fuzzy Glossary Recognizer (Tippfehler & Kürzel)         │
│    • Ignore-Listen-Filterung (Rollen, Grade, Begriffe)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. INTERAKTIVE REVIEW- & LINKING-SCHICHT (NiceGUI)                     │
│    • Entitäten-Bündelung nach kanonischen Begriffen (z. B. 8x "Julia") │
│    • Entity-Linking (Zuweisung von Co-Referenzen & Schreibweisen)      │
│    • Semantische Rollen-Zuordnung (Modi 1, 2, 3)                       │
│    • Reaktive Live-Vorschau im Arbeitsspeicher                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐   ┌────────────────────────────────┐
│ 4. ANONYMISIERTER TEXT            │   │ 5. LOKALE MAPPING-TABELLE      │
│    [PERSON_1_STUDENT_VOLLNAME]    │   │    {"[PERSON_1_...]": "Julia"}│
│    -> Sicher für Cloud-LLMs       │   │    -> Bleibt privat auf Rechner│
└─────────────────┬─────────────────┘   └────────────────┬───────────────┘
                  │                                      │
                  │   [Verarbeitung im Cloud-LLM]        │
                  │   (z. B. ChatGPT / Claude Prompt)    │
                  │                                      │
                  ▼                                      │
┌───────────────────────────────────┐                    │
│ 6. LLM-ANTWORT MIT PLATZHALTERN   │                    │
└─────────────────┬─────────────────┘                    │
                  │                                      │
                  └─────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 7. DETERMINISTISCHE DE-ANONYMISIERUNG                                  │
│    • Single-Pass Regex-Substitution (Längen-absteigend sortiert)       │
│    • 100% exakte Wiederherstellung ohne Kaskadierungsfehler            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       WIEDERHERGESTELLTER KLARTEXT                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kernkomponenten im Detail

### 2.1 Extraktoren (`extractors.py`)
- **`extract_text_from_pdf_bytes`:** Nutzt `pymupdf4llm` zur Umwandlung von PDF-Seiten in semantisch strukturiertes Markdown.
- **In-Memory-Support:** Sämtliche Extraktoren arbeiten nativ auf `bytes` (`read_document_from_bytes`), um das Schreiben temporärer Dateien auf Festplatte vollständig zu vermeiden.
- **Fehlerbehandlung:** Bildbasierte PDFs ohne Textlayer werden über `doc.page_count > 0 and not pages_text` erkannt und werfen `ValueError` mit klarer OCR-Hinweismeldung.

### 2.2 Presidio Analyzer & Custom Recognizers (`recognizers.py`)
- **`GLiNERRecognizer`:**
  - Kapselt `urchade/gliner_multi_pii-v1`.
  - Hält ein Singleton-Klassen-Cache `_MODEL_CACHE`, sodass das PyTorch-Modell nur ein einziges Mal in den RAM geladen wird.
  - Wendet `chunk_text_with_offsets` an, um das 384-Token-Limit von DeBERTa/GLiNER ohne Informationsverlust zu handhaben.
- **`is_sentence_boundary`:**
  - Schützt deutsche und englische Standardabkürzungen (`Dr.`, `Prof.`, `Bahnhofstr.`, `Nr.`, `14. Juli`), damit Sätze nicht mitten im Eigennamen zerschnitten werden.
- **`FuzzyGlossaryRecognizer`:**
  - Nutzt `rapidfuzz.fuzz.ratio` mit Schwellenwerten (High Confidence $\ge 90\%$, Review-Bedarf $\ge 75\%$).
  - Bevorzugt exakte Treffer (Score 1.0) vor Fuzzy-Treffern.

### 2.3 Platzhalter-Engine & Entity-Linking (`anonymizer.py`) *(Geplant für Phase 3 / In Prüfung)*
- **Format-Modi:**
  - **Modus 1:** `[<TYPE>_<N>]`
  - **Modus 2:** `[<TYPE>_<N>_<ROLE>]`
  - **Modus 3:** `[<TYPE>_<ROLE>]` *(mit automatischer Kollisions-Prüfung und Fallback auf Modus 2)*
- **Co-Referenz-Tags:**
  - Verknüpfte Entitäten erhalten ein gemeinsames Präfix `[<TYPE>_<N>_...]` und ein Oberflächen-Tag (`_VOLLNAME`, `_VORNAME`, `_NACHNAME`, `_KURZFORM`).
- **De-Anonymisierung:**
  ```python
  sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)
  pattern = re.compile("|".join(re.escape(k) for k in sorted_placeholders))
  return pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
  ```
  Verhindert, dass z. B. ein wiederhergestellter Name, der zufällig ein Substring eines anderen Platzhalters ist, in einer zweiten Ersetzungsrunde erneut überschrieben wird.

### 2.4 Benutzeroberfläche (`app.py`)
- Aufgebaut mit **NiceGUI** (FastAPI + Vue/Quasar Backend).
- Desktop-Ausführung über `ui.run(native=True)` (nutzt Microsoft WebView2 auf Windows via `pywebview`).
- CPU-intensive NLP-Berechnungen werden per `asyncio.to_thread` vom UI-Event-Loop isoliert.

---

## 3. Sicherheits- und Datenschutzmodell

| Aspekt | Garantie | Technische Umsetzung |
| :--- | :--- | :--- |
| **Datenabfluss** | 0% externe Datenübertragung | Offline-Modus für HuggingFace (`HF_HUB_OFFLINE=1`), kein Telemetrie-Code in NiceGUI/Presidio |
| **Persistenz** | Keine PII auf temporären Datenträgern | Reine RAM-Objekte (`io.BytesIO`, Python-State), Speicherbereinigung nach Session-Ende |
| **Reversibilität** | 100% mathematische Wiederherstellbarkeit | Deterministische JSON-Mappingtabelle, Single-Pass-Regex |
| **Admin-Rechte** | 0% Admin-Rechte erforderlich | Reine User-Space-Dependencies (`uv pip install`) |

---

## 4. Abgrenzung: Warum keine echte PDF-Content-Stream-Redaktion?

> [!NOTE]
> **Bewusste Scope-Entscheidung:**
> `local-anonymizer` führt **keine** native PDF-Content-Stream-Redaktion durch (d. h. keine direkte Entfernung von Vektorglyphen oder Pixel-Übermalung in binären PDF-Dateien). 
> 
> **Begründung:** Der Einsatzzweck des Tools ist ein reiner **Prompt-Privacy-Layer für KI-Workflows**: Textinhalte werden aus Dokumenten extrahiert, lokal bereinigt, an ein LLM übergeben und die LLM-Antwort anschliessend de-anonymisiert. Das Veröffentlichen optisch geschwärzter Original-PDFs ist nicht Gegenstand dieses Tools. Das text- und markdown-basierte Vorgehen stellt sicher, dass keinerlei unsichtbare PDF-Metadaten oder versteckte Textlayer unbemerkt an das LLM übermittelt werden können.
