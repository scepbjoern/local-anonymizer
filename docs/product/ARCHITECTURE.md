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
│    • CH/DE Address Pattern & Swiss Checksum Recognizers                │
│    • IT-System Glossar + GLiNER-Sicherheitsnetz                        │
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
- **`extract_text_from_pdf_bytes`:** Nutzt die erprobte RAG-Layout-Engine von PyMuPDF zur Umwandlung von PDF-Seiten in semantisch sauberes Markdown (erhält Überschriften, Listen, Fettungen und Tabellenstrukturen ohne Wörter in rahmenlosen Tabellen zu zerschneiden). Unterstützt Bildtext-Filterung, wiederkehrende Kopf-/Fußzeilen-Filterung mit Seite-1-Titelschutz sowie Multi-Core-Parallelisierung via `ProcessPoolExecutor`.
- **In-Memory & Lokale Puffer-Bereinigung:** Sämtliche Extraktoren und NLP-Analysen arbeiten grundsätzlich auf In-Memory-`bytes` (`read_document_from_bytes`). Zwei spezifische Vorgänge nutzen das geschützte Anwendungsverzeichnis `~/.local-anonymizer/temp_uploads`:
  1. *Drag-and-Drop im GUI:* Zur zuverlässigen Übertragung großer Dateien (bis 50 MB) ohne WebSocket-Limits als HTTP-Streaming-Puffer.
  2. *Mehrseitige PDF-Extraktion:* Zur Vermeidung von Windows-IPC-Serialisierungs-Overhead bei Multiprocessing-Seiten-Dispatch als kurzzeitige geteilte Zwischendatei.
  Beide Pfade sind durch `try...finally`-Sofortlöschung, Startup-Bereinigung (`cleanup_temp_uploads` / `cleanup_extraction_temp_files`) sowie `atexit`-Handler gegen verwaiste Dateien abgesichert.
- **Fehlerbehandlung:** Bildbasierte PDFs ohne Textlayer werden über `doc.page_count > 0 and not pages_text` erkannt und werfen `ValueError` mit klarer OCR-Hinweismeldung. Bei mehrseitigen PDFs fängt eine 3-Stufen-Fehlerbehandlung Einzelseitenfehler ab, ohne das Gesamtdokument abzubrechen.

### 2.2 Presidio Analyzer, Dual-Modell-Ensemble & Custom Recognizers (`recognizers.py`)
- **`EUPiiRecognizer` (Stufe 2):**
  - Kapselt das vortrainierte europäische PII-Modell `bardsai/eu-pii-anonimization-multilang` (XLM-RoBERTa Token-Klassifikation).
  - Selektiv aktiviert für 4 hochpräzise Kernkategorien: `PERSON`, `LOCATION`, `ID_NUMBER` und `HEALTH_DATA`.
  - Nutzt Fast-Tokenizer-Windowing (`max_length=384`, `stride=64`), BIO-Tag-Aggregation mit subtoken-präziser Wortfortführung und Deduplizierung.
  - Führt vorab eine kanonische Schutzprüfung auf dem Volltext durch (`_get_protected_deterministic_spans`): Überlappt ein ML-Span mit einem gültigen AHV-, UID-, IBAN-, E-Mail-, Adress-, URL-, Datums- oder Telefon-Treffer, wird er verworfen.
  - Verwaltet Offline-First-Zustände thread-sicher über den atomaren Kontextmanager `set_huggingface_offline_mode(offline)` unter `_HF_HUB_LOCK` mit exakter Snapshot-Wiederherstellung.
- **`GLiNERRecognizer` (Stufe 1):**
  - Kapselt `urchade/gliner_multi_pii-v1`.
  - Hält ein Singleton-Klassen-Cache `_MODEL_CACHE`, sodass das PyTorch-Modell nur ein einziges Mal in den RAM geladen wird.
  - Wendet `chunk_text_with_offsets` an, um das 384-Token-Limit von DeBERTa/GLiNER ohne Informationsverlust zu handhaben.
  - Verwendet für `PERSON` präzisere Prompts wie `person's proper name`, `named person` und `proper name`, um generische Rollennomen weniger häufig zu erfassen. `ROLE`/`JOB_TITLE` nutzt die Prompts `job title`, `professional role` und `position` in einem getrennten Modellpass und ist standardmässig deaktiviert.
  - Dient als Primärmodell für `ORGANIZATION` und `ROLE` sowie als flexibles Sicherheitsnetz für offene/ungewöhnliche Diagnosen.
- **4-Stufen Quellenhierarchie (Source Priority):**
  - Bei überlappenden Spans gilt strikt: `Glossar (Stufe 4)` > `Deterministisch / Validatoren / Bibliothek (Stufe 3)` > `EU-PII (Stufe 2)` > `GLiNER (Stufe 1)`.
- **`is_sentence_boundary`:**
  - Schützt deutsche und englische Standardabkürzungen (`Dr.`, `Prof.`, `Bahnhofstr.`, `Nr.`, `14. Juli`), damit Sätze nicht mitten im Eigennamen zerschnitten werden.
- **`FuzzyGlossaryRecognizer` (Stufe 4):**
  - Nutzt `rapidfuzz.fuzz.ratio` mit Schwellenwerten (High Confidence $\ge 90\%$, Review-Bedarf $\ge 75\%$).
  - Bevorzugt exakte Treffer (Score 1.0) vor Fuzzy-Treffern.
  - Explizite Glossar-Treffer werden gemäss der Kategorie-Richtlinie separat zugelassen oder vollständig blockiert. Sie können eingebaute generische Ignore-Begriffe bewusst überschreiben; persönliche Ignore-Einträge behalten immer Vorrang.
- **Deterministische Recognizer & Bibliotheken (Stufe 3):**
  - `ValidatedPhoneRecognizer` bindet Google `phonenumbers` (`libphonenumber`) mit vollem Ländercode- und Prüfziffern-Support ein und vergibt den autoritativen Score 1.0, sodass Telefonnummern nicht durch KI-Zero-Shot-Scores verdrängt werden.
  - `AddressPatternRecognizer` erkennt zusammenhängende Schweizer und deutsche Adressen per Regex und weist die bekannte Kollision zwischen vierstelliger Schweizer PLZ und Jahreszahl konservativ zurück.
  - `AHVNumberRecognizer` validiert die AHV-Kontrollziffer; `UIDNumberRecognizer` validiert CHE/UID nach Modulo 11. Formal korrekte, aber prüfziffern-ungültige Nummern werden nicht als Entitäten ausgegeben.
  - `IT_SYSTEM` wird über das dynamisch aus dem Glossar abgeleitete Zieltypenset sowie separate GLiNER-Sicherheitsnetz-Prompts erkannt.

Die optionale Kategorie `ROLE` wird über die UI-Modi gesteuert und standardmässig nicht aktiviert. Dadurch bleiben fachlich wichtige Formulierungen wie „Der Sachbearbeiter prüft …“ standardmässig erhalten; bei erhöhtem Identifikationsrisiko können Funktionsbezeichnungen wie `CEO` oder `Leiter Prozessmanagement` gezielt anonymisiert werden. Generische Rollennomen werden zusätzlich über die eingebaute Ignore-Liste gegen typische PERSON-Falschpositive geschützt.

Die UI-Schalter für Entitätstypen verwenden bis zu vier Modi: `Aus` blockiert alle Quellen, `Nur Glossar & manuell` beschränkt sich auf explizite Einträge, `Nur Glossar, manuell, deterministisch & EU-PII (ohne GLiNER)` schliesst GLiNER für sensible Kernkategorien gezielt aus, und `Alle Quellen` aktiviert sämtliche Quellen. Glossar-Treffer werden in einem getrennten, direkten Pass geprüft; dadurch kann beispielsweise `IT_SYSTEM` auf „Nur Glossar & manuell“ stehen, während ein vollständiges „Aus“ auch `SAP` blockiert. Manuelle, dokumentbezogene Markierungen folgen derselben Kategorie-Richtlinie.

### 2.3 Platzhalter-Engine & Entity-Linking (`anonymizer.py`) *(Umgesetzt in Phase 3)*
- **Format-Modi:**
  - **Modus 1:** `[<TYPE>_<N>]`
  - **Modus 2:** `[<TYPE>_<N>_<ROLE>]`
  - **Modus 3:** `[<TYPE>_<ROLE>]` *(mit automatischer (Typ, Rolle)-Kollisions-Prüfung und Fallback auf Modus 2)*
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
- Desktop-Ausführung über `ui.run(native=True)` (Windows über Microsoft WebView2). Für macOS sind Startskripte und ein Browser-Fallback vorbereitet, aber noch nicht praktisch getestet.
- CPU-intensive NLP-Berechnungen werden per `asyncio.to_thread` vom UI-Event-Loop isoliert.
- Hierarchische Baum-Ansicht (`build_entity_tree`) zur intuitiven Darstellung verknüpfter Namensvarianten mit sofortiger Trennen-Aktion.

### 2.5 Konfigurations- & OS-Architektur (`config.py`, Startskripte)
- **Persistente Einstellungen:** `~/.local-anonymizer/config.json` speichert Nutzereinstellungen defensiv (mit Fallback auf Standardwerte bei Dateifehlern).
- **Diagnose-Logging:** `~/.local-anonymizer/app.log` dient als Fehlerprotokoll bei unsichtbarem Start.
- **Windows-Integration:** `install_windows.bat` erstellt Desktop- und Startmenü-Verknüpfungen mit geräuschlosem `pythonw`-Start.
- **macOS-Vorbereitung:** `start_mac.command` und `install_mac.command` sowie ein manuell triggerbarer GitHub-Actions-Workflow (`.github/workflows/mac-test.yml`) sind vorhanden. Ein manueller End-to-End-Test auf macOS steht noch aus.

---

## 3. Sicherheits- und Datenschutzmodell

| Aspekt | Garantie | Technische Umsetzung |
| :--- | :--- | :--- |
| **Datenabfluss** | 0% externe Datenübertragung im Normalbetrieb | Thread-sicherer lokaler Cache-First-Modus (`set_huggingface_offline_mode`), temporäre Online-Schaltung nur bei explizit autorisiertem Erstdownload, kein Telemetrie-Code in NiceGUI/Presidio |
| **Persistenz** | Keine absichtliche dauerhafte Speicherung von PII durch die Pipeline | Entitäten und Mappings werden im Python-State verarbeitet. Beim Drag-and-drop kann ein Upload kurzzeitig im lokalen Ordner `temp_uploads` liegen; er wird nach dem RAM-Transfer und zusätzlich beim Beenden bereinigt. |
| **Reversibilität** | 100% mathematische Wiederherstellbarkeit | Deterministische JSON-Mappingtabelle, Single-Pass-Regex |
| **Admin-Rechte** | 0% Admin-Rechte erforderlich | Reine User-Space-Dependencies (`uv pip install`) |

---

## 4. Lokaler LLM-Triage-Layer (Phase 6A)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AppState (app.py)                              │
│  - raw_text                                                                 │
│  - analysis_revision / document_revision                                    │
│  - entity_groups (mit eindeutigen occ_id pro Occurrence)                    │
│  - llm_triage_snapshot (kryptografischer Snapshot-Hash)                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Batching & Token Budgeting (batching.py)                    │
│  - Extraktion kontextueller Snippets um occ_id                              │
│  - Dynamische Partitionierung in sequenzielle Batches (Token-Budget)        │
│  - Adversarial Escaping & Delimiter (Risikominimierung für Fremdtexte)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Local API Provider & Security (provider.py)                 │
│  - Strikte Loopback-Validierung (127.0.0.1, localhost, [::1])               │
│  - Kein Cloud-Fallback, Session-lokaler Concurrency-Lock                    │
│  - Initiales Senden von reasoning_effort: "none"                            │
│  - 1x Fallback-Retry ohne reasoning_effort bei HTTP 400/422                │
│  - 2 MB Streaming-Größenbegrenzung & PII-sichere Fehlerbehandlung           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Pydantic v2 Schema & Envelope (schema.py)                  │
│  - TriageEnvelope (schema_version="1.0", request_id, document_hash)         │
│  - Strikte Validierung: extra="forbid", str_strip_whitespace=True           │
│  - Polymorphe Items: keep, recategorize, discard                            │
│  - Aktionsabhängige Null-Typen (Optional[Literal[None]])                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ApplyService & Rollback (apply_service.py)                  │
│  - Snapshot-Hash-Validierung gegen State-Drift                              │
│  - Vorvalidierung kanonischer Entitätstypen & Auswirkungsberechnung         │
│  - Atomare Mutation von EntityGroup / Occurrences & Group-Splits            │
│  - Automatischer Rollback bei unerwartetem Fehler                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kernprinzipien des Triage-Layers

1. **Strikte Datenschutz- und Netzwerkgrenze:**
   - Provider akzeptiert ausschliesslich lokale Loopback-Verbindungen. LAN-IPs oder Remote-URLs werfen `ValueError`.
   - Bei Modellfehlern oder ungültigem JSON werden keine Modell-Rohantworten oder Validierungsdaten in Logs oder Konsole geschrieben.
2. **Kryptografischer Drift-Schutz (Snapshot-Binding):**
   - Jede Anfrage bindet den SHA-256-Hash des Rohtexts, die Revisionsnummer und die Entitätenkonfiguration (`compute_triage_snapshot`).
   - Ändert der Nutzer den Text während der Analyse, werden verspätet eintreffende Batches sicher verworfen.
3. **Deterministische Batches:**
   - Große Dokumente werden in sequenzielle Batches aufgeteilt.
   - Bricht ein Batch ab, verbleiben ungeprüfte Fundstellen im Zustand `unprocessed`. Sammelübernahmen werden gesperrt; Einzelübernahmen bleiben möglich.
4. **Human-in-the-Loop & Atomare Übernahme:**
   - LLM-Ergebnisse sind reine Vorschläge in `llm_triage_results`.
   - Vor jeder Mutation öffnet sich ein Auswirkungsdialog mit unverbindlicher Vorschau.
   - Die Übernahme erfolgt atomar über `ApplyService.apply_mutations` mit automatischem Rollback.

---

## 5. Abgrenzung: Warum keine echte PDF-Content-Stream-Redaktion?

> [!NOTE]
> **Bewusste Scope-Entscheidung:**
> `local-anonymizer` führt **keine** native PDF-Content-Stream-Redaktion durch (d. h. keine direkte Entfernung von Vektorglyphen oder Pixel-Übermalung in binären PDF-Dateien). 
> 
> **Begründung:** Der Einsatzzweck des Tools ist ein reiner **Prompt-Privacy-Layer für KI-Workflows**: Textinhalte werden aus Dokumenten extrahiert, lokal bereinigt, an ein LLM übergeben und die LLM-Antwort anschliessend de-anonymisiert. Das Veröffentlichen optisch geschwärzter Original-PDFs ist nicht Gegenstand dieses Tools. Das text- und markdown-basierte Vorgehen stellt sicher, dass keinerlei unsichtbare PDF-Metadaten oder versteckte Textlayer unbemerkt an das LLM übermittelt werden können.
