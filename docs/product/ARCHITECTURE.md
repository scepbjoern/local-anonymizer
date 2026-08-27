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
- **`extract_text_from_pdf_bytes`:** Nutzt `pymupdf4llm` zur Umwandlung von PDF-Seiten in semantisch strukturiertes Markdown (inklusive Tabellenbereinigung, Trennung von Bildtext und wiederkehrenden Kopf-/Fußzeilen-Filtern mit Seite-1-Titelschutz).
- **In-Memory & Streaming-Puffer:** Sämtliche Extraktoren und NLP-Analysen arbeiten nativ auf `bytes` (`read_document_from_bytes`). Beim Drag-and-Drop im GUI wird zur zuverlässigen Übertragung großer Dokumente (bis 50 MB) ein temporärer HTTP-Streaming-Puffer (`~/.local-anonymizer/temp_uploads`) verwendet, der per `try...finally` sofort nach dem RAM-Ladevorgang sowie über `atexit` beim Session-Ende bereinigt wird.
- **Fehlerbehandlung:** Bildbasierte PDFs ohne Textlayer werden über `doc.page_count > 0 and not pages_text` erkannt und werfen `ValueError` mit klarer OCR-Hinweismeldung.

### 2.2 Presidio Analyzer & Custom Recognizers (`recognizers.py`)
- **`GLiNERRecognizer`:**
  - Kapselt `urchade/gliner_multi_pii-v1`.
  - Hält ein Singleton-Klassen-Cache `_MODEL_CACHE`, sodass das PyTorch-Modell nur ein einziges Mal in den RAM geladen wird.
  - Wendet `chunk_text_with_offsets` an, um das 384-Token-Limit von DeBERTa/GLiNER ohne Informationsverlust zu handhaben.
  - Verwendet für `PERSON` präzisere Prompts wie `person's proper name`, `named person` und `proper name`, um generische Rollennomen weniger häufig zu erfassen. `ROLE`/`JOB_TITLE` nutzt die Prompts `job title`, `professional role` und `position` in einem getrennten Modellpass und ist standardmässig deaktiviert.
- **`is_sentence_boundary`:**
  - Schützt deutsche und englische Standardabkürzungen (`Dr.`, `Prof.`, `Bahnhofstr.`, `Nr.`, `14. Juli`), damit Sätze nicht mitten im Eigennamen zerschnitten werden.
- **`FuzzyGlossaryRecognizer`:**
  - Nutzt `rapidfuzz.fuzz.ratio` mit Schwellenwerten (High Confidence $\ge 90\%$, Review-Bedarf $\ge 75\%$).
  - Bevorzugt exakte Treffer (Score 1.0) vor Fuzzy-Treffern.
  - Explizite Glossar-Treffer werden gemäss der Kategorie-Richtlinie separat zugelassen oder vollständig blockiert. Sie können eingebaute generische Ignore-Begriffe bewusst überschreiben; persönliche Ignore-Einträge behalten immer Vorrang.
- **Deterministische Phase-B-Recognizer:**
  - `AddressPatternRecognizer` erkennt zusammenhängende Schweizer und deutsche Adressen per Regex und weist die bekannte Kollision zwischen vierstelliger Schweizer PLZ und Jahreszahl konservativ zurück.
  - `AHVNumberRecognizer` validiert die AHV-Kontrollziffer; `UIDNumberRecognizer` validiert CHE/UID nach Modulo 11. Formal korrekte, aber prüfziffern-ungültige Nummern werden nicht als Entitäten ausgegeben.
  - `IT_SYSTEM` wird über das dynamisch aus dem Glossar abgeleitete Zieltypenset sowie separate GLiNER-Sicherheitsnetz-Prompts erkannt.

Die optionale Kategorie `ROLE` wird wie andere Kategorien über die drei UI-Modi gesteuert und standardmässig nicht aktiviert. Dadurch bleiben fachlich wichtige Formulierungen wie „Der Sachbearbeiter prüft …“ standardmässig erhalten; bei erhöhtem Identifikationsrisiko können Funktionsbezeichnungen wie `CEO` oder `Leiter Prozessmanagement` gezielt anonymisiert werden. Generische Rollennomen werden zusätzlich über die eingebaute Ignore-Liste gegen typische PERSON-Falschpositive geschützt.

Die UI-Schalter für Entitätstypen verwenden drei Modi: `Aus` blockiert alle Quellen, `Nur Glossar & manuell` deaktiviert KI-/Bibliotheks-/Regex-Erkennung und `Alle Quellen` aktiviert sämtliche Quellen. Glossar-Treffer werden in einem getrennten, direkten Pass geprüft; dadurch kann beispielsweise `IT_SYSTEM` auf „Nur Glossar & manuell“ stehen, während ein vollständiges „Aus“ auch `SAP` blockiert. Manuelle, dokumentbezogene Markierungen folgen derselben Kategorie-Richtlinie.

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
- Desktop-Ausführung über `ui.run(native=True)` (nutzt Microsoft WebView2 auf Windows bzw. WebKit/Cocoa auf macOS via `pywebview`).
- CPU-intensive NLP-Berechnungen werden per `asyncio.to_thread` vom UI-Event-Loop isoliert.
- Hierarchische Baum-Ansicht (`build_entity_tree`) zur intuitiven Darstellung verknüpfter Namensvarianten mit sofortiger Trennen-Aktion.

### 2.5 Konfigurations- & OS-Architektur (`config.py`, Startskripte)
- **Persistente Einstellungen:** `~/.local-anonymizer/config.json` speichert Nutzereinstellungen defensiv (mit Fallback auf Standardwerte bei Dateifehlern).
- **Diagnose-Logging:** `~/.local-anonymizer/app.log` dient als Fehlerprotokoll bei unsichtbarem Start.
- **Windows-Integration:** `install_windows.bat` erstellt Desktop- und Startmenü-Verknüpfungen mit geräuschlosem `pythonw`-Start.
- **macOS-Integration:** `start_mac.command` und `install_mac.command` bieten native Startskripte. `pyproject.toml` bindet `pywebview>=5.0.0` ein, was macOS-spezifische Cocoa-Bindings out-of-the-box mitliefert.
- **Automatisierte Mac-Prüfung:** GitHub Actions Workflow (`.github/workflows/mac-test.yml`) für manuell triggerbare Tests auf `macos-latest`.

---

## 3. Sicherheits- und Datenschutzmodell

| Aspekt | Garantie | Technische Umsetzung |
| :--- | :--- | :--- |
| **Datenabfluss** | 0% externe Datenübertragung | Offline-Modus für HuggingFace (`HF_HUB_OFFLINE=1`), kein Telemetrie-Code in NiceGUI/Presidio |
| **Persistenz** | Keine PII auf dauerhaften Datenträgern | Reine RAM-Verarbeitung für Entitäten & Mappings (`io.BytesIO`, Python-State). Temporärer Drag-and-Drop HTTP-Puffer (`temp_uploads`) wird per `try...finally` sofort nach RAM-Transfer und via `atexit` bereinigt. |
| **Reversibilität** | 100% mathematische Wiederherstellbarkeit | Deterministische JSON-Mappingtabelle, Single-Pass-Regex |
| **Admin-Rechte** | 0% Admin-Rechte erforderlich | Reine User-Space-Dependencies (`uv pip install`) |

---

## 4. Abgrenzung: Warum keine echte PDF-Content-Stream-Redaktion?

> [!NOTE]
> **Bewusste Scope-Entscheidung:**
> `local-anonymizer` führt **keine** native PDF-Content-Stream-Redaktion durch (d. h. keine direkte Entfernung von Vektorglyphen oder Pixel-Übermalung in binären PDF-Dateien). 
> 
> **Begründung:** Der Einsatzzweck des Tools ist ein reiner **Prompt-Privacy-Layer für KI-Workflows**: Textinhalte werden aus Dokumenten extrahiert, lokal bereinigt, an ein LLM übergeben und die LLM-Antwort anschliessend de-anonymisiert. Das Veröffentlichen optisch geschwärzter Original-PDFs ist nicht Gegenstand dieses Tools. Das text- und markdown-basierte Vorgehen stellt sicher, dass keinerlei unsichtbare PDF-Metadaten oder versteckte Textlayer unbemerkt an das LLM übermittelt werden können.
