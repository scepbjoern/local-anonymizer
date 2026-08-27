# Product Requirements Document (PRD)

## 1. Vision & Ziel

`local-anonymizer` ist eine datenschutzkonforme, 100% lokal lauffähige Software zur **Anonymisierung und De-Anonymisierung von Dokumenten und Texten** für KI- und LLM-Workflows. 

Das Tool ermöglicht es Studierenden, Dozierenden und Wissensarbeitern, vertrauliche Dokumente (Notizen, Verträge, Projektberichte, Gutachten) sicher von personenbezogenen Daten (PII) und unternehmensinternen Begriffen zu bereinigen, bevor diese an externe Cloud-LLMs (z. B. ChatGPT, Claude, Microsoft Copilot) gesendet werden. Nach der Verarbeitung durch das LLM stellt das Tool die Originalnamen und -begriffe über eine lokale, geheime Mapping-Tabelle mit einem Klick vollständig und verlustfrei wieder her.

---

## 2. Kernprinzipien & Design-Vorgaben

1. **🛡️ 100% Lokal & Zero-Telemetry:** Weder Dokumenteninhalte, noch Metadaten oder Telemetrie verlassen das Gerät.
2. **💾 RAM-zentrierte Verarbeitung & transparenter Streaming-Puffer:** Erkannte Entitäten, Mapping-Tabellen und Texttransformationen verbleiben ausschliesslich im RAM. Bei nativer Datei-Auswahl (`tkinter`) erfolgt das Einlesen direkt ohne temporäre Dateien. Beim Drag-and-Drop im GUI wird die Datei über einen lokalen HTTP-Streaming-Puffer (`~/.local-anonymizer/temp_uploads`, max. 50 MB) gestreamt und unmittelbar nach dem Einlesen in den RAM per `try...finally` sofort gelöscht sowie beim Beenden der Anwendung (`atexit`) automatisch bereinigt.
3. **⚡ No-Admin / Geringer Footprint:** Installation ohne Administratorrechte (via `uv` oder `pip`), lauffähig auf Standard-Notebooks (CPU-only).
4. **🔄 Deterministische Reversibilität:** Mathematisch exakte, kaskadenfreie Wiederherstellung (Single-Pass-Substitution).
5. **🎯 High-Recall mit interaktiver Kontrolle:** Das Erkennungsmodell findet im Zweifel lieber zu viel als zu wenig; der Mensch behält im visuellen Review-Interface mühelos die Letztentscheidung.

---

## 3. Zielgruppen & Anwendungsfälle

* **Studierende & Weiterbildungsteilnehmer (z. B. CAS/MAS):** Anonymisierung von realen Firmen-Aktennotizen, Interviewtranskripten oder Masterarbeiten zur Analyse durch Cloud-KIs ohne Verletzung von NDAs.
* **Dozierende & Prüfende:** Bereinigung von Gutachten, Prüfungsdokumenten und Korrektur-Aufzeichnungen.
* **Unternehmen & Berater:** Bereinigung von vertraulichen Kundenprotokollen vor dem LLM-Prompting.

---

## 4. Detaillierte Feature-Spezifikationen

### 4.1 Feature 1: Strukturierte Dokumenten- & Markdown-Extraktion *(Umgesetzt)*

* **Unterstützte Formate:** `.txt`, `.md`, `.docx`, `.pdf`, `.json`, `.csv`.
* **PDF-zu-Markdown (`pymupdf4llm`):**
  * Beibehaltung der Dokumentstruktur:
    * Überschriften werden anhand der Schriftgrössen automatisch als `# H1`, `## H2`, `### H3` formatiert.
    * Tabellen werden in echte Markdown-Tabellen (`| Spalte 1 | Spalte 2 |`) konvertiert.
    * Listen und Aufzählungen (`- Item`, `1. Item`) bleiben strukturiert erhalten.
    * Textauszeichnungen (`**fett**`, `*kursiv*`) bleiben markiert.
* **Word-Extraktion (`python-docx`):** Strukturierte Extraktion von Absätzen und Tabelleninhalten.
* **Robustes Encoding:** Automatisches Fallback über `utf-8-sig`, `utf-8`, `cp1252`, `iso-8859-15` und `latin-1`.
* **Scan-PDF-Erkennung:** Klare Nutzerwarnung (`ValueError`), falls ein PDF rein bildbasiert ist und OCR benötigen würde.

---

### 4.2 Feature 2: Hybride Entitätserkennung (Zero-Shot NER & Glossar) *(Umgesetzt)*

* **Multilinguales Zero-Shot NER:** Basiert auf [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi_pii-v1`) zur Erkennung von:
  * `PERSON`, `ORGANIZATION`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `LOCATION`, `DATE_TIME`, `IBAN_CODE`, `CREDIT_CARD`, `ID_NUMBER`, `FINANCIAL_DATA`, `HEALTH_DATA`, `IP_ADDRESS`, `IT_SYSTEM` und optional `ROLE`.
* **Abkürzungs-bewusstes Chunking:** Zerlegung langer Texte in überlappungsfreie Abschnitte (<800 Zeichen), ohne Satzgrenzen bei typischen Abkürzungen (`Dr.`, `Prof.`, `Bahnhofstr.`, `14. Juli`) zu zerschneiden.
* **Fuzzy-Glossar (RapidFuzz):**
  * Zuordnung interner Firmenkürzel (z. B. `"abcd"` $\rightarrow$ `PERSON`).
  * Fehlertolerantes Matching bei Tippfehlern (z. B. `"ZHW"` $\rightarrow$ `"ZHAW"`).
* **Globale & Session-Ignore-Listen:** Schutz generischer Rollen, Grade und Feldbezeichnungen (`CAS`, `BSc`, `Studierende`, `Dozent`, `Unternehmen`, `E-Mail`, `App`, `Applikation`). Eingebaute Standard-Ignores können durch bewusste Glossar-Einträge überschrieben werden; persönliche Ignore-Einträge behalten immer Vorrang.
* **Dreistufige Erkennungssteuerung:** `Aus` blockiert alle Quellen einer Kategorie, `Nur Glossar & manuell` lässt nur explizite und manuelle Einträge zu, `Alle Quellen` aktiviert KI-, Bibliotheks-, Regex- und explizite Erkennung.
* **Deterministische Schweizer Erweiterungen:** `ADDRESS` erkennt Schweizer und deutsche Adressmuster; `AHV_NUMBER` wird mit der AHV-Kontrollziffer und `UID_NUMBER` mit der offiziellen Modulo-11-Prüfziffer validiert. `IT_SYSTEM` nutzt das Glossar als Primärquelle und GLiNER-Prompts als Sicherheitsnetz.
* **Optionale Rollen-Erkennung:** Präzisere PERSON-Prompts reduzieren Falschpositive bei generischen Rollennomen. `ROLE`/`JOB_TITLE` erkennt Funktionsbezeichnungen, ist aber standardmässig deaktiviert und kann bei Bedarf über KI oder Glossar aktiviert werden.

---

### 4.3 Feature 3: Semantische Rollen-Labels & Format-Modi *(Umgesetzt)*

Um dem nachgelagerten Cloud-LLM optimalen semantischen Kontext zu vermitteln, ohne Klarnamen preiszugeben, unterstützt die Pipeline flexible Platzhalterformate:

| Modus | Name | Schema | Beispiel |
| :--- | :--- | :--- | :--- |
| **Modus 1** | Standard (Nur Nummer) | `[TYP_NUMMER]` | `[PERSON_1]`, `[ORGANIZATION_1]` |
| **Modus 2** | Typ + Nummer + Rolle *(Empfohlen)* | `[TYP_NUMMER_ROLLE]` | `[PERSON_1_STUDENT]`, `[ORGANIZATION_1_ZULIEFERER]` |
| **Modus 3** | Typ + Rolle | `[TYP_ROLLE]` | `[PERSON_STUDENT]`, `[ORGANIZATION_ZULIEFERER]` |

* **Granulare Steuerung:** Jede erkannte Entität kann in der Review-Tabelle mit einer individuellen Rollenbezeichnung versehen werden (z. B. `Student`, `Dozent`, `Eigentümer`, `Kunde`).
* **Fallback:** Bleibt das Rollenfeld leer, greift automatisch Modus 1 mit fortlaufender Nummerierung.
* **Kollisions-Schutz (Modus 3):**
  > [!IMPORTANT]
  > Modus 3 lässt bewusst die Nummer weg. Tragen jedoch **mehrere unterschiedliche Entitäten dieselbe Rolle** (z. B. zwei verschiedene Studierende, beide mit Rolle `STUDENT`), entsteht ein Informationsverlust und ein Widerspruch in der Mapping-Tabelle. 
  > **Schutzmechanismus:** Das System führt bei Modus 3 eine automatische `(Typ, Rolle)`-Eindeutigkeits-Prüfung durch. Bei Rollen-Mehrfachvergabe wird eine Warnung ausgegeben und für die betroffenen Entitäten automatisch auf Modus 2 (`[PERSON_1_STUDENT]`, `[PERSON_2_STUDENT]`) zurückgefallen, um die 100%ige Reversibilität und LLM-Differenzierung sicherzustellen.

---

### 4.4 Feature 4: Co-Referenz & Entity-Linking (Schreibweisen-Verknüpfung) *(Umgesetzt)*

Personen und Organisationen treten in realen Texten oft in unterschiedlichen Schreibweisen auf (z. B. *„Julia Meier“*, *„Julia“*, *„Frau Meier“*).

* **Problem:** Ein einfaches Zusammenführen auf denselben Platzhalter verhindert das exakte Wiederherstellen der ursprünglichen Textform.
* **Lösung via Co-Referenz-Tags:**
  * Das System verknüpft Varianten mit derselben Haupt-Entitätsnummer (`PERSON_1`).
  * Die spezifische Oberflächenform wird als Tag angehängt:
    * *„Julia Meier“* $\rightarrow$ `[PERSON_1_STUDENT_VOLLNAME]`
    * *„Julia“* $\rightarrow$ `[PERSON_1_STUDENT_VORNAME]`
    * *„Frau Meier“* $\rightarrow$ `[PERSON_1_STUDENT_ANREDE]`
  * **Nutzen:** Das LLM versteht, dass es sich um dieselbe Identität handelt, behält aber die grammatikalische Satzstruktur bei. Beim De-Anonymisieren wird jede Stelle mit 100%iger Exaktheit in ihrer ursprünglichen Schreibweise wiederhergestellt.
* **Smart Linking & Undo/Trennen-Funktion:**
  * **Automatischer Vorschlag:** Kürzere Begriffe (z. B. „Julia“) schlagen bei Namensübereinstimmung eine Verknüpfung mit der Langform vor.
  * **Explizite „Trennen“-Funktion:** Fälschlich vorgeschlagene Verknüpfungen (z. B. zwei verschiedene Personen namens „Julia Meier“ und „Julia Suter“) können per Klick auf **„✕ Trennen“** sofort gelöst werden. Dadurch erhalten beide Personen eigenständige IDs (`PERSON_1` und `PERSON_2`).

---

### 4.5 Feature 5: Interaktives Review-GUI (NiceGUI) *(Umgesetzt)*

* **Single-Page Application (`app.py`):**
  * Start als eigenständiges natives Desktop-Fenster (`native=True` via pywebview/WebView2 auf Windows) oder im Webbrowser (`--browser`). Für macOS sind Startskripte und ein Browser-Fallback vorbereitet, aber noch nicht praktisch getestet.
* **Hierarchische Tree-View:** Klare optische Strukturierung: Hauptidentitäten oben, verknüpfte Schreibweisen (`_VORNAME`, `_ANREDE`) visuell eingerückt darunter mit sofortiger `✕ Trennen`-Aktion.
* **Transparente Platzhalter-Badges:** Jede Tabellenzeile zeigt direkt den final zugeordneten Platzhalter (z. B. `[PERSON_1_STUDENT]`).
* **Manuelle Entitätserfassung:** Fehlende Begriffe (Falsch-Negative des NER) können direkt per 1-Klick-Eingabe (`"Remo: PERSON"`) im gesamten Text als Entität forciert werden.
* **Gebündelte Entitäten-Ansicht:** Identische Begriffe (z. B. 8x „Julia“) werden auf eine einzige Zeile mit Mengenzähler (`8x`) aggregiert.
* **Aufklappbare Kontext-Vorschau:** Akkordeon-Ansicht zeigt alle Fundstellen im Satzkontext mit visueller Hervorhebung (`... für <mark>Julia</mark> im Herbst ...`).
* **Sortier-Toolbar:** Umschaltbar nach *Erstes Auftreten*, *Häufigkeit*, *Entitätstyp*, *Alphabetisch* und *Review-Bedarf*.
* **Transparente Fundstellen:** Jede Fundstelle zeigt ihre Erkennungsmethode (`KI`, `Regex`, `Bibliothek`, `Glossar direkt`, `Glossar Fuzzy` oder `manuell`). Gruppenaktionen erlauben Ignorieren, dauerhaftes Übernehmen ins Glossar oder eine nur für den aktuellen Durchlauf gültige manuelle Markierung.
* **Persistente Einstellungen (`config.json`):** Format-Modus, aktivierte Entitäten, Schwellenwerte, Ignore-Listen und Glossare werden automatisch in `~/.local-anonymizer/config.json` gespeichert.
* **Desktop-Export-Aktionen:**
  * 📋 *In Zwischenablage kopieren*
  * 💾 *Format-Wahl (`.txt` oder `.md`)*
  * 📁 *1-Klick-Export aller 3 Dateien (`_anonymized.*`, `_mapping.json`, `_report.json`) in einen Zielordner.*

---

### 4.6 Feature 6: Deterministische 2-Wege De-Anonymisierung *(Umgesetzt)*

* **Single-Pass Regex-Substitution:** Platzhalter werden nach Längen absteigend sortiert und in einem einzigen Regex-Durchlauf ersetzt, um Kaskadierungsfehler mathematisch auszuschliessen.
* **Dokumenten-Upload im Restore-Tab:** LLM-Antworten können direkt als `.docx`, `.md` oder `.txt` hochgeladen werden.
* **Automatischer Mapping-Transfer:** Das im ersten Tab aktive Mapping wird automatisch als Standard im Restore-Tab vorgeschlagen.
* **Word-Export mit echten Formatvorlagen:** Wiederhergestellte Dokumente können als echtes `.docx` mit nativen Absatzformatvorlagen (`Heading 1`, `Heading 2`, `List Bullet`, `Table Grid`) exportiert werden.

---

### 4.7 Feature 7: Plattform- & OS-Integration *(Umgesetzt)*

* **Windows-Integration:** `install_windows.bat` und `start_windows.bat` ermöglichen den Start im Hintergrund (Silent Mode via `pythonw`) mit Desktop- und Startmenü-Verknüpfung sowie Logging unter `~/.local-anonymizer/app.log`.
* **macOS-Vorbereitung:** `start_mac.command` und `install_mac.command` sowie ein manueller GitHub-Actions-Workflow (`.github/workflows/mac-test.yml`) sind vorhanden. Die Anwendung wurde bisher nicht manuell auf macOS getestet; eine vollständige Plattformfreigabe liegt daher noch nicht vor.

---

## 5. Roadmap & Phasen

### ✅ Phase 1: Core Engine & Härtung (Abgeschlossen)
* Presidio-GLiNER Pipeline, RapidFuzz Glossar, Multi-Encoding Extraktoren, Single-Pass De-Anonymisierung, Behebung aller 10 Code-Review Findings.

### ✅ Phase 2: Testsuite & NiceGUI Review-GUI (Abgeschlossen)
* 15/15 Pytest Regressionstests, NiceGUI Desktop-Anwendung, In-Memory File Handling, Entitäten-Bündelung, Kontext-Akkordeons, Sortier-Toolbar, nativer Datei-Export.

### ✅ Phase 3: Semantische Rollen, Entity-Linking & Markdown-Extraktion (Abgeschlossen)
* 21/21 Pytest Regressionstests bestanden.
* Einbindung von `pymupdf4llm` für formatierte Markdown-Extraktion aus PDFs.
* Umsetzung der 3 Platzhalter-Format-Modi (`[TYP_NUMMER]`, `[TYP_NUMMER_ROLLE]`, `[TYP_ROLLE]` mit Kollisionsschutz).
* Co-Referenz-Tags (`VOLLNAME`, `VORNAME`, `NACHNAME`, `KURZFORM`, `ANREDE`).

### ✅ Phase 4: UX-Optimierungen, Tree-View, Genitiv- & Word-Export (Abgeschlossen)
* 27/27 Pytest Regressionstests bestanden (100%).
* Platzhalter-Badges in Tabelle & Baumansicht (Tree-View) für verknüpfte Schreibweisen.
* Genitiv-Erkennung ("Julias", "Meiers") mit Falsch-Positiv-Schutz.
* 1-Klick Manuelle Entitätserfassung ("Remo: PERSON").
* Word-zu-Markdown Überschriften-Extraktion & Markdown-zu-Docx Export mit echten Formatvorlagen.
* Persistente Konfiguration (`config.json`) & Silent-Mode Windows Launcher (`pythonw`).
* macOS Launch-Skripte & manueller GitHub Actions Mac-Workflow.

### 🔮 Phase 5a: Homonym-Zuordnung pro Fundstelle (Geplant nach CAS-Deliverable)
* Granulare Disambiguierung identischer Textstellen innerhalb eines Dokuments direkt über Einzelauswahlen in den aufklappbaren Kontext-Akkordeons des Review-GUIs.

### 🔮 Phase 5b: Projekt-Registry für dokumentübergreifende Mappings (Geplant nach CAS-Deliverable)
* Verschlüsselte, strikt projekt- und kontextbezogene Mapping-Registry (Passphrase-geschützt via KeePass) zur Wiederverwendung konsistenter Pseudonyme über mehrere Dokumente hinweg.
* Projektbezogene Glossare, Ignore-Listen, Entitätseinstellungen und manuell vergebene Rollen, beispielsweise `LEHRPERSON` oder `STUDIENGANGSLEITUNG`, sollen über mehrere Dokumente einer CAS-Abschlussarbeit konsistent verfügbar sein.
* Eine kontrollierte projektübergreifende Wiederverwendung kann später geprüft werden; Speicherform und genaue Berechtigungsgrenzen sind noch offen.

### 🔮 Phase 6: Lokaler SLM-Triage-Layer (Geplant / Ausblick)
* Optionaler lokaler LLM-/SLM-Prüflayer (z. B. via Ollama / LM Studio), der Begriffe, Kontext und Tabellenentscheidung gemeinsam bewertet und Empfehlungen für Entitätstyp, Anonymisierung, Glossar und Rolle abgibt.
* Unterstützung bei schwierigen Fällen sowie optionaler Sicherheitscheck des bereits anonymisierten Dokuments auf verbliebene sensible Angaben.
