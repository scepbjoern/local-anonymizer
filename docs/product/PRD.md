# Product Requirements Document (PRD)

## 1. Vision & Ziel

`local-anonymizer` ist eine datenschutzkonforme, 100% lokal lauffähige Software zur **Anonymisierung und De-Anonymisierung von Dokumenten und Texten** für KI- und LLM-Workflows. 

Das Tool ermöglicht es Studierenden, Dozierenden und Wissensarbeitern, vertrauliche Dokumente (Notizen, Verträge, Projektberichte, Gutachten) sicher von personenbezogenen Daten (PII) und unternehmensinternen Begriffen zu bereinigen, bevor diese an externe Cloud-LLMs (z. B. ChatGPT, Claude, Microsoft Copilot) gesendet werden. Nach der Verarbeitung durch das LLM stellt das Tool die Originalnamen und -begriffe über eine lokale, geheime Mapping-Tabelle mit einem Klick vollständig und verlustfrei wieder her.

---

## 2. Kernprinzipien & Design-Vorgaben

1. **🛡️ 100% Lokal & Zero-Telemetry:** Weder Dokumenteninhalte, noch Metadaten oder Telemetrie verlassen das Gerät.
2. **💾 Reine In-Memory-Verarbeitung:** Hochgeladene Dateien und extrahierte Klartexte werden ausschliesslich im RAM gehalten und niemals ungefragt in temporären Verzeichnissen auf der Festplatte abgelegt.
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

### 4.1 Feature 1: Strukturierte Dokumenten- & Markdown-Extraktion *(Geplant für Phase 3 / In Prüfung)*

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

### 4.2 Feature 2: Hybride Entitätserkennung (Zero-Shot NER & Glossar) *(Umgesetzt in Phase 1 & 2)*

* **Multilinguales Zero-Shot NER:** Basiert auf [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi_pii-v1`) zur Erkennung von:
  * `PERSON`, `ORGANIZATION`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `LOCATION`, `DATE_TIME`, `IBAN_CODE`, `CREDIT_CARD`, `ID_NUMBER`, `FINANCIAL_DATA`, `HEALTH_DATA`, `IP_ADDRESS`.
* **Abkürzungs-bewusstes Chunking:** Zerlegung langer Texte in überlappungsfreie Abschnitte (<800 Zeichen), ohne Satzgrenzen bei typischen Abkürzungen (`Dr.`, `Prof.`, `Bahnhofstr.`, `14. Juli`) zu zerschneiden.
* **Fuzzy-Glossar (RapidFuzz):**
  * Zuordnung interner Firmenkürzel (z. B. `"abcd"` $\rightarrow$ `PERSON`).
  * Fehlertolerantes Matching bei Tippfehlern (z. B. `"ZHW"` $\rightarrow$ `"ZHAW"`).
* **Globale & Session-Ignore-Listen:** Schutz generischer Rollen und Grade (`CAS`, `BSc`, `Studierende`, `Dozent`, `Unternehmen`).

---

### 4.3 Feature 3: Semantische Rollen-Labels & Format-Modi *(Geplant für Phase 3 / In Prüfung)*

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
  > **Schutzmechanismus:** Das System führt bei Modus 3 eine automatische Eindeutigkeits-Prüfung durch. Bei Rollen-Mehrfachvergabe wird eine Warnung ausgegeben und für die betroffenen Entitäten automatisch auf Modus 2 (`[PERSON_1_STUDENT]`, `[PERSON_2_STUDENT]`) zurückgefallen, um die 100%ige Reversibilität und LLM-Differenzierung sicherzustellen.

---

### 4.4 Feature 4: Co-Referenz & Entity-Linking (Schreibweisen-Verknüpfung) *(Geplant für Phase 3 / In Prüfung)*

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
  * **Explizite „Trennen“-Funktion:** Fälschlich vorgeschlagene Verknüpfungen (z. B. zwei verschiedene Personen namens „Julia Meier“ und „Julia Suter“) können per Klick auf **„Trennen / Als eigenständige Person behandeln“** sofort gelöst werden. Dadurch erhalten beide Personen eigenständige IDs (`PERSON_1` und `PERSON_2`).

---

### 4.5 Feature 5: Interaktives Review-GUI (NiceGUI)

* **Single-Page Application (`app.py`):**
  * Start als eigenständiges natives Desktop-Fenster (`native=True` via pywebview/WebView2) oder im Webbrowser (`--browser`).
* **Gebündelte Entitäten-Ansicht:** Identische Begriffe (z. B. 8x „Julia“) werden auf eine einzige Zeile mit Mengenzähler (`8x`) aggregiert.
* **Aufklappbare Kontext-Vorschau:** Akkordeon-Ansicht zeigt alle Fundstellen im Satzkontext mit visueller Hervorhebung (`... für <mark>Julia</mark> im Herbst ...`).
* **Sortier-Toolbar:** Umschaltbar nach *Erstes Auftreten*, *Häufigkeit*, *Entitätstyp*, *Alphabetisch* und *Review-Bedarf*.
* **Reaktive Live-Vorschau:** Text-Vorschau aktualisiert sich bei jedem Klick in der Tabelle ohne Verzögerung im Arbeitsspeicher.
* **Desktop-Export-Aktionen:**
  * 📋 *In Zwischenablage kopieren*
  * 💾 *Native Windows-Dateidialoge („Speichern unter...“)*
  * 📁 *1-Klick-Export aller 3 Dateien (`_anonymized.txt`, `_mapping.json`, `_report.json`) in einen Zielordner inkl. „Im Explorer öffnen“-Aktion.*
* **De-Anonymisierungs-Tab:** Wiederherstellung der Originaltexte durch Hochladen der LLM-Antwort und der lokalen Mapping-Datei.

---

### 4.6 Feature 6: Deterministische 2-Wege De-Anonymisierung

* **Single-Pass Regex-Substitution:** Platzhalter werden nach Längen absteigend sortiert und in einem einzigen Regex-Durchlauf ersetzt, um Kaskadierungsfehler (Überschreiben von Platzhaltern innerhalb ersetzter Klartexte) mathematisch auszuschliessen.
* **Audit-Report:** JSON-Bericht mit Dokument-Metadaten, Entitätszählern, Konfidenzen und Review-Status.

---

## 5. Roadmap & Phasen

### ✅ Phase 1: Core Engine & Härtung (Abgeschlossen)
* Presidio-GLiNER Pipeline, RapidFuzz Glossar, Multi-Encoding Extraktoren, Single-Pass De-Anonymisierung, Behebung aller 10 Code-Review Findings.

### ✅ Phase 2: Testsuite & NiceGUI Review-GUI (Abgeschlossen)
* 15/15 Pytest Regressionstests, NiceGUI Desktop-Anwendung, In-Memory File Handling, Entitäten-Bündelung, Kontext-Akkordeons, Sortier-Toolbar, nativer Datei-Export.

### ⏳ Phase 3: Semantische Rollen, Entity-Linking & Markdown-Extraktion (Aktuell)
* Einbindung von `pymupdf4llm` für formatierte Markdown-Extraktion aus PDFs.
* Umsetzung der 3 Platzhalter-Format-Modi (`[TYP_NUMMER]`, `[TYP_NUMMER_ROLLE]`, `[TYP_ROLLE]`).
* Implementierung des Entity-Linkings für Schreibweisen (`VOLLNAME`, `VORNAME`, `ANREDE`, `KURZFORM`) im Core und im GUI.

### 🔮 Phase 4: Lokaler SLM-Triage-Layer (Geplant / Ausblick)
* Optionaler Triage-Filter mit lokalem SLM (z. B. via Ollama / LM Studio) zur automatischen Vorfilterung von Grenzfall-Entitäten.
