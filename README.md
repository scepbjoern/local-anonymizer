# local-anonymizer 🔒

> **Privacy-first, local PII anonymization and de-anonymization pipeline for AI workflows.**

`local-anonymizer` is a lightweight, local-first Desktop GUI, CLI tool, and Python library that scrubs Personally Identifiable Information (PII) and company-specific sensitive entities from documents and text before sending them to cloud or local LLMs. Once the LLM completes its task (e.g., summarization, rewriting, or extraction), `local-anonymizer` seamlessly restores the original entities using a secure, local-only mapping table.

---

## ✨ Key Features

- **🛡️ 100% Local & Private:** All entity detection, mapping, and text transformations run completely on your local machine. No data or telemetry leaves your computer.
- **🤖 Zero-Shot Multilingual PII Detection:** Powered by [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi_pii-v1`) with intelligent sentence-boundary-aware chunking for German and English PII extraction without document length limits.
- **🏷️ 3 Semantic Placeholder Modes:**
  - **Modus 1 (Classic):** `[PERSON_1]`, `[ORGANIZATION_1]`
  - **Modus 2 (Numbered + Role, Recommended):** `[PERSON_1_STUDENT]`, `[ORGANIZATION_1_ZULIEFERER]`
  - **Modus 3 (Role Only, Compact):** `[PERSON_STUDENT]` (with automatic collision detection & fallback if multiple entities share the same role).
- **🔗 Smart Entity Linking & Co-Reference Resolution:** Links first names (*"Julia"*), honorifics (*"Frau Meier"*), and genitives (*"Julias"*) to master identities (*"Julia Meier"*) using explicit surface tags (`_VOLLNAME`, `_VORNAME`, `_NACHNAME`, `_ANREDE`, `_GENITIV`, `_KURZFORM`) for 100% exact, grammatically consistent restoration.
- **🚻 Gender-Aware Entity Recognition:** Automatically handles German gendered occupational and role suffixes (`-in`, `:in`, `*in`, `_in`, `/in`, `Innen`, `innen`) to prevent truncated or missed entity spans.
- **🔍 Fuzzy Glossary & Overrides:** Uses [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) to catch custom internal terms, project codes, employee acronyms (e.g., mapping `"abcd"` -> `PERSON`), and typos (e.g. `"ZHW"` -> `"ZHAW"`).
- **🚫 Interactive Ignore Lists:** Exclude generic roles, academic degrees, or product names that should never be scrubbed (e.g., `"CAS"`, `"BSc"`, `"Dozent"`).
- **📄 Multi-Format Document Support:** Structured text and Markdown extraction for Word `.docx`, `.pdf`, `.csv`, `.json`, `.txt`, and `.md` with robust multi-encoding fallback (`utf-8-sig`, `cp1252`, `iso-8859-15`).
- **📊 Advanced PDF-to-Markdown Extraction:** Powered by `pymupdf4llm` with table structure cleanup, broken hyphenation repair, picture text extraction toggle, and recurring header/footer suppression with Page-1 title protection.
- **🖥️ Native Desktop GUI:** Responsive, instant-startup NiceGUI interface running as a native desktop window (with `--browser` option for web workflows).
- **⚡ Lightweight & No-Admin:** Optimized for CPU-only laptops without requiring administrative privileges (installable via `uv` or `pip`).

---

## 🚀 Quick Start

### 1. Installation

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/scepbjoern/local-anonymizer.git
cd local-anonymizer
uv sync
```

Or using standard `pip`:

```bash
pip install -e .
```

---

### 2. Interactive Review-GUI (NiceGUI Desktop Application)

Launch the application in native desktop mode:

```bash
# Windows / Desktop: Instant startup as a native window
uv run --extra gui python app.py

# Or launch directly in your default web browser
uv run --extra gui python app.py --browser
```

On Windows, you can also double-click `start_windows.vbs` for a silent background launch without a persistent console window.

> [!NOTE]
> **Data Privacy & In-Memory Processing:**
> All entity analysis, secret mapping tables, and text transformations reside strictly in RAM. When using the native file picker (`tkinter`), files are read directly into memory. When using drag-and-drop in the GUI, large files are streamed to a temporary local upload directory (`~/.local-anonymizer/temp_uploads`, max 50 MB) and are immediately unlinked/deleted from disk as soon as they are loaded into RAM.

> [!TIP]
> **Hinweis für macOS-Nutzer:**
> Auf macOS startet das GUI standardmäßig über das WebKit/Cocoa-Backend von `pywebview`. Falls kein separates Fenster gewünscht ist oder WebKit-Abhängigkeiten fehlen, kann die Anwendung jederzeit mit dem `--browser`-Flag gestartet werden:
> ```bash
> uv run --extra gui python app.py --browser
> ```
> Das Windows-Hilfsskript `start_windows.vbs` ist für Windows-Systeme optimiert; auf macOS erfolgt der Start direkt über das Terminal oder ein Shell-Skript.

---

### 3. Command Line Interface (CLI)

The CLI provides two primary commands: `anonymize` and `restore`.

#### Step 1: Anonymize a Document

Scrub sensitive information from `.docx`, `.pdf`, `.csv`, `.json`, `.txt`, or `.md` files:

```bash
uv run cli.py anonymize path/to/document.docx
```

With custom configuration (glossary, ignore list, entity filters):

```bash
uv run cli.py anonymize path/to/document.docx --config config.example.json
```

**Options:**
- `--config`, `-c`: Path to a JSON configuration file.
- `--output-dir`, `-o`: Directory to save outputs (default: same directory as input).
- `--no-mapping`: Permanent non-reversible scrubbing (no mapping file generated).
- `--language`, `-l`: Language code (default: `de`).

**Generated Files:**
1. `<name>_anonymized.txt` (or `.md`): Anonymized document ready for LLM prompts.
2. `<name>_mapping.json`: Secret local mapping table.
3. `<name>_report.json`: Detailed audit report with confidence scores and review flags.

---

#### Step 2: Restore / De-Anonymize LLM Response

Once your LLM generates a response or analysis containing placeholders, restore the original names:

```bash
uv run cli.py restore path/to/llm_response.txt path/to/original_mapping.json
```

**Options:**
- `--output-file`, `-o`: Destination path for the restored document (default: `<stem>_restored.txt`).

---

### 4. Automated Testing (pytest)

Run the full automated test suite (covers all recognizers, extractors, multi-encoding fallbacks, smart linking, and roundtrip reversibility):

```bash
uv run pytest
```

---

## ⚙️ Configuration (`config.json`)

Customize detection behavior using a simple JSON file:

```json
{
  "language": "de",
  "format_mode": "numbered_role",
  "gliner_threshold": 0.35,
  "enabled_entities": [
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "LOCATION"
  ],
  "glossary": {
    "abcd": "PERSON",
    "efgh": "PERSON",
    "ZHAW": "ORGANIZATION",
    "Zürcher Hochschule für Angewandte Wissenschaften": "ORGANIZATION"
  },
  "ignore_terms": [
    "CAS",
    "DAS",
    "MAS",
    "BSc",
    "MSc"
  ]
}
```

- **`format_mode`**: Select `numbered` (`[TYP_NR]`), `numbered_role` (`[TYP_NR_ROLLE]`), or `role_only` (`[TYP_ROLLE]`).
- **`enabled_entities`**: Choose which entity categories to detect and mask.
- **`glossary`**: Explicitly map internal acronyms and company names to entity types.
- **`ignore_terms`**: Whitelist words to prevent false-positive masking.

---

## 💻 Python API Usage

*For developers integrating the pipeline into custom Python scripts, automations, or web apps:*

```python
from local_anonymizer.pipeline import AnonymizationPipeline

# Initialize pipeline
pipeline = AnonymizationPipeline(
    language="de",
    glossary={"abcd": "PERSON", "ZHAW": "ORGANIZATION"},
    ignore_terms=["CAS"],
    enabled_entities=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE"]
)

# 1. Anonymize document
result = pipeline.process_file("contract.docx")
print("Anonymized Text:
", result.anonymization_result.anonymized_text)

# (Send result.anonymization_result.anonymized_text to LLM...)

# 2. De-anonymize LLM response
restored_file = AnonymizationPipeline.restore_file(
    anonymized_path="llm_output.txt",
    mapping_path=result.mapping_file,
)
```

---

## 🧩 Architecture Overview

```text
Input Document (.docx / .pdf / .csv / .json / .txt / .md)
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Document Extractor (pymupdf4llm / python-docx)         │
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Presidio Analyzer + Chunking Engine                    │
 │  ├── GLiNER Zero-Shot Recognizer (Multi-PII)           │
 │  ├── RapidFuzz Custom Glossary Recognizer              │
 │  ├── Gender Suffix Handler & Sentence Splitter         │
 │  └── Ignore-Terms & Entity-Type Filter                 │
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Local Anonymizer & Entity Linking                      │
 │  ├── Semantic Roles & Co-Reference Linking             │
 │  ├── Replaces PII with [TYP_NR_ROLLE] placeholders     │
 │  └── Stores Secret Mapping Table (Local JSON)          │
 └────────────────────────────────────────────────────────┘
         │
         ├────────────────────────────────┐
         ▼                                ▼
Anonymized Text (Safe for LLM)      Mapping Table (Local Only)
         │                                │
         ▼                                │
[Cloud or Local LLM Processing]           │
         │                                │
         ▼                                ▼
 ┌────────────────────────────────────────────────────────┐
 │ Local De-Anonymizer                                    │
 │  └── Restores original PII into LLM output             │
 └────────────────────────────────────────────────────────┘
         │
         ▼
  Final Restored Output
```

---

## 📄 License

MIT License. Free for academic, personal, and commercial use.
