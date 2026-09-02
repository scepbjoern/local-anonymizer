# local-anonymizer 🔒

> **Privacy-first, local PII anonymization and de-anonymization pipeline for AI workflows.**

`local-anonymizer` is a lightweight, local-first Desktop GUI, CLI tool, and Python library that scrubs Personally Identifiable Information (PII) and company-specific sensitive entities from documents and text before sending them to cloud or local LLMs. Once the LLM completes its task (e.g., summarization, rewriting, or extraction), `local-anonymizer` seamlessly restores the original entities using a secure, local-only mapping table.

Für die praktische Nutzung steht ein deutschsprachiger [Benutzerleitfaden](docs/guide/00-einstieg.md) mit Installation, Review, Export, Wiederherstellung, Datenschutz und Fehlerbehebung bereit.

---

## ✨ Key Features

- **🛡️ 100% Local & Private:** All entity detection, mapping, and text transformations run completely on your local machine. No data or telemetry leaves your computer.
- **🤖 Dual-Model Hybrid AI Ensemble:**
  - **GLiNER Zero-Shot (`urchade/gliner_multi_pii-v1`):** Flexible extraction of organizations, roles, and open-vocabulary entities with chunking for arbitrary text lengths.
  - **European PII Classifier (`bardsai/eu-pii-anonimization-multilang`):** Specialized Token-Classification model for high-precision European person names, locations, IDs, and health data.
- **⚖️ 4-Tier Source Priority Hierarchy:**
  - `Tier 4 (Highest):` User Glossary & Project Vocabulary (Exact & Fuzzy)
  - `Tier 3:` Deterministic Libraries & Validators (Google `phonenumbers`, Checksum-validated AHV, UID, IBAN, Email, Address Regex)
  - `Tier 2:` Specialized EU-PII Model (High precision on `PERSON`, `LOCATION`, `ID_NUMBER`, `HEALTH_DATA`)
  - `Tier 1:` GLiNER Zero-Shot Model (`ORGANIZATION`, `ROLE`, and fallback safety net)
- **🏷️ 3 Semantic Placeholder Modes:**
  - **Modus 1 (Classic):** `[PERSON_1]`, `[ORGANIZATION_1]`
  - **Modus 2 (Numbered + Role, Recommended):** `[PERSON_1_STUDENT]`, `[ORGANIZATION_1_ZULIEFERER]`
  - **Modus 3 (Role Only, Compact):** `[PERSON_STUDENT]` (with automatic collision detection & fallback if multiple entities share the same role).
- **🔗 Smart Entity Linking & Co-Reference Resolution:** Links first names (*"Julia"*), honorifics (*"Frau Meier"*), and genitives (*"Julias"*) to master identities (*"Julia Meier"*) using explicit surface tags (`_VOLLNAME`, `_VORNAME`, `_NACHNAME`, `_ANREDE`, `_GENITIV`, `_KURZFORM`) for 100% exact, grammatically consistent restoration.
- **🚻 Gender-Aware Entity Recognition:** Automatically handles German gendered occupational and role suffixes (`-in`, `:in`, `*in`, `_in`, `/in`, `Innen`, `innen`) to prevent truncated or missed entity spans.
- **🔍 Fuzzy Glossary & Overrides:** Uses [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) to catch custom internal terms, project codes, employee acronyms (e.g., mapping `"abcd"` -> `PERSON`), and typos (e.g. `"ZHW"` -> `"ZHAW"`).
- **🚫 Interactive Ignore Lists:** Exclude generic roles, academic degrees, or product names that should never be scrubbed (e.g., `"CAS"`, `"BSc"`, `"Dozent"`).
- **👔 Optional Role Detection:** `ROLE` recognizes job titles such as `CEO`, `CFO`, or `Leiter Prozessmanagement`; it is off by default because process roles are often meaningful content.
- **🇨🇭 Deterministic Swiss/CH-PII Detection:** Recognizes Swiss/German addresses plus checksum-validated AHV and CHE/UID numbers; internal IT systems are supported through the glossary and GLiNER safety-net prompts.
- **🔎 Transparent Review Sources & Bulk Toggles:** Each finding shows whether it came from `🤖 GLiNER`, `🤖 EU-PII`, `🔤 Regex`, `📚 Bibliothek` (Google `phonenumbers`), `📖 Glossar` (direct/fuzzy), or `✍ Manuell`. The toolbar includes "Alle aktivieren", "Alle abwählen" and an interactive "Alle aufklappen / zuklappen" toggle.
- **🧠 Optional Local LLM Triage Layer:** Verifies detected entities in context via local OpenAI-compatible endpoints (e.g. Ollama, LM Studio). Provides snapshot-protected recommendations for false-positive filtering, category corrections, and role descriptors with explicit, impact-previewed, atomic human approval.
- **📄 Multi-Format Document Support:** Structured text and Markdown extraction for Word `.docx`, `.pdf`, `.csv`, `.json`, `.txt`, and `.md` with robust multi-encoding fallback (`utf-8-sig`, `cp1252`, `iso-8859-15`).
- **📊 Advanced PDF-to-Markdown Extraction:** Structured Markdown extraction powered by PyMuPDF RAG (preserving headers, lists, bold/italic, and clean tables without tearing words across columns), picture text toggle, and recurring header/footer suppression with Page-1 title protection.
- **🖥️ Native Desktop GUI:** Responsive, instant-startup NiceGUI interface running as a native desktop window (with `--browser` option for web workflows).
- **⚡ Lightweight & No-Admin:** Optimized for CPU-only laptops without requiring administrative privileges (installable via `uv` or `pip`).

---

## 🚀 Quick Start

### 1. Installation

#### Option A: Automated Setup via Local AI Coding Assistant (Recommended for AI Workflows)

If you use a local AI coding assistant with terminal and filesystem permissions (such as OpenAI Codex, Claude Code, Cursor, Windsurf, or Google Antigravity), you can delegate the complete installation with a single prompt:

> **Agent Prompt:**
> *"Klone das Repository https://github.com/scepbjoern/local-anonymizer in <Zielordner> und installiere die Applikation."*
> *(Or in English: "Clone https://github.com/scepbjoern/local-anonymizer into <target_folder> and install the application.")*

The AI assistant will inspect [`AGENTS.md`](AGENTS.md) and [`pyproject.toml`](pyproject.toml) to automatically handle environment setup, `uv sync --extra gui`, and verification tests.

> [!NOTE]
> **Empirical Verification & Security Disclosure:**
> This automated installation path was empirically verified on Windows using OpenAI Codex ($n=1$). Delegating execution commands to an AI agent grants it local execution rights on your system. For users who prefer full transparency and step-by-step control, use the manual setup below.

#### Option B: Manual Setup (Standard)

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/scepbjoern/local-anonymizer.git
cd local-anonymizer
uv sync --extra gui
```

Or using standard `pip`:

```bash
pip install -e ".[gui]"
```

#### Optional: Local LLM Triage Layer Extra (`[llm]`)

To enable the optional local LLM review assistance with local OpenAI-compatible endpoints (e.g. Ollama):

```bash
# Via uv:
uv sync --extra gui --extra llm

# Via pip:
pip install -e ".[gui,llm]"
```

**Local Model Setup with Ollama:**
1. Install [Ollama](https://ollama.com/) and start the local service (default: `http://127.0.0.1:11434/v1`).
2. Pull the reference model (separate multi-GB download):
   ```bash
   ollama pull qwen3:8b
   ```
   *(Tested alternatives: `ministral-3:8b`, `qwen3.5:9b`)*
3. Launch with LLM support:
   ```bash
   uv run --extra gui --extra llm python app.py
   # Or browser mode:
   uv run --extra gui --extra llm python app.py --browser
   ```
4. In the Desktop GUI, enable the LLM Review Assistant (top area), select or enter your local model (e.g. `qwen3:8b`), and optionally preload it.

> [!NOTE]
> **Hardware & Performance:**
> The core deterministic anonymization pipeline (GLiNER, EU-PII, Regex, Swiss Checksums) is lightweight and runs efficiently on standard CPU-only laptops. The optional LLM Triage Layer is an accelerated power-user path (for the tested reference setup with ~8B models, GPU acceleration is recommended).

> [!IMPORTANT]
> **Strict Human-in-the-Loop:**
> The LLM Triage Layer makes recommendations only. It never mutates entity groups or placeholders automatically. Changes are only applied when the user explicitly reviews them, selects them, and confirms them in the impact dialog.

---

### 2. Interactive Review-GUI (NiceGUI Desktop Application)

Eine schrittweise Anleitung für Studierende und andere Nutzerinnen und Nutzer
steht im [Benutzerleitfaden](docs/guide/00-einstieg.md). Die folgenden Befehle sind
die technische Kurzreferenz.


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
> All entity analysis, secret mapping tables, and text transformations reside strictly in RAM. When using the native file picker (`tkinter`), files are read directly into memory. When using drag-and-drop in the GUI, large files are streamed to a temporary local upload directory (`~/.local-anonymizer/temp_uploads`, max 50 MB) and are immediately unlinked/deleted from disk as soon as they are loaded into RAM. For multi-page PDFs, a temporary extraction file is written to `temp_uploads` to dispatch page tasks across CPU cores via `ProcessPoolExecutor`, protected against concurrent startup deletion by a 30-minute age boundary and unlinked immediately in `finally` upon completion.

> [!NOTE]
> **First-Run Model Downloads & Offline-First Operation:**
> On the very first run, if required local ML models (GLiNER ~1.10 GB, EU-PII ~1.10 GB) are not yet cached under `~/.cache/huggingface/hub/`, the application displays an interactive confirmation dialog with the exact model names and download sizes before fetching any files. Inferences and document analyses run strictly locally; model loaders attempt local cache access first under thread-safe offline context management (`set_huggingface_offline_mode`), connecting only temporarily during an authorized download.

> [!WARNING]
> **Hinweis für macOS-Nutzer:**
> Die Mac-Startskripte und der automatisierte Testworkflow sind vorbereitet, aber die Anwendung wurde bisher nicht praktisch auf macOS getestet. Der Browser-Modus kann als Versuch verwendet werden:
> ```bash
> uv run --extra gui python app.py --browser
> ```
> Das Windows-Hilfsskript `start_windows.vbs` ist der getestete Referenzweg; auf macOS erfolgt der Start direkt über das Terminal oder ein Shell-Skript.

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
- `--enable-eupii` / `--disable-eupii`: Toggle the European PII token classification model (`bardsai/eu-pii-anonimization-multilang`).
- `--eupii-threshold`: Score threshold for EU-PII predictions (default: `0.50`).

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

Run the full automated test suite (covers all recognizers, extractors, multi-encoding fallbacks, smart linking, LLM triage layer, and roundtrip reversibility):

```bash
uv run pytest
```
* **Full test suite passes cleanly (230+ passed, 2 integration tests deselected by default).**
* Run the live EU-PII model integration test with `uv run pytest -m integration`.

Weitere Dokumentation ist über den [Dokumentations-Wegweiser](docs/INDEX.md)
erreichbar. Die Produkt- und Architekturdokumente beschreiben Anforderungen
und technische Umsetzung; der Benutzerleitfaden beschreibt die Bedienung.

---

## ⚙️ Configuration (`config.json`)

Customize detection behavior using a simple JSON file:

```json
{
  "language": "de",
  "format_mode": "numbered_role",
  "gliner_threshold": 0.55,
  "enable_eupii": true,
  "eupii_threshold": 0.50,
  "eupii_model_name": "bardsai/eu-pii-anonimization-multilang",
  "enabled_entities": [
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "LOCATION",
    "ADDRESS",
    "AHV_NUMBER",
    "UID_NUMBER",
    "ID_NUMBER",
    "HEALTH_DATA",
    "IT_SYSTEM"
  ],
  "enabled_glossary_entities": [
    "PERSON",
    "ORGANIZATION",
    "IT_SYSTEM"
  ],
  "glossary": {
    "abcd": "PERSON",
    "efgh": "PERSON",
    "ZHAW": "ORGANIZATION",
    "Zürcher Hochschule für Angewandte Wissenschaften": "ORGANIZATION",
    "SAP": "IT_SYSTEM"
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
- **`enable_eupii`**: Enable or disable the local European PII token classification model (default: `true`).
- **`eupii_threshold`**: Minimum confidence score for EU-PII detections (default: `0.50`).
- **`enabled_entities`**: Choose which entity categories general AI/library/regex detection should inspect and mask.
- **`enabled_glossary_entities`**: Choose which categories may use explicit glossary entries. An empty list disables glossary matching completely; in the GUI this is the `Aus` mode.
- **`glossary`**: Explicitly map internal acronyms and company names to entity types. Allowed explicit entries take precedence over generic built-in ignore terms.
- **`ignore_terms`**: Whitelist words to prevent false-positive masking. Generic labels such as `Email`, `E-Mail`, `App`, and `Applikation` are protected by default but can still be deliberately added to the glossary.

---

## 💻 Python API Usage

*For developers integrating the pipeline into custom Python scripts, automations, or web apps:*

```python
from local_anonymizer.pipeline import AnonymizationPipeline

# Initialize pipeline
pipeline = AnonymizationPipeline(
    language="de",
    enable_eupii=True,
    glossary={"abcd": "PERSON", "ZHAW": "ORGANIZATION"},
    ignore_terms=["CAS"],
    enabled_entities=["PERSON", "ORGANIZATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE"],
    enabled_glossary_entities=["PERSON", "ORGANIZATION"]
)

# 1. Anonymize document
result = pipeline.process_file("contract.docx")
print("Anonymized Text:\n", result.anonymization_result.anonymized_text)

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
 │ Presidio Analyzer + Ensemble Engine                    │
 │  ├── EU-PII Classifier (bardsai/eu-pii, Tier 2)        │
 │  ├── GLiNER Zero-Shot Recognizer (Multi-PII, Tier 1)   │
 │  ├── RapidFuzz Custom Glossary Recognizer (Tier 4)     │
 │  ├── Swiss Checksum Recognizers (AHV, UID, IBAN)       │
 │  ├── CH/DE Address Regex & Google phonenumbers (Tier 3)│
 │  ├── Gender Suffix Handler & Sentence Splitter         │
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
