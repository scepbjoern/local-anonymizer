# local-anonymizer 🔒

> **Privacy-first, local PII anonymization and de-anonymization pipeline for AI workflows.**

`local-anonymizer` is a lightweight, local-first CLI tool and Python library that scrubs Personally Identifiable Information (PII) and company-specific sensitive entities from documents and text before sending them to cloud or local LLMs. Once the LLM completes its task (e.g., summarization, rewriting, or extraction), `local-anonymizer` seamlessly restores the original entities using a secure, local-only mapping table.

---

## ✨ Features

- **🛡️ 100% Local & Private:** All entity detection, mapping, and text processing run completely on your local machine. No data or telemetry leaves your computer.
- **🤖 Zero-Shot Multilingual PII Detection:** Powered by [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi_pii-v1`) with intelligent chunking for accurate German and English PII extraction without document length limits.
- **🔍 Fuzzy Glossary & Overrides:** Uses [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) to catch custom internal terms, project codes, employee acronyms (e.g., mapping `"abcd"` -> `PERSON`), and typos (e.g. `"ZHW"` -> `"ZHAW"`).
- **🚫 Ignore Lists:** Easily exclude terms that should never be scrubbed (e.g., `"CAS"`, `"BSc"`, product names).
- **🔄 Deterministic 2-Way Mapping:** Replaces sensitive entities with readable placeholders (e.g. `[PERSON_1]`, `[ORGANIZATION_1]`, `[IBAN_CODE_1]`), preserving context for LLMs while guaranteeing 100% exact de-anonymization.
- **📄 Multi-Format Document Support:** Direct text extraction for `.txt`, `.md`, Word `.docx`, and `.pdf` files.
- **⚡ Lightweight & No-Admin:** Designed for CPU-only hardware without requiring administrative privileges (installable via `uv` or `pip`).

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

### 2. Command Line Interface (CLI)

The CLI provides two main subcommands: `anonymize` and `restore`.

#### Step 1: Anonymize a Document

Scrub sensitive information from `.txt`, `.md`, `.docx`, or `.pdf` files:

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
1. `<name>_anonymized.txt`: Anonymized document ready for LLM prompts.
2. `<name>_mapping.json`: Secret local mapping table.
3. `<name>_report.json`: Detailed audit report with confidence scores and review flags.

---

#### Step 2: Restore / De-Anonymize LLM Response

Once your LLM generates a response or reviewed text containing `[CATEGORY_N]` placeholders, restore the original names/terms:

```bash
uv run cli.py restore path/to/llm_response.txt path/to/original_mapping.json
```

**Options:**
- `--output-file`, `-o`: Destination path for the restored document (default: `<stem>_restored.txt`).

---

## ⚙️ Configuration (`config.json`)

You can customize detection behavior using a simple JSON file:

```json
{
  "language": "de",
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
    "ijkl": "PERSON",
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

- **`enabled_entities`**: Only scrub selected categories (e.g. omit `DATE_TIME` if meeting dates should remain visible).
- **`glossary`**: Explicitly map internal acronyms and company names to desired entity types.
- **`ignore_terms`**: Prevent false positives from being scrubbed.

---

## 💻 Python API Usage

*For developers and students integrating the pipeline into custom Python scripts, automations, or web apps:*

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
print("Anonymized Text:\n", result.anonymization_result.anonymized_text)

# (Send result.anonymization_result.anonymized_text to LLM...)

# 2. De-anonymize
restored_file = AnonymizationPipeline.restore_file(
    anonymized_path="llm_output.txt",
    mapping_path=result.mapping_file,
)
```

---

## 🧩 Architecture

```text
Input Document (.txt / .docx / .pdf)
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Document Extractor (python-docx / PyMuPDF)             │
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Presidio Analyzer + Chunking Engine                    │
 │  ├── GLiNER Zero-Shot Recognizer (Multi-PII)           │
 │  ├── RapidFuzz Custom Glossary Recognizer              │
 │  └── Ignore-Terms & Entity-Type Filter                 │
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Local Anonymizer                                       │
 │  ├── Replaces PII with [CATEGORY_N] placeholders       │
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
