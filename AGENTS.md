# AGENTS.md – Guidelines for AI Coding Assistants & Automated Setup

This document provides definitive instructions for autonomous and interactive AI coding agents (e.g., OpenAI Codex, Anthropic Claude Code, Cursor, Windsurf, Google Antigravity) working with the `local-anonymizer` repository.

---

## 1. Project Overview & Environment Requirements

`local-anonymizer` is a privacy-first, 100% offline and local PII anonymization and de-anonymization tool with a NiceGUI desktop interface and a CLI.

* **Python Version:** Python >= 3.11 (see pyproject.toml / .python-version; verified on Python 3.11).
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) is the standard and recommended environment manager.
* **No Admin Privileges Required:** All dependencies install strictly into user space / local virtual environment.

---

## 2. Installation & Setup

When an agent is instructed to set up or install this repository:

### Step 1: Clone Repository (if not already inside workspace)
```bash
git clone https://github.com/scepbjoern/local-anonymizer.git
cd local-anonymizer
```

### Step 2: Install Virtual Environment & GUI Dependencies
Always include the `gui` extra so NiceGUI and desktop components are installed:
```bash
# Recommended via uv:
uv sync --extra gui

# Standard pip alternative:
pip install -e ".[gui]"
```

---

## 3. Starting the Application

* **Native Desktop GUI (Windows / Default):**
  ```bash
  uv run --extra gui python app.py
  ```
* **Web Browser Fallback Mode (e.g. headless, remote, or macOS testing):**
  ```bash
  uv run --extra gui python app.py --browser
  ```
* **Silent Windows Launcher (No console window):**
  ```bash
  wscript.exe start_windows.vbs
  ```
* **CLI Anonymize / Restore:**
  ```bash
  uv run cli.py anonymize path/to/document.docx
  uv run cli.py restore path/to/llm_response.txt path/to/mapping.json
  ```

---

## 4. First-Run Behavior & Model Weights

* **Dual-Model Ensemble Downloads:**
  * **GLiNER (`urchade/gliner_multi_pii-v1`, ~150 MB):** Zero-shot model for organizations, roles, and open-vocabulary safety net.
  * **EU-PII (`bardsai/eu-pii-anonimization-multilang`, ~1.1 GB):** Specialized token-classification model for European names, locations, IDs, and health data.
* **Transparent Confirmation on First Download:** On the very first run, if required models are not yet cached locally under `~/.cache/huggingface/hub/`, the application presents an interactive confirmation dialog detailing the exact model names and download sizes before fetching any weights.
* **100% Offline After Caching:** Once models are downloaded, all subsequent runs operate completely offline (`HF_HUB_OFFLINE=1`).
* **Async Background Warmup:** The NiceGUI application launches immediately (<0.5s startup). Cached ML models initialize asynchronously in a background thread. The status badge in the UI transitions to *"Modell bereit"* once warmup completes.

---

## 5. Verification & Test Suite

Agents making code modifications must ensure all tests pass:
```bash
uv run pytest -q
```
* Expected outcome: 99 passed unit and regression tests (1 integration test deselected by default).
* Run the live EU-PII integration test when modifying recognizer lifecycle:
  ```bash
  uv run pytest -m integration -q
  ```
* Note: A warning regarding `resume_download` from `huggingface_hub` is an upstream deprecation notice and not a test failure.

---

## 6. Critical Architectural Rules & Pitfalls for Agents

1. **Always use `--extra gui`:** Without `--extra gui`, running `app.py` will fail with missing NiceGUI dependencies.
2. **In-Memory Guarantee & Temporary Files Privacy Contract:**
   * Text extraction, NLP analysis, entity linking, and mapping tables operate in-memory.
   * Large drag-and-drop uploads use a local streaming buffer (`~/.local-anonymizer/temp_uploads`), which is immediately unlinked (`try...finally`) after memory transfer.
   * Multi-page PDF extraction writes a temporary file to `~/.local-anonymizer/temp_uploads` to dispatch page tasks across CPU cores via `ProcessPoolExecutor`. This temporary file is unlinked immediately in `finally` and protected against concurrent startup deletion by a 30-minute age cutoff.
   * Stale temporary files older than 30 minutes are cleaned up on startup and shutdown (`atexit`). Do not alter this privacy and operational contract.
3. **Session State Isolation:**
   * In `app.py`, `create_ui()` is decorated with `@ui.page('/')`.
   * Application state is encapsulated per-session in `state = AppState()`. Never store client-specific state in global module variables.
   * Shared ML models are protected by `_model_lock = threading.Lock()`.
   * Multi-page PDF worker environment variables are serialized via `_pdf_env_lock = threading.Lock()`.
4. **Safe Process Termination on Windows:**
   * Never execute a blanket `taskkill /im pythonw.exe` or `Stop-Process -Name pythonw` as it kills unrelated user processes. Always filter by process path / command line matching `*local-anonymizer*` and `*app.py*`.
5. **Deterministic Single-Pass De-Anonymization:**
   * Placeholders in `de_anonymize` must always be sorted by descending length to prevent sub-string collision and cascading replacement errors.
