# local-anonymizer – Dokumentations-Wegweiser

Willkommen in der technischen Dokumentation für `local-anonymizer`.

> **Governance-Hinweis:**
> Dieses Repository enthält die *technische Source of Truth* (Produkt-Spezifikation, Architektur, Komponenten-Doku, Test-Dokumentation).
> Strategische Handoffs, CAS-Protokolle, Architekturentscheide und Projektphasen-Steuerung verbleiben im privaten Obsidian-Vault unter `50-59 Knowledge systems, AI & private development / 52 AI, agents & automation / 52.16 AI workspaces / P_PLAI - Privacy-First Local AI`.

---

## 1. Produkt & Architektur (`product/`)

Globale Dokumente, die das gesamte Repository und die Produktdefinition betreffen:
- [Product Requirements Document (PRD)](product/PRD.md) – Vision, Zielgruppen, Anwendungsfälle, Spezifikation der Format-Modi, Entity-Linking und Phasen-Roadmap.
- [System Architecture](product/ARCHITECTURE.md) – Technische Architektur (In-Memory Processing, GLiNER NER Chunking, Presidio Integration, RapidFuzz Fuzzy Glossary, NiceGUI UI-Schicht, Single-Pass De-Anonymisierung).

---

## 2. Komponenten & Pipeline (`components/`)

Spezifikationen der einzelnen Module:
- `extractors` – Strukturierte Dokumenten-Extraktion (.pdf via `pymupdf4llm`, .docx, .txt, .md).
- `recognizers` – Zero-Shot PII Erkennung (GLiNER) und fehlertolerantes Glossar (RapidFuzz).
- `anonymizer` – Placeholder Substitution Engine, Rollen-Mapping, Entity-Linking & Regex De-Anonymisierung.
- `pipeline` – End-to-End Orchestrierung & Audit-Reporting.
- `gui` – Lokale NiceGUI Review- und Korrektur-Oberfläche.

---

## 3. Entwicklung & Qualitätssicherung

- **Tests:** Ausführen der automatisierten Regressionstestsuite mit `uv run pytest`.
- **Lokale Ausführung:** `uv run --extra gui python app.py` (Desktop) oder `uv run --extra gui python app.py --browser` (Web).
