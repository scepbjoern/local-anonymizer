# local-anonymizer – Dokumentations-Wegweiser

Willkommen in der Dokumentation für `local-anonymizer`.

> **Governance-Hinweis:**
> Dieses Repository enthält die *technische Source of Truth* (Produkt-Spezifikation, Architektur, Feature-Masterpläne, Komponenten-Doku).
> Strategische Handoffs, CAS-Protokolle, persönliche Entscheidungsprovenienz und Projektphasen-Steuerung verbleiben im privaten Obsidian-Vault des Projektinhabers.

---

## 1. Benutzerleitfaden (`guide/`)

Der deutschsprachige Leitfaden ist der beste Einstieg für Studierende und
andere Personen, die das Werkzeug praktisch verwenden möchten:

- [Benutzerleitfaden](guide/00-einstieg.md) – Einstieg, Installation, Bedienung,
  Review-Tabelle, optionale lokale LLM-Triage, Export, LLM-Nutzung, Wiederherstellung,
  Datenschutz, Fehlerbehebung und Ausblick.

## 2. Produkt & Architektur (`product/`)

Globale Dokumente, die das gesamte Repository und die Produktdefinition betreffen:
- [Product Requirements Document (PRD)](product/PRD.md) – Vision, Zielgruppen, Anwendungsfälle, Spezifikation der Format-Modi, Entity-Linking, Homonym-Zuordnung (Phase 5a), lokaler LLM-Triage-Layer (Phase 6A) und Phasen-Roadmap.
- [System Architecture](product/ARCHITECTURE.md) – Technische Architektur (In-Memory Processing, GLiNER NER Chunking, Presidio Integration, RapidFuzz Fuzzy Glossary, NiceGUI UI-Schicht, Single-Pass De-Anonymisierung, lokaler LLM-Triage-Layer mit Snapshot-Schutz, Batching, Provider-Loopback und Apply-Service).
- [Phase 6B Ausgangskontrolle](product/Phase_6B_Ausgangskontrolle.md) – Masterplan für den Postcheck (Nachzügler-Suche).

---

## 3. Komponenten & Pipeline (Quellcode)

Die technische Umsetzung liegt direkt unter `src/local_anonymizer/` und in
`app.py`:

- [`extractors.py`](../src/local_anonymizer/extractors.py) – strukturierte
  Dokumenten-Extraktion;
- [`recognizers.py`](../src/local_anonymizer/recognizers.py) – EU-PII-,
  GLiNER-, Regex-, Bibliotheks- und Glossar-Erkennung;
- [`anonymizer.py`](../src/local_anonymizer/anonymizer.py) – Platzhalter,
  Rollen, Linking und Wiederherstellung;
- [`pipeline.py`](../src/local_anonymizer/pipeline.py) –
  End-to-End-Orchestrierung;
- [`llm/`](../src/local_anonymizer/llm/) – optionaler lokaler LLM-Triage-Layer:
  - [`schema.py`](../src/local_anonymizer/llm/schema.py) – Pydantic-v2-Vertrag mit `extra="forbid"`, Snapshot-/Revision-Bindung;
  - [`batching.py`](../src/local_anonymizer/llm/batching.py) – deterministisches Token-Budgeting, Kontext-Snippets, sequenzielle Batches;
  - [`provider.py`](../src/local_anonymizer/llm/provider.py) – OpenAI-kompatibler Loopback-Client mit HTTP-400/422-Retry und Streaming-Sicherheitsgrenze;
  - [`apply_service.py`](../src/local_anonymizer/llm/apply_service.py) – Snapshot-Validierung, Auswirkungsvorschau, atomare Mutation & Rollback;
- [`app.py`](../app.py) – lokale NiceGUI-Review-Oberfläche mit integriertem LLM-Triage-Panel und Human-in-the-Loop-Workflow.

---

## 4. Entwicklung & Qualitätssicherung

- **Tests:** Ausführen der automatisierten Regressionstestsuite mit `uv run pytest`.
- **Lokale Ausführung:** `uv run --extra gui python app.py` (Desktop) oder `uv run --extra gui python app.py --browser` (Web).
