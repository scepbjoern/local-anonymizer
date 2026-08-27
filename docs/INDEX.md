# local-anonymizer – Dokumentations-Wegweiser

Willkommen in der Dokumentation für `local-anonymizer`.

> **Governance-Hinweis:**
> Dieses Repository enthält die *technische Source of Truth* (Produkt-Spezifikation, Architektur, Komponenten-Doku, Test-Dokumentation).
> Strategische Handoffs, CAS-Protokolle, Architekturentscheide und Projektphasen-Steuerung verbleiben im privaten Obsidian-Vault unter `50-59 Knowledge systems, AI & private development / 52 AI, agents & automation / 52.16 AI workspaces / P_PLAI - Privacy-First Local AI`.

---

## 1. Benutzerleitfaden (`guide/`)

Der deutschsprachige Leitfaden ist der beste Einstieg für Studierende und
andere Personen, die das Werkzeug praktisch verwenden möchten:

- [Benutzerleitfaden](guide/00-einstieg.md) – Einstieg, Installation, Bedienung,
  Review, LLM-Nutzung, Export, Wiederherstellung, Datenschutz, Fehlerbehebung
  und Ausblick.

## 2. Produkt & Architektur (`product/`)

Globale Dokumente, die das gesamte Repository und die Produktdefinition betreffen:
- [Product Requirements Document (PRD)](product/PRD.md) – Vision, Zielgruppen, Anwendungsfälle, Spezifikation der Format-Modi, Entity-Linking und Phasen-Roadmap.
- [System Architecture](product/ARCHITECTURE.md) – Technische Architektur (In-Memory Processing, GLiNER NER Chunking, Presidio Integration, RapidFuzz Fuzzy Glossary, NiceGUI UI-Schicht, Single-Pass De-Anonymisierung).

---

## 3. Komponenten & Pipeline (Quellcode)

Die technische Umsetzung liegt direkt unter `src/local_anonymizer/` und in
`app.py`:

- [`extractors.py`](../src/local_anonymizer/extractors.py) – strukturierte
  Dokumenten-Extraktion;
- [`recognizers.py`](../src/local_anonymizer/recognizers.py) – GLiNER-, Regex-
  und Glossar-Erkennung;
- [`anonymizer.py`](../src/local_anonymizer/anonymizer.py) – Platzhalter,
  Rollen, Linking und Wiederherstellung;
- [`pipeline.py`](../src/local_anonymizer/pipeline.py) –
  End-to-End-Orchestrierung;
- [`app.py`](../app.py) – lokale NiceGUI-Review-Oberfläche.

---

## 4. Entwicklung & Qualitätssicherung

- **Tests:** Ausführen der automatisierten Regressionstestsuite mit `uv run pytest`.
- **Lokale Ausführung:** `uv run --extra gui python app.py` (Desktop) oder `uv run --extra gui python app.py --browser` (Web).
