# Ausblick

Die aktuelle Auslieferung umfasst die lokale hybride Erkennung (GLiNER, EU-PII, Regex, Schweizer Prüfziffern), die Homonym-Trennung (Phase 5a) sowie die optionale lokale LLM-Review-Assistenz (Phase 6A).

Die folgenden Erweiterungen sind Gegenstand künftiger Entwicklungsphasen:

## 1. Finale Ausgangskontrolle (Phase 6B)

Als zweiter optionaler Prüfschritt ist eine automatisierte Sicherheits-Endkontrolle des fertig anonymisierten Textes vorgesehen:
- Das lokale LLM liest das fertige Dokument vor dem Export und sucht nach übersehenen Rest-Identifikatoren oder Kontextlecks (z. B. unmaskierte Funktionsbezeichnungen in Kombination mit seltenen Abteilungsnamen).
- Erkannte Restrisiken werden als Prüfhinweis vor dem Export signalisiert.

## 2. LLM-unterstütztes Smart-Linking

Erweiterung des deterministischen Linking-Algorithmus:
- Erkennung komplexerer Koreferenzen (z. B. Pronomen, Spitznamen, Berufsbezeichnungen im Textfluss), die deterministisch schwer auflösbar sind.
- Vorschlag von Verknüpfungen zur Hauptentität mit menschlicher Freigabe.

## 3. Projektbezogene Mappings & Registry (Phase 5b)

Eine künftige Projektfunktion könnte Einstellungen, Begriffe und Mappings für einen zusammenhängenden Arbeitskontext sichern (z. B. eine mehrteilige CAS-Abschlussarbeit oder eine Fallaktenserie):
- Konsistente Platzhaltervergabe über mehrere Dokumente hinweg (`[PERSON_1_PROJEKTLEITER]` bleibt in Dokument A und B dieselbe reale Person).
- Projektweite Begriffs- und Ignore-Listen.
- Speicherung und Berechtigungsgrenzen (z. B. per SQLite oder verschlüsselter Projekt-JSON) werden zu gegebener Zeit spezifiziert.

