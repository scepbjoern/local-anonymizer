# Ausblick

Die aktuelle Auslieferung konzentriert sich auf lokale Erkennung, menschliche
Prüfung und reversible Anonymisierung. Die folgenden Erweiterungen sind noch
nicht Bestandteil dieser Version.

## Lokales LLM als zusätzliche Prüfung

Später kann ein lokales LLM die Review-Tabelle unterstützen. Es könnte für jede
Fundstelle den Begriff und seinen Kontext betrachten und Empfehlungen abgeben:

- Ist die erkannte Kategorie plausibel?
- Handelt es sich um eine Person, eine Organisation oder einen harmlosen
  Fachbegriff?
- Sollte die Fundstelle anonymisiert werden?
- Ist eine Rolle oder ein Glossar-Eintrag sinnvoll?

Das wäre besonders hilfreich bei kniffligen Fällen wie einem Namen, der als
Organisation erkannt wurde, oder einem internen Begriff, der nur in einem
bestimmten Kontext sensibel ist.

Zusätzlich könnte das lokale LLM nach dem Review einen Sicherheitscheck des
anonymisierten Dokuments durchführen und nach verbliebenen sensiblen Angaben
suchen.

Das LLM würde Empfehlungen liefern. Die endgültige Entscheidung sollte bei der
Person bleiben, die das Dokument kennt und freigibt.

## Projektbezogene Einstellungen

Eine spätere Projektfunktion könnte Einstellungen, Begriffe und Entscheidungen
für einen bestimmten Arbeitszusammenhang sichern, beispielsweise für eine
CAS-Abschlussarbeit.

Dann könnte unter anderem festgelegt werden:

- dass `SAP` in diesem Projekt immer als `IT_SYSTEM` behandelt wird;
- welche Begriffe projektweit ignoriert werden;
- welche Rollen verwendet werden, etwa `LEHRPERSON` oder
  `STUDIENGANGSLEITUNG`;
- welche Person welchem projektbezogenen Platzhalter entspricht;
- wie dieselben Begriffe über mehrere Dokumente hinweg konsistent behandelt
  werden.

Später könnte ein Teil dieser Einstellungen auch projektübergreifend genutzt
werden. Ob die Daten dafür in JSON-Dateien, einer Datenbank oder einer anderen
Form gespeichert werden, ist noch offen.

Diese Funktionen werden erst in späteren Entwicklungsphasen konkret geplant
und umgesetzt.

