# Datenschutz und Grenzen

## Was lokal bleibt

Die Entitätserkennung, die Platzhalter-Zuordnung und die Wiederherstellung
werden lokal ausgeführt. Es gibt keine eingebaute Übertragung an einen
Cloud-Dienst und keine Telemetrie für die Kernpipeline.

Bei der nativen Dateiauswahl werden Dateien direkt in den Arbeitsspeicher
eingelesen. Beim Drag-and-drop kann die Anwendung grössere Uploads kurzzeitig
in einem lokalen temporären Upload-Ordner puffern und löscht diese Datei nach
dem Einlesen wieder.

Das bedeutet nicht, dass eine Datei unter allen Umständen physisch nie auf der
Festplatte liegt. Bei besonders schützenswerten Dokumenten solltest du deshalb
auch lokale Zugriffsrechte, Backups, Virenscanner und temporäre Systemkopien
berücksichtigen.

## Was du weitergeben darfst

Vor der Weitergabe an ein externes LLM sollte nur die anonymisierte Datei
verwendet werden. Die Mapping-Datei, die Originaldatei und der Prüfbericht
bleiben lokal, sofern sie nicht ausdrücklich benötigt werden.

## Keine hundertprozentige automatische Erkennung

Kein automatisches Erkennungssystem findet alle sensiblen Angaben zuverlässig
und klassifiziert alle Fundstellen richtig. Die Anwendung ist deshalb auf einen
Review-Schritt ausgelegt:

- prüfe die Fundstellen im Kontext;
- kontrolliere auch nicht markierte Stellen stichprobenartig;
- ergänze fehlende Begriffe manuell oder im Glossar;
- ignoriere harmlose Fehlklassifikationen bewusst;
- exportiere erst nach deiner Freigabe.

Auch ein hoher Score beweist nicht, dass ein Begriff sachlich richtig
klassifiziert wurde.

## Dokumentformate

PDFs mit digitalem Text werden strukturiert ausgelesen. Ein PDF, das nur aus
gescannten Bildern besteht, benötigt OCR. Dieser Anwendungsfall ist in der
aktuellen Standardversion noch nicht vollständig abgedeckt.

Das Werkzeug anonymisiert Textinhalte. Es ist keine visuelle PDF-Redaktion und
entfernt nicht automatisch alle denkbaren Metadaten aus Originaldateien.

## Plattformgrenzen

Windows ist der praktische Referenzweg dieser Auslieferung. Die Mac-Skripte
und der Mac-Testworkflow sind vorhanden, aber die Anwendung wurde noch nicht
manuell auf macOS getestet. Für macOS kann daher keine gleichwertige
Praxissicherheit behauptet werden.

