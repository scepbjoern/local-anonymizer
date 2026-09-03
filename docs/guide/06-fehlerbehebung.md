# Fehlerbehebung

## Die Anwendung startet nicht

Prüfe zunächst:

1. dass du dich im Projektordner befindest;
2. dass `uv` im Terminal gefunden wird;
3. dass die GUI-Abhängigkeiten installiert wurden:

   ```powershell
   uv sync --extra gui
   ```

4. ob der Browser-Modus funktioniert:

   ```powershell
   uv run --extra gui python app.py --browser
   ```

Wenn der Browser-Modus funktioniert, liegt das Problem wahrscheinlich am
nativen Fenster-Backend und nicht an der Kernpipeline.

## Der erste Lauf dauert lange

Beim ersten Start können Python-Abhängigkeiten und lokale Modellgewichte
geladen werden. Spätere Analysen benötigen normalerweise weniger Zeit.

## Ein Begriff wird nicht erkannt

- Prüfe, ob die Kategorie auf **Alle Quellen** oder **Nur Glossar & manuell**
  steht.
- Markiere den Begriff manuell.
- Füge ihn mit dem passenden Entitätstyp zum Glossar hinzu.
- Starte die Analyse nach einer Änderung der Erkennungseinstellungen neu.

## Ein Begriff wird falsch erkannt

- Deaktiviere die Fundstelle für den aktuellen Lauf.
- Verwende **Ignorieren**, wenn der Begriff künftig grundsätzlich nicht
  anonymisiert werden soll.
- Prüfe, ob ein generischer Begriff in der Ignore-Liste fehlt.
- Kontrolliere die Quelle: Ein KI-Treffer ist anders zu beurteilen als ein
  direkter Glossar-Treffer.

## Glossar direkt oder Fuzzy?

`Glossar · direkt` bedeutet eine exakte Übereinstimmung mit dem gespeicherten
Begriff. `Glossar · Fuzzy` bedeutet, dass die gefundene Schreibweise nur
ähnlich ist, beispielsweise wegen eines Tippfehlers.

Wenn ein Begriff trotz identischer Absicht als Fuzzy erscheint, kontrolliere
insbesondere Sonderzeichen, Leerzeichen und die tatsächliche Schreibweise im
Dokument. `eClaims` und `eClaims+` sind beispielsweise unterschiedliche
Begriffe.

## Probleme bei der lokalen LLM-Review-Assistenz

- **Fehlendes Zusatzpaket:** Erscheint der Hinweis `LLM-Paket nicht verfügbar`, installiere das Extra:
  ```powershell
  uv sync --extra gui --extra llm
  ```
- **Ollama nicht erreichbar:** Überprüfe, ob der Ollama-Dienst im Hintergrund läuft (standardmässig unter `http://127.0.0.1:11434`). Teste im Browser oder Terminal, ob `http://127.0.0.1:11434` antwortet.
- **Modell nicht gefunden / Tippfehler:** Stelle sicher, dass das Modell lokal heruntergeladen ist (`ollama list`) und der Name in den Einstellungen exakt übereinstimmt (z. B. `qwen3:8b`). Falls noch nicht vorhanden: `ollama pull qwen3:8b`.
- **Falscher Endpunkt:** Der Endpunkt muss eine lokale Loopback-Adresse sein und den Pfad `/v1` enthalten (Standard: `http://127.0.0.1:11434/v1`).
- **Unvollständige Batches / Timeout:** Wenn das Modell überlastet ist oder die GPU zu wenig VRAM hat, können einzelne Batches abbrechen. Die Applikation zeigt die ungeprüften Fundstellen transparent an. Du kannst geprüfte Fundstellen einzeln übernehmen oder den Lauf nach Entlastung des Systems wiederholen.

## Probleme bei der Ausgangskontrolle (Postcheck)

- **Start-Button deaktiviert:**
  - Prüfe, ob im oberen Bereich ein Modell konfiguriert ist (z. B. `qwen3:8b`).
  - Bestätige vorab die Checkbox zur 32k-Kontextunterstützung deines lokalen Servers.
  - Stelle sicher, dass zuvor eine Textanalyse durchgeführt wurde und keine andere Operation (z. B. Upload oder Analyse) aktiv ist.
- **Bestätigungs-Checkbox erscheint nach Modell- oder Endpunktwechsel erneut:** Das ist beabsichtigt und kein Fehler. Die 32k-Bestätigung gilt immer nur für die aktuell konfigurierte Modell-/Endpunkt-Kombination; wechselst du das Modell oder den API-Endpunkt, musst du einmal erneut bestätigen. Ein blosses erneutes Bereitwerden desselben Modells (z. B. nach Ablauf des Ollama-Keep-alive) verlangt dagegen **keine** erneute Bestätigung.
- **Ausgangskontrolle meldet unvollständigen Lauf (fehlerhafte Textpositionen):**
  - Kleine lokale Sprachmodelle können bei langen Dokumenten oder komplexen Tabellen ungenaue Zeichen-Offsets liefern.
  - Das System schützt dich vor Fehlplatzierungen: Entspricht der vom Modell gemeldete Zeichenbereich im Text nicht exakt dem Fundbegriff (`anonymisierter_text[start:end] == text`), wird der Treffer sicherheitsgerichtet verworfen und eine entsprechende Warnung angezeigt.
  - Überprüfe in solchen Fällen den Text stichprobenartig manuell oder teste ein größeres bzw. höher parametrisiertes lokales Modell.

## macOS

Die Anwendung wurde bisher nicht praktisch auf macOS getestet. Wenn der
native Start fehlschlägt, kann der Browser-Modus versucht werden. Ergebnisse
und Fehlermeldungen sollten für einen späteren Kompatibilitätstest festgehalten
werden.

