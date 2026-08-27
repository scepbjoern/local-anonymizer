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

## macOS

Die Anwendung wurde bisher nicht praktisch auf macOS getestet. Wenn der
native Start fehlschlägt, kann der Browser-Modus versucht werden. Ergebnisse
und Fehlermeldungen sollten für einen späteren Kompatibilitätstest festgehalten
werden.

