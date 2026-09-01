# Installation und Start

## Für wen dieser Abschnitt gedacht ist

Wenn du noch nie ein Repository von GitHub verwendet hast, ist das kein
Problem. Du musst weder Git noch Python selbst kennen. Folge den Schritten
unten in der angegebenen Reihenfolge. Bei Problemen kannst du später den
Abschnitt [Fehlerbehebung](06-fehlerbehebung.md) oder den Hinweis zur Nutzung
eines Programmierassistenten lesen.

## Voraussetzungen

Für die Kernpipeline brauchst du:

- einen Windows-Rechner; Windows 10 oder 11 ist der Referenzweg für diese
  Anleitung;
- eine Internetverbindung für die erstmalige Einrichtung und den
  Modell-Download;
- genügend freien Speicher für die lokale Python-Umgebung und die
  NLP-Modellgewichte;
- keine Administratorrechte, sofern die lokale Geräteverwaltung die
  Installation von `uv` nicht zusätzlich einschränkt.

Die Kernpipeline (GLiNER, EU-PII, Regex, Schweizer Prüfziffern) benötigt kein lokales Chat-LLM und läuft flüssig auf Standard-Laptops mit reiner CPU. Für den optionalen lokalen LLM-Triage-Layer (Phase 6A mit ~8B-Modellen) wird für den getesteten Referenzweg eine GPU-Beschleunigung empfohlen.

## Option A: Einrichtung durch einen Programmierassistenten

Wenn du mit Terminals und Python noch nicht vertraut bist, kannst du einen
Programmierassistenten wie Codex oder Anthropic Claude um Hilfe bitten. Das
kann sinnvoller sein, als mehrere Installationsschritte selbst zu erraten.

Gib dem Assistenten aber nur Zugriff auf den Ordner dieses Projekts. Er sollte
nicht auf deinen gesamten Benutzerordner, private Dokumente, Browserdaten,
Passwörter, Cloud-Laufwerke oder SSH-Schlüssel zugreifen können. Prüfe jeden
vorgeschlagenen Befehl, bevor du ihn ausführst, und stoppe ihn, wenn er Dateien
ausserhalb des Projektordners löschen, versenden oder verändern würde.

Ein geeigneter Auftrag wäre beispielsweise:

> Du arbeitest nur im Ordner dieses lokalen `local-anonymizer`-Repositories.
> Richte die Anwendung unter Windows ohne Administratorrechte ein. Zeige mir
> jeden Befehl vor der Ausführung, erkläre kurz seinen Zweck und verwende keine
> privaten Dateien. Installiere nur die Abhängigkeiten des Projekts, starte
> danach die GUI und melde mir verständlich, falls etwas fehlschlägt.

Der Assistent sollte nicht deine Originaldokumente zur Fehleranalyse an einen
Cloud-Dienst hochladen. Verwende für Rückfragen zunächst die fiktive Datei
[`cas-abschlussarbeit-beispiel.md`](../../examples/cas-abschlussarbeit-beispiel.md).

## Option B: Windows Schritt für Schritt selbst einrichten

### 1. Repository von GitHub herunterladen

Du brauchst nicht zwingend die Git-Befehle zu verwenden. Öffne im Browser die
Projektseite:

<https://github.com/scepbjoern/local-anonymizer>

Klicke dort auf **Code** und anschliessend auf **Download ZIP**. Speichere die
ZIP-Datei beispielsweise im Ordner **Downloads** und entpacke sie danach in
einen eigenen Ordner. Vermeide einen Pfad mit sehr langen Namen oder
Sonderzeichen, zum Beispiel:

```text
C:\Users\DeinName\local-anonymizer
```

Öffne anschliessend den entpackten Ordner. Du erkennst den richtigen
Projektordner daran, dass darin unter anderem `pyproject.toml`, `app.py` und
`README.md` liegen.

### 2. Ein Terminal im Projektordner öffnen

Am einfachsten geht es im Datei-Explorer:

1. Öffne den Projektordner.
2. Klicke oben in die Adresszeile.
3. Tippe `powershell` ein.
4. Drücke **Enter**.

Es öffnet sich ein PowerShell-Fenster, dessen Eingabeaufforderung bereits auf
den Projektordner zeigt. Du kannst das mit folgendem Befehl prüfen:

```powershell
Get-Location
```

Der angezeigte Pfad sollte auf deinen Ordner `local-anonymizer` enden. Falls
nicht, wechsle mit `Set-Location` in den richtigen Ordner:

```powershell
Set-Location "C:\Users\DeinName\local-anonymizer"
```

Alternativ kannst du im Explorer mit der rechten Maustaste in eine freie Stelle
des Ordners klicken und **Im Terminal öffnen** auswählen. Je nach Windows-
Version heisst der Eintrag **Open in Terminal**.

### 3. `uv` installieren

`uv` richtet die lokale Python-Umgebung ein und installiert die für das
Projekt benötigten Pakete. Es verändert nicht dein System-Python und benötigt
für diesen Installationsweg normalerweise keine Administratorrechte.

Prüfe zuerst, ob `uv` bereits verfügbar ist:

```powershell
uv --version
```

Wenn eine Versionsnummer angezeigt wird, kannst du direkt mit dem nächsten
Schritt fortfahren. Wenn der Befehl nicht gefunden wird, installiere `uv` mit
dem offiziellen Windows-Installationsskript:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Die offizielle Installationsseite ist unter
<https://docs.astral.sh/uv/getting-started/installation/> verfügbar. Der
Befehl lädt ein Skript aus dem Internet und führt es aus. Wenn du besonders
vorsichtig vorgehen möchtest, öffne zuerst nur die offizielle Seite, lies die
Installationsmöglichkeiten und entscheide danach bewusst.

Schliesse das PowerShell-Fenster nach der Installation und öffne ein neues
Terminal im Projektordner. Prüfe erneut:

```powershell
uv --version
```

### 4. Projektumgebung einrichten

Führe im Projektordner aus:

```powershell
uv sync --extra gui
```

Dabei werden eine lokale `.venv`-Umgebung und die benötigten Abhängigkeiten
angelegt. Der erste Vorgang kann je nach Internetverbindung mehrere Minuten
dauern.

#### Optional: Lokalen LLM-Review-Assistenten einrichten (`[llm]`)

Wenn du den optionalen lokalen LLM-Review-Assistenten (beschleunigter Power-User-Pfad) nutzen möchtest:

1. Installiere das Zusatzpaket:
   ```powershell
   uv sync --extra gui --extra llm
   ```
2. Installiere und starte [Ollama](https://ollama.com/) (Standard: `http://127.0.0.1:11434/v1`).
3. Lade das getestete Referenzmodell herunter (mehrere GB Download):
   ```powershell
   ollama pull qwen3:8b
   ```
   *(Als Alternativen wurden auch `ministral-3:8b` und `qwen3.5:9b` getestet; `qwen3:8b` ist die Standardempfehlung).*
4. Starte die Anwendung mit aktiviertem LLM-Extra:
   ```powershell
   uv run --extra gui --extra llm python app.py
   # Oder im Browser-Modus:
   uv run --extra gui --extra llm python app.py --browser
   ```
5. Aktiviere in den Einstellungen der Applikation die Option **Lokale LLM-Review-Assistenz** und trage `qwen3:8b` ein.

### 5. Anwendung starten (Standard-Kernpfad)

Starte die GUI mit:

```powershell
uv run --extra gui python app.py
```

Wenn alles funktioniert, öffnet sich ein eigenes Anwendungsfenster. Alternativ
kannst du auch `start_windows.vbs` im Explorer doppelt anklicken. Dieses Skript
startet die Anwendung ohne dauerhaft sichtbares Terminalfenster.

### 6. Ersten Funktionstest durchführen

Lade danach die fiktive Datei
[`cas-abschlussarbeit-beispiel.md`](../../examples/cas-abschlussarbeit-beispiel.md)
und führe den Ablauf aus [Kapitel 02](02-erster-anonymisierungslauf.md) durch.
So testest du zuerst die Installation, ohne vertrauliche Originaldokumente zu
verwenden.

## Kurzform für bereits erfahrene Nutzerinnen und Nutzer

```powershell
git clone https://github.com/scepbjoern/local-anonymizer.git
Set-Location local-anonymizer
uv sync --extra gui
uv run --extra gui python app.py
```

Die ZIP-Variante oben ist für Personen ohne Git-Erfahrung gedacht. Für die
erste Einrichtung kann zusätzlich `install_windows.bat` verwendet werden. Dieses
Skript richtet die lokale Umgebung ein und legt Verknüpfungen an.

## Browser-Modus

Wenn du kein natives Anwendungsfenster verwenden möchtest, kannst du die
Anwendung im Standardbrowser starten:

```powershell
uv run --extra gui python app.py --browser
```

## macOS: vorbereiteter, aber ungetesteter Weg

Im Repository liegen `install_mac.command` und `start_mac.command`. Sie sind
als vorbereitete Startmöglichkeit vorhanden, wurden aber noch nicht auf einem
Mac praktisch geprüft.

Ein möglicher Versuch ist:

```bash
./install_mac.command
./start_mac.command
```

Falls das native Fenster nicht startet, ist zusätzlich der Browser-Modus
vorbereitet:

```bash
uv run --extra gui python app.py --browser
```

Fehler auf macOS bitte nicht als Beweis einer falschen Bedienung interpretieren;
für diese Plattform fehlt noch ein echter Praxistest.

## Aufräumen

Die lokale Umgebung liegt im Projektordner. Zum Entfernen genügt es, den
Projektordner zu löschen. Der optionale `uv`-Cache liegt getrennt im
Benutzerprofil und kann bei Bedarf mit `uv cache clean` bereinigt werden.

## Was beim ersten Start passiert

Beim ersten Analyse-Lauf werden die benötigten lokalen Modellkomponenten
geladen. Das kann deutlich länger dauern als spätere Läufe. Die Daten werden
nicht an einen Cloud-Dienst gesendet.
