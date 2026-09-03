# Phase 6B: LLM-Ausgangskontrolle (Postcheck)

**Status:** Implementiert, inkl. Minimaler Entkopplung, F1 Client-Binding sowie U1/U2-Nachbesserung
(Review-Bedienelemente im Review-Assistenten, Kontextanzeige nach Modell-/Endpunktwechsel). Automatisierte
Regressionstests grün; finale GUI-Abnahme durch Björn/PLAI-00b steht für den Folgecommit noch aus.
**Basis-Commit:** `cbf06c006a37a08e3e7745e7980a5ffc6f21a990`

---

## 1. Zweck und Abgrenzung (Non-Goals)
Die Ausgangskontrolle (Phase 6B) dient als zusätzliches, manuell gestartetes Prüfverfahren. Das lokale LLM liest den bereits *anonymisierten Ausgabetext* und sucht nach PII (Nachzüglern), die von den regulären Scannern (und der Phase-6A-Triage) übersehen wurden.

**Gemeinsame Konfiguration & Entkopplung:**
- Die Ausgangskontrolle nutzt dieselben lokalen Modelleinstellungen (Ollama/Generic Endpoint, Modellname, Vorladen, Test) wie die Review-Assistenz.
- Sie ist jedoch **vollständig entkoppelt**: Der Postcheck kann unabhängig davon gestartet werden, ob der LLM-Review (Stufe 2) in den Einstellungen aktiviert ist oder übersprungen wurde.

**Abgrenzung:**
- Dies ist **kein** Ersatz für die reguläre Erkennung oder die Triage (Phase 6A), deren bestehendes Verhalten unverändert erhalten bleibt.
- Es gibt **keine** automatische Übernahme von Funden oder einen Endlos-Prüfkreislauf.
- Smart-Linking oder globale Muster-Ersetzungen sind nicht Teil dieses MVP.

---

## 2. Benutzerablauf (Start, Fehler, Schließen)
1. **Startvoraussetzung & Erwerb der Sperre:** Unter der Anonymisierungsvorschau erscheint ein neuer Button `"Ausgangskontrolle starten (Nachzügler suchen)"`. Der Start ist nur möglich, wenn keine kollidierenden Arbeiten laufen (kein Upload, keine Extraktion, keine reguläre Analyse, keine laufende Phase-6A-Triage). Der Start erwirbt eine sessionlokale Sperre.
2. **LLM-Lauf & Bearbeitungssperre:** Mit dem Klick friert die App den aktuellen sichtbaren Ausgabetext (`current_anon_text`) sowie dessen Segmentzuordnung für diesen Lauf unveränderlich ein und sendet ihn an den lokalen Provider. Gleichzeitig werden das gesamte Rohtextfeld, die Haupt-Review-Tabelle und relevante Einstellungen (Profil, Kategorien, Ignore, Format) im UI gesperrt (`readonly`/`disabled`) und serverseitig gegen Mutation geschützt.
3. **Vorschlagsliste & Auswahlphase:** Die App präsentiert die Ergebnisse in einer separaten Tabelle (Postcheck-Vorschläge). Die Bearbeitungssperre bleibt während der gesamten Auswahlphase aktiv. Abbruch und Schließen bleiben erreichbar.
4. **Sammelübernahme:** Der Benutzer wählt per Checkbox einen oder mehrere Funde aus und klickt auf `"Ausgewählte übernehmen"`. Die Funde werden atomar in die regulären Entitäten eingefügt und die Vorschau erneuert.
5. **Abschluss, Abbruch & Entsperrung:** 
   - Nach erfolgreicher Übernahme verfallen alle verbliebenen, nicht ausgewählten Postcheck-Vorschläge und das UI wird entsperrt.
   - Bei explizitem Verwerfen/Schließen, Fehler, Timeout, Disconnect oder Null-Funden greift ein definierter Abschluss- und Entsperrpfad. 
   - Ein bloßer Abbruch erfordert kein Zurückschreiben einer kompletten alten AppState-Kopie, da der bestehende Zustand während der Sperre unverändert blieb.
   - Alte Antworten, alte Cleanup-Aktionen (z.B. verzögerte `finally`-Aufrufe) oder veraltete Apply-Aktionen dürfen einen neu gestarteten Lauf niemals beeinflussen oder entsperren.

---

## 3. Scope, Suchumfang und Kategorienvalidierung
Die Ausgangskontrolle erzwingt das eingestellte Kategorienprofil und Ignorierregeln serverseitig:
- **Erlaubte Modi:** Bei `all` und `explicit_eupii` darf das LLM zusätzliche Funde suchen und vorschlagen.
- **Blockierte Modi:** Bei `explicit_only` (nur Glossar/Manuell) und `off` blockiert die App neue Postcheck-Ergebnisse vollständig.
- **Ignore-Schutz:** Explizit ignorierte Begriffe (`ignored_terms`) und manuell deaktivierte Fundstellen werden niemals durch das LLM übersteuert.
- **Serverseitige Validierung:** Eine Pydantic-Strukturvalidierung allein genügt nicht. Die App validiert serverseitig:
  - Zulässigkeit der vorgeschlagenen Kategorie im aktiven Profil.
  - Strikte Integer-Indizes (`0 <= start < end <= Textlänge`).
  - Exakter Slice-Match auf den eingefrorenen Prüftext des aktiven Laufs.
  - Fund liegt vollständig in einem zulässigen, unveränderten Textsegment.

---

## 4. Vertrag zur Sammelübernahme (Batch Apply)
Die Übernahme erfolgt für alle ausgewählten Vorschläge gemeinsam:
- **Konfliktprüfung:** Alle ausgewählten Funde werden vorab sowohl gegeneinander (keine gegenseitigen Intervall-Überschneidungen) als auch gegen bestehende Platzhalter/Entitäten auf Kollisionen geprüft.
- **Atomarität:** Das Einfügen ist strikt atomar (Alles oder nichts). Schlägt auch nur ein Fund fehl (z.B. späte Konflikte), bricht der gesamte Vorgang ab, ohne `entity_groups`, Overrides, Zähler oder Revisionen auch nur teilweise zu verändern.
- **Homonym-Sicherheit:** Für die neuen Funde werden kollisionsfreie kanonische IDs (`occ_id`) generiert und bestehende Homonym-Regeln beachtet.

---

## 5. Sperr-, Mapping- und Lifecycle-Vertrag (UI-Sperre & Guard)
Um Inkonsistenzen zu verhindern, wird die App während des Postchecks gesperrt.

**Bearbeitungssperre & Guard:**
- **UI- & Server-Sperre:** Während des Laufs und der Auswahlphase sind alle zustandsverändernden Operationen (Textänderung, Upload/Reset, Profil/Kategorien/Ignore-Listen, Formatwechsel, Triage-Übernahme) gesperrt. Änderungsfunktionen prüfen eine sessionlokale Sperre und weisen Mutationen ab. Lesen und Kopieren bleibt möglich.
- **Run-ID / Generation Guard:** Es wird eine sessionlokale Lauf-ID (Generation-Guard) geführt. Verspätete Antworten abgebrochener Läufe oder Cleanup-Aktionen alter Läufe (z.B. aus dem `finally`-Block) werden verworfen und dürfen den aktiven Lauf nicht stören oder vorzeitig entsperren.

**Rückführung (Back-Mapping) & Schemavalidierung:**
Der beim Start eingefrorene Prüftext und dessen Segmentzuordnung bleiben unveränderlich an den Lauf gebunden.
Das LLM liefert den übersehenen Text (`text`) sowie exklusive Python-Indizes (`output_start`, `output_end`) bezogen auf den Anonymisierungstext.
1. Strikte Schema- und Typprüfung: `schema_version`, `request_id` und `items` sind Pflichtfelder. Indizes müssen strikte Integer sein (Booleans, Strings und Floats werden abgewiesen). Der Fundtext wird nicht global getrimmt, um exakte Slices zu bewahren.
2. `frozen_anon_text[output_start:output_end] == text`.
3. Das Intervall muss vollständig in *einem* unveränderten Textsegment liegen.
4. Über das Segment-Offset wird `raw_start`:`raw_end` berechnet.
5. `raw_text[raw_start:raw_end] == text`. Überschneidungen mit bestehenden Platzhaltern/Entity-Intervallen oder manuell deaktivierten Entitäten führen zur Ablehnung.
6. Unterscheidung von Null-Funden vs. Modellfehlern: Nur eine valide leere Liste (`items: []`) gilt als "keine Nachzügler". Treten Slice-Fehler auf, wird dies wahrheitsgetreu als unvollständige Prüfung mit fehlerhaften Modellpositionen gemeldet (keine falsche Entwarnung). Reines Filtern durch Profil/Ignore wird gesondert ausgewiesen.

**Gruppenidentität & Rollback-Garantie:**
- Eindeutige Gruppen-IDs: Bei Namensgleichheit mit abweichendem Typ (Homonyme) oder abweichenden Rollen werden keine identischen `group_id`s vergeben, sondern nach Phase 5a eindeutige Split-Gruppen mit `OccurrenceOverride` angelegt.
- Deaktivierte Gruppen bleiben geschützt und werden nicht unabsichtlich reaktiviert.
- Vollständiger Rollback: Bei Fehlern im Vorschau- oder Übernahmepfad werden sämtliche veränderten Felder (`entity_groups`, `occurrence_overrides`, `mapping`, `anon_text`, `report`, `selected_ids`, `colliding_roles`, `status_msg`, `run_id`, `is_postcheck_active`) atomar auf den Vorzustand zurückgesetzt.

---

## 6. Budget- und Provider-Vertrag (32.000 Tokens Limit)
Die Ausgangskontrolle nutzt Einzelaufrufe ohne Chunking und setzt ausreichenden Speicher und praktikable Laufzeit für längere Kontexte voraus.

**Vertragsdetails & Limitierungen:**
- **Gesamtbudget:** `32.000` Tokens (ohne stille Textkürzung, kein Chunking/Aufteilen in mehrere Aufrufe). Tokenzählung ist eine gekennzeichnete Schätzung (`estimate_tokens`).
- **Antwortreserve:** `4.096` Tokens. Diese wird im tatsächlichen Postcheck-Request explizit durchgesetzt (`max_tokens=4096`), um Overflow zu verhindern.
- **Maximaler Input:** Es verbleiben max. `27.904` Tokens für die vollständige Eingabe (System-Prompt, Instruktionen, JSON-Schema, Chat-Puffer und Dokumenttext). Wird dieser überschritten, erfolgt ein sofortiger, harter Abbruch vor dem API-Call.
- **Provider-Trennung & Grenzen:** Der Aufruf ändert das bestehende Triage-Verhalten (Phase 6A) nicht. Bestehende lokale Provider- und Loopbackgrenzen bleiben erhalten.
- **Abschluss-Status:** Ein fehlender, unerwarteter oder unvollständiger Abschlussstatus (z. B. `finish_reason == "length"` oder Schema-Abbruch) wird zwingend abgewiesen, auch bei syntaktisch zufällig gültigem JSON. Eine Teilantwort wird niemals als erfolgreiche Vollprüfung ausgewiesen.

**Kontextbestätigung (Nutzerverantwortung):**
- Ist der tatsächliche Serverkontext des Providers programmatisch unbekannt, darf eine ausdrückliche Nutzerbestätigung den Start ermöglichen.
- Die Bestätigung wird sichtbar gekennzeichnet als *"vom Nutzer bestätigt, nicht technisch geprüft"*.
- Ändert sich das Modell, der Endpoint oder eine relevante Konfiguration, verfällt die Bestätigung (`invalidate_llm_config`). Anzeige (Badge/Checkbox) und Serverprüfung teilen sich dieselbe Gültigkeitsregel (`compute_postcheck_bound_key` / `is_postcheck_context_confirmed`), damit beide nie auseinanderlaufen.
- Ein reines Verlieren der Modellbereitschaft (Ollama-Keep-alive-Ablauf, ein fehlgeschlagener Verbindungstest, ein nicht verifizierbares Vorladen) ist **keine** Konfigurationsänderung und darf die Bestätigung nicht verwerfen (`invalidate_llm_ready` bleibt readiness-only; siehe Handoff 20260903-1256, U2).
- Bekannte, zu kleine Grenzen dürfen durch die Bestätigung nicht übergangen werden.
- Dem Nutzer wird transparent erläutert, dass eine serverseitige Kürzung durch den Provider möglich ist (Restrisiko ohne Garantie für Seitenzahl oder Vollständigkeit).

---

## 7. Konsolidierte Akzeptanzkriterien
*(Pflichtprüfungen zu Zustand/Offsets/Atomarität sind automatisiert zu testen, GUI-Abläufe separat manuell)*

1. **Mapping:** Identisches Wort als Person und Nicht-Person, Wortteil sowie verschobene Offsets nach Platzhaltern werden exakt zurückgeführt.
2. **Intervall-Sicherheit:** Falsche Output-Slices, die im Original-Slice rein zufällig passen, ungültige Indizes oder Überschneidungen mit Platzhaltern werden serverseitig hart abgewiesen.
3. **Kategorien- und Scope-Prüfung:** Alle vier Modi (`all`, `explicit_eupii`, `explicit_only`, `off`) sowie explizite Ignore-Werte werden serverseitig erzwungen.
4. **Sperre & Zustand:** Sämtliche zustandsverändernden Funktionen (Upload, Text, Profil, Kategorien, Ignore, Triage-Übernahme) werden serverseitig und im UI verweigert, solange der Postcheck aktiv oder in der Auswahlphase ist.
5. **Lifecycle & Guard:** Verspätete Antworten alter Läufe nach Neustart werden abgewiesen. Der alte Cleanup (aus `finally`) entsperrt keinen neu gestarteten Lauf. Ende der Auswahlphase oder Abbruch entsperrt das UI sicher ohne Full-State-Restore.
6. **Atomarität / Fehler:** Bei einem Apply-Fehler (z.B. späte Konflikte) erfolgt keinerlei Teilmutation. Das Mapping, der Restore und eine Neuanalyse bleiben absolut konsistent. Wiederholungs-Apply wird sicher behandelt.
7. **Sammelübernahme:** Überlappende Funde innerhalb desselben Batch-Applys blockieren sich gegenseitig sicher. Nach erfolgreicher Übernahme verfallen alle verbliebenen, nicht ausgewählten Vorschläge sofort.
8. **Budget & Reserve:** Budgetüberschreitung bricht ohne Request ab. Der tatsächliche API-Request setzt `max_tokens=4096` durch. Truncation (`finish_reason == "length"` oder fehlender/unerwarteter Status) führt zum Fehlerabbruch.
9. **Kontextbestätigung:** Die Abfrage erscheint bei unbekanntem Limit, verfällt bei einem echten Modell-/Endpoint-Wechsel und warnt vor serverseitiger Kürzung. Bekannte zu kleine Limits blockieren weiterhin. Ein reiner Bereitschafts-/Keep-alive-Verlust bei unverändertem Modell/Endpoint verfällt die Bestätigung **nicht**, und Anzeige (Badge/Checkbox) sowie Serverprüfung bleiben nach einem Wechsel konsistent (kein veraltetes Badge bei gleichzeitig blockiertem Start).
10. **Phase-6A-Regression:** Alle bestehenden Phase-6A-Funktionen und -Tests bleiben vollständig und unverändert erhalten.
11. **Minimale Entkopplung:** Die Ausgangskontrolle ist auch bei deaktiviertem LLM-Review vollumfänglich start- und bedienbar. Die Modelleinstellungen und Statusanzeigen bleiben für beide Funktionen zentral zugänglich.
12. **NiceGUI-Client-Binding (F1):** Alle asynchronen Hintergrund-Worker-Callbacks binden explizit an den aktiven UI-Client-Kontext (`with client:`), um Re-Rendering- und Notification-Abbrüche zu verhindern.
13. **Review-Bedienplatzierung (U1):** Der Schalter „LLM-Review aktivieren“ und die Checkbox „LLM-Review direkt an die Textanalyse anschließen“ sind Teil des Review-Assistenten-Panels (nicht der gemeinsamen Modelleinstellungen), bleiben dort auch bei ausgeschaltetem Review sichtbar/bedienbar, und bilden keine zweite Konfigurationskopie (dieselben `state.config`-Felder wie zuvor).
