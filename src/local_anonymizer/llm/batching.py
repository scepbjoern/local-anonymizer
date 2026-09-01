"""Token-budgeted batching and prompt preparation for LLM Triage Layer (Phase 6A)."""

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


SYSTEM_TRIAGE_PROMPT = """Du bist ein hochpräziser Assistent zur Überprüfung von erkannten personenbezogenen Daten (PII) und Entitäten in Texten (Triage-Layer).
Deine Aufgabe:
1. Bewerte die vorgelegten Entitäts-Kandidaten im jeweiligen Kontext.
2. Entscheide für jeden Kandidaten (identifiziert über seine eindeutige 'occ_id'):
   - "keep": Die Entitäts-Kategorie ist korrekt. Optional kannst du eine präzisierende Rolle/Deskriptor als 'descriptor_suggestion' (z. B. "Kläger", "Projektleiter", "Kunde") vorschlagen.
   - "recategorize": Die Entität gehört zu einer anderen Kategorie (z. B. von LOCATION zu ORGANIZATION oder PERSON). Gib 'new_entity_type' an und optional 'descriptor_suggestion'.
   - "discard": Es handelt sich um ein False Positive (z. B. ein normales Wort, Berufsbezeichnung ohne Personenname, etc.) und soll ignoriert werden.
3. Gib für jede Entscheidung eine kurze Begründung ('reasoning', max. 1-2 Sätze) und eine Konfidenz ('high', 'medium', 'low') an.
4. Antworte AUSSCHLIESSLICH im geforderten JSON-Schema mit schema_version "1.0". Gib GENAU für jede übergebene occ_id einen Eintrag im Array 'items' zurück. Keine Markdown-Formatierung um das JSON herum."""


@dataclass
class TriageBatch:
    batch_index: int
    total_batches: int
    request_id: str
    document_revision: int
    document_hash: str
    occ_ids: List[str]
    candidates: List[Dict[str, Any]]
    system_prompt: str
    user_prompt: str

    @property
    def occ_id_set(self) -> Set[str]:
        return set(self.occ_ids)


def estimate_tokens(text: str) -> int:
    """Conservative token estimator (~3.5 chars per token for German + safety margin)."""
    if not text:
        return 0
    return int(math.ceil(len(text) / 3.5))


def build_candidate_prompt(
    candidates: List[Dict[str, Any]],
    request_id: str,
    doc_rev: int,
    doc_hash: str,
) -> str:
    """Build formatted user prompt containing candidate entries and strict JSON contract."""
    candidate_lines = []
    for idx, c in enumerate(candidates, start=1):
        occ_id = c["occ_id"]
        orig = c.get("original_text", "")
        ent_type = c.get("entity_type", "")
        role = c.get("role", "")
        ctx = c.get("context_snippet", orig)

        role_str = f", Aktuelle Rolle: '{role}'" if role else ""
        candidate_lines.append(
            f"Kandidat {idx}:\n"
            f"  - occ_id: \"{occ_id}\"\n"
            f"  - Erkannter Begriff: \"{orig}\"\n"
            f"  - Aktueller Typ: {ent_type}{role_str}\n"
            f"  - Kontext: \"{ctx}\""
        )

    candidates_formatted = "\n\n".join(candidate_lines)

    prompt = f"""Bitte überprüfe die folgenden {len(candidates)} Fundstellen:

{candidates_formatted}

Erstelle die Antwort als striktes JSON-Objekt mit folgender Struktur:
{{
  "schema_version": "1.0",
  "request_id": "{request_id}",
  "document_revision": {doc_rev},
  "document_hash": "{doc_hash}",
  "items": [
    {{
      "occ_id": "<jeweilige_occ_id>",
      "action": "keep" | "recategorize" | "discard",
      "new_entity_type": "<nur bei recategorize>",
      "descriptor_suggestion": "<optionaler Rollenvorschlag>",
      "reasoning": "<kurze Begründung>",
      "confidence": "high" | "medium" | "low"
    }}
  ]
}}"""
    return prompt


def prepare_triage_batches(
    candidates: List[Dict[str, Any]],
    document_revision: int,
    document_hash: str,
    max_tokens_per_batch: int = 3000,
    max_items_per_batch: int = 15,
) -> List[TriageBatch]:
    """
    Split a list of candidate occurrences into token-budgeted sequential batches.
    Each batch receives an exact set of occ_ids and strict envelope bindings.
    """
    if not candidates:
        return []

    system_tokens = estimate_tokens(SYSTEM_TRIAGE_PROMPT)
    base_envelope_tokens = 150  # overhead for JSON structure

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_tokens = system_tokens + base_envelope_tokens

    for cand in candidates:
        ctx = cand.get("context_snippet", "")
        orig = cand.get("original_text", "")
        # Estimate prompt tokens + expected response tokens (~80 tokens per item)
        item_prompt_tokens = estimate_tokens(ctx) + estimate_tokens(orig) + 50
        item_response_tokens = 80
        total_item_tokens = item_prompt_tokens + item_response_tokens

        would_exceed_tokens = (current_tokens + total_item_tokens) > max_tokens_per_batch
        would_exceed_count = len(current_batch) >= max_items_per_batch

        if current_batch and (would_exceed_tokens or would_exceed_count):
            batches.append(current_batch)
            current_batch = [cand]
            current_tokens = system_tokens + base_envelope_tokens + total_item_tokens
        else:
            current_batch.append(cand)
            current_tokens += total_item_tokens

    if current_batch:
        batches.append(current_batch)

    total_batches = len(batches)
    result_batches: List[TriageBatch] = []

    for b_idx, batch_cands in enumerate(batches, start=1):
        req_id = f"req_{uuid.uuid4().hex[:12]}"
        user_prompt = build_candidate_prompt(
            batch_cands,
            request_id=req_id,
            doc_rev=document_revision,
            doc_hash=document_hash,
        )
        occ_ids = [c["occ_id"] for c in batch_cands]
        result_batches.append(
            TriageBatch(
                batch_index=b_idx,
                total_batches=total_batches,
                request_id=req_id,
                document_revision=document_revision,
                document_hash=document_hash,
                occ_ids=occ_ids,
                candidates=batch_cands,
                system_prompt=SYSTEM_TRIAGE_PROMPT,
                user_prompt=user_prompt,
            )
        )

    return result_batches
