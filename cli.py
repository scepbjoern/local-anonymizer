"""Command Line Interface for local-anonymizer."""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress background library warnings to keep CLI output clean
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from local_anonymizer.pipeline import AnonymizationPipeline


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration file if provided, otherwise return empty dict."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"⚠️ Warning: Config file not found at '{config_path}'. Proceeding with defaults.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse config file '{config_path}': {e}. Proceeding with defaults.")
        return {}


def handle_anonymize(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    config = load_config(args.config)

    language = args.language or config.get("language", "de")
    glossary = config.get("glossary", {})
    ignore_terms = config.get("ignore_terms", [])
    enabled_entities = config.get("enabled_entities", None)
    custom_labels = config.get("custom_labels", None)
    threshold = config.get("threshold", 0.55)
    model = args.model or config.get("model", "urchade/gliner_multi_pii-v1")

    print("=" * 70)
    print("🔒 LOCAL ANONYMIZER - PII SCRUBBING")
    print("=" * 70)
    print(f"📄 Input File : {input_path}")
    print(f"🌐 Language   : {language}")
    print(f"🤖 Model      : {model}")
    print(f"🎯 Threshold  : {threshold}")
    print(f"🔄 Reversible : {'No (--no-mapping enabled: permanent redaction)' if args.no_mapping else 'Yes (mapping table saved)'}")

    if glossary:
        print(f"📚 Glossary   : {len(glossary)} custom terms loaded")
        for term, cat in list(glossary.items())[:5]:
            print(f"   • '{term}' -> {cat}")
        if len(glossary) > 5:
            print(f"   • ... and {len(glossary) - 5} more")

    if custom_labels:
        print(f"🏷️  Custom Labels: {len(custom_labels)} defined")
        for label, cat in custom_labels.items():
            print(f"   • '{label}' -> {cat}")

    if ignore_terms:
        print(f"🚫 Ignored    : {', '.join(ignore_terms)}")

    if enabled_entities:
        print(f"🎯 Entities   : {', '.join(enabled_entities)}")
    print("-" * 70)

    print("⏳ Processing document...")
    try:
        pipeline = AnonymizationPipeline(
            language=language,
            glossary=glossary,
            ignore_terms=ignore_terms,
            enabled_entities=enabled_entities,
            custom_labels=custom_labels,
            gliner_threshold=threshold,
            gliner_model=model,
        )

        result = pipeline.process_file(
            input_path=input_path,
            output_dir=args.output_dir,
            no_mapping=args.no_mapping,
        )
    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        sys.exit(1)

    print("\n✅ DETECTED ENTITIES:")
    entities = result.anonymization_result.entities
    if not entities:
        print("   (No sensitive entities detected)")
    else:
        for ent in entities:
            flag = " ⚠️ [NEEDS REVIEW]" if ent.needs_review else ""
            display_val = "[REDACTED]" if args.no_mapping else f"'{ent.original_text}'"
            print(
                f"   • {ent.placeholder:<18} | {ent.entity_type:<15} | Score: {ent.score:.2f} | {display_val}{flag}"
            )

    print("\n" + "=" * 70)
    print("📝 ANONYMIZED TEXT PREVIEW:")
    print("=" * 70)
    lines = result.anonymization_result.anonymized_text.splitlines()
    preview = "\n".join(lines[:20])
    print(preview)
    if len(lines) > 20:
        print(f"\n... [{len(lines) - 20} more lines]")
    print("=" * 70)

    print("\n💾 SAVED ARTIFACTS:")
    print(f"   1. Anonymized Text : {result.anonymized_file}")
    if result.mapping_file:
        print(f"   2. Mapping Table   : {result.mapping_file}")
    else:
        print("   2. Mapping Table   : [DISABLED (--no-mapping)]")
    print(f"   3. Audit Report    : {result.report_file}")
    print("=" * 70)


def handle_restore(args: argparse.Namespace) -> None:
    anon_path = Path(args.anonymized_file)
    map_path = Path(args.mapping_file)

    if not anon_path.exists():
        print(f"❌ Error: Anonymized / LLM output file not found: {anon_path}")
        sys.exit(1)
    if not map_path.exists():
        print(f"❌ Error: Mapping file not found: {map_path}")
        sys.exit(1)

    print("=" * 70)
    print("🔓 LOCAL ANONYMIZER - DE-ANONYMIZATION / RESTORE")
    print("=" * 70)
    print(f"📄 Target File  : {anon_path}")
    print(f"🗺️  Mapping File : {map_path}")
    print("-" * 70)

    try:
        restored_file = AnonymizationPipeline.restore_file(
            anonymized_path=anon_path,
            mapping_path=map_path,
            output_path=args.output_file,
        )
    except Exception as e:
        print(f"\n❌ Error restoring file: {e}")
        sys.exit(1)

    print("✅ Successfully restored original entities!")
    print(f"💾 Restored File : {restored_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Local Privacy-First PII Anonymization & Restoration CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: anonymize
    anon_parser = subparsers.add_parser("anonymize", help="Anonymize a document (.txt, .docx, .pdf)")
    anon_parser.add_argument("input_file", help="Path to input document")
    anon_parser.add_argument("--output-dir", "-o", default=None, help="Directory to save outputs")
    anon_parser.add_argument("--config", "-c", default=None, help="Path to config.json (glossary, ignore terms, entity filters)")
    anon_parser.add_argument("--language", "-l", default=None, help="Language code (default: de)")
    anon_parser.add_argument("--model", "-m", default=None, help="GLiNER model identifier")
    anon_parser.add_argument("--no-mapping", action="store_true", help="Scrub PII permanently without saving mapping table")

    # Command: restore
    restore_parser = subparsers.add_parser("restore", help="De-anonymize / restore text using a mapping table")
    restore_parser.add_argument("anonymized_file", help="Path to document containing [CATEGORY_N] placeholders")
    restore_parser.add_argument("mapping_file", help="Path to *_mapping.json file")
    restore_parser.add_argument("--output-file", "-o", default=None, help="Path to save restored document")

    # Handle direct execution without subcommand (e.g. `uv run cli.py document.docx`)
    if len(sys.argv) > 1 and sys.argv[1] not in ["anonymize", "restore", "-h", "--help"]:
        sys.argv.insert(1, "anonymize")

    args = parser.parse_args()

    if args.command == "anonymize":
        handle_anonymize(args)
    elif args.command == "restore":
        handle_restore(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
