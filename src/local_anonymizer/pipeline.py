"""Pipeline orchestrator for end-to-end anonymization workflows."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from local_anonymizer.anonymizer import AnonymizationResult, LocalAnonymizer
from local_anonymizer.extractors import read_document


@dataclass
class PipelineResult:
    """Outcome of running the anonymization pipeline on a file."""
    input_file: Path
    anonymized_file: Path
    mapping_file: Optional[Path]
    report_file: Path
    anonymization_result: AnonymizationResult
    reversible: bool = True


class AnonymizationPipeline:
    """End-to-end anonymization pipeline for documents and text."""

    def __init__(
        self,
        language: str = "de",
        glossary: Optional[Dict[str, str]] = None,
        ignore_terms: Optional[Sequence[str]] = None,
        enabled_entities: Optional[Sequence[str]] = None,
        custom_labels: Optional[Dict[str, str]] = None,
        gliner_model: str = "urchade/gliner_multi_pii-v1",
        gliner_threshold: float = 0.55,
        enabled_glossary_entities: Optional[Sequence[str]] = None,
    ):
        self.anonymizer = LocalAnonymizer(
            language=language,
            glossary=glossary,
            ignore_terms=ignore_terms,
            enabled_entities=enabled_entities,
            custom_labels=custom_labels,
            gliner_model=gliner_model,
            gliner_threshold=gliner_threshold,
            enabled_glossary_entities=enabled_glossary_entities,
        )

    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        no_mapping: bool = False,
    ) -> PipelineResult:
        """
        Process a document: extract text, anonymize PII, and save outputs.

        Args:
            input_path: Path to input document.
            output_dir: Optional custom output directory.
            no_mapping: If True, do not persist a mapping table (permanent, non-reversible anonymization).

        Creates:
        - <stem>_anonymized.txt
        - <stem>_mapping.json (unless no_mapping is True)
        - <stem>_report.json
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        out_directory = Path(output_dir) if output_dir else input_file.parent
        out_directory.mkdir(parents=True, exist_ok=True)

        stem = input_file.stem
        anonymized_file = out_directory / f"{stem}_anonymized.txt"
        mapping_file = None if no_mapping else (out_directory / f"{stem}_mapping.json")
        report_file = out_directory / f"{stem}_report.json"

        # 1. Extraction
        raw_text = read_document(input_file)

        # 2. Anonymization
        result = self.anonymizer.anonymize(raw_text)

        # 3. Save outputs
        anonymized_file.write_text(result.anonymized_text, encoding="utf-8")
        if not no_mapping and mapping_file is not None:
            LocalAnonymizer.save_mapping(result.mapping, mapping_file)

        # 4. Save report with entity stats and review flags
        result_dict = result.to_dict()
        if no_mapping:
            # Clear mapping from report for true non-reversible mode
            result_dict["mapping"] = {}
            for entity_dict in result_dict.get("entities", []):
                entity_dict["original"] = "[REDACTED]"

        report_data = {
            "source_file": str(input_file.resolve()),
            "anonymized_file": str(anonymized_file.resolve()),
            "mapping_file": str(mapping_file.resolve()) if mapping_file else None,
            "reversible": not no_mapping,
            **result_dict,
        }
        report_file.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return PipelineResult(
            input_file=input_file,
            anonymized_file=anonymized_file,
            mapping_file=mapping_file,
            report_file=report_file,
            anonymization_result=result,
            reversible=not no_mapping,
        )

    @staticmethod
    def restore_file(
        anonymized_path: Union[str, Path],
        mapping_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Restore an anonymized text file using its mapping table.
        Runs entirely in-memory without loading NER/ML models.
        """
        anon_file = Path(anonymized_path)
        map_file = Path(mapping_path)

        if not anon_file.exists():
            raise FileNotFoundError(f"Anonymized file not found: {anon_file}")
        if not map_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {map_file}")

        if output_path:
            out_file = Path(output_path)
        else:
            stem = anon_file.stem.replace("_anonymized", "")
            out_file = anon_file.parent / f"{stem}_restored.txt"

        anonymized_text = anon_file.read_text(encoding="utf-8")
        mapping = LocalAnonymizer.load_mapping(map_file)

        restored_text = LocalAnonymizer.de_anonymize(anonymized_text, mapping)
        out_file.write_text(restored_text, encoding="utf-8")

        return out_file
