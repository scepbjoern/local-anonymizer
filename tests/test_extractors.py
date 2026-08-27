import pytest
from local_anonymizer.extractors import read_document, extract_text_from_pdf, extract_text_from_txt, UnsupportedFileFormatError
from pathlib import Path

def test_scanned_pdf_error_handling(mocker):
    # Mock pymupdf to return empty text
    mock_doc = mocker.MagicMock()
    mock_doc.__enter__.return_value = mock_doc
    mock_page = mocker.MagicMock()
    mock_page.get_text.return_value = "   \n"
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.page_count = 1
    
    mocker.patch("pymupdf.open", return_value=mock_doc)
    
    with pytest.raises(ValueError, match="scanned/image-based PDF requiring OCR"):
        extract_text_from_pdf(Path("dummy.pdf"))

def test_multi_encoding_txt(tmp_path):
    txt_file = tmp_path / "test.txt"
    # Write in cp1252
    txt_file.write_bytes("Müller äöü €".encode("cp1252"))
    
    text = extract_text_from_txt(txt_file)
    assert "Müller äöü €" in text

def test_unsupported_file_format(tmp_path):
    unknown_file = tmp_path / "dummy.unknown"
    unknown_file.write_text("content", encoding="utf-8")
    with pytest.raises(UnsupportedFileFormatError):
        read_document(unknown_file)


def test_pdf_to_markdown_structure(tmp_path):
    import pymupdf
    pdf_path = tmp_path / "sample_struct.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hauptueberschrift", fontsize=24)
    page.insert_text((50, 100), "Dies ist ein normaler Textabsatz.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    extracted = extract_text_from_pdf(pdf_path)
    assert "Hauptueberschrift" in extracted
    assert "Dies ist ein normaler Textabsatz." in extracted


def test_anonymize_within_docx_markdown_headings_preserves_structure(tmp_path):
    import docx
    from local_anonymizer.extractors import extract_text_from_docx
    from local_anonymizer.anonymizer import LocalAnonymizer

    docx_path = tmp_path / "headings.docx"
    doc = docx.Document()
    doc.add_heading("Projektleiter Julia Meier Bericht", level=1)
    doc.save(str(docx_path))

    extracted = extract_text_from_docx(docx_path)
    assert "# Projektleiter Julia Meier Bericht" in extracted

    anonymizer = LocalAnonymizer()
    roles = {"julia meier": "STUDENT"}
    result = anonymizer.anonymize(extracted, format_mode="numbered_role", roles=roles)

    assert "# Projektleiter [PERSON_1_STUDENT] Bericht" in result.anonymized_text
    
    restored = LocalAnonymizer.de_anonymize(result.anonymized_text, result.mapping)
    assert restored == extracted


def test_anonymize_within_markdown_tables_preserves_structure():
    from local_anonymizer.anonymizer import LocalAnonymizer

    anonymizer = LocalAnonymizer()
    table_md = (
        "| Name | Rolle | Organisation |\n"
        "| :--- | :--- | :--- |\n"
        "| Julia Meier | Studentin | ZHAW |\n"
        "| Max Mustermann | Dozent | ETH |\n"
    )
    roles = {"julia meier": "STUDENT", "max mustermann": "DOZENT"}
    result = anonymizer.anonymize(table_md, format_mode="numbered_role", roles=roles)

    assert "[PERSON_1_STUDENT]" in result.anonymized_text
    assert "[PERSON_2_DOZENT]" in result.anonymized_text
    lines = result.anonymized_text.strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        assert line.startswith("|") and line.endswith("|")

    # Roundtrip check
    restored = LocalAnonymizer.de_anonymize(result.anonymized_text, result.mapping)
    assert restored == table_md


def test_create_docx_from_markdown_with_native_styles():
    import docx
    from local_anonymizer.extractors import create_docx_from_markdown, save_markdown_to_docx_bytes

    md_content = (
        "# Hauptkapitel\n\n"
        "Dies ist ein normaler Textabsatz mit Informationen.\n\n"
        "## Unterabschnitt\n\n"
        "- Punkt 1\n"
        "- Punkt 2\n\n"
        "| Spalte A | Spalte B |\n"
        "| :--- | :--- |\n"
        "| Wert 1 | Wert 2 |\n"
    )

    doc = create_docx_from_markdown(md_content)
    # Verify paragraph styles and text
    paras = [p for p in doc.paragraphs if p.text]
    assert len(paras) >= 4
    assert paras[0].text == "Hauptkapitel"
    assert "Heading 1" in paras[0].style.name or "Überschrift 1" in paras[0].style.name or "heading 1" in paras[0].style.name.lower()
    assert paras[1].text == "Dies ist ein normaler Textabsatz mit Informationen."
    assert paras[2].text == "Unterabschnitt"
    assert "Heading 2" in paras[2].style.name or "Überschrift 2" in paras[2].style.name or "heading 2" in paras[2].style.name.lower()

    # Verify table
    assert len(doc.tables) == 1
    assert doc.tables[0].cell(0, 0).text == "Spalte A"
    assert doc.tables[0].cell(1, 0).text == "Wert 1"

    # Verify raw bytes generation
    raw_bytes = save_markdown_to_docx_bytes(md_content)
    assert len(raw_bytes) > 0

