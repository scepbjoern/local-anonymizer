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


def test_create_docx_from_markdown_inline_formatting():
    from local_anonymizer.extractors import create_docx_from_markdown

    md = "Hier ist **fetter Text**, *kursiver Text*, ***fett-kursiv*** und `Consolas-Code`."
    doc = create_docx_from_markdown(md)
    p = doc.paragraphs[0]

    assert "**" not in p.text
    assert "*" not in p.text
    assert "`" not in p.text
    assert p.text == "Hier ist fetter Text, kursiver Text, fett-kursiv und Consolas-Code."

    bold_runs = [r for r in p.runs if r.bold and not r.italic]
    assert any("fetter Text" in r.text for r in bold_runs)

    italic_runs = [r for r in p.runs if r.italic and not r.bold]
    assert any("kursiver Text" in r.text for r in italic_runs)

    bi_runs = [r for r in p.runs if r.bold and r.italic]
    assert any("fett-kursiv" in r.text for r in bi_runs)

    code_runs = [r for r in p.runs if r.font.name == "Consolas"]
    assert any("Consolas-Code" in r.text for r in code_runs)


def test_docx_custom_outline_levels_and_lists(tmp_path):
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from local_anonymizer.extractors import extract_text_from_docx

    docx_path = tmp_path / "custom_doc.docx"
    doc = docx.Document()

    # 1. Custom heading via outline level
    p_h1 = doc.add_paragraph("Firmeninterne Hauptueberschrift")
    pPr = p_h1._p.get_or_add_pPr()
    outlineLvl = OxmlElement("w:outlineLvl")
    outlineLvl.set(qn("w:val"), "0")
    pPr.append(outlineLvl)

    # 2. German Heading 2 style
    p_h2 = doc.add_heading("Zweiter Unterabschnitt", level=2)

    # 3. Bullet item
    doc.add_paragraph("Erster Aufzählungspunkt", style="List Bullet")

    # 4. Numbered item
    doc.add_paragraph("Nummerierter Punkt Eins", style="List Number")

    doc.save(str(docx_path))

    extracted = extract_text_from_docx(docx_path)
    assert "# Firmeninterne Hauptueberschrift" in extracted
    assert "## Zweiter Unterabschnitt" in extracted
    assert "- Erster Aufzählungspunkt" in extracted
    assert "1. Nummerierter Punkt Eins" in extracted


def test_docx_bold_paragraph_false_positive_prevention(tmp_path):
    import docx
    from docx.shared import Pt
    from local_anonymizer.extractors import extract_text_from_docx

    docx_path = tmp_path / "warning_doc.docx"
    doc = docx.Document()

    # Regular paragraph with bold text that is NOT a heading (has full sentence with period, normal size)
    p_warn = doc.add_paragraph()
    run = p_warn.add_run("WICHTIGER HINWEIS: Dieser Text ist fett und wichtig, aber ein normaler Fließtextabsatz.")
    run.bold = True
    run.font.size = Pt(11)

    doc.save(str(docx_path))

    extracted = extract_text_from_docx(docx_path)
    # Must NOT start with '#'
    assert not extracted.startswith("#")
    assert "WICHTIGER HINWEIS: Dieser Text ist fett und wichtig, aber ein normaler Fließtextabsatz." in extracted


def test_anonymize_within_csv_markdown_tables_preserves_structure(tmp_path):
    from local_anonymizer.extractors import read_document
    from local_anonymizer.anonymizer import LocalAnonymizer

    csv_path = tmp_path / "team.csv"
    csv_path.write_text(
        "Name,Rolle,Hochschule\n"
        "Julia Meier,Studentin,ZHAW\n"
        "Max Mustermann,Dozent,ETH\n",
        encoding="utf-8"
    )

    extracted_md = read_document(csv_path)
    # Verify CSV was parsed into markdown table
    assert "| Name | Rolle | Hochschule |" in extracted_md
    assert "| :--- | :--- | :--- |" in extracted_md
    assert "| Julia Meier | Studentin | ZHAW |" in extracted_md

    anonymizer = LocalAnonymizer()
    roles = {"julia meier": "STUDENT", "max mustermann": "DOZENT"}
    result = anonymizer.anonymize(extracted_md, format_mode="numbered_role", roles=roles)

    assert "[PERSON_1_STUDENT]" in result.anonymized_text
    assert "[PERSON_2_DOZENT]" in result.anonymized_text
    assert "[ORGANIZATION_1]" in result.anonymized_text or "ZHAW" in result.anonymized_text

    # Structure check: every line should be a markdown table row
    for line in result.anonymized_text.strip().splitlines():
        assert line.startswith("|") and line.endswith("|")

    # Roundtrip reversibility check
    restored = LocalAnonymizer.de_anonymize(result.anonymized_text, result.mapping)
    assert restored == extracted_md


def test_read_document_from_bytes_with_progress():
    import pymupdf
    from local_anonymizer.extractors import read_document_from_bytes

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Page 1 Content")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Page 2 Content")
    pdf_bytes = doc.tobytes()
    doc.close()

    progress_calls = []
    def on_progress(curr, total, msg):
        progress_calls.append((curr, total, msg))

    text = read_document_from_bytes(pdf_bytes, "test.pdf", progress_callback=on_progress)
    assert "Page 1 Content" in text
    assert "Page 2 Content" in text
    assert len(progress_calls) >= 2
    assert progress_calls[-1][0] == 2
    assert progress_calls[-1][1] == 2


def test_clean_extracted_pdf_markdown():
    from local_anonymizer.extractors import clean_extracted_pdf_markdown

    sample = (
        "## Dokumenttitel\n\n"
        "<mark>Dr. Andreas Schönenberger</mark> leitet das Projekt.\n"
        "Quelle: <u>www.sanitas.com</u>\n\n"
        "<!-- Start of picture text -->\n"
        "CEO<br>Dr. Andreas<br>Schönenberger<br>CMO / New<br>Corporate  CFO<br>\n"
        "Elias Frühauf Jan Schultz Kaspar  WandhovenWolfgang  Tobias Caluori<br>Trachsel<br>\n"
        "<!-- End of picture text -->"
    )

    # 1. With extract_picture_text=True
    cleaned = clean_extracted_pdf_markdown(sample, extract_picture_text=True)
    assert "<mark>" not in cleaned and "</mark>" not in cleaned
    assert "<u>" not in cleaned and "</u>" not in cleaned
    assert "Dr. Andreas Schönenberger leitet das Projekt." in cleaned
    assert "Quelle: www.sanitas.com" in cleaned
    assert "<!-- Start of picture text -->" not in cleaned
    assert "<br>" not in cleaned
    assert "Wandhoven Wolfgang" in cleaned

    # 2. With extract_picture_text=False (picture text completely omitted)
    cleaned_no_pic = clean_extracted_pdf_markdown(sample, extract_picture_text=False)
    assert "Dr. Andreas Schönenberger leitet das Projekt." in cleaned_no_pic
    assert "Quelle: www.sanitas.com" in cleaned_no_pic
    assert "Wandhoven" not in cleaned_no_pic
    assert "Jan Schultz" not in cleaned_no_pic


def test_pdf_headers_footers_toggle():
    import pymupdf
    from local_anonymizer.extractors import extract_text_from_pdf_bytes

    doc = pymupdf.open()
    for i in range(1, 3):
        page = doc.new_page(width=595, height=842)
        # Header
        page.insert_text((50, 30), "Vertrauliche Kopfzeile Sanitas", fontsize=9)
        # Body
        page.insert_text((50, 150), f"Haupttext des Dokuments auf Seite {i}.", fontsize=11)
        # Footer
        page.insert_text((50, 800), f"Seite {i} von 2 - Fußzeile", fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()

    # Default (include_headers_footers=False): Suppresses repeated running headers/footers
    text_no_hf = extract_text_from_pdf_bytes(pdf_bytes, include_headers_footers=False)
    assert "Haupttext des Dokuments auf Seite 1." in text_no_hf
    assert "Haupttext des Dokuments auf Seite 2." in text_no_hf
    assert "Vertrauliche Kopfzeile Sanitas" not in text_no_hf
    assert "Fußzeile" not in text_no_hf

    # With headers and footers (include_headers_footers=True)
    text_with_hf = extract_text_from_pdf_bytes(pdf_bytes, include_headers_footers=True)
    assert "Haupttext des Dokuments auf Seite 1." in text_with_hf
    assert "Vertrauliche Kopfzeile Sanitas" in text_with_hf
    assert "Fußzeile" in text_with_hf


def test_docx_headers_footers_toggle():
    import io
    import docx
    from local_anonymizer.extractors import extract_text_from_docx_bytes

    doc = docx.Document()
    sec = doc.sections[0]
    sec.header.paragraphs[0].text = "Header Text in Word"
    doc.add_paragraph("Body paragraph in Word.")
    sec.footer.paragraphs[0].text = "Footer Text in Word"

    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    # Default (include_headers_footers=False)
    text_no_hf = extract_text_from_docx_bytes(docx_bytes, include_headers_footers=False)
    assert "Body paragraph in Word." in text_no_hf
    assert "Header Text in Word" not in text_no_hf
    assert "Footer Text in Word" not in text_no_hf

    # With headers and footers (include_headers_footers=True)
    text_with_hf = extract_text_from_docx_bytes(docx_bytes, include_headers_footers=True)
    assert "Body paragraph in Word." in text_with_hf
    assert "Header Text in Word" in text_with_hf
    assert "Footer Text in Word" in text_with_hf




