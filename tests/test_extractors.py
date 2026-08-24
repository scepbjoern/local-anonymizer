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
