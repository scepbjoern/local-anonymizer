"""Document extraction utilities for various file formats."""

import io
from pathlib import Path
from typing import Union
import pymupdf  # PyMuPDF
import pymupdf4llm
import docx


class UnsupportedFileFormatError(ValueError):
    """Raised when an unsupported file format is provided to the extractor."""
    pass


def extract_text_from_txt_bytes(raw_bytes: bytes) -> str:
    """
    Extract text from plain text or markdown bytes with robust encoding detection.
    Tries UTF-8 (with BOM), UTF-8, Windows CP1252, ISO-8859-15, and Latin-1 in order.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "iso-8859-15", "latin-1"]
    for enc in encodings:
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Ultimate fallback with character replacement
    return raw_bytes.decode("utf-8", errors="replace")


def extract_text_from_txt(path: Path) -> str:
    """Extract text from plain text or markdown files."""
    return extract_text_from_txt_bytes(path.read_bytes())


def extract_text_from_docx_bytes(raw_bytes: bytes) -> str:
    """Extract text from Microsoft Word .docx bytes."""
    doc = docx.Document(io.BytesIO(raw_bytes))
    full_text = []
    for para in doc.paragraphs:
        if para.text:
            full_text.append(para.text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))

    return "\n\n".join(full_text)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from Microsoft Word .docx files."""
    return extract_text_from_docx_bytes(path.read_bytes())


def extract_text_from_pdf_bytes(raw_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Extract structured Markdown text from PDF bytes using pymupdf4llm.
    Preserves headings, lists, tables, and bold/italic styles.
    Raises ValueError if PDF contains pages but zero extractable text (e.g. scanned image PDF).
    """
    doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    doc_pages = doc.page_count

    # Check if there is any extractable text across pages
    has_text = any(page.get_text().strip() for page in doc)
    if doc_pages > 0 and not has_text:
        doc.close()
        raise ValueError(
            f"No extractable text found in PDF '{filename}' ({doc_pages} pages). "
            f"The file appears to be a scanned/image-based PDF requiring OCR, or an empty document."
        )

    try:
        md_text = pymupdf4llm.to_markdown(doc)
    except Exception:
        # Fallback to plain get_text if markdown conversion fails
        pages_text = [page.get_text().strip() for page in doc if page.get_text().strip()]
        md_text = "\n\n--- Page Break ---\n\n".join(pages_text)
    finally:
        doc.close()

    return md_text.strip()


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract structured Markdown text from PDF files using pymupdf4llm.
    Raises ValueError if PDF contains pages but zero extractable text (e.g. scanned image PDF).
    """
    doc = pymupdf.open(str(path))
    doc_pages = doc.page_count

    # Check if there is any extractable text across pages
    has_text = any(page.get_text().strip() for page in doc)
    if doc_pages > 0 and not has_text:
        doc.close()
        raise ValueError(
            f"No extractable text found in PDF '{path.name}' ({doc_pages} pages). "
            f"The file appears to be a scanned/image-based PDF requiring OCR, or an empty document."
        )

    try:
        md_text = pymupdf4llm.to_markdown(doc)
    except Exception:
        # Fallback to plain get_text if markdown conversion fails
        pages_text = [page.get_text().strip() for page in doc if page.get_text().strip()]
        md_text = "\n\n--- Page Break ---\n\n".join(pages_text)
    finally:
        doc.close()

    return md_text.strip()


def read_document_from_bytes(data: bytes, filename: str) -> str:
    """Unified document reader from in-memory bytes."""
    ext = Path(filename).suffix.lower()
    if ext in [".txt", ".md", ".json", ".csv"]:
        return extract_text_from_txt_bytes(data)
    elif ext == ".docx":
        return extract_text_from_docx_bytes(data)
    elif ext == ".pdf":
        return extract_text_from_pdf_bytes(data, filename=filename)
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format: '{ext}'. Supported formats: .txt, .md, .json, .csv, .docx, .pdf"
        )


def read_document(file_path: Union[str, Path]) -> str:
    """Unified document reader supporting .txt, .md, .json, .csv, .docx, and .pdf."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in [".txt", ".md", ".json", ".csv"]:
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format: '{ext}'. Supported formats: .txt, .md, .json, .csv, .docx, .pdf"
        )

