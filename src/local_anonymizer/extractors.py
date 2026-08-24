"""Document extraction utilities for various file formats."""

from pathlib import Path
from typing import Union
import pymupdf  # PyMuPDF
import docx


def extract_text_from_txt(path: Path) -> str:
    """Extract text from plain text or markdown files."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def extract_text_from_docx(path: Path) -> str:
    """Extract text from Microsoft Word .docx files."""
    doc = docx.Document(path)
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


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF files using PyMuPDF."""
    doc = pymupdf.open(str(path))
    pages_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages_text.append(text.strip())
    doc.close()
    return "\n\n--- Page Break ---\n\n".join(pages_text)


def read_document(file_path: Union[str, Path]) -> str:
    """Unified document reader supporting .txt, .md, .docx, and .pdf."""
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
        raise ValueError(f"Unsupported file format: '{ext}'. Supported: .txt, .md, .docx, .pdf")
