"""Document extraction utilities for various file formats."""

import csv
import io
import json
import re
from pathlib import Path
from typing import Union
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


import sys
import logging


def safe_read_bytes(file_path: Union[str, Path]) -> bytes:
    """
    Safely read bytes from a file path. On Windows, uses Win32 API with full sharing
    (FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE) so that files currently open
    in Microsoft Word, Excel, OneDrive, or other editors can be read without PermissionError.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            GENERIC_READ = 0x80000000
            FILE_SHARE_READ = 1
            FILE_SHARE_WRITE = 2
            FILE_SHARE_DELETE = 4
            OPEN_EXISTING = 3
            FILE_ATTRIBUTE_NORMAL = 0x80
            INVALID_HANDLE_VALUE = -1

            handle = kernel32.CreateFileW(
                str(path.resolve()),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                None,
            )
            if handle != INVALID_HANDLE_VALUE:
                try:
                    size = kernel32.GetFileSize(handle, None)
                    if size > 0:
                        buf = ctypes.create_string_buffer(size)
                        bytes_read = wintypes.DWORD()
                        if kernel32.ReadFile(handle, buf, size, ctypes.byref(bytes_read), None):
                            return buf.raw[:bytes_read.value]
                finally:
                    kernel32.CloseHandle(handle)
        except Exception as e:
            logging.debug(f"Win32 safe_read_bytes fallback: {e}")

    # Standard fallback (macOS, Linux, or if Win32 wasn't used)
    return path.read_bytes()


def extract_text_from_txt(path: Path) -> str:
    """Extract text from plain text or markdown files."""
    return extract_text_from_txt_bytes(safe_read_bytes(path))


def extract_text_from_json_bytes(data: bytes) -> str:
    """Extract JSON bytes as cleanly formatted JSON text."""
    raw_str = extract_text_from_txt_bytes(data)
    try:
        parsed = json.loads(raw_str)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return raw_str


def extract_text_from_csv_bytes(data: bytes) -> str:
    """Extract CSV bytes as a structured Markdown table."""
    text_content = extract_text_from_txt_bytes(data)
    lines = [l for l in text_content.splitlines() if l.strip()]
    if not lines:
        return ""

    # Detect delimiter: try comma, semicolon, tab
    sample = "\n".join(lines[:5])
    delimiter = ","
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=";,|\t")
        delimiter = dialect.delimiter
    except Exception:
        if ";" in sample and "," not in sample:
            delimiter = ";"
        elif "\t" in sample:
            delimiter = "\t"

    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return text_content

    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return text_content

    # Normalize row lengths
    norm_rows = []
    for r in rows:
        padded = [c.strip().replace("\n", " ") for c in r]
        while len(padded) < max_cols:
            padded.append("")
        norm_rows.append(padded)

    header = norm_rows[0]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] * max_cols) + " |",
    ]
    for r in norm_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n".join(md_lines)


def extract_text_from_docx_bytes(raw_bytes: bytes) -> str:
    """
    Extract text from Microsoft Word .docx bytes with:
    - Level 1: XML Outline Level (w:outlineLvl 0..3) for custom corporate heading templates
    - Level 2: Multilingual style names (Heading 1..4, Überschrift 1..4, Title, Subtitle)
    - Level 3: Bullet & numbered lists via w:numPr and list styles
    - Level 4: Strict false-positive protection for bold text (only < 60 chars, no ending period, >= 16pt)
    - Level 5: Markdown tables
    """
    import docx
    doc = docx.Document(io.BytesIO(raw_bytes))
    full_text = []
    for para in doc.paragraphs:
        if not para.text or not para.text.strip():
            continue
        text = para.text.strip()
        style_name = para.style.name.lower() if para.style and para.style.name else ""

        # Stufe 1: XML Gliederungsebene w:outlineLvl
        outline_val = None
        try:
            outline_nodes = para._element.xpath("./w:pPr/w:outlineLvl/@w:val")
            if outline_nodes:
                outline_val = int(outline_nodes[0])
        except Exception:
            pass

        # Stufe 2: Listen über XML (w:numPr) oder Style-Namen
        has_num_pr = False
        try:
            has_num_pr = bool(para._element.xpath("./w:pPr/w:numPr"))
        except Exception:
            pass

        is_bullet = "bullet" in style_name or "listenpunkt" in style_name or "aufzählung" in style_name
        is_number = "number" in style_name or "nummerierung" in style_name or "list 2" in style_name or "list 3" in style_name

        if outline_val is not None and 0 <= outline_val <= 5:
            prefix = "#" * (outline_val + 1)
            text = f"{prefix} {text}"
        elif "heading 1" in style_name or "überschrift 1" in style_name or style_name in ["title", "titel"]:
            text = f"# {text}"
        elif "heading 2" in style_name or "überschrift 2" in style_name or style_name in ["subtitle", "untertitel"]:
            text = f"## {text}"
        elif "heading 3" in style_name or "überschrift 3" in style_name:
            text = f"### {text}"
        elif "heading 4" in style_name or "überschrift 4" in style_name:
            text = f"#### {text}"
        elif is_bullet:
            text = f"- {text}"
        elif is_number or has_num_pr:
            text = f"1. {text}"
        elif "quote" in style_name or "zitat" in style_name:
            text = f"> {text}"
        else:
            # Stufe 4: Direktauszeichnungs-Heuristik mit striktem Falsch-Positiv-Schutz:
            # Nur kurze Absätze (< 60 Zeichen), ohne Schlusspunkt, mindestens 16pt groß und ganz fett.
            is_bold_title = False
            try:
                runs = [r for r in para.runs if r.text and r.text.strip()]
                if runs and all(r.bold for r in runs) and len(text) < 60 and not text.endswith("."):
                    sizes = [r.font.size.pt for r in runs if r.font and r.font.size and r.font.size.pt]
                    if sizes and min(sizes) >= 16:
                        is_bold_title = True
            except Exception:
                pass

            if is_bold_title:
                text = f"# {text}"

        full_text.append(text)

    # Also extract text from tables as structured Markdown tables
    for table in doc.tables:
        rows_text = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(row_cells):
                rows_text.append(row_cells)
        if rows_text:
            header = rows_text[0]
            table_md = ["| " + " | ".join(header) + " |", "| " + " | ".join([":---"] * len(header)) + " |"]
            for r in rows_text[1:]:
                while len(r) < len(header):
                    r.append("")
                table_md.append("| " + " | ".join(r[:len(header)]) + " |")
            full_text.append("\n".join(table_md))

    return "\n\n".join(full_text)


def extract_text_from_docx(path: Path) -> str:
    """Extract text from Microsoft Word .docx files."""
    return extract_text_from_docx_bytes(safe_read_bytes(path))


def create_docx_from_markdown(md_text: str):
    """
    Convert Markdown text into a native Word .docx document using real Word paragraph styles
    (Heading 1, Heading 2, Heading 3, Heading 4, List Bullet, List Number, Normal).
    Does NOT output literal '#' or Markdown markers into the document.
    """
    import docx
    doc = docx.Document()
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        trimmed = line.strip()

        if not trimmed:
            i += 1
            continue

        # Headings
        if trimmed.startswith("#### "):
            doc.add_heading(trimmed[5:].strip(), level=4)
        elif trimmed.startswith("### "):
            doc.add_heading(trimmed[4:].strip(), level=3)
        elif trimmed.startswith("## "):
            doc.add_heading(trimmed[3:].strip(), level=2)
        elif trimmed.startswith("# "):
            doc.add_heading(trimmed[2:].strip(), level=1)
        # Lists
        elif trimmed.startswith("- ") or trimmed.startswith("* "):
            doc.add_paragraph(trimmed[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\.\s+", trimmed):
            text_part = re.sub(r"^\d+\.\s+", "", trimmed)
            doc.add_paragraph(text_part.strip(), style="List Number")
        # Blockquotes
        elif trimmed.startswith("> "):
            doc.add_paragraph(trimmed[2:].strip(), style="Quote" if "Quote" in doc.styles else "Normal")
        # Tables (Markdown | col1 | col2 |)
        elif trimmed.startswith("|") and trimmed.endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1

            rows_data = []
            for t_line in table_lines:
                cells = [c.strip() for c in t_line.strip("|").split("|")]
                if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    continue
                rows_data.append(cells)

            if rows_data:
                num_cols = max(len(r) for r in rows_data)
                table = doc.add_table(rows=len(rows_data), cols=num_cols)
                try:
                    table.style = "Table Grid"
                except Exception:
                    pass
                for r_idx, r_data in enumerate(rows_data):
                    for c_idx, cell_val in enumerate(r_data):
                        table.cell(r_idx, c_idx).text = cell_val
        else:
            doc.add_paragraph(trimmed)

        i += 1

    return doc


def save_markdown_to_docx_bytes(md_text: str) -> bytes:
    """Convert Markdown text to .docx and return raw bytes."""
    doc = create_docx_from_markdown(md_text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def extract_text_from_pdf_bytes(raw_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Extract structured Markdown text from PDF bytes using pymupdf4llm.
    Preserves headings, lists, tables, and bold/italic styles.
    Raises ValueError if PDF contains pages but zero extractable text (e.g. scanned image PDF).
    """
    import pymupdf
    import pymupdf4llm
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
    import pymupdf
    import pymupdf4llm
    doc = pymupdf.open(str(path))
    doc_pages = doc.page_count

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
        pages_text = [page.get_text().strip() for page in doc if page.get_text().strip()]
        md_text = "\n\n--- Page Break ---\n\n".join(pages_text)
    finally:
        doc.close()

    return md_text.strip()


def read_document_from_bytes(data: bytes, filename: str) -> str:
    """Unified document reader from in-memory bytes."""
    ext = Path(filename).suffix.lower()
    if ext in [".txt", ".md"]:
        return extract_text_from_txt_bytes(data)
    elif ext == ".json":
        return extract_text_from_json_bytes(data)
    elif ext == ".csv":
        return extract_text_from_csv_bytes(data)
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
    raw_data = safe_read_bytes(path)
    return read_document_from_bytes(raw_data, path.name)
