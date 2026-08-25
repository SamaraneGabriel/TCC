from pathlib import Path


def parse_pdf(source: str | Path | bytes) -> str:
    """Extract plain text from a PDF using PyMuPDF (fitz). `source` pode ser um
    caminho de arquivo ou os bytes crus do PDF (ex: upload em memória)."""
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf") from e

    doc = fitz.open(stream=source, filetype="pdf") if isinstance(source, bytes) else fitz.open(str(source))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()
