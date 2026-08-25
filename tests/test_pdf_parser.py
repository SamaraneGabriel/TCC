import fitz

from src.parsers.pdf_parser import parse_pdf


def _make_pdf_bytes(texto: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), texto)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_from_bytes():
    data = _make_pdf_bytes("Ola mundo")
    assert "Ola mundo" in parse_pdf(data)


def test_parse_pdf_from_path(tmp_path):
    data = _make_pdf_bytes("Teste de caminho")
    caminho = tmp_path / "teste.pdf"
    caminho.write_bytes(data)
    assert "Teste de caminho" in parse_pdf(caminho)


def test_parse_pdf_from_path_as_string(tmp_path):
    data = _make_pdf_bytes("Teste string")
    caminho = tmp_path / "teste2.pdf"
    caminho.write_bytes(data)
    assert "Teste string" in parse_pdf(str(caminho))
