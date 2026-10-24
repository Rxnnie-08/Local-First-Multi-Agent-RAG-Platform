from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.core.exceptions import ParseError
from agentic_rag.ingestion.parsers import (
    classify,
    discover_files,
    parse_file,
    sha256_of,
    source_name,
)
from fakes import make_pdf

MAX = 5 * 1_048_576


def parse(path: Path, *, max_bytes: int = MAX):  # type: ignore[no-untyped-def]
    return parse_file(
        path, source=path.name, file_hash=sha256_of(path, max_bytes), max_bytes=max_bytes
    )


def test_classify() -> None:
    assert classify(Path("a.MD")) == "markdown"
    assert classify(Path("a.pdf")) == "pdf"
    assert classify(Path("a.py")) == "code"
    assert classify(Path("a.txt")) == "text"
    assert classify(Path("a.exe")) is None


def test_discover_skips_vendored_dirs_and_unsupported(tmp_path: Path) -> None:
    for rel in ["b.md", "a.py", "img.png", ".git/x.md", "node_modules/pkg/y.py", "sub/c.txt"]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    found = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert found == ["a.py", "b.md", "sub/c.txt"]


def test_discover_single_unsupported_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "photo.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ParseError, match="Unsupported"):
        discover_files(f)


def test_source_name(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "a.md"
    f.write_text("x")
    assert source_name(root, f) == "notes/sub/a.md"
    assert source_name(f, f) == "a.md"


def test_markdown_heading_paths(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(
        "Preamble text.\n\n# Title\n\nIntro.\n\n## Part A\n\nA body.\n\n### Deep\n\nDeep body.\n\n"
        "## Part B\n\nB body.\n"
    )
    sections = parse(f).sections
    got = [(s.metadata.get("heading_path"), s.text) for s in sections]
    assert got == [
        (None, "Preamble text."),
        ("Title", "Intro."),
        ("Title > Part A", "A body."),
        ("Title > Part A > Deep", "Deep body."),
        ("Title > Part B", "B body."),
    ]


def test_markdown_hash_inside_code_fence_is_not_a_heading(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Real\n\n```bash\n# just a comment\necho hi\n```\n\nAfter.\n")
    sections = parse(f).sections
    assert len(sections) == 1
    assert "# just a comment" in sections[0].text
    assert sections[0].metadata["heading_path"] == "Real"


def test_markdown_front_matter_stripped(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("---\ntitle: X\ntags: [a]\n---\n# Head\n\nBody.\n")
    sections = parse(f).sections
    assert [s.text for s in sections] == ["Body."]


def test_code_language_and_crlf(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_bytes(b"def f():\r\n    return 1\r\n")
    doc = parse(f)
    assert doc.doc_type == "code"
    assert doc.metadata["language"] == "python"
    assert "\r" not in doc.sections[0].text


def test_binary_file_rejected(tmp_path: Path) -> None:
    f = tmp_path / "bin.txt"
    f.write_bytes(b"abc\x00def")
    with pytest.raises(ParseError, match="binary"):
        parse(f)


def test_size_limit(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("x" * 5000)
    with pytest.raises(ParseError, match="limit"):
        sha256_of(f, 1000)


def test_empty_file_rejected(tmp_path: Path) -> None:
    f = tmp_path / "empty.md"
    f.write_text("   \n\n")
    with pytest.raises(ParseError, match="no extractable text"):
        parse(f)


def test_pdf_pages_become_sections(tmp_path: Path) -> None:
    f = tmp_path / "paper.pdf"
    f.write_bytes(make_pdf(["Hello from page one", "Greetings from page two"]))
    doc = parse(f)
    assert doc.doc_type == "pdf"
    assert [s.metadata["page"] for s in doc.sections] == [1, 2]
    assert "page one" in doc.sections[0].text
    assert "page two" in doc.sections[1].text


def test_pdf_without_text_rejected(tmp_path: Path) -> None:
    f = tmp_path / "blank.pdf"
    f.write_bytes(make_pdf([""]))
    with pytest.raises(ParseError, match="no extractable text"):
        parse(f)


def test_corrupt_pdf_rejected(tmp_path: Path) -> None:
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(ParseError, match="PDF"):
        parse(f)
