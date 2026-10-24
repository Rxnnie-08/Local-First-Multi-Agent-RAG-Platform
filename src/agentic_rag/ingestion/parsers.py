"""File discovery and parsing into sections (text plus the metadata that locates it)."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_rag.core.exceptions import ParseError

DocType = Literal["markdown", "pdf", "code", "text"]

MARKDOWN_EXT = frozenset({".md", ".markdown", ".mdx"})
PDF_EXT = frozenset({".pdf"})
TEXT_EXT = frozenset({".txt", ".rst"})
CODE_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".sh": "shell",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}
IGNORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode",
        "target", ".next", "site-packages",
    }
)  # fmt: skip


class Section(BaseModel):
    """A contiguous span of text with its location metadata (heading path, page, ...)."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    source: str
    doc_type: DocType
    file_hash: str
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)


def classify(path: Path) -> DocType | None:
    ext = path.suffix.lower()
    if ext in MARKDOWN_EXT:
        return "markdown"
    if ext in PDF_EXT:
        return "pdf"
    if ext in CODE_EXT:
        return "code"
    if ext in TEXT_EXT:
        return "text"
    return None


def discover_files(root: Path) -> list[Path]:
    """Supported files under `root` (or `root` itself), sorted, skipping vendored/hidden dirs."""
    if root.is_file():
        if classify(root) is None:
            raise ParseError(f"Unsupported file type: {root.suffix or root.name}")
        return [root]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            if classify(candidate) is not None:
                found.append(candidate)
    return found


def source_name(root: Path, file: Path) -> str:
    """Stable identifier stored in the DB: '<root-folder>/<relative/posix/path>'."""
    if root.is_file():
        return file.name
    return f"{root.name}/{file.relative_to(root).as_posix()}"


def read_bytes(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise ParseError(
                f"{path.name} is {size / 1_048_576:.1f} MB, above the "
                f"{max_bytes / 1_048_576:g} MB limit (INGEST_MAX_FILE_MB)"
            )
        return path.read_bytes()
    except OSError as exc:
        raise ParseError(f"Cannot read {path.name}: {exc}") from exc


def sha256_of(path: Path, max_bytes: int) -> str:
    return hashlib.sha256(read_bytes(path, max_bytes)).hexdigest()


def parse_file(path: Path, *, source: str, file_hash: str, max_bytes: int) -> ParsedDocument:
    """Blocking; call via `asyncio.to_thread`."""
    doc_type = classify(path)
    if doc_type is None:
        raise ParseError(f"Unsupported file type: {path.suffix or path.name}")

    if doc_type == "pdf":
        sections = _parse_pdf(path, max_bytes)
        metadata: dict[str, Any] = {}
    else:
        raw = read_bytes(path, max_bytes)
        if b"\x00" in raw[:8192]:
            raise ParseError(f"{path.name} looks like a binary file")
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        if doc_type == "markdown":
            sections = _parse_markdown(text)
            metadata = {}
        elif doc_type == "code":
            sections = [Section(text=text)] if text.strip() else []
            metadata = {"language": CODE_EXT[path.suffix.lower()]}
        else:
            sections = [Section(text=text)] if text.strip() else []
            metadata = {}

    if not sections:
        raise ParseError(f"{path.name} contains no extractable text")
    return ParsedDocument(
        source=source,
        doc_type=doc_type,
        file_hash=file_hash,
        sections=sections,
        metadata=metadata,
    )


# --- Markdown ---------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            nl = text.find("\n", end + 1)
            return text[nl + 1 :] if nl != -1 else ""
    return text


def _parse_markdown(text: str) -> list[Section]:
    """Split on headings (ignoring '#' lines inside code fences), tracking the heading path."""
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    sections: list[Section] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            meta = {"heading_path": " > ".join(h for _, h in stack)} if stack else {}
            sections.append(Section(text=body, metadata=meta))

    for line in _strip_front_matter(text).split("\n"):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            buffer.append(line)
            continue
        heading = _HEADING_RE.match(line) if fence is None else None
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
        else:
            buffer.append(line)
    flush()
    return sections


# --- PDF --------------------------------------------------------------------


def _parse_pdf(path: Path, max_bytes: int) -> list[Section]:
    from pypdf import PasswordType, PdfReader  # imported lazily: only PDF ingestion needs it

    read_bytes(path, max_bytes)  # enforces the size limit with a friendly message
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise ParseError(f"{path.name} is password-protected")
        sections: list[Section] = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                sections.append(Section(text=text, metadata={"page": number}))
    except ParseError:
        raise
    except Exception as exc:  # pypdf raises many unrelated types on malformed files
        raise ParseError(f"Cannot parse PDF {path.name}: {type(exc).__name__}: {exc}") from exc
    if not sections:
        raise ParseError(f"{path.name} has no extractable text (scanned PDF? OCR is not supported)")
    return sections
