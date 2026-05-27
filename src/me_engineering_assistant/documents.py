from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DOCUMENT_FILENAMES = (
    "ECU-700_Series_Manual.md",
    "ECU-800_Series_Base.md",
    "ECU-800_Series_Plus.md",
)


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: dict[str, str]


def project_root() -> Path:
    return PROJECT_ROOT


def default_document_paths(base_path: str | Path | None = None) -> list[Path]:
    root = Path(base_path).expanduser().resolve() if base_path else PROJECT_ROOT
    paths: list[Path] = []
    for name in DEFAULT_DOCUMENT_FILENAMES:
        direct = root / name
        nested = root / "readmes" / name
        paths.append(direct if direct.exists() else nested)
    return paths


def load_source_documents(
    base_path: str | Path | None = None,
    document_paths: Sequence[str | Path] | None = None,
) -> list[SourceDocument]:
    paths = [Path(path).expanduser().resolve() for path in document_paths] if document_paths else default_document_paths(base_path)
    documents: list[SourceDocument] = []
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ECU source document(s): {', '.join(missing)}")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        documents.append(SourceDocument(path=path, text=text, metadata=infer_metadata(path, text)))
    return documents


def infer_metadata(path: Path, text: str) -> dict[str, str]:
    filename = path.name.lower()
    raw = f"{path.name}\n{text}".lower()
    if "series_plus" in filename or "plus" in filename:
        model = "ECU-850b"
        series = "ECU-800"
        doc_type = "variant"
    elif "series_base" in filename or "base" in filename:
        model = "ECU-850"
        series = "ECU-800"
        doc_type = "base"
    elif "ecu-850b" in raw:
        model = "ECU-850b"
        series = "ECU-800"
        doc_type = "variant"
    elif "ecu-850" in raw:
        model = "ECU-850"
        series = "ECU-800"
        doc_type = "base"
    elif "ecu-750" in raw or "ecu-700" in raw:
        model = "ECU-750"
        series = "ECU-700"
        doc_type = "legacy_manual"
    else:
        model = "unknown"
        series = "unknown"
        doc_type = "unknown"

    title = _first_heading(text) or path.stem
    return {
        "source": path.name,
        "path": str(path),
        "title": title,
        "model": model,
        "series": series,
        "doc_type": doc_type,
    }


def chunk_documents(documents: Iterable[SourceDocument], max_chars: int = 1_100) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        sections = _markdown_sections(document.text)
        for index, section in enumerate(sections):
            for part_index, content in enumerate(_split_long_section(section, max_chars=max_chars)):
                metadata = dict(document.metadata)
                metadata["section"] = _first_heading(content) or document.metadata["title"]
                chunk_id = f"{document.metadata['model']}::{index:02d}:{part_index:02d}"
                chunks.append(DocumentChunk(chunk_id=chunk_id, content=content.strip(), metadata=metadata))
    return chunks


def _first_heading(text: str) -> str | None:
    match = re.search(r"(?m)^#{1,6}\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _markdown_sections(text: str) -> list[str]:
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        is_heading = bool(re.match(r"^#{1,3}\s+", line))
        if is_heading and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(section).strip() for section in sections if "\n".join(section).strip()]


def _split_long_section(section: str, max_chars: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    paragraphs = re.split(r"\n\s*\n", section)
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph.strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = paragraph.strip()
    if current:
        parts.append(current)
    return parts
