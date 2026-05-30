from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "content": self.content, "metadata": self.metadata}


@dataclass(frozen=True)
class CatalogEntry:
    source: str
    model: str
    series: str
    title: str
    doc_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "model": self.model,
            "series": self.series,
            "title": self.title,
            "doc_type": self.doc_type,
        }


@dataclass(frozen=True)
class ModelFieldEvidence:
    model: str
    series: str
    field: str
    field_label: str
    value: str
    source: str
    section: str
    chunk_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model": self.model,
            "series": self.series,
            "field": self.field,
            "field_label": self.field_label,
            "value": self.value,
            "source": self.source,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }


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
    raw = f"{path.name}\n{text}"
    model = _infer_model_identifier(raw) or "unknown"
    series = _series_for_model(model)
    doc_type = _infer_doc_type(path, text)
    title = _first_heading(text) or path.stem
    return {
        "source": path.name,
        "path": str(path),
        "title": title,
        "model": model,
        "series": series,
        "doc_type": doc_type,
    }


def _infer_model_identifier(text: str) -> str | None:
    identifiers = _ecu_identifiers(text)
    model_identifiers = [identifier for identifier in identifiers if not _is_series_identifier(identifier)]
    return model_identifiers[0] if model_identifiers else (identifiers[0] if identifiers else None)


def _ecu_identifiers(text: str) -> list[str]:
    identifiers = []
    for match in re.findall(r"(?<![A-Za-z0-9])ecu[-_\s]?(\d{3}[a-z]?)(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
        identifier = _canonical_ecu_identifier(match)
        if identifier not in identifiers:
            identifiers.append(identifier)
    return identifiers


def _canonical_ecu_identifier(suffix: str) -> str:
    suffix = suffix.upper()
    if suffix.endswith("B"):
        return f"ECU-{suffix[:-1]}b"
    return f"ECU-{suffix}"


def _is_series_identifier(identifier: str) -> bool:
    match = re.fullmatch(r"ECU-(\d{3})([a-z]?)", identifier, flags=re.IGNORECASE)
    return bool(match and not match.group(2) and match.group(1).endswith("00"))


def _series_for_model(model: str) -> str:
    match = re.fullmatch(r"ECU-(\d{3})([a-z]?)", model, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    series_number = f"{match.group(1)[0]}00"
    return f"ECU-{series_number}"


def _infer_doc_type(path: Path, text: str) -> str:
    filename = path.name.lower()
    if "plus" in filename or "addendum" in filename or "variant" in filename:
        return "variant"
    if "base" in filename or "baseline" in filename:
        return "base"

    raw = f"{path.name}\n{_first_heading(text) or ''}\n{text[:500]}".lower()
    if "legacy" in raw or "manual" in raw:
        return "legacy_manual"
    if "base" in raw or "baseline" in raw:
        return "base"
    if "variant" in raw or "addendum" in raw or "enhanced" in raw or "plus" in raw:
        return "variant"
    return "spec"


def chunk_documents(documents: Iterable[SourceDocument], max_chars: int = 1_100) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        sections = _markdown_sections(document.text)
        for index, section in enumerate(sections):
            section_name = _first_heading(section) or document.metadata["title"]
            for part_index, content in enumerate(_split_long_section(section, max_chars=max_chars)):
                metadata = dict(document.metadata)
                metadata["section"] = _first_heading(content) or section_name
                metadata["field"] = ""
                metadata["field_label"] = ""
                metadata["value"] = ""
                metadata["chunk_type"] = "section"
                chunk_id = f"{document.metadata['model']}::{index:02d}:{part_index:02d}"
                chunks.append(DocumentChunk(chunk_id=chunk_id, content=content.strip(), metadata=metadata))
            chunks.extend(_field_chunks(document, section, section_name=section_name, section_index=index))
    return chunks


def build_document_catalog(documents: Iterable[SourceDocument]) -> list[CatalogEntry]:
    entries = []
    for document in documents:
        metadata = document.metadata
        entries.append(
            CatalogEntry(
                source=metadata["source"],
                model=metadata["model"],
                series=metadata["series"],
                title=metadata["title"],
                doc_type=metadata["doc_type"],
            )
        )
    return entries


def build_model_field_table(chunks: Iterable[DocumentChunk]) -> list[ModelFieldEvidence]:
    chunk_list = list(chunks)
    rows = []
    for chunk in chunk_list:
        metadata = chunk.metadata
        if metadata.get("chunk_type") != "field" or not metadata.get("field"):
            continue
        rows.append(
            ModelFieldEvidence(
                model=metadata.get("model", "unknown"),
                series=metadata.get("series", "unknown"),
                field=metadata.get("field", ""),
                field_label=metadata.get("field_label", ""),
                value=metadata.get("value", ""),
                source=metadata.get("source", "unknown"),
                section=metadata.get("section", "unknown"),
                chunk_id=chunk.chunk_id,
            )
        )
    rows.extend(_inherited_variant_rows(rows, chunk_list))
    return rows


def _inherited_variant_rows(
    rows: Sequence[ModelFieldEvidence],
    chunks: Sequence[DocumentChunk],
) -> list[ModelFieldEvidence]:
    inherited = []
    base_by_series = {
        chunk.metadata.get("series"): chunk.metadata.get("model")
        for chunk in chunks
        if chunk.metadata.get("doc_type") == "base"
    }
    inheriting_variants = _variant_inheritance_sources(chunks)
    fields_by_model = {}
    for row in rows:
        fields_by_model.setdefault(row.model, set()).add(row.field)

    for variant_model, variant_metadata in inheriting_variants.items():
        base_model = base_by_series.get(variant_metadata.get("series"))
        if not base_model:
            continue
        variant_fields = fields_by_model.setdefault(variant_model, set())
        for base_row in rows:
            if base_row.model != base_model or base_row.field in variant_fields:
                continue
            inherited.append(
                ModelFieldEvidence(
                    model=variant_model,
                    series=variant_metadata.get("series", base_row.series),
                    field=base_row.field,
                    field_label=base_row.field_label,
                    value=f"Inherited from base {base_model}: {base_row.value}",
                    source=variant_metadata.get("source", base_row.source),
                    section="Inherited base features",
                    chunk_id=f"{variant_model}::inherited::{base_row.chunk_id}",
                )
            )
            variant_fields.add(base_row.field)
    return inherited


def _variant_inheritance_sources(chunks: Sequence[DocumentChunk]) -> dict[str, dict[str, str]]:
    variants = {}
    for chunk in chunks:
        metadata = chunk.metadata
        if metadata.get("doc_type") != "variant":
            continue
        if not re.search(r"\bincludes\s+all\s+features\s+of\s+(?:the\s+)?base\b", chunk.content, re.IGNORECASE):
            continue
        variants[metadata.get("model", "")] = metadata
    return {model: metadata for model, metadata in variants.items() if model}


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


def _field_chunks(
    document: SourceDocument,
    section: str,
    *,
    section_name: str,
    section_index: int,
) -> list[DocumentChunk]:
    chunks = []
    for row_index, row in enumerate(_extract_field_rows(section)):
        metadata = dict(document.metadata)
        metadata.update(
            {
                "section": section_name,
                "field": row["field"],
                "field_label": row["field_label"],
                "value": row["value"],
                "chunk_type": "field",
            }
        )
        chunk_id = f"{document.metadata['model']}::{section_index:02d}:field:{row_index:02d}"
        content = (
            f"Model: {metadata['model']}\n"
            f"Series: {metadata['series']}\n"
            f"Field: {row['field_label']}\n"
            f"Value: {row['value']}\n"
            f"Source: {metadata['source']}\n"
            f"Section: {section_name}"
        )
        chunks.append(DocumentChunk(chunk_id=chunk_id, content=content, metadata=metadata))
    return chunks


def _extract_field_rows(section: str) -> list[dict[str, str]]:
    rows = []
    for line in section.splitlines():
        parsed = _parse_markdown_table_row(line)
        if parsed is not None:
            rows.append(parsed)
            continue
        parsed = _parse_label_value_bullet(line)
        if parsed is not None:
            rows.append(parsed)
            continue
        parsed = _parse_plain_feature_bullet(line)
        if parsed is not None:
            rows.append(parsed)
    rows.extend(_parse_sentence_fields(section))
    return rows


def _parse_markdown_table_row(line: str) -> dict[str, str] | None:
    if "|" not in line:
        return None
    stripped = line.strip()
    if not stripped or re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped):
        return None

    cells = [_clean_markdown(cell) for cell in stripped.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return None

    label, value = cells[0], cells[1]
    if label.lower() in {"feature", "field", "parameter"} or value.lower() in {"specification", "value"}:
        return None
    return _field_row(label, value)


def _parse_label_value_bullet(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    match = re.match(r"^[-*]\s+\*\*(.+?)\*\*:\s+(.+)$", stripped)
    if not match:
        return None
    return _field_row(_clean_markdown(match.group(1)), _clean_markdown(match.group(2)))


def _parse_plain_feature_bullet(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    match = re.match(r"^[-*]\s+(.+)$", stripped)
    if not match or ":" in match.group(1):
        return None
    label = _clean_markdown(match.group(1))
    if len(label.split()) < 2:
        return None
    return _field_row(label, "Listed feature")


def _parse_sentence_fields(section: str) -> list[dict[str, str]]:
    rows = []
    for command in re.findall(r"me-driver-ctl\s+[a-zA-Z0-9_\-= ]+", section):
        rows.append(_field_row("NPU Enable Command", _clean_markdown(command)))
    for sentence in re.split(r"(?<=[.!?])\s+", _clean_markdown(section)):
        if re.search(r"\bover-the-air\b|\bota\b", sentence, flags=re.IGNORECASE):
            rows.append(_field_row("OTA Support", sentence))
    return rows


def _field_row(label: str, value: str) -> dict[str, str]:
    return {
        "field": slugify_field(label),
        "field_label": label,
        "value": value,
    }


def slugify_field(label: str) -> str:
    cleaned = _clean_markdown(label)
    tokens = re.findall(r"[a-z0-9]+", cleaned.lower())
    return "_".join(tokens)


def _clean_markdown(value: str) -> str:
    cleaned = re.sub(r"`{1,3}", "", value)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
