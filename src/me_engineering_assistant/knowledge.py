from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from me_engineering_assistant.documents import SourceDocument


@dataclass(frozen=True)
class ECUSpec:
    model: str
    series: str
    source: str
    processor: str | None = None
    memory_ram: str | None = None
    storage: str | None = None
    can_interface: str | None = None
    ethernet: str | None = None
    power_consumption: str | None = None
    operating_temperature: str | None = None
    connectors: str | None = None
    npu: str | None = None
    ota_supported: bool | None = None
    npu_enable_command: str | None = None
    safety: str | None = None


ROW_PATTERNS = (
    re.compile(r"^\|\s*\*\*(?P<key>[^*]+)\*\*\s*\|\s*(?P<value>.*?)\s*\|?\s*$"),
    re.compile(r"^\s*\*\*(?P<key>[^*]+)\*\*\s*\|\s*(?P<value>.*?)\s*\|?\s*$"),
)


def extract_specs(documents: Iterable[SourceDocument]) -> dict[str, ECUSpec]:
    specs: dict[str, ECUSpec] = {}
    for document in documents:
        fields = _extract_table_fields(document.text)
        model = document.metadata["model"]
        text_lower = document.text.lower()
        ota_supported: bool | None = None
        if re.search(r"ota\)?\s+updates?\s+are\s+not\s+supported", text_lower):
            ota_supported = False
        elif "over-the-air (ota) update capability" in text_lower or model == "ECU-850b":
            ota_supported = True

        specs[model] = ECUSpec(
            model=model,
            series=document.metadata["series"],
            source=document.metadata["source"],
            processor=fields.get("processor"),
            memory_ram=fields.get("memory_ram"),
            storage=fields.get("storage"),
            can_interface=fields.get("can_interface"),
            ethernet=fields.get("ethernet"),
            power_consumption=fields.get("power_consumption"),
            operating_temperature=fields.get("operating_temperature"),
            connectors=fields.get("connectors"),
            npu=fields.get("npu"),
            ota_supported=ota_supported,
            npu_enable_command=_extract_npu_command(document.text),
            safety=_extract_safety(document.text),
        )
    return specs


def _extract_table_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for pattern in ROW_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            key = _canonical_feature_key(match.group("key"))
            if key:
                fields[key] = _clean_markdown(match.group("value"))
            break
    return fields


def _canonical_feature_key(key: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()
    if normalized == "processor":
        return "processor"
    if "memory" in normalized or "ram" in normalized:
        return "memory_ram"
    if normalized == "storage":
        return "storage"
    if "can interface" in normalized:
        return "can_interface"
    if normalized == "ethernet":
        return "ethernet"
    if "power consumption" in normalized:
        return "power_consumption"
    if "operating temp" in normalized or "operating temperature" in normalized:
        return "operating_temperature"
    if normalized == "connectors":
        return "connectors"
    if normalized == "npu":
        return "npu"
    return None


def _clean_markdown(value: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = value.replace("**", "")
    return re.sub(r"\s+", " ", value).strip()


def _extract_npu_command(text: str) -> str | None:
    match = re.search(r"(?m)^\s*(me-driver-ctl\s+--enable-npu\s+--mode=performance)\s*$", text)
    return match.group(1).strip() if match else None


def _extract_safety(text: str) -> str | None:
    match = re.search(r"certified for\s+(.+?)\.", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None
