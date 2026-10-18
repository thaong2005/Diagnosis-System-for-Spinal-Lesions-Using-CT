"""Read clinical reference labels without exporting identifying fields."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO


ALLOWED_FRACTURE_CLASSES = frozenset({"A1", "A3", "A4", "B2", "B3", "C"})
REQUIRED_COLUMNS = frozenset({"Đốt gãy", "ID", "phân loại"})
_LEVEL_PATTERN = re.compile(r"([CDTLS])(\d{1,2})")


def normalize_text(value: object) -> str:
    """Normalize whitespace and Unicode while retaining Vietnamese text."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.replace("\ufeff", "").split()).strip()


def parse_identifiers(value: object) -> tuple[str, ...]:
    """Return candidate clinical IDs for in-memory matching only."""
    normalized = normalize_text(value)
    identifiers = tuple(
        part.strip()
        for part in normalized.split("$$$")
        if part.strip()
    )
    if not identifiers:
        raise ValueError("Nhãn tham chiếu thiếu ID.")
    return identifiers


def parse_vertebral_levels(value: object) -> tuple[str, ...]:
    """Parse Vietnamese D/T/L notation and canonicalize D to thoracic T."""
    normalized = normalize_text(value).upper()
    normalized = re.sub(r"^GÃY\s+", "", normalized)
    levels = []
    for prefix, number in _LEVEL_PATTERN.findall(normalized):
        canonical_prefix = "T" if prefix == "D" else prefix
        level = f"{canonical_prefix}{int(number)}"
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError(f"Không đọc được mức đốt từ nhãn: {value!r}")
    return tuple(levels)


@dataclass(frozen=True)
class ReferenceLabel:
    """One development-only label row.

    Identifiers stay in memory for matching and are hidden from repr to reduce
    accidental disclosure in logs.
    """

    row_number: int
    vertebral_levels: tuple[str, ...]
    fracture_class: str
    identifiers: tuple[str, ...] = field(repr=False)
    has_notes: bool = False


def parse_reference_labels(stream: TextIO) -> list[ReferenceLabel]:
    """Parse and validate reference labels from an open CSV text stream."""
    reader = csv.DictReader(stream)
    columns = set(reader.fieldnames or ())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            "CSV thiếu cột bắt buộc: " + ", ".join(sorted(missing))
        )

    labels = []
    for row_number, row in enumerate(reader, start=2):
        fracture_class = normalize_text(row.get("phân loại")).upper()
        if fracture_class not in ALLOWED_FRACTURE_CLASSES:
            raise ValueError(
                f"Dòng {row_number} có phân loại không hỗ trợ: "
                f"{fracture_class!r}"
            )
        labels.append(
            ReferenceLabel(
                row_number=row_number,
                vertebral_levels=parse_vertebral_levels(row.get("Đốt gãy")),
                fracture_class=fracture_class,
                identifiers=parse_identifiers(row.get("ID")),
                has_notes=bool(normalize_text(row.get("kèm theo"))),
            )
        )
    return labels


def load_reference_labels(path: str | Path) -> list[ReferenceLabel]:
    """Load and validate reference labels from the clinical CSV."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return parse_reference_labels(stream)


def summarize_reference_labels(
    labels: Iterable[ReferenceLabel],
) -> dict[str, object]:
    """Produce aggregate, non-identifying counts safe for logs/reports."""
    records = list(labels)
    return {
        "row_count": len(records),
        "class_counts": dict(
            sorted(Counter(item.fracture_class for item in records).items())
        ),
        "level_counts": dict(
            sorted(
                Counter(
                    level
                    for item in records
                    for level in item.vertebral_levels
                ).items()
            )
        ),
        "multi_level_rows": sum(
            len(item.vertebral_levels) > 1 for item in records
        ),
        "rows_with_notes": sum(item.has_notes for item in records),
    }
