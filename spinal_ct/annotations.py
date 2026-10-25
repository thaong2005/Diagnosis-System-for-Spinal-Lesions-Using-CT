"""Schema and validation for development-only 3D vertebral annotations.

Annotations are never production inference inputs.  They describe reviewed
development data and keep both voxel-index and physical LPS coordinates so a
ROI can be checked against the exact CT geometry that produced it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from spinal_ct.labels import parse_vertebral_levels


SCHEMA_VERSION = 1
REVIEW_STATUSES = frozenset({"draft", "reviewed", "rejected"})
_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_AXES = ("z", "y", "x")


class AnnotationValidationError(ValueError):
    """Raised when an annotation does not satisfy the 3D schema."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Annotation không hợp lệ: " + "; ".join(errors))


def _finite_number(value: Any, field_name: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field_name} phải là số hữu hạn")
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{field_name} phải là số hữu hạn")
    return number


def _positive_int(value: Any, field_name: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{field_name} phải là số nguyên dương")
        return 0
    return value


def _vector(
    value: Any,
    field_name: str,
    errors: list[str],
    *,
    length: int,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        errors.append(f"{field_name} phải có {length} phần tử")
        return tuple(0.0 for _ in range(length))
    return tuple(
        _finite_number(item, f"{field_name}[{index}]", errors)
        for index, item in enumerate(value)
    )


def _bbox(
    value: Any,
    field_name: str,
    errors: list[str],
    *,
    axes: tuple[str, ...],
    numeric: bool,
) -> dict[str, tuple[int | float, int | float]]:
    if not isinstance(value, Mapping):
        errors.append(f"{field_name} phải là object")
        return {}

    result: dict[str, tuple[int | float, int | float]] = {}
    for axis in axes:
        min_key = f"{axis}_min"
        max_key = f"{axis}_max_exclusive" if not numeric else f"{axis}_max"
        if min_key not in value or max_key not in value:
            errors.append(f"{field_name} thiếu {min_key}/{max_key}")
            continue

        if numeric:
            minimum = _finite_number(value[min_key], f"{field_name}.{min_key}", errors)
            maximum = _finite_number(value[max_key], f"{field_name}.{max_key}", errors)
        else:
            minimum = value[min_key]
            maximum = value[max_key]
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
            ):
                errors.append(f"{field_name}.{axis} phải là số nguyên")
                continue

        if maximum <= minimum or (not numeric and minimum < 0):
            constraint = "0 <= min < max" if not numeric else "min < max"
            errors.append(f"{field_name}.{axis} phải thỏa {constraint}")
        result[axis] = (minimum, maximum)
    return result


def validate_annotation(payload: Mapping[str, Any]) -> list[str]:
    """Return all schema and geometry errors without exposing patient data."""
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["annotation phải là object"]

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version phải là {SCHEMA_VERSION}")

    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not _CASE_ID_PATTERN.fullmatch(case_id):
        errors.append("case_id phải là mã ẩn danh dạng chữ/số, _, ., -")

    try:
        levels = parse_vertebral_levels(payload.get("vertebral_levels", []))
        if not levels:
            errors.append("vertebral_levels không được rỗng")
    except (TypeError, ValueError):
        errors.append("vertebral_levels phải chứa mức đốt hợp lệ")

    if payload.get("target") != "vertebral_body":
        errors.append("target phải là 'vertebral_body'")

    status = payload.get("review_status")
    if status not in REVIEW_STATUSES:
        errors.append("review_status phải là draft, reviewed hoặc rejected")

    voxel_bbox = _bbox(
        payload.get("bbox_zyx_voxel"),
        "bbox_zyx_voxel",
        errors,
        axes=_AXES,
        numeric=False,
    )
    physical_bbox = _bbox(
        payload.get("bbox_lps_mm"),
        "bbox_lps_mm",
        errors,
        axes=("x", "y", "z"),
        numeric=True,
    )

    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        errors.append("geometry phải là object")
        geometry = {}

    shape = geometry.get("shape_zyx")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        errors.append("geometry.shape_zyx phải có 3 phần tử")
        shape_values = (0, 0, 0)
    else:
        shape_values = tuple(
            _positive_int(value, f"geometry.shape_zyx[{index}]", errors)
            for index, value in enumerate(shape)
        )

    _vector(geometry.get("spacing_zyx_mm"), "geometry.spacing_zyx_mm", errors, length=3)
    _vector(geometry.get("origin_lps_mm"), "geometry.origin_lps_mm", errors, length=3)
    _vector(geometry.get("direction_lps"), "geometry.direction_lps", errors, length=9)

    for axis_index, axis in enumerate(_AXES):
        if axis in voxel_bbox and shape_values[axis_index] > 0:
            _, maximum = voxel_bbox[axis]
            if maximum > shape_values[axis_index]:
                errors.append(
                    f"bbox_zyx_voxel.{axis}_max_exclusive vượt geometry.shape_zyx"
                )

    if status == "rejected" and not payload.get("rejection_reason"):
        errors.append("annotation rejected phải có rejection_reason")
    if status != "rejected" and payload.get("rejection_reason"):
        errors.append("rejection_reason chỉ được dùng khi review_status là rejected")

    return errors


@dataclass(frozen=True)
class Annotation3D:
    """Validated, development-only 3D annotation."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Annotation3D":
        errors = validate_annotation(payload)
        if errors:
            raise AnnotationValidationError(errors)
        return cls(payload=dict(payload))

    @classmethod
    def load(cls, path: str | Path) -> "Annotation3D":
        annotation_path = Path(path)
        with annotation_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
