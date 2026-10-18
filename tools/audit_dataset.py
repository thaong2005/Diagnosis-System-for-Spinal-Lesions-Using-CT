"""Audit local CT labels and DICOM headers without exporting patient identity."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pydicom

from spinal_ct.labels import load_reference_labels, normalize_text
from spinal_ct.labels import summarize_reference_labels


def iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {"", ".dcm"}:
            yield path


def read_header(path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Incorrect value for Specific Character Set.*",
            category=UserWarning,
            module="pydicom.charset",
        )
        try:
            return pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def audit_dicom(root: Path, known_identifiers: set[str]) -> dict[str, object]:
    series = defaultdict(int)
    kernels = Counter()
    patient_ids = set()
    matched_patient_ids = set()
    candidate_files = 0
    unreadable = 0
    non_ct_files = 0
    invalid_charset_files = 0
    dicom_images = 0

    for path in iter_candidate_files(root):
        candidate_files += 1
        try:
            dataset = read_header(path)
        except Exception:
            unreadable += 1
            continue

        required = ("SOPInstanceUID", "SeriesInstanceUID", "Rows", "Columns")
        if not all(hasattr(dataset, field) for field in required):
            unreadable += 1
            continue
        if normalize_text(getattr(dataset, "Modality", "")) != "CT":
            non_ct_files += 1
            continue

        charset = normalize_text(getattr(dataset, "SpecificCharacterSet", ""))
        if charset == "ISO IR 192":
            invalid_charset_files += 1

        dicom_images += 1
        series[str(dataset.SeriesInstanceUID)] += 1
        kernels[normalize_text(getattr(dataset, "ConvolutionKernel", "Unknown"))] += 1
        patient_id = normalize_text(getattr(dataset, "PatientID", ""))
        if patient_id:
            patient_ids.add(patient_id)
            if patient_id in known_identifiers:
                matched_patient_ids.add(patient_id)

    return {
        "candidate_files": candidate_files,
        "ct_image_files": dicom_images,
        "unreadable_or_non_image_files": unreadable,
        "non_ct_image_files": non_ct_files,
        "invalid_specific_character_set_files": invalid_charset_files,
        "ct_series_count": len(series),
        "distinct_dicom_patient_ids": len(patient_ids),
        "matched_dicom_patient_ids": len(matched_patient_ids),
        "unmatched_dicom_patient_ids": len(patient_ids - matched_patient_ids),
        "kernel_image_counts": dict(sorted(kernels.items())),
        "series_size": {
            "min": min(series.values(), default=0),
            "max": max(series.values(), default=0),
        },
    }


def build_audit(raw_root: Path, labels_path: Path) -> dict[str, object]:
    labels = load_reference_labels(labels_path)
    known_identifiers = {
        identifier
        for label in labels
        for identifier in label.identifiers
    }
    return {
        "schema_version": 1,
        "contains_patient_identifiers": False,
        "labels": summarize_reference_labels(labels),
        "dicom": audit_dicom(raw_root, known_identifiers),
        "warnings": [
            "Aggregate audit only; matching by PatientID must be clinically verified.",
            "Do not treat unlabelled vertebrae as confirmed negative examples.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.raw_root, args.labels)
    content = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
