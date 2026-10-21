"""Validate one development-only 3D annotation JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinal_ct.annotations import Annotation3D
from spinal_ct.annotations import AnnotationValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        annotation = Annotation3D.load(args.annotation)
    except AnnotationValidationError as error:
        print(f"INVALID: {args.annotation}")
        for item in error.errors:
            print(f"- {item}")
        return 1
    except (OSError, ValueError) as error:
        print(f"INVALID: không đọc được annotation ({error})")
        return 1

    case_id = annotation.payload["case_id"]
    status = annotation.payload["review_status"]
    print(f"VALID: case_id={case_id}, review_status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
