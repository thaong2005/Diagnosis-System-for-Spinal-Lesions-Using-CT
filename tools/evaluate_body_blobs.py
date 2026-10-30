"""Evaluate LoG centre proposals against approved body regions of one CT."""
import argparse
import json
from pathlib import Path
import numpy as np
from spinal_ct.ct_io import load_ct
from spinal_ct.review import ct_fingerprint
from spinal_ct.morphometry import load_session
from spinal_ct.blob_detection import detect_blobs, assess_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--series-uid')
    parser.add_argument('--regions', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file')
    parser.add_argument('--tolerance-mm', type=float, required=True, help='Choose before evaluation; centroid distance criterion')
    parser.add_argument('--min-sigma-mm', type=float, default=2)
    parser.add_argument('--max-sigma-mm', type=float, default=12)
    parser.add_argument('--threshold', type=float, default=.03)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error('Choose a new output file and a positive finite matching tolerance.')
    try:
        image = load_ct(args.input, args.series_uid)
        fingerprint = ct_fingerprint(image)
        rows, signatures = [], set()
        for folder in args.regions:
            session = load_session(folder, image, fingerprint=fingerprint)
            if not session.reviewed:
                raise ValueError('All evaluation regions must be confirmed first.')
            # Do not count reruns of an identical region as independent examples.
            signature = session._signature()
            if signature in signatures:
                raise ValueError('Duplicate evaluation region.')
            signatures.add(signature)
            detection = detect_blobs(session.ct, session.plane.spacing_xy_mm, session.valid,
                min_sigma_mm=args.min_sigma_mm, max_sigma_mm=args.max_sigma_mm, threshold=args.threshold)
            comparison = assess_candidates(detection, session.mask, session.plane.spacing_xy_mm,
                                            tolerance_mm=args.tolerance_mm)
            rows.append(dict(region_signature=signature, level=session.level, age_group=session.age_group,
                             source_kind=session.source_kind, detection=detection, **comparison))
        distances = [r['nearest_distance_mm'] for r in rows if r['nearest_distance_mm'] is not None]
        report = dict(schema_version=1, task='body_blob_centre_evaluation', input_fingerprint=fingerprint,
                      regions=len(rows), matched_roi_fraction=sum(r['matched'] for r in rows)/len(rows),
                      mean_nearest_distance_mm=float(np.mean(distances)) if distances else None,
                      no_candidate_rois=sum(r['candidate_count'] == 0 for r in rows),
                      extra_candidates=sum(r['extra_candidates'] for r in rows), per_region=rows,
                      interpretation='Matching to user-confirmed mask centroids; not clinical detection sensitivity. '
                                     'Regions may be correlated within a patient; lock parameters on independent validation data.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        print(f'Blob centre evaluation saved: {args.output}')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Blob evaluation failed: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
