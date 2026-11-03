"""Evaluate automated vertebral body segmentation and landmarks against manual regions."""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct
from spinal_ct.review import ct_fingerprint
from spinal_ct.body_region import sample_plane
from spinal_ct.localization import LEVELS, validate_label_image
from spinal_ct.morphometry import (
    load_session,
    measure_heights,
    auto_segment_body,
    auto_detect_landmarks,
    BodyAutoParams,
    LANDMARK_NAMES,
)


HEIGHT_KEYS = (
    'anterior_mm',
    'middle_mm',
    'posterior_mm',
    'anterior_posterior_ratio',
    'middle_posterior_ratio',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CT image (NIfTI file or DICOM directory)')
    parser.add_argument('--series-uid', help='Optional DICOM series UID')
    parser.add_argument('--labels', type=Path, help='Optional NIfTI vertebral label map (vertebrae.nii.gz)')
    parser.add_argument('--regions', nargs='+', type=Path, required=True, help='User-confirmed manual region directories')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file')
    parser.add_argument('--inset-mm', type=float, default=2.5, help='Landmark inset from anterior/posterior walls in mm')
    parser.add_argument('--anterior-fraction', type=float, default=0.65, help='Cutoff fraction for pedicle cutting')
    parser.add_argument('--neck-ratio', type=float, default=0.60, help='Ratio to anterior reference height for neck detection')
    args = parser.parse_args()

    if args.output.exists():
        parser.error('Output file already exists; choose a new file.')

    try:
        params = BodyAutoParams(
            anterior_fraction=args.anterior_fraction,
            neck_ratio=args.neck_ratio,
            inset_mm=args.inset_mm,
        )

        image = load_ct(args.input, args.series_uid)
        fingerprint = ct_fingerprint(image)

        labels_img = None
        if args.labels is not None:
            labels_img = sitk.ReadImage(str(args.labels))
            validate_label_image(image, labels_img)

        rows = []
        signatures = set()

        for folder in args.regions:
            session = load_session(folder, image, fingerprint=fingerprint)
            if not session.reviewed:
                raise ValueError('All evaluation regions must be confirmed first.')
            if len(session.landmarks) != 6:
                raise ValueError('All evaluation regions must have 6 height landmarks.')

            signature = session._signature()
            if signature in signatures:
                raise ValueError('Duplicate evaluation region.')
            signatures.add(signature)

            if labels_img is not None:
                level_idx = LEVELS.index(session.level) + 1
                _, _, hint = sample_plane(image, session.plane, labels_img, level_idx)
                auto_mask = auto_segment_body(
                    session.ct, session.valid, hint, session.plane.spacing_xy_mm, params=params
                )
            else:
                auto_mask = session.mask.copy()

            auto_landmarks = auto_detect_landmarks(auto_mask, session.plane, params=params)

            manual_heights = measure_heights(session.landmarks, session.plane)
            auto_heights = measure_heights(auto_landmarks, session.plane)

            # Segmentation overlap
            intersection = int(np.logical_and(session.mask, auto_mask).sum())
            union = int(np.logical_or(session.mask, auto_mask).sum())
            manual_px = int(session.mask.sum())
            auto_px = int(auto_mask.sum())
            pixel_area = float(session.plane.spacing_xy_mm[0] * session.plane.spacing_xy_mm[1])

            dice = float(2.0 * intersection / (manual_px + auto_px)) if (manual_px + auto_px) > 0 else 0.0
            iou = float(intersection / union) if union > 0 else 0.0
            manual_area_mm2 = float(manual_px * pixel_area)
            auto_area_mm2 = float(auto_px * pixel_area)
            area_diff_mm2 = float(auto_area_mm2 - manual_area_mm2)

            # Landmarks error (Euclidean distance in mm)
            man_pts = np.asarray(session.landmarks, float)
            auto_pts = np.asarray(auto_landmarks, float)
            pt_diffs = np.linalg.norm(auto_pts - man_pts, axis=1)

            landmark_errors_mm = {
                name: float(err) for name, err in zip(LANDMARK_NAMES, pt_diffs)
            }
            mean_pt_err = float(np.mean(pt_diffs))
            max_pt_err = float(np.max(pt_diffs))
            over_2mm = int(np.sum(pt_diffs > 2.0))

            # Heights comparison
            height_metrics = {}
            for k in HEIGHT_KEYS:
                man_v = float(manual_heights[k])
                auto_v = float(auto_heights[k])
                diff_v = float(auto_v - man_v)
                height_metrics[k] = dict(
                    manual=man_v,
                    auto=auto_v,
                    diff=diff_v,
                    abs_diff=float(abs(diff_v)),
                )

            rows.append(dict(
                region_signature=signature,
                level=session.level,
                age_group=session.age_group,
                source_kind=session.source_kind,
                dice=dice,
                iou=iou,
                manual_area_mm2=manual_area_mm2,
                auto_area_mm2=auto_area_mm2,
                area_diff_mm2=area_diff_mm2,
                landmark_errors_mm=landmark_errors_mm,
                mean_landmark_error_mm=mean_pt_err,
                max_landmark_error_mm=max_pt_err,
                landmarks_over_2mm=over_2mm,
                heights=height_metrics,
            ))

        # Overall summary
        mean_dice = float(np.mean([r['dice'] for r in rows]))
        mean_iou = float(np.mean([r['iou'] for r in rows]))
        mean_lm_err = float(np.mean([r['mean_landmark_error_mm'] for r in rows]))
        max_lm_err = float(np.max([r['max_landmark_error_mm'] for r in rows]))
        total_over_2mm = int(np.sum([r['landmarks_over_2mm'] for r in rows]))

        # Bland-Altman statistics for heights & ratios
        bland_altman = {}
        for k in HEIGHT_KEYS:
            diffs = np.array([r['heights'][k]['diff'] for r in rows], float)
            bias = float(np.mean(diffs))
            sd = float(np.std(diffs))
            loa95 = [float(bias - 1.96 * sd), float(bias + 1.96 * sd)]
            mae = float(np.mean(np.abs(diffs)))
            bland_altman[k] = dict(
                bias=bias,
                sd=sd,
                loa95=loa95,
                mae=mae,
            )

        summary = dict(
            mean_dice=mean_dice,
            mean_iou=mean_iou,
            mean_landmark_error_mm=mean_lm_err,
            max_landmark_error_mm=max_lm_err,
            total_landmarks_over_2mm=total_over_2mm,
            bland_altman=bland_altman,
        )

        report = dict(
            schema_version=1,
            task='body_auto_vs_manual_evaluation',
            input_fingerprint=fingerprint,
            regions_evaluated=len(rows),
            summary=summary,
            per_region=rows,
            parameters=params.to_dict(),
            interpretation='Comparison between automated segmentation/landmarks and approved manual ground truth. '
                           'No patient-identifiable data or local paths included.',
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)

        print(f'Evaluation report saved: {args.output}')
        return 0

    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Evaluation failed: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

