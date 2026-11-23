"""Extract and filter clean normal vertebrae from TotalSegmentator v201 (Step 2.4 - Direction B).

Adheres to:
1. Zero Patient Leakage: External dataset is completely independent from local patients.
2. Anti-Noise Filtering:
   - Excludes FOV truncated vertebrae (touching axial volume boundaries z=0 or z=z_max).
   - Excludes vertebrae with metal / cement artifacts (99th percentile HU >= 1500).
   - Excludes pathological / fractured vertebrae (wedging/fish-mouth: ha_hp or hm_hp not in [0.85, 1.20]).
   - Excludes collapsed vertebrae relative to adjacent levels (relative_collapse_ratio < 0.88).
3. Standard 2.5D Dual-ROI with Fixed Physical FOV 80mm and 16-D L2-normalized feature vectors.
"""
from typing import Any
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import SimpleITK as sitk

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spinal_ct.morphometry import auto_segment_body
from spinal_ct.aospine_dataset import (
    extract_body_canal_roi_25d,
    extract_posterior_arch_roi_25d,
    DEFAULT_PHYSICAL_FOV_MM,
)
from spinal_ct.prototype_bank import (
    extract_roi_prototype_features,
    SPINE_LEVELS,
)

LEVEL_ORDER = ['T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']


def compute_mid_sagittal_heights(b_mask: np.ndarray, sz_mm: float) -> tuple[float, float, float, float]:
    """Compute anterior, posterior, middle height and total height on 2D body mask.

    Parameters
    ----------
    b_mask : np.ndarray
        2D binary mask of vertebral body (axis 0: Z/height, axis 1: Y/depth).
    sz_mm : float
        Voxel size in Z direction (mm).

    Returns
    -------
    tuple of float
        (ha_hp_ratio, hm_hp_ratio, h_total_mm, h_post_mm)
    """
    occ_rows, occ_cols = np.where(b_mask)
    if len(occ_rows) < 10:
        return 1.0, 1.0, 1.0, 1.0

    c_min, c_max = int(occ_cols.min()), int(occ_cols.max())
    col_w = max(1, c_max - c_min)
    ant_cols = (occ_cols <= c_min + int(0.25 * col_w))
    post_cols = (occ_cols >= c_max - int(0.25 * col_w))
    mid_cols = (occ_cols > c_min + int(0.25 * col_w)) & (occ_cols < c_max - int(0.25 * col_w))

    h_ant = float(occ_rows[ant_cols].max() - occ_rows[ant_cols].min() + 1) if ant_cols.any() else 1.0
    h_post = float(occ_rows[post_cols].max() - occ_rows[post_cols].min() + 1) if post_cols.any() else 1.0
    h_mid = float(occ_rows[mid_cols].max() - occ_rows[mid_cols].min() + 1) if mid_cols.any() else 1.0

    ha_hp = float(h_ant / max(h_post, 1.0))
    hm_hp = float(h_mid / max(h_post, 1.0))
    h_total_mm = float((h_ant + h_mid + h_post) * sz_mm)
    h_post_mm = float(h_post * sz_mm)

    return ha_hp, hm_hp, h_total_mm, h_post_mm


def extract_external_normals(
    external_dir: Path | str,
    output_dir: Path | str,
    max_per_level: int = 40,
    fov_mm: float = DEFAULT_PHYSICAL_FOV_MM,
    target_size: tuple[int, int] = (128, 128),
    save_npy: bool = True,
    case_limit: int | None = None,
) -> dict[str, Any]:
    """Process TotalSegmentator external dataset and extract certified clean normal vertebrae.

    Parameters
    ----------
    external_dir : Path or str
        Root directory containing TotalSegmentator cases (sXXXX/).
    output_dir : Path or str
        Destination directory for extracted .npy and manifest.json.
    max_per_level : int
        Maximum number of clean normals to retain per spinal level (T10..L5).
    fov_mm : float
        Fixed physical field-of-view in mm (default 80.0 mm).
    target_size : tuple of int
        Target spatial resolution (128, 128).
    save_npy : bool
        Whether to save body and arch .npy files to output_dir.
    case_limit : int, optional
        Optional cap on cases scanned (useful for testing/smoke runs).

    Returns
    -------
    dict[str, Any]
        Extraction manifest summary.
    """
    ext_root = Path(external_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ext_root.is_dir():
        raise FileNotFoundError(f"External directory not found: {ext_root}")

    # Discover TotalSegmentator cases
    case_dirs = sorted([d for d in ext_root.iterdir() if d.is_dir() and d.name.startswith('s')])
    if case_limit:
        case_dirs = case_dirs[:case_limit]

    print(f"Discovered {len(case_dirs)} TotalSegmentator cases in {ext_root}")

    clean_normals_by_level = {lvl: 0 for lvl in LEVEL_ORDER}
    rejections = {
        'missing_ct': 0,
        'missing_mask': 0,
        'mask_too_small': 0,
        'fov_truncated': 0,
        'metal_or_cement': 0,
        'morphology_wedge_or_biconcave': 0,
        'relative_collapse': 0,
        'extraction_error': 0,
    }
    samples: list[dict[str, Any]] = []
    cases_scanned = 0

    for c_dir in case_dirs:
        # Check if all levels reached max_per_level
        if all(clean_normals_by_level[lvl] >= max_per_level for lvl in LEVEL_ORDER):
            print("Target quota reached for all levels T10-L5. Stopping scan.")
            break

        ct_path = c_dir / "ct.nii.gz"
        seg_dir = c_dir / "segmentations"
        if not ct_path.is_file() or not seg_dir.is_dir():
            rejections['missing_ct'] += 1
            continue

        # Check which levels are needed and present
        needed_levels = [
            lvl for lvl in LEVEL_ORDER
            if clean_normals_by_level[lvl] < max_per_level
            and (seg_dir / f"vertebrae_{lvl}.nii.gz").is_file()
        ]
        if not needed_levels:
            continue

        cases_scanned += 1
        case_name = c_dir.name

        try:
            ct_img = sitk.ReadImage(str(ct_path))
            ct_img = sitk.DICOMOrient(ct_img, 'LPS')
            ct_arr = sitk.GetArrayFromImage(ct_img)
            sp_xyz = ct_img.GetSpacing()  # (sx, sy, sz)
            z_dim, y_dim, x_dim = ct_arr.shape
            spacing_2d = (float(sp_xyz[1]), float(sp_xyz[2]))
        except Exception as e:
            print(f"Error reading CT for case {case_name}: {e}")
            rejections['extraction_error'] += 1
            continue

        # Candidate storage for this case
        case_candidates: dict[str, dict[str, Any]] = {}

        for lvl in needed_levels:
            mask_path = seg_dir / f"vertebrae_{lvl}.nii.gz"
            try:
                m_img = sitk.ReadImage(str(mask_path))
                m_img = sitk.DICOMOrient(m_img, 'LPS')
                m_arr = sitk.GetArrayFromImage(m_img)
            except Exception:
                rejections['missing_mask'] += 1
                continue

            # Check 1: Volume too small or empty
            mask_sum = int(np.sum(m_arr > 0))
            if mask_sum < 500:
                rejections['mask_too_small'] += 1
                continue

            # Check 2: FOV Truncated (touching z=0 or z=z_dim-1)
            z_idx, y_idx, x_idx = np.where(m_arr > 0)
            if z_idx.min() == 0 or z_idx.max() == z_dim - 1:
                rejections['fov_truncated'] += 1
                continue

            # Check 3: Metal / Cement artifacts (99th percentile >= 1500 HU)
            vert_voxels = ct_arr[m_arr > 0]
            if float(np.percentile(vert_voxels, 99)) >= 1500.0:
                rejections['metal_or_cement'] += 1
                continue

            # Check 4: Mid-sagittal slice & body segmentation
            mid_x = int(np.median(x_idx))
            slice_ct = ct_arr[:, :, mid_x].astype(np.float32)
            slice_mask = (m_arr[:, :, mid_x] > 0)
            if not slice_mask.any() or slice_mask.sum() < 25:
                rejections['mask_too_small'] += 1
                continue

            try:
                det = auto_segment_body(
                    slice_ct,
                    np.ones_like(slice_mask, bool),
                    slice_mask,
                    spacing_2d,
                    return_details=True,
                )
                b_m = det['body_mask']
                arch_m = det.get('posterior_arch_mask')
                if arch_m is None or not arch_m.any():
                    arch_m = np.zeros_like(b_m)

                ha_hp, hm_hp, h_total, h_post = compute_mid_sagittal_heights(b_m, sp_xyz[2])

                # Strict Morphometry Filter: 0.85 <= ha_hp <= 1.20 and 0.85 <= hm_hp <= 1.20
                if not (0.85 <= ha_hp <= 1.20 and 0.85 <= hm_hp <= 1.20):
                    rejections['morphology_wedge_or_biconcave'] += 1
                    continue

                case_candidates[lvl] = {
                    'level': lvl,
                    'mid_x': mid_x,
                    'body_mask': b_m,
                    'arch_mask': arch_m,
                    'ha_hp': ha_hp,
                    'hm_hp': hm_hp,
                    'h_total': h_total,
                    'h_post': h_post,
                }
            except Exception:
                rejections['extraction_error'] += 1
                continue

        # Relative collapse check across adjacent levels in this case
        valid_candidates: dict[str, dict[str, Any]] = {}
        for lvl, c_info in case_candidates.items():
            lvl_idx = LEVEL_ORDER.index(lvl)
            adj_heights: list[float] = []

            # Check superior level
            if lvl_idx > 0:
                sup_lvl = LEVEL_ORDER[lvl_idx - 1]
                if sup_lvl in case_candidates:
                    adj_heights.append(case_candidates[sup_lvl]['h_total'])

            # Check inferior level
            if lvl_idx < len(LEVEL_ORDER) - 1:
                inf_lvl = LEVEL_ORDER[lvl_idx + 1]
                if inf_lvl in case_candidates:
                    adj_heights.append(case_candidates[inf_lvl]['h_total'])

            if adj_heights:
                adj_mean = float(np.mean(adj_heights))
                rel_collapse = float(c_info['h_total'] / max(adj_mean, 1e-4))
                if rel_collapse < 0.88:
                    rejections['relative_collapse'] += 1
                    continue
                c_info['relative_collapse_ratio'] = round(rel_collapse, 3)
            else:
                c_info['relative_collapse_ratio'] = 1.0

            valid_candidates[lvl] = c_info

        # Extract ROIs and features for verified clean normals
        for lvl, c_info in valid_candidates.items():
            if clean_normals_by_level[lvl] >= max_per_level:
                continue

            sample_id = f"ext_totalseg_{case_name}_{lvl}"
            try:
                b_roi = extract_body_canal_roi_25d(
                    ct_arr,
                    c_info['body_mask'],
                    sp_xyz,
                    slice_x_index=c_info['mid_x'],
                    fov_mm=fov_mm,
                    target_size=target_size,
                )
                a_roi = extract_posterior_arch_roi_25d(
                    ct_arr,
                    c_info['arch_mask'],
                    sp_xyz,
                    body_mask=c_info['body_mask'],
                    slice_x_index=c_info['mid_x'],
                    fov_mm=fov_mm,
                    target_size=target_size,
                )
                feats = extract_roi_prototype_features(b_roi, a_roi)

                b_rel_path = ""
                a_rel_path = ""
                if save_npy:
                    b_file = out_dir / f"{sample_id}_body.npy"
                    a_file = out_dir / f"{sample_id}_arch.npy"
                    np.save(str(b_file), b_roi.astype(np.float32))
                    np.save(str(a_file), a_roi.astype(np.float32))
                    b_rel_path = f"{sample_id}_body.npy"
                    a_rel_path = f"{sample_id}_arch.npy"

                samples.append({
                    'sample_id': sample_id,
                    'patient_id': f"totalseg_{case_name}",
                    'level': lvl,
                    'aospine_class': 'Normal',
                    'ha_hp_ratio': round(c_info['ha_hp'], 3),
                    'hm_hp_ratio': round(c_info['hm_hp'], 3),
                    'relative_collapse_ratio': c_info.get('relative_collapse_ratio', 1.0),
                    'prototype_features': feats.tolist(),
                    'body_roi_path': b_rel_path,
                    'arch_roi_path': a_rel_path,
                })
                clean_normals_by_level[lvl] += 1
            except Exception as e:
                print(f"Error extracting ROIs for {sample_id}: {e}")
                rejections['extraction_error'] += 1

    # Compile and export manifest.json
    total_clean = sum(clean_normals_by_level.values())
    manifest = {
        'total_cases_scanned': cases_scanned,
        'clean_normals_by_level': clean_normals_by_level,
        'total_clean_normals': total_clean,
        'max_per_level': max_per_level,
        'rejections': rejections,
        'samples': samples,
    }

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n" + "=" * 60)
    print("TOTAL-SEGMENTATOR NORMAL EXTRACTION COMPLETE")
    print(f"Cases scanned: {cases_scanned}")
    print(f"Total clean normals extracted: {total_clean}")
    print("Level breakdown:")
    for lvl, cnt in clean_normals_by_level.items():
        print(f"  {lvl}: {cnt} / {max_per_level}")
    print("Rejection reasons:")
    for r, cnt in rejections.items():
        print(f"  {r}: {cnt}")
    print(f"Manifest written to: {manifest_file}")
    print("=" * 60)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract clean normal vertebrae from TotalSegmentator Small v201.")
    parser.add_argument('--external-dir', type=str, default='data/external/totalsegmentator_small',
                        help='Directory with TotalSegmentator cases.')
    parser.add_argument('--output-dir', type=str, default='data/processed/external_normal_rois',
                        help='Output directory for manifest and ROIs.')
    parser.add_argument('--max-per-level', type=int, default=40,
                        help='Maximum clean vertebrae per level T10-L5 (default: 40).')
    parser.add_argument('--no-save-npy', action='store_true',
                        help='Skip saving .npy files (save features in manifest only).')
    parser.add_argument('--case-limit', type=int, default=None,
                        help='Optional case limit for quick dry-runs.')
    args = parser.parse_args()

    extract_external_normals(
        external_dir=args.external_dir,
        output_dir=args.output_dir,
        max_per_level=args.max_per_level,
        save_npy=not args.no_save_npy,
        case_limit=args.case_limit,
    )


if __name__ == '__main__':
    main()
