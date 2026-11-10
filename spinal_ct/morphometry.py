"""Editable body measurement sessions, review invalidation and reproducible export."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from spinal_ct.body_region import BodyPlane, binary_mask, sample_plane, spacing_xy
from spinal_ct.blob_analysis import analyze_blob, normalize_region
from spinal_ct.area_monte_carlo import estimate_area, validate_sampling
from spinal_ct.localization import LEVELS
from spinal_ct.review import ct_fingerprint
from spinal_ct.aospine_rules import evaluate_aospine_decision_tree, rank_and_classify_spine
from spinal_ct.aospine_fusion import fuse_and_classify_spine

LANDMARK_NAMES = ('anterior_superior', 'anterior_inferior', 'middle_superior',
                  'middle_inferior', 'posterior_superior', 'posterior_inferior')


@dataclass(frozen=True)
class BodyAutoParams:
    anterior_fraction: float = 0.65
    neck_ratio: float = 0.60
    ant_ref_span: float = 0.45
    search_start: float = 0.35
    search_end: float = 0.85
    min_component_ratio: float = 0.15
    inset_mm: float = 2.5
    band_half_mm: float = 1.5
    min_hu: float | None = None
    min_normal_ap_ratio: float = 0.80
    max_normal_ap_ratio: float = 1.30

    def __post_init__(self):
        if not np.isfinite(self.anterior_fraction) or not (0.35 <= self.anterior_fraction <= 1.0):
            raise ValueError('anterior_fraction must be between 0.35 and 1.0.')
        if not np.isfinite(self.neck_ratio) or not (0.10 <= self.neck_ratio <= 0.95):
            raise ValueError('neck_ratio must be between 0.10 and 0.95.')
        if not np.isfinite(self.ant_ref_span) or self.ant_ref_span <= 0:
            raise ValueError('ant_ref_span must be positive.')
        if not np.isfinite(self.search_start) or not np.isfinite(self.search_end) or self.search_start >= self.search_end:
            raise ValueError('search_start must be less than search_end.')
        if not np.isfinite(self.min_component_ratio) or self.min_component_ratio <= 0:
            raise ValueError('min_component_ratio must be positive.')
        if not np.isfinite(self.inset_mm) or self.inset_mm < 0:
            raise ValueError('inset_mm must be non-negative.')
        if not np.isfinite(self.band_half_mm) or self.band_half_mm <= 0:
            raise ValueError('band_half_mm must be positive.')
        if self.min_hu is not None and not np.isfinite(self.min_hu):
            raise ValueError('min_hu must be finite when specified.')
        if not np.isfinite(self.min_normal_ap_ratio) or not np.isfinite(self.max_normal_ap_ratio) or self.min_normal_ap_ratio >= self.max_normal_ap_ratio:
            raise ValueError('Invalid normal AP ratio range.')

    def to_dict(self) -> dict:
        return asdict(self)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal 2D holes in a boolean mask without modifying input in place."""
    clean = binary_mask(mask).copy()
    if not clean.any():
        return clean
    filled = sitk.BinaryFillhole(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    return sitk.GetArrayFromImage(filled).astype(bool)


def _largest_anterior_component(mask: np.ndarray, min_component_ratio: float = 0.15) -> np.ndarray:
    """Select the anterior vertebral body component and discard posterior arch fragments."""
    clean = binary_mask(mask).copy()
    if not clean.any():
        return clean
    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    labels = sitk.GetArrayFromImage(cc_img)
    max_label = int(labels.max())
    if max_label <= 1:
        return clean
    counts = np.bincount(labels.ravel())[1:]
    max_area = int(counts.max())
    candidates = []
    for label_id, area in enumerate(counts, start=1):
        if area >= max(3, int(min_component_ratio * max_area)):
            rows, cols = np.nonzero(labels == label_id)
            candidates.append((float(cols.min()), float(cols.mean()), -int(area), label_id))
    candidates.sort()
    return labels == candidates[0][3]


def extract_posterior_arch_mask(full_vert_mask_2d: np.ndarray, cut_col: int | None,
                                body_mask: np.ndarray | None = None) -> np.ndarray:
    """Extract posterior arch mask (col > cut_col) from whole vertebra sagittal mask.

    Parameters
    ----------
    full_vert_mask_2d : np.ndarray
        2D binary mask of whole vertebra.
    cut_col : int | None
        Column index where anterior body was severed from posterior arch.
        If None and body_mask is provided, extracts posterior structures by subtracting body_mask.
        If None or non-positive without body_mask, returns an all-False mask of identical shape.
    body_mask : np.ndarray | None
        Optional 2D binary mask of the segmented anterior vertebral body.

    Returns
    -------
    np.ndarray
        Boolean 2D mask containing only the posterior arch structures (col > cut_col).
    """
    clean = binary_mask(full_vert_mask_2d).copy()
    if cut_col is not None and 0 < cut_col < clean.shape[1] - 1:
        arch = clean.copy()
        arch[:, :cut_col + 1] = False
        return arch
    if body_mask is not None and binary_mask(body_mask).any():
        arch = clean & (~binary_mask(body_mask))
        body_cols = np.flatnonzero(binary_mask(body_mask).any(axis=0))
        if len(body_cols) > 0:
            arch[:, :int(body_cols.max()) + 1] = False
        return arch
    return np.zeros_like(clean, dtype=bool)


def find_spinous_process_tip(posterior_arch_mask: np.ndarray,
                             spacing_mm: tuple[float, float]) -> dict:
    """Find the spinous process tip (max col, min row) and segmentation quality.

    In the 2D LPS Sagittal coordinate convention:
    - col = 0 is Anterior, col increases Posteriorly.
    - row = 0 is Inferior, row increases Superiorly.
    The spinous process projects posteriorly and inferiorly. The tip is represented
    by the most posterior column (max col), taking the most inferior row (min row)
    among occupied pixels at that column.

    Parameters
    ----------
    posterior_arch_mask : np.ndarray
        2D boolean mask of the posterior arch structures.
    spacing_mm : tuple[float, float]
        Pixel spacing (sx, sy) in mm along (col, row).

    Returns
    -------
    dict
        tip_xy_mm : list[float] | None
            Physical coordinates [x_mm, y_mm] of the spinous tip.
        tip_rc : tuple[int, int] | None
            Matrix indices (row, col) of the spinous tip.
        q_seg : float
            Segmentation quality metric in [0.0, 1.0] based on size and continuity.
        area_mm2 : float
            Physical area of posterior arch mask in mm^2.
        continuity : float
            Ratio of largest connected component area to total area.
        is_valid : bool
            True if tip was detected on a valid mask.
    """
    sx, sy = spacing_xy(spacing_mm)
    mask = binary_mask(posterior_arch_mask).copy()
    occupied_rows, occupied_cols = np.nonzero(mask)
    if len(occupied_cols) == 0:
        return dict(
            tip_xy_mm=None,
            tip_rc=None,
            q_seg=0.0,
            area_mm2=0.0,
            continuity=0.0,
            is_valid=False,
        )

    max_col = int(np.max(occupied_cols))
    tip_rows = occupied_rows[occupied_cols == max_col]
    # Spinous process slants inferior-posteriorly: take min row (most inferior)
    tip_row = int(np.min(tip_rows))

    tip_x_mm = float(max_col * sx)
    tip_y_mm = float(tip_row * sy)
    area_mm2 = float(len(occupied_rows) * sx * sy)

    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)), True)
    labels = sitk.GetArrayViewFromImage(cc_img)
    counts = np.bincount(labels.ravel())[1:]
    if len(counts) == 0:
        continuity = 0.0
    else:
        continuity = float(np.max(counts) / np.sum(counts))

    # Quality metric: product of continuity and size sufficiency (typical arch >= 80 mm^2)
    size_factor = float(np.clip(area_mm2 / 80.0, 0.0, 1.0))
    q_seg = float(np.clip(continuity * size_factor, 0.0, 1.0))

    return dict(
        tip_xy_mm=[round(tip_x_mm, 2), round(tip_y_mm, 2)],
        tip_rc=(tip_row, max_col),
        q_seg=round(q_seg, 4),
        area_mm2=round(area_mm2, 2),
        continuity=round(continuity, 4),
        is_valid=True,
    )


def auto_segment_body(ct: np.ndarray, valid: np.ndarray, hint: np.ndarray,
                      spacing: tuple, *, params: BodyAutoParams | None = None,
                      return_details: bool = False) -> np.ndarray | dict:
    """Isolate the 2D vertebral body mask from whole-vertebra sagittal hint and CT.
    
    If return_details is True, returns a dict with 'body_mask', 'posterior_arch_mask',
    and 'cut_col' for AO Spine biomechanical analysis. Otherwise returns body mask.
    """
    params = params or BodyAutoParams()
    sx, sy = spacing_xy(spacing)
    ct_arr = np.asarray(ct, dtype=float)
    valid_mask = binary_mask(valid).copy()
    hint_mask = binary_mask(hint).copy()
    if ct_arr.shape != valid_mask.shape or ct_arr.shape != hint_mask.shape:
        raise ValueError('CT, valid FOV and hint masks must share the same 2D shape.')
    if not np.isfinite(params.anterior_fraction) or not 0.35 <= params.anterior_fraction <= 1.0:
        raise ValueError('anterior_fraction must be between 0.35 and 1.0.')
    if not np.isfinite(params.neck_ratio) or not 0.10 <= params.neck_ratio <= 0.95:
        raise ValueError('neck_ratio must be between 0.10 and 0.95.')

    candidate = hint_mask & valid_mask
    if params.min_hu is not None:
        if not np.isfinite(params.min_hu):
            raise ValueError('min_hu must be finite when specified.')
        candidate = candidate & (ct_arr >= float(params.min_hu))

    if not candidate.any():
        raise ValueError('No vertebral candidate pixels found within the valid CT field of view.')

    candidate = _fill_holes(candidate) & valid_mask

    # Close small internal 1-2 px horizontal gaps within candidate structures (e.g. vascular grooves or noise)
    # while leaving true canal spaces (>= 5 mm) untouched
    gap_close_px = max(1, int(round(1.5 / sx)))
    if gap_close_px >= 1:
        sitk_cand = sitk.GetImageFromArray(candidate.astype(np.uint8))
        closed_sitk = sitk.BinaryMorphologicalClosing(sitk_cand, [gap_close_px, 1], sitk.sitkBall)
        closed_candidate = sitk.GetArrayFromImage(closed_sitk).astype(bool)
        candidate = (closed_candidate | candidate) & valid_mask

    body = _largest_anterior_component(candidate, params.min_component_ratio)

    # Check if candidate already contains distinct posterior components separated by spinal canal
    other = candidate & (~body)
    has_separated_posterior_arch = False
    arch_component = None
    if other.any():
        body_cols = np.flatnonzero(body.any(axis=0))
        if len(body_cols) > 0:
            b_max = int(body_cols[-1])
            cc_other = sitk.ConnectedComponent(sitk.GetImageFromArray(other.astype(np.uint8)), True)
            labels_other = sitk.GetArrayFromImage(cc_other)
            max_lbl = int(labels_other.max())
            arch_cand_labels = []
            min_sep_px = max(2, int(round(2.5 / sx)))
            for lbl in range(1, max_lbl + 1):
                mask_lbl = (labels_other == lbl)
                lbl_cols = np.flatnonzero(mask_lbl.any(axis=0))
                if len(lbl_cols) > 0 and int(lbl_cols.min()) >= b_max + min_sep_px:
                    arch_cand_labels.append(lbl)
            if arch_cand_labels:
                has_separated_posterior_arch = True
                arch_component = np.isin(labels_other, arch_cand_labels)

    cols = np.flatnonzero(body.any(axis=0))
    cut_col = None
    if (not has_separated_posterior_arch) and len(cols) >= 6:
        c_min, c_max = int(cols[0]), int(cols[-1])
        span = c_max - c_min + 1
        col_heights = body[:, c_min:c_max + 1].sum(axis=0).astype(float) * sy
        ant_span = max(2, int(round(params.ant_ref_span * span)))
        ant_ref_height = float(np.percentile(col_heights[:ant_span], 75))

        s_start = max(2, int(round(params.search_start * span)))
        s_end = min(span - 1, max(s_start + 1, int(round(params.search_end * span))))
        if ant_ref_height > 0 and s_end > s_start:
            window = col_heights[s_start:s_end]
            min_idx_rel = int(np.argmin(window))
            valley_height = float(window[min_idx_rel])

            # Behind the valley, check if there is a posterior peak (height rises back up)
            post_window = col_heights[s_start + min_idx_rel:]
            post_peak_height = float(np.max(post_window)) if len(post_window) > 0 else 0.0
            has_posterior_peak = (post_peak_height >= 0.40 * ant_ref_height) or (post_peak_height >= 1.4 * max(valley_height, 1e-3))

            if (valley_height <= params.neck_ratio * ant_ref_height) and has_posterior_peak:
                drop_indices = np.flatnonzero(window <= params.neck_ratio * ant_ref_height)
                neck_rel = int(drop_indices[0]) if len(drop_indices) else min_idx_rel
                cut_col = c_min + min(s_start + min_idx_rel,
                                      max(s_start + neck_rel, int(round(params.anterior_fraction * span))))
                trimmed = body.copy()
                trimmed[:, cut_col:] = False
                if trimmed.any():
                    body = _largest_anterior_component(trimmed, params.min_component_ratio)

    body = _fill_holes(body) & valid_mask
    if int(body.sum()) < 3:
        raise ValueError('Segmented vertebral body region is too small (< 3 pixels).')
    if return_details:
        if arch_component is not None and arch_component.any():
            posterior_arch = arch_component
        else:
            posterior_arch = extract_posterior_arch_mask(candidate, cut_col, body_mask=body)
        return dict(body_mask=body, posterior_arch_mask=posterior_arch, cut_col=cut_col)
    return body


def _wall_x(mask: np.ndarray, sx: float) -> tuple[float, float]:
    """Anterior/posterior wall x (mm) from the middle half of body height."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 2:
        raise ValueError('Mask needs at least 2 occupied rows to estimate anterior/posterior walls.')
    lo = len(rows) // 4
    hi = max(lo + 1, (3 * len(rows)) // 4)
    mid = rows[lo:hi]
    first = [int(np.flatnonzero(mask[r])[0]) for r in mid]
    last = [int(np.flatnonzero(mask[r])[-1]) for r in mid]
    return float(np.median(first) * sx), float(np.median(last) * sx)


def _band_edges(mask: np.ndarray, sx: float, sy: float, x_mm: float, half_mm: float = 1.5) -> tuple[float, float]:
    """Median superior/inferior contour y (mm) in a vertical band, filtering staircase edge steps."""
    occupied_cols = np.flatnonzero(mask.any(axis=0))
    eff_half = max(float(half_mm), 0.6 * float(sx))
    cols = [int(c) for c in occupied_cols if abs(c * sx - x_mm) <= eff_half]
    if not cols:
        raise ValueError('No mask pixels near landmark column.')
    ys = [np.flatnonzero(mask[:, c]) for c in cols]
    heights = np.array([r[-1] - r[0] for r in ys])
    keep = heights >= np.median(heights)
    ys = [r for r, k in zip(ys, keep) if k]
    return float(np.median([r[-1] for r in ys]) * sy), float(np.median([r[0] for r in ys]) * sy)


def auto_detect_landmarks(mask: np.ndarray, plane: BodyPlane, *,
                          params: BodyAutoParams | None = None) -> list[list[float]]:
    """Detect 6 landmarks along vertical columns inset from the anterior and posterior walls."""
    from spinal_ct.body_region import spacing_xy
    params = params or BodyAutoParams()
    clean = binary_mask(mask).copy()
    if tuple(plane.size_xy[::-1]) != clean.shape:
        raise ValueError('Mask shape does not match plane dimensions.')
    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    if int(sitk.GetArrayViewFromImage(cc_img).max()) > 1:
        raise ValueError('Mask contains multiple disconnected components.')
    sx, sy = spacing_xy(plane.spacing_xy_mm)
    xa, xp = _wall_x(clean, sx)
    width_mm = xp - xa
    if width_mm <= 2 * sx:
        raise ValueError('Vertebral body mask is too narrow to place anterior/middle/posterior landmarks.')
    eff_inset = min(float(params.inset_mm), 0.25 * width_mm)
    pts: list[list[float]] = []
    for x in (xa + eff_inset, 0.5 * (xa + xp), xp - eff_inset):
        sup, inf = _band_edges(clean, sx, sy, x, half_mm=params.band_half_mm)
        pts.extend([[float(x), float(sup)], [float(x), float(inf)]])
    measure_heights(pts, plane)
    return pts


def mask_digest(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def measure_heights(points, plane):
    if not points:
        return None
    points = np.asarray(points, float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError('Height measurements need all six finite landmarks.')
    spacing = np.asarray(plane.spacing_xy_mm)
    if (points < -spacing/2).any() or (points > (np.array(plane.size_xy)-.5)*spacing).any():
        raise ValueError('Height landmarks extend beyond the plane.')
    paired = points.reshape(3, 2, 2)
    if (paired[:, 0, 1] <= paired[:, 1, 1]).any() or (np.diff(paired.mean(1)[:, 0]) <= 0).any():
        raise ValueError('Landmarks must be superior/inferior pairs in anterior → middle → posterior order.')
    heights = np.linalg.norm(paired[:, 0]-paired[:, 1], axis=1)
    return dict(anterior_mm=float(heights[0]), middle_mm=float(heights[1]), posterior_mm=float(heights[2]),
                anterior_posterior_ratio=float(heights[0]/heights[2]),
                middle_posterior_ratio=float(heights[1]/heights[2]),
                definition='Euclidean distances between manually confirmed superior/inferior landmark pairs.')


def project_point_between_planes(
    point_xy_mm: list[float] | tuple[float, float],
    source_plane: BodyPlane | None,
    target_plane: BodyPlane | None,
) -> list[float]:
    """Project a 2D point [x, y] in mm from source_plane to target_plane via 3D LPS space.

    Parameters
    ----------
    point_xy_mm : list[float] | tuple[float, float]
        2D coordinate [x, y] in source_plane in millimeters.
    source_plane : BodyPlane | None
        Source plane containing origin_lps_mm, u_lps, and v_lps.
    target_plane : BodyPlane | None
        Target plane containing origin_lps_mm, u_lps, and v_lps.

    Returns
    -------
    list[float]
        Projected [x, y] coordinate in target_plane in millimeters.
    """
    if source_plane is None or target_plane is None or source_plane is target_plane:
        return [round(float(point_xy_mm[0]), 2), round(float(point_xy_mm[1]), 2)]
    p3d = np.asarray(source_plane.point_lps(point_xy_mm), dtype=float)
    origin_tgt = np.asarray(target_plane.origin_lps_mm, dtype=float)
    u_tgt = np.asarray(target_plane.u_lps, dtype=float)
    v_tgt = np.asarray(target_plane.v_lps, dtype=float)
    rel = p3d - origin_tgt
    x_tgt = float(np.dot(rel, u_tgt))
    y_tgt = float(np.dot(rel, v_tgt))
    return [round(x_tgt, 2), round(y_tgt, 2)]


def compute_endplate_deformities(
    landmarks: list | dict | None,
    body_mask: np.ndarray | None = None,
    spacing_mm: tuple | None = None,
) -> tuple[float, float]:
    """Compute upper and lower endplate deformity in mm from 6 landmarks.

    From 6 landmarks [AS, AI, MS, MI, PS, PI]:
    - upper_def: perpendicular distance / sagitta from MS to line AS-PS
    - lower_def: perpendicular distance / sagitta from MI to line AI-PI

    Returns
    -------
    tuple[float, float]
        (upper_endplate_deformity_mm, lower_endplate_deformity_mm)
    """
    if not landmarks:
        return 0.0, 0.0
    if isinstance(landmarks, dict):
        as_pt = landmarks.get('anterior_superior')
        ai_pt = landmarks.get('anterior_inferior')
        ms_pt = landmarks.get('middle_superior')
        mi_pt = landmarks.get('middle_inferior')
        ps_pt = landmarks.get('posterior_superior')
        pi_pt = landmarks.get('posterior_inferior')
    elif isinstance(landmarks, (list, tuple)) and len(landmarks) >= 6:
        as_pt, ai_pt, ms_pt, mi_pt, ps_pt, pi_pt = landmarks[:6]
    else:
        return 0.0, 0.0

    if any(pt is None for pt in (as_pt, ai_pt, ms_pt, mi_pt, ps_pt, pi_pt)):
        return 0.0, 0.0

    # Upper endplate deformity (perpendicular distance & sagitta from MS to line AS-PS)
    dx_up = float(ps_pt[0] - as_pt[0])
    dy_up = float(ps_pt[1] - as_pt[1])
    l_up = float(np.hypot(dx_up, dy_up))
    if l_up > 1e-4:
        up_def = abs(dy_up * ms_pt[0] - dx_up * ms_pt[1] + ps_pt[0] * as_pt[1] - ps_pt[1] * as_pt[0]) / l_up
    else:
        up_def = abs(float(ms_pt[1]) - 0.5 * (float(as_pt[1]) + float(ps_pt[1])))

    # Lower endplate deformity (perpendicular distance & sagitta from MI to line AI-PI)
    dx_lo = float(pi_pt[0] - ai_pt[0])
    dy_lo = float(pi_pt[1] - ai_pt[1])
    l_lo = float(np.hypot(dx_lo, dy_lo))
    if l_lo > 1e-4:
        lo_def = abs(dy_lo * mi_pt[0] - dx_lo * mi_pt[1] + pi_pt[0] * ai_pt[1] - pi_pt[1] * ai_pt[0]) / l_lo
    else:
        lo_def = abs(float(mi_pt[1]) - 0.5 * (float(ai_pt[1]) + float(pi_pt[1])))

    return round(float(up_def), 2), round(float(lo_def), 2)


def is_vertebra_fov_truncated(session_or_data) -> bool:
    """Check if vertebra is truncated at the CT field of view boundary (< 10 mm height or < 150 mm2 area)."""
    if hasattr(session_or_data, 'landmarks') and getattr(session_or_data, 'landmarks', None) is not None:
        lm = session_or_data.landmarks
        if len(lm) == 6:
            plane = getattr(session_or_data, 'plane', None)
            if plane is not None:
                h = measure_heights(lm, plane)
                max_h = max(h.get('anterior_mm', 0.0), h.get('middle_mm', 0.0), h.get('posterior_mm', 0.0))
                if 0.0 < max_h < 10.0:
                    return True
        mask = getattr(session_or_data, 'mask', None)
        plane = getattr(session_or_data, 'plane', None)
        if mask is not None and mask.any() and plane is not None:
            sx, sy = getattr(plane, 'spacing_xy_mm', (1.0, 1.0))
            area_mm2 = float(np.sum(mask)) * sx * sy
            if area_mm2 < 150.0:
                return True
    elif isinstance(session_or_data, dict):
        h = session_or_data.get('heights') or {}
        ha = h.get('anterior_mm', 0.0)
        hm = h.get('middle_mm', 0.0)
        hp = h.get('posterior_mm', 0.0)
        max_h = max(ha, hm, hp)
        if 0.0 < max_h < 10.0:
            return True
        area = session_or_data.get('area_mm2', 999.0)
        if area < 150.0:
            return True
    return False


def measure_posterior_wall_retropulsion(
    ct_slice: np.ndarray,
    spacing_mm: tuple[float, float],
    curr_vert_landmarks: list[list[float]] | dict,
    upper_vert_pi: list[float] | tuple[float, float] | None = None,
    lower_vert_ps: list[float] | tuple[float, float] | None = None,
    *,
    body_mask: np.ndarray | None = None,
    ct_volume: np.ndarray | None = None,
    parasagittal_slices: list[np.ndarray] | None = None,
    exclude_mask: np.ndarray | None = None,
    min_bone_hu: float = 250.0,
    canal_depth_mm: float = 15.0,
    search_depth_mm: float = 6.0,
    max_gap_mm: float = 2.5,
    safety_margin_mm: float = 2.0,
) -> dict:
    """Measure retropulsion of posterior wall / bone fragments into the spinal canal.

    Constructs an Anterior Spinal Canal Line using adjacent vertebrae:
    - upper_vert_pi: Posterior-Inferior landmark of vertebra (i+1)
    - lower_vert_ps: Posterior-Superior landmark of vertebra (i-1)
    If adjacent vertebrae are not available, falls back to the current vertebra's
    posterior landmarks (PS and PI).

    Bone pixels (HU >= min_bone_hu) posterior to the canal line within the search window
    are detected to quantify retropulsion delta_posterior_mm and canal_encroachment_ratio.

    Parameters
    ----------
    ct_slice : np.ndarray
        2D CT sagittal image (HU).
    spacing_mm : tuple[float, float]
        Pixel spacing (sx, sy) in mm along (col, row).
    curr_vert_landmarks : list[list[float]] | dict
        6 landmarks of current vertebra in mm, or dict with landmark names.
    upper_vert_pi : list[float] | tuple[float, float] | None
        [x, y] mm of adjacent superior vertebra posterior-inferior corner.
    lower_vert_ps : list[float] | tuple[float, float] | None
        [x, y] mm of adjacent inferior vertebra posterior-superior corner.
    ct_volume : np.ndarray | None
        Optional 3D volume for multi-planar analysis.
    parasagittal_slices : list[np.ndarray] | None
        Optional list of parasagittal CT 2D slices (e.g. at +/- 4 mm).
    min_bone_hu : float
        HU threshold for cortical/cancellous bone fragments (default 250.0).
    canal_depth_mm : float
        Normal AP diameter of spinal canal in mm (default 15.0 mm).
    search_depth_mm : float
        Depth of canal search window posterior to the canal line (default 12.0 mm).
    safety_margin_mm : float
        Margin around vertebral height in mm.

    Returns
    -------
    dict
        delta_posterior_mm : float
            Maximum bone retropulsion distance into canal (mm).
        canal_encroachment_ratio : float
            Ratio of delta_posterior_mm to canal_depth_mm (0.0 to 1.0).
        canal_line_points : list[list[float]]
            [[x_lower, y_lower], [x_upper, y_upper]] defining reference line in mm.
        reference_mode : str
            'adjacent_vertebrae' or 'current_vertebra_fallback'.
        max_retropulsion_point_xy_mm : list[float] | None
            Coordinates [x, y] in mm of most encroaching bone pixel.
        retropulsed_pixel_count : int
            Number of bone pixels exceeding the canal line on mid-sagittal.
        retropulsed_bone_mask : np.ndarray
            Boolean 2D mask of encroaching bone pixels on ct_slice.
    """
    sx, sy = spacing_xy(spacing_mm)
    ct_arr = np.asarray(ct_slice, dtype=float)
    h, w = ct_arr.shape

    if isinstance(curr_vert_landmarks, dict):
        curr_ps = curr_vert_landmarks.get('posterior_superior')
        curr_pi = curr_vert_landmarks.get('posterior_inferior')
        curr_as = curr_vert_landmarks.get('anterior_superior')
        curr_ai = curr_vert_landmarks.get('anterior_inferior')
    elif isinstance(curr_vert_landmarks, (list, tuple)) and len(curr_vert_landmarks) >= 6:
        curr_as = curr_vert_landmarks[0]
        curr_ai = curr_vert_landmarks[1]
        curr_ps = curr_vert_landmarks[4]
        curr_pi = curr_vert_landmarks[5]
    else:
        raise ValueError('curr_vert_landmarks must be a 6-point list or a dict with standard landmark names.')

    if curr_ps is None or curr_pi is None:
        raise ValueError('Current vertebra must have valid posterior_superior and posterior_inferior landmarks.')

    if upper_vert_pi is not None and lower_vert_ps is not None:
        p_ref_lower = [float(lower_vert_ps[0]), float(lower_vert_ps[1])]
        p_ref_upper = [float(upper_vert_pi[0]), float(upper_vert_pi[1])]
        reference_mode = 'adjacent_vertebrae'
    else:
        p_ref_lower = [float(curr_pi[0]), float(curr_pi[1])]
        p_ref_upper = [float(curr_ps[0]), float(curr_ps[1])]
        reference_mode = 'current_vertebra_fallback'

    if p_ref_lower[1] > p_ref_upper[1]:
        p_ref_lower, p_ref_upper = p_ref_upper, p_ref_lower

    if body_mask is not None and body_mask.any():
        rows = np.flatnonzero(body_mask.any(axis=1))
        if len(rows) >= 4:
            quarter = max(1, len(rows) // 4)
            r_top = rows[:quarter]
            r_bot = rows[-quarter:]
            c_top = max(np.flatnonzero(body_mask[r, :])[-1] for r in r_top if body_mask[r, :].any())
            c_bot = max(np.flatnonzero(body_mask[r, :])[-1] for r in r_bot if body_mask[r, :].any())
            x_top_body = c_top * sx
            x_bot_body = c_bot * sx

            # Ensure canal line reference does not sit inside body mask cortical corners
            if p_ref_upper[0] < x_top_body:
                p_ref_upper[0] = x_top_body
            if p_ref_lower[0] < x_bot_body:
                p_ref_lower[0] = x_bot_body

    x1, y1 = p_ref_lower
    x2, y2 = p_ref_upper
    dx = x2 - x1
    dy = y2 - y1
    line_len = float(np.hypot(dx, dy))
    if line_len < 1e-4 or abs(dy) < 1e-4:
        dx, dy, line_len = 0.0, 1.0, 1.0

    y_min_mm = min(float(curr_pi[1]), float(curr_ai[1]) if curr_ai else float(curr_pi[1])) - safety_margin_mm
    y_max_mm = max(float(curr_ps[1]), float(curr_as[1]) if curr_as else float(curr_ps[1])) + safety_margin_mm

    r_start = max(0, int(np.floor(y_min_mm / sy)))
    r_end = min(h, int(np.ceil(y_max_mm / sy)) + 1)

    slices_to_scan = [ct_arr]
    if parasagittal_slices:
        for s in parasagittal_slices:
            if s is not None and s.shape == ct_arr.shape:
                slices_to_scan.append(np.asarray(s, dtype=float))

    # Calculate baseline cortex offset:
    # When landmarks are inset into the body (by BodyAutoParams.inset_mm = 2.5 mm),
    # the canal line lies inside the body. Normal uninjured cortical bone extends
    # continuously from the canal line outward across all vertebral rows.
    # To avoid false-positive retropulsion on intact cortical bone, we determine the
    # baseline cortical thickness across the vertebral rows.
    wall_depths = []
    for r in range(r_start, r_end):
        y_mm = r * sy
        x_line_mm = x1 + (dx / dy) * (y_mm - y1)
        c_line = int(round(x_line_mm / sx))
        depth = 0.0
        c = c_line
        while 0 <= c < w:
            hu_val = ct_arr[r, c]
            is_excluded = bool(exclude_mask[r, c]) if exclude_mask is not None and 0 <= r < exclude_mask.shape[0] and 0 <= c < exclude_mask.shape[1] else False
            if (hu_val >= min_bone_hu) and (not is_excluded):
                depth += sx
                c += 1
            else:
                break
        wall_depths.append(depth)

    baseline_offset_mm = 0.0
    if len(wall_depths) >= 3:
        p20 = float(np.percentile(wall_depths, 20))
        if p20 >= 1.0:
            baseline_offset_mm = min(p20, 3.5)

    max_delta = 0.0
    best_pt = None
    retropulsed_mask = np.zeros((h, w), dtype=bool)
    retropulsed_pixel_count = 0

    for slice_idx, s_arr in enumerate(slices_to_scan):
        for r in range(r_start, r_end):
            y_mm = r * sy
            x_line_mm = x1 + (dx / dy) * (y_mm - y1)
            c_line = x_line_mm / sx

            c_start = max(0, int(np.floor(c_line)))
            c_end = min(w, int(np.ceil((x_line_mm + search_depth_mm) / sx)) + 1)
            if c_end <= c_start:
                continue

            gap_pixels = 0
            for col_offset in range(c_end - c_start):
                c = c_start + col_offset
                if c >= w:
                    break

                hu_val = s_arr[r, c]
                is_excluded = bool(exclude_mask[r, c]) if exclude_mask is not None and 0 <= r < exclude_mask.shape[0] and 0 <= c < exclude_mask.shape[1] else False
                is_bone = (hu_val >= min_bone_hu) and (not is_excluded)

                if not is_bone:
                    gap_pixels += 1
                    if gap_pixels * sx >= max_gap_mm:
                        # Reached CSF / soft tissue gap: stop ray before encountering posterior lamina
                        break
                    continue

                gap_pixels = 0
                x_mm = c * sx
                # Normal vector pointing into canal (increasing x): (dy, -dx) / line_len
                d_perp = ((x_mm - x1) * dy - (y_mm - y1) * dx) / line_len
                eff_delta = max(0.0, d_perp - baseline_offset_mm)
                if eff_delta > 0.0:
                    if slice_idx == 0:
                        retropulsed_mask[r, c] = True
                        retropulsed_pixel_count += 1
                    if eff_delta > max_delta:
                        max_delta = float(eff_delta)
                        best_pt = [round(x_mm, 2), round(y_mm, 2)]

    encroachment_ratio = float(np.clip(max_delta / max(canal_depth_mm, 0.1), 0.0, 1.0))

    y_line_lower = min(y1, y_min_mm)
    y_line_upper = max(y2, y_max_mm)
    x_line_lower = x1 + (dx / dy) * (y_line_lower - y1)
    x_line_upper = x1 + (dx / dy) * (y_line_upper - y1)
    canal_line_points = [[round(x_line_lower, 2), round(y_line_lower, 2)], [round(x_line_upper, 2), round(y_line_upper, 2)]]

    return dict(
        delta_posterior_mm=round(max_delta, 3),
        canal_encroachment_ratio=round(encroachment_ratio, 4),
        canal_line_points=canal_line_points,
        reference_mode=reference_mode,
        max_retropulsion_point_xy_mm=best_pt,
        retropulsed_pixel_count=int(retropulsed_pixel_count),
        retropulsed_bone_mask=retropulsed_mask,
    )


def compute_intervertebral_metrics(vertebrae_data: list[dict],
                                   *,
                                   baseline_isp_mm: float | None = None) -> list[dict]:
    """Compute intervertebral morphometric metrics across a sequence of vertebrae (e.g. T10 to L5).

    Computes 4 essential clinical biomechanical metrics for AO Spine classification:
    a) isp_ratio: Interspinous Distance Ratio (B2 criterion)
    b) anterior_disc_angle_deg & disc_opening_ratio: Anterior disc gaping (B3 criterion)
    c) translation_mm: Vertebral translation / slip (Type C criterion)
    d) relative_collapse_ratio: Symmetric body collapse vs adjacent levels (A4 burst criterion)

    Parameters
    ----------
    vertebrae_data : list[dict]
        Sequence of vertebra dictionaries. Each dict should contain:
        - 'landmarks': 6 landmarks [[x, y], ...] in mm, or dict with landmark names
        - 'level': optional level name (e.g. 'T10', 'L1')
        - 'spinous_tip': optional [x, y] in mm (from find_spinous_process_tip)
        - 'heights': optional dict with anterior_mm, middle_mm, posterior_mm
        - 'centroid_xy': optional [x, y] in mm
    baseline_isp_mm : float | None
        Optional reference interspinous distance in mm. If None, estimated automatically
        from adjacent pairs (using minimum for <=2 pairs, median for >=3 pairs to resist outliers).

    Returns
    -------
    list[dict]
        Updated list of vertebra dictionaries with computed intervertebral metrics.
    """
    if not vertebrae_data:
        return []

    # Standardize individual vertebra geometry
    parsed: list[dict] = []
    for item in vertebrae_data:
        lm = item.get('landmarks')
        if isinstance(lm, dict):
            as_pt = lm['anterior_superior']
            ai_pt = lm['anterior_inferior']
            ms_pt = lm['middle_superior']
            mi_pt = lm['middle_inferior']
            ps_pt = lm['posterior_superior']
            pi_pt = lm['posterior_inferior']
        elif isinstance(lm, (list, tuple)) and len(lm) >= 6:
            as_pt, ai_pt, ms_pt, mi_pt, ps_pt, pi_pt = lm[:6]
        else:
            raise ValueError('Each vertebra must provide 6 valid landmarks.')

        h_info = item.get('heights')
        if not h_info or not all(k in h_info for k in ('anterior_mm', 'middle_mm', 'posterior_mm')):
            ha = float(np.linalg.norm(np.array(as_pt) - np.array(ai_pt)))
            hm = float(np.linalg.norm(np.array(ms_pt) - np.array(mi_pt)))
            hp = float(np.linalg.norm(np.array(ps_pt) - np.array(pi_pt)))
            h_info = dict(
                anterior_mm=ha,
                middle_mm=hm,
                posterior_mm=hp,
                anterior_posterior_ratio=float(ha / max(hp, 1e-4)),
                middle_posterior_ratio=float(hm / max(hp, 1e-4)),
            )
        else:
            h_info = dict(h_info)
            if 'anterior_posterior_ratio' not in h_info:
                h_info['anterior_posterior_ratio'] = float(h_info['anterior_mm'] / max(h_info['posterior_mm'], 1e-4))
            if 'middle_posterior_ratio' not in h_info:
                h_info['middle_posterior_ratio'] = float(h_info['middle_mm'] / max(h_info['posterior_mm'], 1e-4))

        h_total = float(h_info['anterior_mm'] + h_info['middle_mm'] + h_info['posterior_mm'])

        c_xy = item.get('centroid_xy')
        if not c_xy or len(c_xy) < 2:
            c_xy = np.mean([as_pt, ai_pt, ms_pt, mi_pt, ps_pt, pi_pt], axis=0).tolist()

        tip = item.get('spinous_tip')
        if isinstance(tip, dict):
            tip_xy = tip.get('tip_xy_mm')
        elif isinstance(tip, (list, tuple)) and len(tip) >= 2:
            tip_xy = [float(tip[0]), float(tip[1])]
        else:
            tip_xy = None

        plane = item.get('plane')
        sp_lps = item.get('spinous_tip_lps')
        if sp_lps is None and plane is not None and tip_xy is not None and hasattr(plane, 'point_lps'):
            sp_lps = plane.point_lps(tip_xy)

        lm_lps = item.get('landmarks_lps')
        if lm_lps is None and plane is not None and hasattr(plane, 'point_lps'):
            lm_lps = [plane.point_lps(pt) for pt in (as_pt, ai_pt, ms_pt, mi_pt, ps_pt, pi_pt)]

        up_d = item.get('upper_endplate_deformity_mm')
        lo_d = item.get('lower_endplate_deformity_mm')
        if up_d is None or lo_d is None:
            up_d, lo_d = compute_endplate_deformities(lm)

        is_trunc = bool(item.get('is_fov_truncated', False)) or is_vertebra_fov_truncated(item)

        parsed.append(dict(
            raw=item,
            as_pt=[float(as_pt[0]), float(as_pt[1])],
            ai_pt=[float(ai_pt[0]), float(ai_pt[1])],
            ms_pt=[float(ms_pt[0]), float(ms_pt[1])],
            mi_pt=[float(mi_pt[0]), float(mi_pt[1])],
            ps_pt=[float(ps_pt[0]), float(ps_pt[1])],
            pi_pt=[float(pi_pt[0]), float(pi_pt[1])],
            heights=h_info,
            h_total=h_total,
            centroid_xy=[float(c_xy[0]), float(c_xy[1])],
            spinous_tip=tip_xy,
            spinous_tip_lps=sp_lps,
            landmarks_lps=lm_lps,
            upper_endplate_deformity_mm=round(float(up_d), 2),
            lower_endplate_deformity_mm=round(float(lo_d), 2),
            is_fov_truncated=is_trunc,
        ))

    n = len(parsed)

    # Determine list ordering along cranial-caudal axis (superior first vs caudal first)
    lvl_0 = parsed[0]['raw'].get('level')
    lvl_last = parsed[-1]['raw'].get('level')
    if lvl_0 in LEVELS and lvl_last in LEVELS and lvl_0 != lvl_last:
        if LEVELS.index(lvl_0) < LEVELS.index(lvl_last):
            sup_offset, inf_offset = -1, 1
        else:
            sup_offset, inf_offset = 1, -1
    elif n >= 2 and parsed[0]['centroid_xy'][1] < parsed[-1]['centroid_xy'][1]:
        # List is caudal-to-cranial: neighbor above has index + 1
        sup_offset, inf_offset = 1, -1
    else:
        # List is cranial-to-caudal: neighbor above has index - 1
        sup_offset, inf_offset = -1, 1

    def _get_neighbor(idx: int, offset: int) -> dict | None:
        target = idx + offset
        if not (0 <= target < n):
            return None
        lvl_curr = parsed[idx]['raw'].get('level')
        lvl_target = parsed[target]['raw'].get('level')
        if lvl_curr in LEVELS and lvl_target in LEVELS:
            if abs(LEVELS.index(lvl_curr) - LEVELS.index(lvl_target)) != 1:
                return None
        if parsed[target].get('is_fov_truncated'):
            return None
        return parsed[target]

    def _get_tip_pt(item_parsed: dict) -> np.ndarray | None:
        if item_parsed.get('spinous_tip_lps') is not None:
            return np.asarray(item_parsed['spinous_tip_lps'], dtype=float)
        if item_parsed.get('spinous_tip') is not None:
            return np.asarray(item_parsed['spinous_tip'], dtype=float)
        return None

    def _spinous_ap_length(item_parsed: dict) -> float | None:
        """Calculate AP length of spinous process from posterior body wall to tip (mm)."""
        tip_lps = item_parsed.get('spinous_tip_lps')
        lm_lps = item_parsed.get('landmarks_lps')
        if tip_lps is not None and lm_lps is not None and len(lm_lps) >= 6:
            # In LPS: Y is Anterior (-) to Posterior (+)
            ps_lps = np.asarray(lm_lps[4], dtype=float)
            tip_lps = np.asarray(tip_lps, dtype=float)
            return float(abs(tip_lps[1] - ps_lps[1]))

        tip_2d = item_parsed.get('spinous_tip')
        ps_2d = item_parsed.get('ps_pt')
        if tip_2d is not None and ps_2d is not None:
            # 2D local slice: x coordinate (col) is anterior to posterior
            return float(abs(tip_2d[0] - ps_2d[0]))
        return None

    def _is_spinous_pair_reliable(item1: dict, item2: dict) -> bool:
        """Verify spinous tip pair consistency between adjacent slices.

        If difference in spinous process depth from posterior wall exceeds 10.0 mm,
        one slice missed the true spinous tip (cut through lamina/articular process).
        Such pairs cannot reliably assess interspinous widening.
        """
        len1 = _spinous_ap_length(item1)
        len2 = _spinous_ap_length(item2)
        if len1 is not None and len2 is not None:
            if abs(len1 - len2) > 10.0:
                return False
        return True

    def _get_body_center(item_parsed: dict) -> np.ndarray:
        if item_parsed.get('landmarks_lps'):
            return np.mean(item_parsed['landmarks_lps'], axis=0)
        return np.array(item_parsed['centroid_xy'] + [0.0], dtype=float)

    # Step 1: Compute pairwise adjacent interspinous distances to establish baseline
    pair_gaps_t: list[float] = []
    pair_gaps_l: list[float] = []
    for i in range(n):
        if parsed[i].get('is_fov_truncated') or parsed[i]['raw'].get('level') == 'S1':
            continue
        inf_neigh = _get_neighbor(i, inf_offset)
        if inf_neigh and not inf_neigh.get('is_fov_truncated') and inf_neigh['raw'].get('level') != 'S1':
            if not _is_spinous_pair_reliable(parsed[i], inf_neigh):
                continue
            p_curr = _get_tip_pt(parsed[i])
            p_inf = _get_tip_pt(inf_neigh)
            q_curr = parsed[i]['raw'].get('q_seg', 1.0)
            q_inf = inf_neigh['raw'].get('q_seg', 1.0)
            if p_curr is not None and p_inf is not None and q_curr >= 0.50 and q_inf >= 0.50:
                d_sp = float(np.linalg.norm(p_curr - p_inf))
                d_b = float(np.linalg.norm(_get_body_center(parsed[i]) - _get_body_center(inf_neigh)))
                if d_sp > 0 and d_b > 0:
                    norm_g = d_sp / max(d_b, 15.0)
                    lvl_c = parsed[i]['raw'].get('level', '')
                    if str(lvl_c).startswith('L'):
                        pair_gaps_l.append(norm_g)
                    else:
                        pair_gaps_t.append(norm_g)

    def _get_ref_isp(lvl_str: str) -> float:
        if baseline_isp_mm is not None and baseline_isp_mm > 0:
            return float(baseline_isp_mm)
        gaps = pair_gaps_l if str(lvl_str).startswith('L') else pair_gaps_t
        if not gaps:
            gaps = pair_gaps_l + pair_gaps_t
        if len(gaps) >= 3:
            return float(np.median(gaps))
        elif len(gaps) > 0:
            return float(np.min(gaps))
        return 1.0

    # Step 2: Helper for disc metrics between superior and inferior vertebra
    def _calc_disc(v_sup: dict, v_inf: dict) -> tuple[float, float]:
        lvl_u = v_sup['raw'].get('level')
        lvl_l = v_inf['raw'].get('level')
        if lvl_u == 'S1' or lvl_l == 'S1':
            return 0.0, 1.0

        if v_sup.get('landmarks_lps') and v_inf.get('landmarks_lps'):
            lm_sup = v_sup['landmarks_lps']
            lm_inf = v_inf['landmarks_lps']
            as_sup, ai_sup = np.array(lm_sup[0], dtype=float), np.array(lm_sup[1], dtype=float)
            ps_sup, pi_sup = np.array(lm_sup[4], dtype=float), np.array(lm_sup[5], dtype=float)
            as_inf, ai_inf = np.array(lm_inf[0], dtype=float), np.array(lm_inf[1], dtype=float)
            ps_inf, pi_inf = np.array(lm_inf[4], dtype=float), np.array(lm_inf[5], dtype=float)
        else:
            as_sup, ai_sup = np.array(v_sup['as_pt'], dtype=float), np.array(v_sup['ai_pt'], dtype=float)
            ps_sup, pi_sup = np.array(v_sup['ps_pt'], dtype=float), np.array(v_sup['pi_pt'], dtype=float)
            as_inf, ai_inf = np.array(v_inf['as_pt'], dtype=float), np.array(v_inf['ai_pt'], dtype=float)
            ps_inf, pi_inf = np.array(v_inf['ps_pt'], dtype=float), np.array(v_inf['pi_pt'], dtype=float)

        v_u = pi_sup - ai_sup
        v_l = ps_inf - as_inf
        len_u, len_l = np.linalg.norm(v_u), np.linalg.norm(v_l)
        if len_u > 1e-4 and len_l > 1e-4:
            cos_th = float(np.clip(np.dot(v_u, v_l) / (len_u * len_l), -1.0, 1.0))
            angle_deg = float(np.rad2deg(np.arccos(cos_th)))
        else:
            angle_deg = 0.0

        d_ant = float(np.linalg.norm(ai_sup - as_inf))
        d_post = float(np.linalg.norm(pi_sup - ps_inf))
        ratio = float(d_ant / max(d_post, 0.5))

        # Only count anterior disc opening angle when anterior height exceeds posterior height
        opening_angle = angle_deg if d_ant > d_post else 0.0
        return opening_angle, ratio

    # Step 2b: Helper for intervertebral translation (step-off) perpendicular to vertebral wall tangents & spinal axis
    def _calc_step_off(v_sup: dict, v_inf: dict) -> float:
        lvl_u = v_sup['raw'].get('level')
        lvl_l = v_inf['raw'].get('level')
        if lvl_u == 'S1' or lvl_l == 'S1':
            return 0.0

        if v_sup.get('landmarks_lps') and v_inf.get('landmarks_lps'):
            lm_sup = v_sup['landmarks_lps']
            lm_inf = v_inf['landmarks_lps']
            as_sup, ai_sup = np.array(lm_sup[0], dtype=float), np.array(lm_sup[1], dtype=float)
            ps_sup, pi_sup = np.array(lm_sup[4], dtype=float), np.array(lm_sup[5], dtype=float)
            as_inf, ai_inf = np.array(lm_inf[0], dtype=float), np.array(lm_inf[1], dtype=float)
            ps_inf, pi_inf = np.array(lm_inf[4], dtype=float), np.array(lm_inf[5], dtype=float)
            c_sup = np.mean([as_sup, ai_sup, ps_sup, pi_sup], axis=0)
            c_inf = np.mean([as_inf, ai_inf, ps_inf, pi_inf], axis=0)
            u_spine = c_sup - c_inf
            len_s = float(np.linalg.norm(u_spine))
            d_post = pi_sup - ps_inf
            d_ant = ai_sup - as_inf
            if len_s > 1e-4:
                e_tan = u_spine / len_s
                d_post_par = np.dot(d_post, e_tan) * e_tan
                d_ant_par = np.dot(d_ant, e_tan) * e_tan
                step_post = float(np.linalg.norm(d_post - d_post_par))
                step_ant = float(np.linalg.norm(d_ant - d_ant_par))
            else:
                step_post = float(np.linalg.norm(d_post))
                step_ant = float(np.linalg.norm(d_ant))
            return float(min(step_post, step_ant))
        else:
            # Posterior wall step-off across disc space
            ps_inf = np.array(v_inf['ps_pt'], dtype=float)
            pi_inf = np.array(v_inf['pi_pt'], dtype=float)
            ps_sup = np.array(v_sup['ps_pt'], dtype=float)
            pi_sup = np.array(v_sup['pi_pt'], dtype=float)
            d_post = pi_sup - ps_inf

            # Anterior wall step-off across disc space
            as_inf = np.array(v_inf['as_pt'], dtype=float)
            ai_inf = np.array(v_inf['ai_pt'], dtype=float)
            as_sup = np.array(v_sup['as_pt'], dtype=float)
            ai_sup = np.array(v_sup['ai_pt'], dtype=float)
            d_ant = ai_sup - as_inf

            # Local spinal axis across the two adjacent vertebrae
            u_spine = np.array(v_sup['centroid_xy'], dtype=float) - np.array(v_inf['centroid_xy'], dtype=float)
            len_s = float(np.linalg.norm(u_spine))
            if len_s > 1e-4:
                e_tan = u_spine / len_s
                e_norm = np.array([-e_tan[1], e_tan[0]])
                step_post = abs(float(np.dot(d_post, e_norm)))
                step_ant = abs(float(np.dot(d_ant, e_norm)))
            else:
                step_post = abs(float(d_post[0]))
                step_ant = abs(float(d_ant[0]))

            # True spinal dislocation involves displacement across both walls perpendicular to spinal axis
            return float(min(step_post, step_ant))

    # Step 3: Compute all 4 metrics for each vertebra
    results: list[dict] = []
    for i, curr in enumerate(parsed):
        c_i = curr['centroid_xy']
        if curr.get('is_fov_truncated'):
            out = dict(curr['raw'])
            out.update({
                'is_fov_truncated': True,
                'is_indeterminate': True,
                'predicted_class': 'Ngoài vùng khảo sát (Chạm biên CT)',
                'lesion_severity_score': 0.0,
                'delta_posterior_mm': 0.0,
                'isp_ratio': 1.0,
                'anterior_disc_angle_deg': 0.0,
                'disc_opening_ratio': 1.0,
                'translation_mm': 0.0,
                'relative_collapse_ratio': 1.0,
                'upper_endplate_deformity_mm': 0.0,
                'lower_endplate_deformity_mm': 0.0,
                'heights': curr['heights'],
                'centroid_xy': [round(c_i[0], 2), round(c_i[1], 2)],
            })
            results.append(out)
            continue

        if curr['raw'].get('level') == 'S1':
            out = dict(curr['raw'])
            out.update({
                'is_indeterminate': False,
                'predicted_class': 'S1 (Tham chiếu)',
                'lesion_severity_score': 0.0,
                'delta_posterior_mm': 0.0,
                'isp_ratio': 1.0,
                'anterior_disc_angle_deg': 0.0,
                'disc_opening_ratio': 1.0,
                'translation_mm': 0.0,
                'relative_collapse_ratio': 1.0,
                'upper_endplate_deformity_mm': 0.0,
                'lower_endplate_deformity_mm': 0.0,
                'heights': curr['heights'],
                'centroid_xy': [round(c_i[0], 2), round(c_i[1], 2)],
            })
            results.append(out)
            continue

        v_sup = _get_neighbor(i, sup_offset)
        v_inf = _get_neighbor(i, inf_offset)

        # a) Interspinous distance ratio (isp_ratio)
        p_curr = _get_tip_pt(curr)
        q_curr = curr['raw'].get('q_seg', 1.0)
        norm_adjacent: list[float] = []
        if p_curr is not None and q_curr >= 0.50 and not curr.get('is_fov_truncated') and curr['raw'].get('level') != 'S1':
            c_curr = _get_body_center(curr)
            if v_sup and v_sup['raw'].get('q_seg', 1.0) >= 0.50 and v_sup['raw'].get('level') != 'S1':
                if _is_spinous_pair_reliable(curr, v_sup):
                    p_sup = _get_tip_pt(v_sup)
                    if p_sup is not None:
                        d_sp = float(np.linalg.norm(p_curr - p_sup))
                        d_b = float(np.linalg.norm(c_curr - _get_body_center(v_sup)))
                        norm_adjacent.append(d_sp / max(d_b, 15.0))
            if v_inf and v_inf['raw'].get('q_seg', 1.0) >= 0.50 and v_inf['raw'].get('level') != 'S1':
                if _is_spinous_pair_reliable(curr, v_inf):
                    p_inf = _get_tip_pt(v_inf)
                    if p_inf is not None:
                        d_sp = float(np.linalg.norm(p_curr - p_inf))
                        d_b = float(np.linalg.norm(c_curr - _get_body_center(v_inf)))
                        norm_adjacent.append(d_sp / max(d_b, 15.0))

        ref_isp = _get_ref_isp(curr['raw'].get('level', ''))
        if norm_adjacent and ref_isp > 1e-3:
            isp_ratio = float(max(norm_adjacent) / ref_isp)
        else:
            isp_ratio = 1.0

        # b) Anterior disc angle & disc opening ratio (B3)
        disc_angles: list[float] = []
        disc_ratios: list[float] = []
        if v_sup:
            ang, r = _calc_disc(v_sup, curr)
            disc_angles.append(ang)
            disc_ratios.append(r)
        if v_inf:
            ang, r = _calc_disc(curr, v_inf)
            disc_angles.append(ang)
            disc_ratios.append(r)

        ant_disc_angle = max(disc_angles) if disc_angles else 0.0
        disc_open_ratio = max(disc_ratios) if disc_ratios else 1.0

        # c) Translation mm (Type C: step-off perpendicular to vertebral wall tangents & spinal trajectory)
        trans_candidates: list[float] = []
        if v_sup:
            trans_candidates.append(_calc_step_off(v_sup, curr))
        if v_inf:
            trans_candidates.append(_calc_step_off(curr, v_inf))

        step_raw = max(trans_candidates) if trans_candidates else 0.0

        # Curvature compensation: if enclosed by both adjacent neighbors in a sequence (excluding sacrum S1)
        if v_sup and v_inf and v_sup['raw'].get('level') != 'S1' and v_inf['raw'].get('level') != 'S1':
            if curr.get('landmarks_lps') and v_sup.get('landmarks_lps') and v_inf.get('landmarks_lps'):
                c_sup = np.mean(v_sup['landmarks_lps'], axis=0)
                c_inf = np.mean(v_inf['landmarks_lps'], axis=0)
                c_curr = np.mean(curr['landmarks_lps'], axis=0)

                a_sup = 0.5 * (np.array(v_sup['landmarks_lps'][0], dtype=float) + np.array(v_sup['landmarks_lps'][1], dtype=float))
                a_inf = 0.5 * (np.array(v_inf['landmarks_lps'][0], dtype=float) + np.array(v_inf['landmarks_lps'][1], dtype=float))
                a_curr = 0.5 * (np.array(curr['landmarks_lps'][0], dtype=float) + np.array(curr['landmarks_lps'][1], dtype=float))
            else:
                c_sup = np.array(v_sup['centroid_xy'], dtype=float)
                c_inf = np.array(v_inf['centroid_xy'], dtype=float)
                c_curr = np.array(curr['centroid_xy'], dtype=float)

                a_sup = 0.5 * (np.array(v_sup['as_pt'], dtype=float) + np.array(v_sup['ai_pt'], dtype=float))
                a_inf = 0.5 * (np.array(v_inf['as_pt'], dtype=float) + np.array(v_inf['ai_pt'], dtype=float))
                a_curr = 0.5 * (np.array(curr['as_pt'], dtype=float) + np.array(curr['ai_pt'], dtype=float))

            u = c_sup - c_inf
            len_u_sq = float(np.dot(u, u))
            if len_u_sq > 1e-4:
                t = float(np.dot(c_curr - c_inf, u) / len_u_sq)
                c_exp = c_inf + t * u
                dev_c = float(np.linalg.norm(c_curr - c_exp))
            else:
                dev_c = 0.0

            u_a = a_sup - a_inf
            len_a_sq = float(np.dot(u_a, u_a))
            if len_a_sq > 1e-4:
                t_a = float(np.dot(a_curr - a_inf, u_a) / len_a_sq)
                a_exp = a_inf + t_a * u_a
                dev_a = float(np.linalg.norm(a_curr - a_exp))
            else:
                dev_a = 0.0

            dev_curve = min(dev_c, dev_a)
            # If vertebra is genuinely dislocated from the spinal trajectory (dev_curve >= 4.0),
            # use max to capture the dislocation; if smoothly following spinal curvature (< 3.0), clamp to dev_curve
            if dev_curve >= 4.0:
                translation_mm = max(step_raw, dev_curve)
            elif dev_curve < 3.0:
                translation_mm = min(step_raw, dev_curve)
            else:
                translation_mm = step_raw
        else:
            translation_mm = step_raw

        # d) Relative collapse ratio (A4)
        if v_sup and v_inf:
            adj_mean = 0.5 * (v_sup['h_total'] + v_inf['h_total'])
            rel_collapse = float(curr['h_total'] / max(adj_mean, 0.1))
        elif v_sup:
            rel_collapse = float(curr['h_total'] / max(v_sup['h_total'], 0.1))
        elif v_inf:
            rel_collapse = float(curr['h_total'] / max(v_inf['h_total'], 0.1))
        else:
            rel_collapse = 1.0

        out = dict(curr['raw'])
        out.update({
            'isp_ratio': round(isp_ratio, 3),
            'anterior_disc_angle_deg': round(ant_disc_angle, 2),
            'disc_opening_ratio': round(disc_open_ratio, 3),
            'translation_mm': round(translation_mm, 2),
            'relative_collapse_ratio': round(rel_collapse, 3),
            'upper_endplate_deformity_mm': curr.get('upper_endplate_deformity_mm', 0.0),
            'lower_endplate_deformity_mm': curr.get('lower_endplate_deformity_mm', 0.0),
            'heights': curr['heights'],
            'centroid_xy': [round(c_i[0], 2), round(c_i[1], 2)],
        })
        results.append(out)

    return results


@dataclass
class BodySession:
    input_fingerprint: str
    level: str
    source_kind: str
    plane: BodyPlane
    ct: np.ndarray
    valid: np.ndarray
    hint: np.ndarray
    mask: np.ndarray | None = None
    landmarks: list = field(default_factory=list)
    age_group: str = 'unspecified'
    candidates: dict | None = None
    sampling_config: dict = field(default_factory=lambda: dict(samples=100_000, seed=42))
    auto_provenance: dict | None = None
    _approved_signature: str | None = None
    result: dict | None = None
    posterior_arch_mask: np.ndarray | None = None
    spinous_tip: dict | None = None
    retropulsion: dict | None = None
    adjacent_sessions: dict | list | None = None
    aospine: dict | None = None
    doctor_confirmed_class: str | None = None

    def __post_init__(self):
        shape = tuple(self.plane.size_xy[::-1])
        if (self.level not in LEVELS or self.source_kind not in ('Baseline', 'Consensus')
                or self.age_group not in ('unspecified', 'adult', 'pediatric')
                or self.ct.shape != shape or self.valid.shape != shape or self.hint.shape != shape
                or not np.isfinite(self.ct).all()):
            raise ValueError('Invalid body measurement session.')
        if self.mask is None:
            self.mask = np.zeros(shape, bool)
        self.set_mask(self.mask, clear_landmarks=False)

    def _signature(self):
        details = dict(plane=self.plane.to_dict(), level=self.level, source_kind=self.source_kind,
                       fingerprint=self.input_fingerprint, landmarks=self.landmarks, age_group=self.age_group)
        return hashlib.sha256((mask_digest(self.mask)+json.dumps(details, sort_keys=True)).encode()).hexdigest()

    @property
    def reviewed(self):
        return self._approved_signature is not None and self._approved_signature == self._signature()

    def invalidate(self):
        self._approved_signature = None
        self.result = None

    def set_candidates(self, detection):
        # LoG is ancillary to area/height measurements on the unchanged body mask.
        self.candidates = detection
        if self.result is not None:
            self.result['blob_detection'] = detection

    def set_mask(self, mask, *, clear_landmarks=True):
        value = binary_mask(mask)
        if value.shape != self.ct.shape or (value & ~self.valid).any():
            raise ValueError('Body mask must stay within the sampled CT field of view.')
        self.mask = value.copy()
        if clear_landmarks:
            self.landmarks = []
        self.auto_provenance = None
        self.invalidate()

    def set_landmarks(self, points):
        points = np.asarray(points, float)
        if points.size == 0:
            self.landmarks = []
        elif points.ndim == 2 and points.shape[1] == 2 and len(points) <= 6 and np.isfinite(points).all():
            lower = -np.asarray(self.plane.spacing_xy_mm)/2
            upper = (np.asarray(self.plane.size_xy)-.5)*self.plane.spacing_xy_mm
            if (points < lower).any() or (points > upper).any():
                raise ValueError('Landmark lies outside the plane.')
            self.landmarks = points.tolist()
        else:
            raise ValueError('Expected up to six finite plane landmarks.')
        self.auto_provenance = None
        self.invalidate()
        if len(self.landmarks) == 6:
            self._ensure_retropulsion(force=True)
            self.recompute_aospine_classification()

    def _ensure_posterior_arch(self):
        if self.spinous_tip is not None and self.spinous_tip.get('tip_xy_mm') is not None:
            return
        if self.posterior_arch_mask is None and self.hint is not None and self.hint.any():
            cut_col = None
            if self.mask is not None and self.mask.any():
                cols = np.where(self.mask)[1]
                cut_col = int(cols.max())
            else:
                hint_cols = np.where(self.hint)[1]
                if hint_cols.size > 0:
                    cut_col = int(np.percentile(hint_cols, 40))
            if cut_col is not None or (self.mask is not None and self.mask.any()):
                self.posterior_arch_mask = extract_posterior_arch_mask(self.hint, cut_col, body_mask=self.mask)
                self.spinous_tip = find_spinous_process_tip(self.posterior_arch_mask, self.plane.spacing_xy_mm)

    def _ensure_retropulsion(self, *, force: bool = False, upper_vert_pi: tuple[float, float] | None = None, lower_vert_ps: tuple[float, float] | None = None):
        if self.retropulsion is not None and not force:
            return
        if len(self.landmarks) != 6:
            self.retropulsion = {
                'delta_posterior_mm': 0.0,
                'canal_encroachment_ratio': 0.0,
                'canal_line_points': [],
                'reference_mode': 'none',
                'max_retropulsion_point_xy_mm': None,
                'retropulsed_pixel_count': 0,
                'retropulsed_bone_mask': np.zeros(self.ct.shape, bool),
            }
            return
        if upper_vert_pi is None or lower_vert_ps is None:
            if self.adjacent_sessions:
                from spinal_ct.localization import LEVELS
                if self.level in LEVELS:
                    idx = LEVELS.index(self.level)
                    sup_lvl = LEVELS[idx - 1] if idx > 0 else None
                    inf_lvl = LEVELS[idx + 1] if idx < len(LEVELS) - 1 else None
                    if isinstance(self.adjacent_sessions, dict):
                        sup_s = self.adjacent_sessions.get(sup_lvl)
                        inf_s = self.adjacent_sessions.get(inf_lvl)
                    else:
                        sup_s = next((s for s in self.adjacent_sessions if getattr(s, 'level', None) == sup_lvl), None)
                        inf_s = next((s for s in self.adjacent_sessions if getattr(s, 'level', None) == inf_lvl), None)
                    if sup_s and getattr(sup_s, 'landmarks', None) and len(sup_s.landmarks) == 6:
                        pt = sup_s.landmarks[5]
                        if getattr(sup_s, 'plane', None) and getattr(self, 'plane', None):
                            upper_vert_pi = tuple(project_point_between_planes(pt, sup_s.plane, self.plane))
                        else:
                            upper_vert_pi = tuple(pt)
                    if inf_s and getattr(inf_s, 'landmarks', None) and len(inf_s.landmarks) == 6:
                        pt = inf_s.landmarks[4]
                        if getattr(inf_s, 'plane', None) and getattr(self, 'plane', None):
                            lower_vert_ps = tuple(project_point_between_planes(pt, inf_s.plane, self.plane))
                        else:
                            lower_vert_ps = tuple(pt)
        self.retropulsion = measure_posterior_wall_retropulsion(
            self.ct, self.plane.spacing_xy_mm, self.landmarks,
            upper_vert_pi=upper_vert_pi, lower_vert_ps=lower_vert_ps,
            exclude_mask=self.posterior_arch_mask,
            body_mask=self.mask,
        )
        if len(self.landmarks) == 6:
            h = measure_heights(self.landmarks, self.plane)
            ha_hp = float(h.get('anterior_posterior_ratio', 1.0))
            hm_hp = float(h.get('middle_posterior_ratio', 1.0))
            up_d, lo_d = compute_endplate_deformities(self.landmarks, body_mask=self.mask, spacing_mm=self.plane.spacing_xy_mm)
            if ha_hp >= 0.80 and hm_hp >= 0.80 and up_d < 2.0 and lo_d < 2.0:
                if self.retropulsion:
                    self.retropulsion['delta_posterior_mm'] = 0.0
                    self.retropulsion['canal_encroachment_ratio'] = 0.0
        if self.level == 'S1' or is_vertebra_fov_truncated(self):
            if self.retropulsion:
                self.retropulsion['delta_posterior_mm'] = 0.0
                self.retropulsion['canal_encroachment_ratio'] = 0.0

    def recompute_aospine_classification(self, spine_sessions: list | dict | None = None) -> dict:
        """Fast (< 10 ms) real-time recomputation of AO Spine classification."""
        self._ensure_posterior_arch()
        self._ensure_retropulsion()

        if len(self.landmarks) != 6:
            if getattr(self, 'doctor_confirmed_class', None):
                eval_res = {
                    'level': self.level,
                    'predicted_class': self.doctor_confirmed_class,
                    'confidence': 1.0,
                    'secondary_subtype': None,
                    'class_probabilities': {c: 1.0 if c == self.doctor_confirmed_class else 0.0
                                            for c in ('Normal', 'A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'C')},
                    'concept_probs': {k: 0.0 for k in ('p_trans', 'p_ant_tb', 'p_post_tb', 'p_pw', 'p_ep2', 'p_fracture', 'p_split')},
                    'raw_metrics': {
                        'delta_posterior_mm': 0.0,
                        'isp_ratio': 1.0,
                        'translation_mm': 0.0,
                        'relative_collapse_ratio': 1.0,
                        'anterior_disc_angle_deg': 0.0,
                        'ha_hp_ratio': 1.0,
                        'hm_hp_ratio': 1.0,
                    },
                    'modifier_m1': False,
                    'modifier_m2': False,
                    'modifiers': [],
                    'is_indeterminate': False,
                    'rejection_reasons': [],
                    'clinical_rationale_vi': f"{self.level} - Bác sĩ đã xác nhận phân loại {self.doctor_confirmed_class}.",
                    'rank': 1,
                    'total_ranked': 1,
                    'lesion_severity_score': 0.0,
                    'patient_summary': dict(
                        primary_lesion_level=self.level,
                        primary_aospine_class=self.doctor_confirmed_class,
                        confidence=1.0,
                        severity_score=0.0,
                        top2_levels=[self.level],
                        has_surgical_instability=False,
                        modifiers=[],
                        is_indeterminate=False,
                        clinical_rationale_vi=f"{self.level} - Bác sĩ đã xác nhận phân loại {self.doctor_confirmed_class}.",
                    ),
                }
            else:
                eval_res = {
                    'level': self.level,
                    'predicted_class': 'Chưa đặt đủ 6 mốc',
                    'confidence': 0.0,
                    'secondary_subtype': None,
                    'class_probabilities': {c: 0.0 for c in ('Normal', 'A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'C')},
                    'concept_probs': {k: 0.0 for k in ('p_trans', 'p_ant_tb', 'p_post_tb', 'p_pw', 'p_ep2', 'p_fracture', 'p_split')},
                    'raw_metrics': {
                        'delta_posterior_mm': 0.0,
                        'isp_ratio': 1.0,
                        'translation_mm': 0.0,
                        'relative_collapse_ratio': 1.0,
                        'anterior_disc_angle_deg': 0.0,
                        'ha_hp_ratio': 1.0,
                        'hm_hp_ratio': 1.0,
                    },
                    'modifier_m1': False,
                    'modifier_m2': False,
                    'modifiers': [],
                    'is_indeterminate': True,
                    'rejection_reasons': ['Chưa đặt đủ 6 mốc chiều cao'],
                    'clinical_rationale_vi': f"{self.level} - Chưa đặt đủ 6 mốc chiều cao để phân loại AO Spine.",
                    'rank': 1,
                    'total_ranked': 1,
                    'lesion_severity_score': 0.0,
                    'patient_summary': dict(
                        primary_lesion_level=self.level,
                        primary_aospine_class='Chưa đặt đủ 6 mốc',
                        confidence=0.0,
                        severity_score=0.0,
                        top2_levels=[self.level],
                        has_surgical_instability=False,
                        modifiers=[],
                        is_indeterminate=True,
                        clinical_rationale_vi=f"{self.level} - Chưa đặt đủ 6 mốc chiều cao để phân loại AO Spine.",
                    ),
                }
            self.aospine = eval_res
            return self.aospine

        target_sessions = spine_sessions if spine_sessions is not None else self.adjacent_sessions

        if target_sessions:
            if isinstance(target_sessions, dict):
                session_list = list(target_sessions.values())
            else:
                session_list = list(target_sessions)

            if self not in session_list and not any(getattr(s, 'level', None) == self.level for s in session_list):
                session_list.append(self)

            def _get_level_idx(s):
                lvl = getattr(s, 'level', None) or (s.get('level') if isinstance(s, dict) else '')
                return LEVELS.index(lvl) if lvl in LEVELS else 999

            session_list = sorted(session_list, key=_get_level_idx)

            v_metrics_list = []
            for s in session_list:
                if isinstance(s, BodySession):
                    if len(s.landmarks) != 6:
                        continue
                    s._ensure_posterior_arch()
                    s._ensure_retropulsion()
                    h = measure_heights(s.landmarks, s.plane)
                    pw_delta = s.retropulsion.get('delta_posterior_mm', 0.0) if s.retropulsion else 0.0
                    sp_pt = s.spinous_tip.get('tip_xy_mm') if s.spinous_tip else None
                    q_val = s.spinous_tip.get('q_seg', 1.0) if s.spinous_tip else 1.0
                    plane = getattr(s, 'plane', None)
                    sp_lps = plane.point_lps(sp_pt) if (plane is not None and sp_pt is not None and hasattr(plane, 'point_lps')) else None
                    lm_lps = [plane.point_lps(pt) for pt in s.landmarks] if (plane is not None and s.landmarks and hasattr(plane, 'point_lps')) else None
                    up_d, lo_d = compute_endplate_deformities(s.landmarks, body_mask=s.mask, spacing_mm=s.plane.spacing_xy_mm)
                    is_trunc = is_vertebra_fov_truncated(s)
                    v_metrics_list.append({
                        'level': s.level,
                        'landmarks': s.landmarks,
                        'landmarks_lps': lm_lps,
                        'spinous_tip': sp_pt,
                        'spinous_tip_lps': sp_lps,
                        'plane': plane,
                        'q_seg': q_val,
                        'delta_posterior_mm': pw_delta,
                        'heights': h,
                        'upper_endplate_deformity_mm': up_d,
                        'lower_endplate_deformity_mm': lo_d,
                        'is_fov_truncated': is_trunc,
                    })
                elif isinstance(s, dict) and len(s.get('landmarks', [])) == 6:
                    v_metrics_list.append(s)

            if v_metrics_list:
                inter_metrics = compute_intervertebral_metrics(v_metrics_list)
                ranked_result = fuse_and_classify_spine(inter_metrics)

                tot = len(ranked_result['ranked_vertebrae'])
                for r_idx, ranked_eval in enumerate(ranked_result['ranked_vertebrae'], 1):
                    lvl = ranked_eval['level']
                    for s in session_list:
                        if isinstance(s, BodySession) and s.level == lvl:
                            eval_copy = dict(ranked_eval)
                            eval_copy['rank'] = r_idx
                            eval_copy['total_ranked'] = tot
                            eval_copy['patient_summary'] = ranked_result['patient_summary']
                            s.aospine = eval_copy
                            if self.level == lvl:
                                self.aospine = eval_copy

                if self.aospine is None:
                    self.aospine = evaluate_aospine_decision_tree({'level': self.level})
                    self.aospine['rank'] = 1
                    self.aospine['total_ranked'] = tot
                    self.aospine['patient_summary'] = ranked_result['patient_summary']
            else:
                self.aospine = evaluate_aospine_decision_tree({'level': self.level})
                self.aospine['rank'] = 1
                self.aospine['total_ranked'] = 1
        else:
            h = measure_heights(self.landmarks, self.plane)
            pw_delta = self.retropulsion.get('delta_posterior_mm', 0.0) if self.retropulsion else 0.0
            sp_pt = self.spinous_tip.get('tip_xy_mm') if self.spinous_tip else None
            q_val = self.spinous_tip.get('q_seg', 1.0) if self.spinous_tip else 1.0

            up_d, lo_d = compute_endplate_deformities(self.landmarks, body_mask=self.mask, spacing_mm=self.plane.spacing_xy_mm)
            is_trunc = is_vertebra_fov_truncated(self)

            v_metrics = {
                'level': self.level,
                'landmarks': self.landmarks,
                'spinous_tip': sp_pt,
                'q_seg': q_val,
                'delta_posterior_mm': pw_delta,
                'heights': h,
                'upper_endplate_deformity_mm': up_d,
                'lower_endplate_deformity_mm': lo_d,
                'is_fov_truncated': is_trunc,
            }
            inter_metrics = compute_intervertebral_metrics([v_metrics])
            ranked_result = fuse_and_classify_spine(inter_metrics)
            eval_res = ranked_result['ranked_vertebrae'][0]
            eval_res['rank'] = 1
            eval_res['total_ranked'] = 1
            eval_res['patient_summary'] = ranked_result['patient_summary']
            self.aospine = eval_res

        return self.aospine

    def auto_populate(self, *, params: BodyAutoParams | None = None, approve: bool = False):
        params = params or BodyAutoParams()
        res = auto_segment_body(self.ct, self.valid, self.hint, self.plane.spacing_xy_mm,
                                params=params, return_details=True)
        mask = res['body_mask']
        cut_col = res.get('cut_col')
        landmarks = auto_detect_landmarks(mask, self.plane, params=params)
        self.set_mask(mask, clear_landmarks=True)
        self.set_landmarks(landmarks)
        self.auto_provenance = dict(method='auto_v1', params=params.to_dict())
        if 'posterior_arch_mask' in res and res['posterior_arch_mask'] is not None and res['posterior_arch_mask'].any():
            self.posterior_arch_mask = res['posterior_arch_mask']
            self.spinous_tip = find_spinous_process_tip(self.posterior_arch_mask, self.plane.spacing_xy_mm)
        elif self.hint is not None and self.hint.any():
            self.posterior_arch_mask = extract_posterior_arch_mask(self.hint, cut_col, body_mask=mask)
            self.spinous_tip = find_spinous_process_tip(self.posterior_arch_mask, self.plane.spacing_xy_mm)
        self._ensure_retropulsion(force=True)
        self.recompute_aospine_classification()
        if approve:
            self.approve()

    def approve(self):
        features = analyze_blob(self.mask, self.plane.spacing_xy_mm)
        if len(self.landmarks) not in (0, 6):
            raise ValueError('Finish all six height landmarks or clear them before review.')
        measure_heights(self.landmarks, self.plane)
        if self.landmarks:
            indices = np.floor(np.array(self.landmarks)/self.plane.spacing_xy_mm+.5).astype(int)
            indices = np.clip(indices, [0, 0], np.array(self.plane.size_xy)-1)
            if not self.valid[indices[:, 1], indices[:, 0]].all():
                raise ValueError('Height landmarks must lie within the CT field of view.')
        if features['pixel_count'] < 3:
            raise ValueError('Region is too small to confirm.')
        self._approved_signature = self._signature()
        self.result = None

    def measure(self, *, samples=100_000, seed=42):
        if not self.reviewed:
            raise ValueError('Confirm the current body region and plane before measuring.')
        configuration = validate_sampling(samples, seed)
        features = analyze_blob(self.mask, self.plane.spacing_xy_mm)
        warnings = list(features['warnings'])
        params = (BodyAutoParams(**self.auto_provenance['params'])
                  if self.auto_provenance and 'params' in self.auto_provenance
                  else BodyAutoParams())
        heights = measure_heights(self.landmarks, self.plane)
        if heights is None:
            warnings.append('height_landmarks_missing')
        else:
            if not (params.min_normal_ap_ratio <= heights['anterior_posterior_ratio'] <= params.max_normal_ap_ratio):
                warnings.append('abnormal_ap_height_ratio')
            if heights['middle_posterior_ratio'] < params.min_normal_ap_ratio:
                warnings.append('low_middle_posterior_ratio')
        if self.age_group == 'unspecified':
            warnings.append('age_group_unspecified')
        eroded_valid = self.valid.copy()
        for axis in (0, 1):
            for shift in (-1, 1):
                eroded_valid &= np.roll(self.valid, shift, axis=axis)
        if (self.mask & ~eroded_valid).any():
            warnings.append('touches_ct_field_boundary')
        normalization = normalize_region(self.ct, self.mask, self.plane.spacing_xy_mm, features)
        aospine = self.recompute_aospine_classification()
        self.result = dict(schema_version=1, task='vertebral_body_morphometry',
                           input_fingerprint=self.input_fingerprint, level=self.level, source_kind=self.source_kind,
                           review_status='confirmed_by_user', review_required=True,
                           status='needs_review' if warnings else 'measured',
                           region_signature=self._signature(), mask_sha256=mask_digest(self.mask),
                           age_group=self.age_group, plane=self.plane.to_dict(),
                           features=features, heights=heights,
                           landmarks_xy_mm=self.landmarks, landmark_names=list(LANDMARK_NAMES),
                           normalization=normalization['transform'], warnings=warnings,
                           auto_provenance=self.auto_provenance,
                           monte_carlo_area=estimate_area(self.mask, self.plane.spacing_xy_mm,
                                                         samples=samples, seed=seed, features=features),
                           blob_detection=self.candidates,
                           aospine=aospine,
                           interpretation='Shape measurements on a user-defined body region; no diagnostic or AO classification.')
        self.sampling_config = configuration
        return self.result


def save_session(session, output, *, samples=100_000, seed=42):
    """Publish a complete new folder. Drafts retain their unconfirmed status."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    if not session.mask.any():
        raise ValueError('Draw a region before saving.')
    configuration = validate_sampling(samples, seed)
    report = session.measure(samples=samples, seed=seed) if session.reviewed else None
    document = dict(schema_version=1, task='vertebral_body_region',
                    input_fingerprint=session.input_fingerprint, level=session.level,
                    source_kind=session.source_kind, plane=session.plane.to_dict(), age_group=session.age_group,
                    landmarks_xy_mm=session.landmarks, mask='body_mask.npy',
                    mask_sha256=mask_digest(session.mask),
                    auto_provenance=session.auto_provenance,
                    region_signature=session._signature(), reviewed=session.reviewed,
                    blob_detection_config=(session.candidates or {}).get('config'),
                    area_sampling=configuration,
                    doctor_confirmed_class=getattr(session, 'doctor_confirmed_class', None),
                    body_definition='User outline of the entire body cross-section, including its enclosed interior.',
                    created_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.body-', dir=output.parent) as temporary:
        root = Path(temporary)/'result'
        root.mkdir()
        np.save(root/'body_mask.npy', session.mask.astype(np.uint8), allow_pickle=False)
        (root/'region.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if report:
            (root/'measurements.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            normalized = normalize_region(session.ct, session.mask, session.plane.spacing_xy_mm, report['features'])
            np.save(root/'normalized_ct.npy', normalized['ct'], allow_pickle=False)
            np.save(root/'normalized_mask.npy', normalized['mask'].astype(np.uint8), allow_pickle=False)
            aospine = report.get('aospine') or {}
            raw_metrics = aospine.get('raw_metrics') or {}
            flat = dict(level=session.level, source_kind=session.source_kind, age_group=session.age_group,
                        area_mm2=report['features']['area_mm2'], fill_ratio=report['features']['fill_ratio'],
                        solidity=report['features']['solidity'], height_width_ratio=report['features']['height_width_ratio'],
                        **{key: (report['heights'] or {}).get(key) for key in (
                            'anterior_mm', 'middle_mm', 'posterior_mm',
                            'anterior_posterior_ratio', 'middle_posterior_ratio')},
                        aospine_class=aospine.get('predicted_class', ''),
                        aospine_confidence=aospine.get('confidence', ''),
                        aospine_secondary=aospine.get('secondary_subtype', ''),
                        aospine_rank=aospine.get('rank', 1),
                        delta_posterior_mm=raw_metrics.get('delta_posterior_mm', (session.retropulsion or {}).get('delta_posterior_mm', 0.0)),
                        isp_ratio=raw_metrics.get('isp_ratio', 1.0),
                        translation_mm=raw_metrics.get('translation_mm', 0.0),
                        relative_collapse_ratio=raw_metrics.get('relative_collapse_ratio', 1.0),
                        doctor_confirmed_class=getattr(session, 'doctor_confirmed_class', None) or '',
                        aospine_modifiers=';'.join(aospine.get('modifiers', [])),
                        aospine_rationale=aospine.get('clinical_rationale_vi', ''),
                        mc_area_mm2=report['monte_carlo_area']['area_mm2'],
                        mc_elbow_area_mm2=(report['monte_carlo_area'].get('elbow') or {}).get('area_mm2'),
                        mc_elbow_samples=(report['monte_carlo_area'].get('elbow') or {}).get('samples'),
                        mc_samples=samples, mc_seed=seed, warnings=';'.join(report['warnings']))
            with (root/'features.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
                writer.writerow(flat)
            from spinal_ct.morphometry_render import export_preview
            export_preview(session, report, root/'preview.png')
            rows = ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>' for k, v in flat.items())
            aospine_html = ''
            if aospine:
                cls_name = aospine.get('predicted_class', 'Normal')
                conf = float(aospine.get('confidence', 0.0))
                sec = f" (kèm {aospine['secondary_subtype']})" if aospine.get('secondary_subtype') else ''
                rank = aospine.get('rank', 1)
                rationale = html.escape(str(aospine.get('clinical_rationale_vi', '')))
                mods = '<br>'.join(html.escape(m) for m in aospine.get('modifiers', []))
                mod_row = f'<tr><th>Cảnh báo Modifiers</th><td>{mods}</td></tr>' if mods else ''
                aospine_html = (
                    f'<h2>Phân loại tổn thương AO Spine</h2><table>'
                    f'<tr><th>Mã phân loại</th><td><strong style="color:#d9534f;font-size:18px">{html.escape(cls_name)}{sec}</strong> (Độ tin cậy: {conf:.1%})</td></tr>'
                    f'<tr><th>Hạng nghi ngờ tổn thương</th><td>#{rank}</td></tr>'
                    f'<tr><th>Lý giải lâm sàng</th><td>{rationale}</td></tr>'
                    f'{mod_row}</table>'
                )
            (root/'report.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8">'
                '<title>Đo thân đốt sống</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}'
                'th,td{padding:8px;text-align:left;border-bottom:1px solid #ccc}table{width:100%;margin-bottom:20px}img{width:100%}</style>'
                '<h1>Đo hình dạng thân đốt sống & Phân loại AO Spine</h1><p>Vùng đo được người dùng xác nhận; kết quả cần kiểm tra. '
                'Khoảng Monte Carlo chỉ mô tả sai số lấy mẫu trên mask.</p>' + aospine_html +
                '<h2>Chi tiết hình thái học</h2><table>'+rows+
                '</table><img src="preview.png" alt="Vùng đo, chuẩn hóa và hội tụ diện tích"></html>', encoding='utf-8')
        root.rename(output)
    return output


def load_session(folder, source, *, fingerprint=None):
    folder = Path(folder)
    document = json.loads((folder/'region.json').read_text(encoding='utf-8'))
    fingerprint = fingerprint or ct_fingerprint(source)
    if (document.get('schema_version') != 1 or document.get('task') != 'vertebral_body_region'
            or document.get('input_fingerprint') != fingerprint or document.get('mask') != 'body_mask.npy'):
        raise ValueError('Region schema or CT fingerprint mismatch.')
    plane = BodyPlane(**document['plane'])
    ct, valid, hint = sample_plane(source, plane)
    mask = binary_mask(np.load(folder/'body_mask.npy', allow_pickle=False))
    if mask_digest(mask) != document.get('mask_sha256'):
        raise ValueError('Body mask has changed since this region was saved.')
    session = BodySession(fingerprint, document['level'], document['source_kind'], plane, ct, valid, hint,
                          mask=mask, age_group=document['age_group'])
    session.set_landmarks(document['landmarks_xy_mm'])
    session.auto_provenance = document.get('auto_provenance')
    if session._signature() != document.get('region_signature'):
        raise ValueError('Region geometry, landmarks or provenance changed since saving.')
    if document.get('reviewed') is True:
        session.approve()
    session.doctor_confirmed_class = document.get('doctor_confirmed_class')
    configuration = document.get('area_sampling')
    if configuration is None and (folder/'measurements.json').is_file():
        # Older schema-1 regions stored sampling settings only in the report.
        previous = json.loads((folder/'measurements.json').read_text(encoding='utf-8'))
        if previous.get('region_signature') == session._signature():
            mc = previous.get('monte_carlo_area', {})
            configuration = dict(samples=mc.get('samples'), seed=mc.get('seed'))
    if configuration is not None:
        session.sampling_config = validate_sampling(**configuration)
    if document.get('blob_detection_config'):
        from spinal_ct.blob_detection import detect_blobs
        session.set_candidates(detect_blobs(session.ct, session.plane.spacing_xy_mm, session.valid,
                                             **document['blob_detection_config']))
    # Measurements are recomputed from the current data, not trusted from a saved report.
    return session


@dataclass
class SpineSession:
    """Manages multi-vertebra spine sessions (T10-L5) with AO Spine ranking and triage."""
    patient_id: str
    sessions: dict[str, BodySession] = field(default_factory=dict)
    ranked_vertebrae: list[dict] = field(default_factory=list)
    patient_summary: dict = field(default_factory=dict)

    def add_session(self, session: BodySession) -> None:
        self.sessions[session.level] = session
        session.adjacent_sessions = self.sessions

    def recompute_aospine_classification(self) -> dict:
        """Fast (< 10 ms) real-time recomputation for all vertebrae in session."""
        if not self.sessions:
            return dict(ranked_vertebrae=[], patient_summary={})
        from spinal_ct.localization import LEVELS
        sorted_sessions = sorted(
            self.sessions.values(),
            key=lambda s: LEVELS.index(s.level) if s.level in LEVELS else 999
        )
        first_session = sorted_sessions[0]
        first_session.recompute_aospine_classification(sorted_sessions)

        self.ranked_vertebrae = []
        for s in sorted_sessions:
            if s.aospine:
                self.ranked_vertebrae.append(s.aospine)

        def _spine_rank_sort_key(ev: dict):
            cls = ev.get('predicted_class', '')
            is_excluded = (
                cls in ('Ngoài vùng khảo sát (Chạm biên CT)', 'S1 (Tham chiếu)', 'Chưa đặt đủ 6 mốc')
                or ev.get('is_fov_truncated', False)
                or ev.get('is_indeterminate', False)
            )
            lvl = ev.get('level', '')
            lvl_idx = LEVELS.index(lvl) if lvl in LEVELS else 999
            return (1 if is_excluded else 0, -ev.get('lesion_severity_score', 0.0), lvl_idx)

        self.ranked_vertebrae.sort(key=_spine_rank_sort_key)
        for rank_idx, v in enumerate(self.ranked_vertebrae, 1):
            v['rank'] = rank_idx
            if v['level'] in self.sessions and self.sessions[v['level']].aospine:
                self.sessions[v['level']].aospine['rank'] = rank_idx

        self.patient_summary = (first_session.aospine or {}).get('patient_summary', {})
        return dict(ranked_vertebrae=self.ranked_vertebrae, patient_summary=self.patient_summary)

    def save_session(self, output_dir: Path | str, *, samples=100_000, seed=42) -> Path:
        """Export individual reviewed sessions and whole-spine patient summary."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.recompute_aospine_classification()
        for lvl, s in self.sessions.items():
            if s.mask.any():
                save_session(s, out_path / lvl, samples=samples, seed=seed)
        summary_doc = dict(
            patient_id=self.patient_id,
            patient_summary=self.patient_summary,
            ranked_vertebrae=self.ranked_vertebrae,
            created_utc=datetime.now(timezone.utc).isoformat()
        )
        (out_path / 'spine_patient_summary.json').write_text(
            json.dumps(summary_doc, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        return out_path

