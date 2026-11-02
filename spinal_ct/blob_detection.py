"""Experimental scale-normalized bright-blob LoG, with sigma in millimetres."""
import numpy as np
import SimpleITK as sitk
from spinal_ct.body_region import spacing_xy


def detect_blobs(ct, spacing, roi=None, *, min_sigma_mm=2., max_sigma_mm=12., scales=8,
                 threshold=0.03, max_candidates=30, window_hu=(-200., 1500.)):
    sx, sy = spacing_xy(spacing)
    ct = np.asarray(ct)
    if ct.ndim != 2 or min(ct.shape) < 4 or not np.isfinite(ct).all():
        raise ValueError('LoG needs a finite CT plane at least 4 × 4 pixels.')
    if (not np.isfinite([min_sigma_mm, max_sigma_mm, threshold, *window_hu]).all()
            or min_sigma_mm < max(sx, sy) or max_sigma_mm < min_sigma_mm
            or not isinstance(scales, int) or not 2 <= scales <= 32
            or not 0 < threshold <= 1 or not 1 <= max_candidates <= 100
            or window_hu[1] <= window_hu[0]):
        raise ValueError('Invalid LoG configuration (minimum sigma must be at least one pixel per axis).')
    allowed = np.ones(ct.shape, bool) if roi is None else np.asarray(roi, bool)
    if allowed.shape != ct.shape:
        raise ValueError('LoG ROI shape differs from CT.')
    values = np.clip((ct.astype(np.float32)-window_hu[0])/(window_hu[1]-window_hu[0]), 0, 1)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing([float(sx), float(sy)])
    sigmas = np.geomspace(min_sigma_mm, max_sigma_mm, scales)
    responses = np.array([-sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(
        image, float(sigma), True)) for sigma in sigmas])
    # Local maxima across 3x3 pixels and the adjacent scales. Suppress plateaus later.
    padded = np.pad(responses, ((1, 1), (1, 1), (1, 1)), constant_values=-np.inf)
    peak = (responses >= threshold) & allowed[None]
    for ds in range(3):
        for dr in range(3):
            for dc in range(3):
                neighbour = padded[ds:ds+scales, dr:dr+ct.shape[0], dc:dc+ct.shape[1]]
                peak &= responses >= neighbour
    peak[:, [0, -1], :] = False
    peak[:, :, [0, -1]] = False
    indices = np.argwhere(peak)
    if len(indices):
        scores = responses[tuple(indices.T)]
        indices = indices[np.argsort(-scores, kind='stable')]
    candidates = []
    for k, row, col in indices:
        point = np.array([col*sx, row*sy])
        sigma = float(sigmas[k])
        if any(np.linalg.norm(point-np.array(p['centre_xy_mm'])) < min(sigma, p['sigma_mm'])
               for p in candidates):
            continue
        candidates.append(dict(centre_xy_mm=point.tolist(), sigma_mm=sigma,
                               display_radius_mm=float(np.sqrt(2)*sigma), response=float(responses[k, row, col])))
        if len(candidates) >= max_candidates:
            break
    return dict(method='scale_normalized_bright_LoG', candidates=candidates,
                config=dict(min_sigma_mm=min_sigma_mm, max_sigma_mm=max_sigma_mm, scales=scales,
                            threshold=threshold, max_candidates=max_candidates, window_hu=list(window_hu)),
                interpretation='Candidate centres/scales only; displayed circles are not body contours.')


def assess_candidates(detection, reference_mask, spacing, *, tolerance_mm):
    """One body per ROI; centre reference is the approved mask centroid."""
    from spinal_ct.blob_analysis import analyze_blob
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('Choose a positive centre matching tolerance in mm before evaluation.')
    centre = np.asarray(analyze_blob(reference_mask, spacing)['centroid_xy_mm'])
    distances = [float(np.linalg.norm(np.asarray(p['centre_xy_mm'])-centre)) for p in detection['candidates']]
    matched = any(d <= tolerance_mm for d in distances)
    return dict(reference_centre_xy_mm=centre.tolist(), tolerance_mm=tolerance_mm,
                candidate_count=len(distances), matched=matched,
                nearest_distance_mm=min(distances) if distances else None,
                first_candidate_distance_mm=distances[0] if distances else None,
                extra_candidates=len(distances)-int(matched),
                candidate_distances_mm=distances)
