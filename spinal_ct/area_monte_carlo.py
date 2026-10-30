"""Uniform point sampling for mask area, separate from HU perturbation MC."""
import numpy as np
from spinal_ct.body_region import binary_mask, spacing_xy
from spinal_ct.blob_analysis import analyze_blob


def estimate_area(mask, spacing, *, samples=100_000, seed=42, features=None):
    mask = binary_mask(mask)
    sx, sy = spacing_xy(spacing)
    if (not isinstance(samples, (int, np.integer)) or not 100 <= samples <= 10_000_000
            or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**63-1):
        raise ValueError('MC requires 100–10,000,000 points and a nonnegative integer seed.')
    features = features or analyze_blob(mask, spacing)
    box = features['oriented_box']
    axes = np.asarray(box['axes_xy'])
    lower, upper = np.asarray(box['min_uv_mm']), np.asarray(box['max_uv_mm'])
    box_area = float(np.prod(upper-lower))
    rng = np.random.default_rng(seed)
    checkpoints = sorted(set([n for n in (100, 1000, 10_000, samples) if n <= samples]))
    hits, done, curve = 0, 0, []
    preview = []
    for target in checkpoints:
        while done < target:
            count = min(65_536, target-done)
            points = rng.uniform(lower, upper, (count, 2)) @ axes.T
            pixels = np.floor(points/[sx, sy]+.5).astype(int)
            inside = ((pixels >= 0).all(1) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] < mask.shape[0]))
            selected = np.flatnonzero(inside)
            inside[selected] = mask[pixels[selected, 1], pixels[selected, 0]]
            if done < 1000:
                take = min(count, 1000-done)
                preview.extend(np.column_stack((points[:take], inside[:take])).tolist())
            hits += int(inside.sum())
            done += count
        p = hits/target
        # Wilson binomial interval remains non-degenerate with zero/all hits.
        z = 1.959963984540054
        denominator = 1+z*z/target
        centre = (p+z*z/(2*target))/denominator
        half = z*np.sqrt(p*(1-p)/target+z*z/(4*target*target))/denominator
        area = p*box_area
        curve.append(dict(samples=target, hits=hits, fill_ratio=p, area_mm2=area,
                          standard_error_mm2=float(box_area*np.sqrt(p*(1-p)/target)),
                          sampling_interval95_mm2=[max(0., centre-half)*box_area, min(1., centre+half)*box_area],
                          error_vs_pixel_area_mm2=area-features['area_mm2']))
    return dict(method='uniform_point_area', seed=int(seed), box_area_mm2=box_area,
                reference_pixel_area_mm2=features['area_mm2'], **curve[-1], convergence=curve,
                preview_points_xy_mm_hit=preview,
                interpretation='Sampling error conditional on this pixel mask; excludes segmentation and clinical error.')
