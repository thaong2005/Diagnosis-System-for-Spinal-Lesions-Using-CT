"""Uniform point sampling for mask area, separate from HU perturbation MC."""
import numpy as np
from spinal_ct.body_region import binary_mask, spacing_xy
from spinal_ct.blob_analysis import analyze_blob


BASE_CHECKPOINTS = (
    100, 150, 200, 300, 400, 500, 700, 1000,
    1500, 2000, 3000, 4000, 5000, 7000, 10000,
    15000, 20000, 30000, 50000, 70000, 100000,
    200000, 500000, 1000000, 2000000, 5000000, 10000000
)


def validate_sampling(samples, seed):
    if (not isinstance(samples, (int, np.integer)) or not 100 <= samples <= 10_000_000
            or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**63-1):
        raise ValueError('MC requires 100–10,000,000 points and a nonnegative integer seed.')
    return dict(samples=int(samples), seed=int(seed))


def find_elbow_point(curve):
    """Find the elbow / knee point where Monte Carlo sampling enters the stable convergence regime."""
    if len(curve) <= 2:
        return curve[-1]
    samples_arr = np.array([c['samples'] for c in curve])
    se_arr = np.array([c['standard_error_mm2'] for c in curve])
    x = np.log10(samples_arr)
    xn = (x - x[0]) / (x[-1] - x[0])
    yn_se = (se_arr - se_arr[-1]) / (se_arr[0] - se_arr[-1] + 1e-12)
    dist_se = 1.0 - xn - yn_se
    elbow_idx = int(np.argmax(dist_se))
    return curve[elbow_idx]


def estimate_area(mask, spacing, *, samples=100_000, seed=42, features=None):
    mask = binary_mask(mask)
    sx, sy = spacing_xy(spacing)
    validate_sampling(samples, seed)
    features = features or analyze_blob(mask, spacing)
    box = features['oriented_box']
    axes = np.asarray(box['axes_xy'])
    lower, upper = np.asarray(box['min_uv_mm']), np.asarray(box['max_uv_mm'])
    box_area = float(np.prod(upper-lower))
    rng = np.random.default_rng(seed)
    checkpoints = sorted(set([n for n in BASE_CHECKPOINTS if n <= samples] + [samples]))
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
    elbow = find_elbow_point(curve)
    return dict(method='uniform_point_area', seed=int(seed), box_area_mm2=box_area,
                reference_pixel_area_mm2=features['area_mm2'], **curve[-1],
                elbow=elbow, convergence=curve,
                preview_points_xy_mm_hit=preview,
                interpretation='Sampling error conditional on this pixel mask; excludes segmentation and clinical error.')
