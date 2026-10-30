"""Shape descriptors of a pixel-cell region, in plane millimetres."""
import numpy as np
import SimpleITK as sitk

from spinal_ct.body_region import binary_mask, spacing_xy


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def half(items):
        result = []
        for p in items:
            def turn():
                a = np.subtract(result[-1], result[-2])
                b = np.subtract(p, result[-1])
                return a[0]*b[1]-a[1]*b[0]
            while len(result) > 1 and turn() <= 0:
                result.pop()
            result.append(p)
        return result
    return np.array(half(points)[:-1] + half(points[::-1])[:-1], float)


def analyze_blob(mask, spacing):
    mask = binary_mask(mask)
    sx, sy = spacing_xy(spacing)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError('Draw a non-empty vertebral body mask first.')
    points = np.column_stack((cols*sx, rows*sy))
    centre = points.mean(0)
    # Add the second moments of each occupied rectangular pixel cell.
    covariance = (points-centre).T @ (points-centre)/len(points) + np.diag([sx*sx, sy*sy])/12
    eigenvalues, vectors = np.linalg.eigh(covariance)
    major = vectors[:, 1]
    if major[0] < 0 or (abs(major[0]) < 1e-12 and major[1] < 0):
        major = -major
    axes = np.column_stack((major, [-major[1], major[0]]))
    projected = points @ axes
    half_cell = np.abs(axes).T @ np.array([sx, sy])/2
    lower, upper = projected.min(0)-half_cell, projected.max(0)+half_cell
    corners = np.array([[lower[0], lower[1]], [upper[0], lower[1]],
                        [upper[0], upper[1]], [lower[0], upper[1]]]) @ axes.T
    # Pixel-edge points define a cell-union hull, rather than a centre-only hull.
    padded = np.pad(mask, 1)
    interior = (padded[1:-1, :-2] & padded[1:-1, 2:]
                & padded[:-2, 1:-1] & padded[2:, 1:-1])
    edge = mask & ~interior
    er, ec = np.nonzero(edge)
    boundary = np.column_stack((ec*sx, er*sy))
    hull = convex_hull(np.concatenate([boundary + [i*sx/2, j*sy/2]
                                      for i in (-1, 1) for j in (-1, 1)]))
    hull_area = abs(np.dot(hull[:, 0], np.roll(hull[:, 1], 1))
                    - np.dot(hull[:, 1], np.roll(hull[:, 0], 1)))/2
    components = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)), True)
    component_count = int(sitk.GetArrayViewFromImage(components).max())
    area = float(len(points)*sx*sy)
    warnings = []
    if len(points) < 16:
        warnings.append('small_region')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        warnings.append('touches_plane_boundary')
    if component_count > 1:
        warnings.append('multiple_components')
    reliable = bool(eigenvalues[1]/eigenvalues[0] > 1.05)
    if not reliable:
        warnings.append('ellipse_orientation_unstable')
    return dict(area_mm2=area, pixel_count=len(points), component_count=component_count,
                centroid_xy_mm=centre.tolist(), ellipse=dict(
                    major_length_mm=float(4*np.sqrt(eigenvalues[1])),
                    minor_length_mm=float(4*np.sqrt(eigenvalues[0])),
                    axis_ratio=float(np.sqrt(eigenvalues[1]/eigenvalues[0])),
                    angle_deg=float(np.rad2deg(np.arctan2(major[1], major[0]))),
                    orientation_reliable=reliable,
                    definition='Ellipse with the same second moments as the union of occupied pixel cells.'),
                oriented_box=dict(axes_xy=axes.tolist(), min_uv_mm=lower.tolist(), max_uv_mm=upper.tolist(),
                                  corners_xy_mm=corners.tolist(), area_mm2=float(np.prod(upper-lower)),
                                  definition='Pixel-cell bounds along moment axes; not a minimum-area box.'),
                fill_ratio=float(area/np.prod(upper-lower)), solidity=float(area/hull_area),
                height_width_ratio=float((np.ptp(rows)+1)*sy/((np.ptp(cols)+1)*sx)),
                height_width_definition='Superior/posterior extent in the confirmed plane, before ellipse alignment.',
                warnings=warnings)


def normalize_region(ct, mask, spacing, features=None, size=128, padding=4):
    mask = binary_mask(mask)
    if ct.shape != mask.shape or not np.isfinite(ct).all():
        raise ValueError('CT plane must be finite and match the body mask.')
    if size < 16 or not 0 <= padding < size/2-1:
        raise ValueError('Invalid normalized image size or padding.')
    features = features or analyze_blob(mask, spacing)
    centre = np.asarray(features['centroid_xy_mm'])
    box = features['oriented_box']
    axes = np.asarray(box['axes_xy'])
    centre_uv = centre @ axes
    radius = max(np.max(np.abs(np.asarray(box['min_uv_mm'])-centre_uv)),
                 np.max(np.abs(np.asarray(box['max_uv_mm'])-centre_uv)))
    step = 2*radius/(size-2*padding)
    origin = centre - axes @ np.array([(size-1)/2*step]*2)
    output = []
    for array, interpolation, fill in ((ct.astype(np.float32), sitk.sitkLinear, -1024),
                                        (mask.astype(np.uint8), sitk.sitkNearestNeighbor, 0)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(list(map(float, spacing)))
        image = sitk.Resample(image, [size, size], sitk.Transform(), interpolation,
                              origin.tolist(), [step, step], axes.ravel().tolist(), fill)
        output.append(sitk.GetArrayFromImage(image))
    transform = np.eye(3)
    transform[:2, :2] = axes*step
    transform[:2, 2] = origin
    normalized_mask = output[1].astype(bool)
    return dict(ct=np.where(normalized_mask, output[0], -1024).astype(np.float32), mask=normalized_mask,
                transform=dict(output_pixel_to_plane_mm=transform.tolist(), spacing_mm=step,
                               size=size, padding=padding, interpolation_ct='linear', interpolation_mask='nearest',
                               outside_body_hu=-1024))
