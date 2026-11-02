"""Physical sagittal planes and explicit, user-defined vertebral body regions."""
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path as PolygonPath

from spinal_ct.review import same_geometry


def spacing_xy(spacing):
    value = np.asarray(spacing, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError('Spacing must contain two positive finite values in mm.')
    return value


def binary_mask(mask):
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) < 1 or value.size > 4_000_000:
        raise ValueError('Expected a 2D mask of at most four million pixels.')
    if not np.isin(value, [0, 1]).all():
        raise ValueError('Body mask must contain only 0 and 1.')
    return value.astype(bool)


@dataclass(frozen=True)
class BodyPlane:
    # origin is the centre of pixel (column=0, row=0), not its lower corner.
    origin_lps_mm: tuple
    u_lps: tuple
    v_lps: tuple
    spacing_xy_mm: tuple
    size_xy: tuple

    def __post_init__(self):
        origin = np.asarray(self.origin_lps_mm, float)
        u, v = np.asarray(self.u_lps, float), np.asarray(self.v_lps, float)
        spacing_xy(self.spacing_xy_mm)
        size = np.asarray(self.size_xy)
        if (origin.shape != (3,) or u.shape != (3,) or v.shape != (3,)
                or not np.isfinite([origin, u, v]).all()
                or not np.allclose(np.array([u, v]) @ np.array([u, v]).T, np.eye(2), atol=1e-6)
                or size.shape != (2,) or not np.isfinite(size).all()
                or (size != size.astype(int)).any() or (size < 2).any()
                or np.prod(size) > 4_000_000):
            raise ValueError('Invalid body plane geometry.')
        # This tool's six-point convention requires posterior-right/superior-up.
        if u[1] < .5 or v[2] < .5:
            raise ValueError('Plane axes must point predominantly posterior and superior.')

    def to_dict(self):
        return asdict(self)

    def reference(self):
        image = sitk.Image([int(x) for x in self.size_xy] + [1], sitk.sitkFloat32)
        image.SetOrigin(list(self.origin_lps_mm))
        image.SetSpacing([*self.spacing_xy_mm, 1.])
        image.SetDirection(np.column_stack((self.u_lps, self.v_lps,
                                           np.cross(self.u_lps, self.v_lps))).ravel().tolist())
        return image

    def point_lps(self, xy_mm):
        return (np.asarray(self.origin_lps_mm) + xy_mm[0]*np.asarray(self.u_lps)
                + xy_mm[1]*np.asarray(self.v_lps)).tolist()


def propose_plane(source, labels, label_id, *, offset_mm=0., yaw_deg=0., tilt_deg=0., margin_mm=5.):
    """Propose from whole-vertebra bounds; the user must confirm the body plane."""
    if not same_geometry(source, labels):
        raise ValueError('CT and localization geometry differ.')
    if not np.isfinite([offset_mm, yaw_deg, tilt_deg, margin_mm]).all() or margin_mm < 0:
        raise ValueError('Invalid plane parameters.')
    if abs(yaw_deg) > 45 or abs(tilt_deg) > 45:
        raise ValueError('Plane angles must be within ±45 degrees.')
    if not 1 <= label_id <= 25:
        raise ValueError('Invalid vertebral label.')
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(False)
    stats.Execute(labels)
    if not stats.HasLabel(label_id):
        raise ValueError('Selected vertebra is absent from the label map.')
    box = np.asarray(stats.GetBoundingBox(label_id))
    corners = np.array([source.TransformContinuousIndexToPhysicalPoint(tuple(map(float, p)))
                        for p in product(*zip(box[:3]-.5, box[:3]+box[3:]-.5))])
    yaw, tilt = np.deg2rad([yaw_deg, tilt_deg])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    rotation = rz @ ry
    u, v, normal = rotation[:, 1], rotation[:, 2], rotation[:, 0]
    centre = np.asarray(stats.GetCentroid(label_id)) + normal*offset_mm
    bounds = (corners-centre) @ np.column_stack((u, v))
    lower, upper = bounds.min(0)-margin_mm, bounds.max(0)+margin_mm
    step = float(min(source.GetSpacing()))
    size = np.maximum(2, np.ceil((upper-lower)/step).astype(int))
    origin = centre + u*(lower[0]+step/2) + v*(lower[1]+step/2)
    return BodyPlane(tuple(origin), tuple(u), tuple(v), (step, step), tuple(map(int, size)))


def sample_plane(source, plane, labels=None, label_id=None):
    """Resample CT linearly; labels use nearest neighbour. Keep the FOV validity."""
    reference = plane.reference()
    image = sitk.Resample(source, reference, sitk.Transform(), sitk.sitkLinear,
                          float('nan'), sitk.sitkFloat32)
    ct = sitk.GetArrayFromImage(image)[0]
    valid = np.isfinite(ct)
    ct = np.where(valid, ct, -1024).astype(np.float32)
    hint = np.zeros(ct.shape, bool)
    if labels is not None:
        if not same_geometry(source, labels):
            raise ValueError('CT and label geometry differ.')
        sampled = sitk.Resample(labels, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        hint = sitk.GetArrayFromImage(sampled)[0] == label_id
    return ct, valid, hint


def plane_parameters(source, labels, label_id, plane):
    """Recover the editor settings only when they reproduce the saved plane."""
    u, v = np.asarray(plane.u_lps), np.asarray(plane.v_lps)
    yaw = np.arctan2(-u[0], u[1])
    tilt = np.arctan2(v[0]*np.cos(yaw)+v[1]*np.sin(yaw), v[2])
    base = propose_plane(source, labels, label_id, yaw_deg=float(np.rad2deg(yaw)),
                         tilt_deg=float(np.rad2deg(tilt)))
    normal = np.cross(u, v)
    delta = np.asarray(plane.origin_lps_mm)-base.origin_lps_mm
    offset = float(delta @ normal)
    if (tuple(plane.size_xy) != base.size_xy
            or not np.allclose(plane.spacing_xy_mm, base.spacing_xy_mm, atol=1e-6)
            or not np.allclose(u, base.u_lps, atol=1e-6)
            or not np.allclose(v, base.v_lps, atol=1e-6)
            or not np.allclose(delta, offset*normal, atol=1e-5)):
        raise ValueError('Saved custom plane cannot be reproduced by the editor controls.')
    return dict(offset_mm=offset, yaw_deg=float(np.rad2deg(yaw)), tilt_deg=float(np.rad2deg(tilt)))


def polygon_mask(vertices_xy_mm, shape, spacing):
    """Rasterize the user's body outline at pixel centres; retain raw vertices."""
    vertices = np.asarray(vertices_xy_mm, float)
    sx, sy = spacing_xy(spacing)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
            or not np.isfinite(vertices).all()):
        raise ValueError('A body outline requires at least three finite points.')
    lower, upper = np.array([-.5*sx, -.5*sy]), (np.array(shape[::-1])-.5)*[sx, sy]
    if (vertices < lower).any() or (vertices > upper).any():
        raise ValueError('Outline extends beyond the plane.')
    y, x = np.indices(shape)
    return PolygonPath(vertices).contains_points(np.column_stack((x.ravel()*sx, y.ravel()*sy))).reshape(shape)
