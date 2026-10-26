"""CT-only vertebral localization; all coordinates in physical LPS millimetres."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk

LEVELS = tuple([f'C{i}' for i in range(1, 8)] +
               [f'T{i}' for i in range(1, 13)] +
               [f'L{i}' for i in range(1, 6)] + ['S1'])

# Frozen TotalSegmentator 2.18.0 / task total / v2 IDs. These are NOT our IDs:
# backend C1=50, ..., L1=31, ..., S1=26; project C1=1, ..., L1=20, S1=25.
BACKEND_LABEL_IDS = dict(zip(LEVELS, range(50, 25, -1)))


def validate_label_image(image: sitk.Image, labels: sitk.Image) -> None:
    from spinal_ct.review import same_geometry
    if not same_geometry(image, labels):
        raise ValueError('Vertebral label geometry mismatch.')
    values = sitk.GetArrayViewFromImage(labels)
    # Bound validation temporaries for large CT volumes, including float inputs.
    for plane in values:
        if not np.isin(plane, np.arange(len(LEVELS)+1)).all():
            raise ValueError('Invalid vertebral label in label map.')


def load_backend_labels(image: sitk.Image, path: Path) -> sitk.Image:
    """Validate and remap a --ml output without interpolation or changing geometry."""
    from spinal_ct.review import same_geometry
    raw = sitk.ReadImage(str(path))
    if not same_geometry(image, raw):
        raise ValueError('Backend label geometry mismatch.')
    values = sitk.GetArrayViewFromImage(raw)
    lookup = np.zeros(51, dtype=np.uint8)
    for project_id, level in enumerate(LEVELS, 1):
        lookup[BACKEND_LABEL_IDS[level]] = project_id
    combined = np.empty(values.shape, dtype=np.uint8)
    allowed = [0, *BACKEND_LABEL_IDS.values()]
    for z, plane in enumerate(values):
        if not np.isin(plane, allowed).all():
            raise ValueError('Unexpected label in TotalSegmentator 2.18.0 vertebral output.')
        combined[z] = lookup[plane.astype(np.uint8)]
    labels = sitk.GetImageFromArray(combined)
    labels.CopyInformation(image)
    return labels


def label_statistics(labels: sitk.Image, reference: sitk.Image):
    """One label-statistics pass; do not compute unused, expensive surface measures."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(False)
    stats.SetComputeFeretDiameter(False)
    stats.SetComputeOrientedBoundingBox(False)
    measured = sitk.Cast(labels, sitk.sitkUInt8)
    # NIfTI-1 rounds geometry to float32 on disk. Measurements must use the
    # original CT geometry, as the binary-mask implementation does.
    measured.CopyInformation(reference)
    stats.Execute(measured)
    return stats


def summarize_labels(image: sitk.Image, labels: sitk.Image,
                     mask_path: str = 'vertebrae.nii.gz') -> dict:
    """Schema 2: each prediction references a shared label map plus its label_id."""
    validate_label_image(image, labels)
    stats = label_statistics(labels, image)
    predictions, warnings = [], []
    for label_id, level in enumerate(LEVELS, 1):
        if not stats.HasLabel(label_id):
            continue
        box = stats.GetBoundingBox(label_id)
        lower = np.array(box[:3])
        upper = lower+box[3:]
        corners = np.asarray([image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, p)))
                              for p in product(*zip(lower-.5, upper-.5))])
        boundary = bool((lower == 0).any() or (upper == image.GetSize()).any())
        if boundary:
            warnings.append(f'{level}: mask touches image boundary; anatomy may be truncated.')
        predictions.append(dict(level=level, label_id=label_id, mask=mask_path,
            mask_encoding='label_map', confidence=None,
            centroid_lps_mm=list(stats.GetCentroid(label_id)),
            bbox_voxel_xyz=dict(min=lower.tolist(), max_exclusive=upper.tolist()),
            roi_lps_mm=dict(min=corners.min(axis=0).tolist(), max=corners.max(axis=0).tolist()),
            volume_mm3=float(stats.GetNumberOfPixels(label_id)*np.prod(image.GetSpacing())),
            touches_boundary=boundary))
    if any(a['centroid_lps_mm'][2] <= b['centroid_lps_mm'][2]
           for a, b in zip(predictions, predictions[1:])):
        warnings.append('Non-descending superior-to-inferior level order; manual review required.')
    indices = [p['label_id'] for p in predictions]
    if any(b-a > 1 for a, b in zip(indices, indices[1:])):
        warnings.append('Missing levels between detected vertebrae; manual review required.')
    if not predictions:
        warnings.append('No vertebrae detected.')
    return dict(schema_version=2, status='indeterminate' if warnings else 'ok',
        review_required=True, task='vertebral_localization', predictions=predictions,
        warnings=warnings, confidence_note='Backend provides hard masks; confidence is unavailable.',
        geometry=dict(coordinate_system='LPS', size_xyz=list(image.GetSize()),
                      spacing_xyz_mm=list(image.GetSpacing()), origin_lps_mm=list(image.GetOrigin()),
                      direction_lps=list(image.GetDirection())),
        label_map={str(i): level for i, level in enumerate(LEVELS, 1)})


def export_binary_masks(labels: sitk.Image, folder: Path) -> None:
    """Optional compatibility export. Never parallelize full-volume saving on Windows."""
    folder.mkdir()
    for label_id, level in enumerate(LEVELS, 1):
        sitk.WriteImage(labels == label_id, str(folder/f'vertebrae_{level}.nii.gz'))


def summarize_masks(image: sitk.Image, mask_dir: Path) -> tuple[dict, sitk.Image]:
    """Validate complete model outputs, then derive per-level masks and physical boxes."""
    combined = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
    predictions, warnings = [], []
    for label_id, level in enumerate(LEVELS, 1):
        path = mask_dir / f'vertebrae_{level}.nii.gz'
        if not path.is_file():
            raise ValueError(f'Model output missing for {level}.')
        mask = sitk.ReadImage(str(path))
        if (mask.GetSize() != image.GetSize() or mask.GetNumberOfComponentsPerPixel() != 1
                or not np.allclose(mask.GetSpacing(), image.GetSpacing(), atol=1e-5)
                or not np.allclose(mask.GetOrigin(), image.GetOrigin(), atol=1e-4)
                or not np.allclose(mask.GetDirection(), image.GetDirection(), atol=1e-5)):
            raise ValueError(f'Mask geometry mismatch for {level}.')
        values = sitk.GetArrayFromImage(mask)
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f'Expected a binary mask for {level}.')
        foreground = values > 0
        if not foreground.any():
            continue
        if (combined[foreground] != 0).any():
            raise ValueError(f'Overlapping vertebral masks at {level}.')
        combined[foreground] = label_id
        xyz = np.argwhere(foreground)[:, ::-1]
        lower, upper = xyz.min(axis=0), xyz.max(axis=0) + 1
        corners = [image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, p)))
                   for p in product(*zip(lower - 0.5, upper - 0.5))]
        corners = np.asarray(corners)
        centroid = image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, xyz.mean(axis=0))))
        boundary = bool((lower == 0).any() or (upper == image.GetSize()).any())
        if boundary:
            warnings.append(f'{level}: mask touches image boundary; anatomy may be truncated.')
        predictions.append(dict(level=level, label_id=label_id,
            mask=f'masks/vertebrae_{level}.nii.gz', confidence=None,
            centroid_lps_mm=list(centroid),
            bbox_voxel_xyz=dict(min=lower.tolist(), max_exclusive=upper.tolist()),
            roi_lps_mm=dict(min=corners.min(axis=0).tolist(), max=corners.max(axis=0).tolist()),
            volume_mm3=float(foreground.sum() * np.prod(image.GetSpacing())),
            touches_boundary=boundary))
    if any(a['centroid_lps_mm'][2] <= b['centroid_lps_mm'][2]
           for a, b in zip(predictions, predictions[1:])):
        warnings.append('Non-descending superior-to-inferior level order; manual review required.')
    indices = [p['label_id'] for p in predictions]
    if any(b - a > 1 for a, b in zip(indices, indices[1:])):
        warnings.append('Missing levels between detected vertebrae; manual review required.')
    if not predictions:
        warnings.append('No vertebrae detected.')
    result = sitk.GetImageFromArray(combined)
    result.CopyInformation(image)
    report = dict(schema_version=1, status='indeterminate' if warnings else 'ok',
        review_required=True, task='vertebral_localization', predictions=predictions,
        warnings=warnings, confidence_note='Backend provides hard masks; confidence is unavailable.',
        geometry=dict(coordinate_system='LPS', size_xyz=list(image.GetSize()),
                      spacing_xyz_mm=list(image.GetSpacing()), origin_lps_mm=list(image.GetOrigin()),
                      direction_lps=list(image.GetDirection())),
        label_map={str(i): level for i, level in enumerate(LEVELS, 1)})
    return report, result


def render_preview(image: sitk.Image, labels: sitk.Image, path: Path,
                   *, disagreement: sitk.Image | None = None) -> None:
    """Write a sagittal QC image; the 3D masks remain the authoritative output."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from spinal_ct.overlays import draw_vertebral_overlay
    for overlay in (labels, disagreement):
        if overlay is None:
            continue
        if (overlay.GetSize() != image.GetSize()
                or not np.allclose(overlay.GetSpacing(), image.GetSpacing())
                or not np.allclose(overlay.GetOrigin(), image.GetOrigin())
                or not np.allclose(overlay.GetDirection(), image.GetDirection())):
            raise ValueError('Preview overlay geometry mismatch.')
    ct = sitk.DICOMOrient(image, 'LPS')
    mask = sitk.DICOMOrient(labels, 'LPS')
    # Resample oblique images to an axis-aligned physical grid for a true sagittal view.
    corners = np.array([ct.TransformIndexToPhysicalPoint(tuple(map(int, p)))
                        for p in product(*[(0, n-1) for n in ct.GetSize()])])
    origin = corners.min(axis=0)
    spacing = np.asarray(ct.GetSpacing())
    size = np.ceil((corners.max(axis=0)-origin)/spacing).astype(int)+1
    reference = sitk.Image([int(n) for n in size], sitk.sitkFloat32)
    reference.SetOrigin(origin.tolist())
    reference.SetSpacing(spacing.tolist())
    ct = sitk.Resample(ct, reference, sitk.Transform(), sitk.sitkLinear, -1024)
    mask = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    data, segmentation = sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(mask)
    counts = (segmentation > 0).sum(axis=(0, 1))
    x = int(counts.argmax()) if counts.any() else data.shape[2]//2
    plane = segmentation[:, :, x]
    fig = Figure(figsize=(12 if disagreement is not None else 7, 10), facecolor='#101820')
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, 2 if disagreement is not None else 1))
    fig.subplots_adjust(left=.09, right=.94, top=.91, bottom=.11, wspace=.35)
    ax = axes[0]
    height, width = plane.shape
    extent = (-spacing[1]/2, (width-.5)*spacing[1],
              -spacing[2]/2, (height-.5)*spacing[2])
    for panel in axes:
        panel.set_facecolor('#101820')
        panel.imshow(data[:, :, x], cmap='gray', vmin=-200, vmax=1200,
                     origin='lower', extent=extent, interpolation='bilinear')
        panel.set_xlim(extent[:2])
        panel.set_ylim(extent[2:])
        panel.tick_params(colors='#b8c7d4', labelsize=8)
        panel.set_xlabel('Anterior → Posterior (mm)', color='#b8c7d4')
        panel.set_ylabel('Inferior → Superior (mm)', color='#b8c7d4')
        for spine in panel.spines.values():
            spine.set_color('#41515e')
    draw_vertebral_overlay(ax, plane, (spacing[1], spacing[2]), names=True)
    ax.set_title('Consensus boundaries' if disagreement is not None else 'Baseline boundaries',
                 color='white', pad=12)
    if disagreement is not None:
        uncertainty = sitk.Resample(disagreement, reference, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, 0)
        values = sitk.GetArrayViewFromImage(uncertainty)[:, :, x]
        heat = axes[1].imshow(np.ma.masked_less_equal(values, 0), cmap='magma',
                             vmin=0, vmax=1, origin='lower', extent=extent,
                             interpolation='nearest', alpha=.8)
        axes[1].set_title('Monte Carlo disagreement', color='white', pad=12)
        bar = fig.colorbar(heat, ax=axes[1], fraction=.035, pad=.03)
        bar.set_label('1 − highest label frequency', color='#b8c7d4')
        bar.ax.tick_params(colors='#b8c7d4')
    fig.suptitle('Vertebral localization · review required', color='white', fontsize=15)
    fig.text(.5, .025, 'Sagittal view · distances relative to image grid · contours follow stored masks\n'
             'Display only: no mask smoothing · disagreement is not calibrated confidence',
             ha='center', color='#b8c7d4', fontsize=8)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    fig.clear()
