"""CT-only vertebral localization; all coordinates in physical LPS millimetres."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk

LEVELS = tuple([f'C{i}' for i in range(1, 8)] +
               [f'T{i}' for i in range(1, 13)] +
               [f'L{i}' for i in range(1, 6)] + ['S1'])


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
    from matplotlib import colormaps
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
    ax = axes[0]
    height, width = plane.shape
    extent = (-spacing[1]/2, (width-.5)*spacing[1],
              -spacing[2]/2, (height-.5)*spacing[2])
    for panel in axes:
        panel.set_facecolor('#101820')
        panel.imshow(data[:, :, x], cmap='gray', vmin=-200, vmax=1200,
                     origin='lower', extent=extent, interpolation='bilinear')
        panel.tick_params(colors='#b8c7d4', labelsize=8)
        panel.set_xlabel('Anterior → Posterior (mm)', color='#b8c7d4')
        panel.set_ylabel('Inferior → Superior (mm)', color='#b8c7d4')
        for spine in panel.spines.values():
            spine.set_color('#41515e')
    annotations = []
    for label_id in np.unique(plane):
        if label_id == 0:
            continue
        region = plane == label_id
        color = colormaps['tab20']((int(label_id)-1) % 20)
        # Padding closes contours at the field-of-view edge. Coordinates describe
        # voxel centres, so level 0.5 traces the boundary between voxel centres.
        ax.contour(np.arange(-1, width+1)*spacing[1],
                   np.arange(-1, height+1)*spacing[2],
                   np.pad(region.astype(float), 1), levels=[0.5],
                   colors=[color], linewidths=1.1, antialiased=True)
        coords = np.argwhere(region)
        z = float(np.median(coords[:, 0]))*spacing[2]
        y = float(coords[:, 1].min())*spacing[1]
        annotations.append((z, y, int(label_id), color))
    # Reserve a label gutter; separate adjacent names even for small fragments.
    previous_z = -np.inf
    for z, y, label_id, color in sorted(annotations):
        label_z = max(z, previous_z + 9)
        previous_z = label_z
        ax.annotate(LEVELS[label_id-1], xy=(y, z), xytext=(-12, label_z),
                    fontsize=9, fontweight='bold', color=color, ha='right', va='center',
                    arrowprops=dict(arrowstyle='-', color=color, lw=.7),
                    annotation_clip=False)
    ax.set_xlim(-28, extent[1])
    ax.set_ylim(extent[2], max(extent[3], previous_z+8))
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
    fig.subplots_adjust(left=.09, right=.94, top=.91, bottom=.11, wspace=.35)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    fig.clear()
