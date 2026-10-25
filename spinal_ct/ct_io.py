"""Shared CT loading and geometry validation; no reference labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def validate_geometry(image: sitk.Image) -> None:
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Input must be a scalar 3D CT volume.')
    if min(image.GetSize()) < 2:
        raise ValueError('A single CT slice is insufficient for 3D localization.')
    spacing = np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if (not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.isfinite(image.GetOrigin()).all()
            or not np.isfinite(direction).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid physical CT geometry.')
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError('CT contains non-finite intensities.')


def load_prepared_ct(path: Path) -> sitk.Image:
    """Read the project's HU array and geometry only; never consume reference labels."""
    metadata = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError('Prepared CT metadata must be an object.')
    volume = np.load(path, mmap_mode='r', allow_pickle=False)
    if volume.ndim != 3 or volume.dtype.kind not in 'iuf':
        raise ValueError('Prepared CT must be a numeric 3D HU array.')
    if list(volume.shape) != metadata.get('shape_zyx'):
        raise ValueError('Prepared CT shape does not match metadata.')
    try:
        spacing = np.asarray(metadata.get('spacing_zyx_mm'), dtype=float)
        origin = np.asarray(metadata.get('origin_lps_mm'), dtype=float)
        if spacing.shape != (3,) or origin.shape != (3,):
            raise ValueError('Prepared CT needs spacing_zyx_mm and origin_lps_mm.')
        direction = None
        if 'image_orientation_patient' in metadata:
            orientation = np.asarray(metadata['image_orientation_patient'], dtype=float)
            if orientation.shape != (6,):
                raise ValueError('Invalid prepared CT orientation.')
            direction = np.column_stack((orientation[:3], orientation[3:],
                                         np.cross(orientation[:3], orientation[3:])))
        if 'direction_lps' in metadata:
            stored = np.asarray(metadata['direction_lps'], dtype=float)
            if stored.shape not in ((9,), (3, 3)):
                raise ValueError('direction_lps must contain a 3x3 direction matrix.')
            stored = stored.reshape(3, 3)
            if direction is not None and not np.allclose(stored, direction, atol=1e-5):
                raise ValueError('Prepared CT orientation fields disagree.')
            direction = stored
        if direction is None:
            raise ValueError('Prepared CT is missing direction metadata.')
        if (not np.isfinite(spacing).all() or (spacing <= 0).any()
                or not np.isfinite(origin).all() or not np.isfinite(direction).all()
                or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)):
            raise ValueError('Invalid prepared CT physical geometry.')
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid numeric prepared CT geometry.') from exc
    image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    image.SetSpacing(spacing[::-1].tolist())
    image.SetOrigin(origin.tolist())
    image.SetDirection(direction.ravel().tolist())
    validate_geometry(image)
    return image


def dicom_groups(path: Path) -> dict:
    """Collect classic CT series without using clinical labels or directory names."""
    groups = {}
    for candidate in path.rglob('*'):
        if not candidate.is_file():
            continue
        try:
            ds = pydicom.dcmread(candidate, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(ds, 'Modality', '') != 'CT' or 'LOCALIZER' in getattr(ds, 'ImageType', []):
            continue
        uid = str(getattr(ds, 'SeriesInstanceUID', ''))
        if uid:
            groups.setdefault(uid, []).append((candidate, ds))
    return groups



def _ordered_dicom_items(items):
    """Validate classic CT headers and return slices in physical order."""
    items = list(items)
    required = ('ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing',
                'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept')
    if len(items) < 2 or any(int(getattr(d, 'NumberOfFrames', 1)) != 1 or
                            not all(hasattr(d, key) for key in required) for _, d in items):
        raise ValueError('Need at least two classic CT slices with geometry and HU rescale metadata; enhanced multi-frame CT is not supported.')
    first = items[0][1]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    spacing = np.asarray(first.PixelSpacing, dtype=float)
    if orientation.shape != (6,) or spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid DICOM orientation or pixel spacing.')
    normal = np.cross(orientation[:3], orientation[3:])
    basis = np.column_stack((orientation[:3], orientation[3:], normal))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
        raise ValueError('DICOM orientation is not orthonormal.')
    for _, ds in items:
        if ((ds.Rows, ds.Columns) != (first.Rows, first.Columns)
                or not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-5)
                or not np.allclose(ds.PixelSpacing, spacing, atol=1e-5)):
            raise ValueError('Inconsistent slice geometry in CT series.')
    items.sort(key=lambda item: np.dot(item[1].ImagePositionPatient, normal))
    positions = np.array([ds.ImagePositionPatient for _, ds in items], dtype=float)
    steps = np.diff(positions, axis=0)
    gaps = steps @ normal
    if (not np.isfinite(positions).all() or (gaps <= 0).any()
            or not np.allclose(gaps, np.median(gaps), atol=0.01, rtol=0)
            or not np.allclose(steps, gaps[:, None] * normal, atol=0.01, rtol=0)):
        raise ValueError('Duplicate/irregular slices or gantry shear; resampling is required.')
    return items, normal


def list_ct_series(path: Path) -> list[dict]:
    """Technical series choices for the UI; no names, patient IDs or descriptions."""
    choices = []
    for uid, items in dicom_groups(path).items():
        first = items[0][1]
        axial = False
        geometry_error = None
        try:
            _, normal = _ordered_dicom_items(items)
            # Automatic selection is limited to stacks within 15 degrees of axial.
            axial = abs(normal[2]) >= np.cos(np.deg2rad(15))
        except (ValueError, TypeError, AttributeError, IndexError) as exc:
            geometry_error = str(exc)
        choices.append(dict(uid=uid, slices=len(items),
                            axial=bool(axial), geometry_error=geometry_error,
                            rows=int(getattr(first, 'Rows', 0)),
                            columns=int(getattr(first, 'Columns', 0)),
                            kernel=str(getattr(first, 'ConvolutionKernel', 'Unknown')),
                            thickness=str(getattr(first, 'SliceThickness', '?'))))
    return choices


def select_sample_series(choices: list[dict]) -> dict | None:
    """Suggest the unique largest valid axial stack, without scanner/kernel rules.

    Slice count is only a technical default, not an assessment of anatomy or quality.
    Ties and unsupported geometry require explicit series selection in the UI.
    """
    candidates = [item for item in choices
                  if item.get('axial') and item.get('geometry_error') is None
                  and item['slices'] >= 2]
    if not candidates:
        return None
    largest = max(item['slices'] for item in candidates)
    matches = [item for item in candidates if item['slices'] == largest]
    return matches[0] if len(matches) == 1 else None


def load_ct(path: Path, series_uid: str | None = None) -> sitk.Image:
    """Read classic single-frame CT DICOM or a preconverted HU NIfTI volume.

    Reject irregular stacks and gantry shear instead of silently changing geometry.
    SimpleITK applies DICOM rescale slope/intercept when decoding pixels.
    """
    path = Path(path)
    if path.is_file() and path.suffix.lower() == '.npy':
        if series_uid:
            raise ValueError('--series-uid only applies to DICOM directories.')
        return load_prepared_ct(path)
    if path.is_file() and path.name.lower().endswith(('.nii', '.nii.gz')):
        if series_uid:
            raise ValueError('--series-uid only applies to DICOM directories.')
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    elif path.is_dir():
        groups = dicom_groups(path)
        if series_uid is None:
            if len(groups) != 1:
                raise ValueError(f'Found {len(groups)} CT series; select one with --series-uid.')
            series_uid = next(iter(groups))
        if series_uid not in groups:
            raise ValueError('Selected CT series was not found.')
        items = groups[series_uid]
        items, _ = _ordered_dicom_items(items)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(p) for p, _ in items])
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()
    else:
        raise ValueError('Input must be DICOM, HU NIfTI or a prepared HU .npy with metadata.json, not a screenshot.')
    validate_geometry(image)
    # Remove identifying metadata before persisting or passing to an external model.
    for key in image.GetMetaDataKeys():
        image.EraseMetaData(key)
    return image
