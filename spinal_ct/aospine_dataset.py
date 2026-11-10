"""2.5D Dual-ROI Data Pipeline for AO Spine Deep Learning Models (Phase 2).

Provides:
- apply_ct_window: Hounsfield Unit (HU) windowing with standard presets.
- extract_body_canal_roi_25d: 5-channel ROI for Body + Spinal Canal (shape (5, 128, 128)).
- extract_posterior_arch_roi_25d: 5-channel ROI for Posterior Arch (shape (5, 128, 128)).
"""
from typing import Any
import numpy as np
from PIL import Image

# Standard Radiologic CT Window Presets
BONE_WINDOW_LEVEL = 400.0
BONE_WINDOW_WIDTH = 1800.0

SOFT_TISSUE_WINDOW_LEVEL = 50.0
SOFT_TISSUE_WINDOW_WIDTH = 350.0


def apply_ct_window(hu_array: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Normalize CT Hounsfield Units into [0.0, 1.0] using specified Window Level (WL) and Window Width (WW).

    Parameters
    ----------
    hu_array : np.ndarray
        Raw CT intensity array in Hounsfield Units.
    wl : float
        Window center/level in HU.
    ww : float
        Window width in HU (must be > 0).

    Returns
    -------
    np.ndarray
        Float32 array in [0.0, 1.0].
    """
    if ww <= 0:
        raise ValueError("Window width (ww) must be strictly positive.")
    arr = np.asarray(hu_array, dtype=np.float32)
    lower = float(wl - ww / 2.0)
    upper = float(wl + ww / 2.0)
    clipped = np.clip(arr, lower, upper)
    norm = (clipped - lower) / float(ww)
    return np.asarray(norm, dtype=np.float32)


def _resize_2d(arr: np.ndarray, target_size: tuple[int, int] = (128, 128), is_mask: bool = False) -> np.ndarray:
    """Resize 2D float32 array to target_size (rows, cols) using PIL."""
    target_w, target_h = target_size[1], target_size[0]
    im = Image.fromarray(arr.astype(np.float32), mode='F')
    resample_mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = im.resize((target_w, target_h), resample=resample_mode)
    return np.array(resized, dtype=np.float32)


def extract_body_canal_roi_25d(
    ct_volume_or_slice: np.ndarray,
    body_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, ...],
    *,
    curr_vert_landmarks: list[list[float]] | None = None,
    slice_x_index: int | None = None,
    target_size: tuple[int, int] = (128, 128),
    canal_depth_mm: float = 12.0,
    vertical_margin_mm: float = 6.0,
) -> np.ndarray:
    """Extract 5-channel 2.5D Body + Canal ROI centered on vertebral body and spinal canal.

    Channels:
    - Ch 0: Mid-sagittal slice with Bone Window (WL=400, WW=1800).
    - Ch 1: Parasagittal slice at -4.0 mm (Bone Window) [fallback: horizontal gradient edge].
    - Ch 2: Parasagittal slice at +4.0 mm (Bone Window) [fallback: vertical gradient edge].
    - Ch 3: Axial slice at upper 1/3 of vertebral body (Bone Window) [fallback: Soft-tissue window].
    - Ch 4: Reference Mask: Body + Canal zone (values in [0.0, 1.0]).

    Parameters
    ----------
    ct_volume_or_slice : np.ndarray
        3D CT volume (Z, Y, X) or 2D sagittal slice (Z, Y).
    body_mask : np.ndarray
        2D binary mask of target vertebral body (shape matches 2D sagittal slice).
    spacing_xyz_mm : tuple
        Voxel spacing (sx, sy, sz) in mm for 3D, or (sy, sz) for 2D.
    curr_vert_landmarks : list of list of float, optional
        6 landmark physical coordinates [AS, AI, MS, MI, PS, PI].
    slice_x_index : int, optional
        Left-Right slice index for 3D volume.
    target_size : tuple of int
        Output spatial shape (rows, cols), default (128, 128).
    canal_depth_mm : float
        Distance (mm) to extend ROI posteriorly into the spinal canal (default 12.0 mm).
    vertical_margin_mm : float
        Margin (mm) to expand superiorly and inferiorly (default 6.0 mm).

    Returns
    -------
    np.ndarray
        Float32 array of shape (5, target_size[0], target_size[1]) in range [0.0, 1.0].
    """
    is_3d = ct_volume_or_slice.ndim == 3
    if is_3d:
        z_dim, y_dim, x_dim = ct_volume_or_slice.shape
        sx = float(spacing_xyz_mm[0]) if len(spacing_xyz_mm) >= 3 else 1.0
        sy = float(spacing_xyz_mm[1]) if len(spacing_xyz_mm) >= 3 else 1.0
        sz = float(spacing_xyz_mm[2]) if len(spacing_xyz_mm) >= 3 else 1.0
        mid_x = slice_x_index if slice_x_index is not None else x_dim // 2
        mid_x = int(np.clip(mid_x, 0, x_dim - 1))
        mid_sag_hu = ct_volume_or_slice[:, :, mid_x].astype(np.float32)
    else:
        mid_sag_hu = ct_volume_or_slice.astype(np.float32)
        sy = float(spacing_xyz_mm[0]) if len(spacing_xyz_mm) >= 2 else 1.0
        sz = float(spacing_xyz_mm[1]) if len(spacing_xyz_mm) >= 2 else 1.0
        sx = 1.0

    h_img, w_img = mid_sag_hu.shape
    b_mask = np.asarray(body_mask, dtype=bool)
    if b_mask.shape != (h_img, w_img):
        raise ValueError(f"body_mask shape {b_mask.shape} does not match sagittal slice shape {(h_img, w_img)}.")

    occupied_rows, occupied_cols = np.where(b_mask)
    if len(occupied_rows) == 0:
        # Fallback if empty mask
        r_min, r_max = h_img // 4, 3 * h_img // 4
        c_min, c_max = w_img // 4, 3 * w_img // 4
    else:
        r_min, r_max = int(occupied_rows.min()), int(occupied_rows.max())
        c_min, c_max = int(occupied_cols.min()), int(occupied_cols.max())

    # Bounding Box calculations in (row, col) = (Z, Y):
    # row = Z (inferior to superior), col = Y (anterior to posterior)
    r_pad = max(2, int(round(vertical_margin_mm / max(sz, 0.1))))
    c_pad_canal = max(4, int(round(canal_depth_mm / max(sy, 0.1))))
    c_pad_ant = max(2, int(round(3.0 / max(sy, 0.1))))

    r_start = max(0, r_min - r_pad)
    r_end = min(h_img, r_max + r_pad + 1)
    c_start = max(0, c_min - c_pad_ant)
    c_end = min(w_img, c_max + c_pad_canal + 1)

    # 1. Ch 0: Mid-sagittal Bone Window
    ch0_full = apply_ct_window(mid_sag_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    ch0 = _resize_2d(ch0_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 2 & 3. Ch 1 & Ch 2: Parasagittal slices (-4mm and +4mm)
    if is_3d:
        dx_vox = max(1, int(round(4.0 / max(sx, 0.1))))
        x_left = int(np.clip(mid_x - dx_vox, 0, x_dim - 1))
        x_right = int(np.clip(mid_x + dx_vox, 0, x_dim - 1))
        ch1_full = apply_ct_window(ct_volume_or_slice[:, :, x_left], BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2_full = apply_ct_window(ct_volume_or_slice[:, :, x_right], BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch1 = _resize_2d(ch1_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)
        ch2 = _resize_2d(ch2_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)
    else:
        # Fallback for 2D: Sobel gradient edges
        grad_r = np.abs(np.diff(ch0_full, axis=0, prepend=ch0_full[:1, :]))
        grad_c = np.abs(np.diff(ch0_full, axis=1, prepend=ch0_full[:, :1]))
        ch1 = _resize_2d(np.clip(grad_r * 5.0, 0.0, 1.0)[r_start:r_end, c_start:c_end], target_size, is_mask=False)
        ch2 = _resize_2d(np.clip(grad_c * 5.0, 0.0, 1.0)[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 4. Ch 3: Axial slice at upper 1/3 (or Soft-tissue window fallback)
    if is_3d and r_max > r_min:
        z_axial = int(np.clip(round(r_min + 0.67 * (r_max - r_min)), 0, z_dim - 1))
        ax_hu = ct_volume_or_slice[z_axial, :, :].astype(np.float32)
        ax_bone = apply_ct_window(ax_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        # Crop around vertebral body and canal in Y and X
        x_start = max(0, mid_x - int(round(25.0 / max(sx, 0.1))))
        x_end = min(x_dim, mid_x + int(round(25.0 / max(sx, 0.1))) + 1)
        ax_crop = ax_bone[c_start:c_end, x_start:x_end]
        ch3 = _resize_2d(ax_crop, target_size, is_mask=False)
    else:
        # 2D fallback: Soft-tissue window of mid-sagittal slice
        ch3_full = apply_ct_window(mid_sag_hu, SOFT_TISSUE_WINDOW_LEVEL, SOFT_TISSUE_WINDOW_WIDTH)
        ch3 = _resize_2d(ch3_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 5. Ch 4: Reference Mask: Body + Canal Reference Zone
    ref_mask = np.zeros((h_img, w_img), dtype=np.float32)
    ref_mask[b_mask] = 1.0
    # Add canal reference zone (columns from c_max to c_end across occupied rows)
    if len(occupied_rows) > 0:
        for r in range(r_min, r_max + 1):
            ref_mask[r, c_max:c_end] = 0.5  # 0.5 designates the canal intrusion zone
    ch4 = _resize_2d(ref_mask[r_start:r_end, c_start:c_end], target_size, is_mask=True)

    roi_5d = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
    return np.clip(roi_5d, 0.0, 1.0)


def extract_posterior_arch_roi_25d(
    ct_volume_or_slice: np.ndarray,
    posterior_arch_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, ...],
    *,
    body_mask: np.ndarray | None = None,
    slice_x_index: int | None = None,
    target_size: tuple[int, int] = (128, 128),
    margin_mm: float = 6.0,
) -> np.ndarray:
    """Extract 5-channel 2.5D Posterior Arch ROI covering spinous processes and tension band.

    Channels:
    - Ch 0: Mid-sagittal slice with Bone Window (WL=400, WW=1800).
    - Ch 1: Parasagittal slice at -4.0 mm (Bone Window) [fallback: horizontal gradient edge].
    - Ch 2: Parasagittal slice at +4.0 mm (Bone Window) [fallback: vertical gradient edge].
    - Ch 3: Mid-sagittal slice with Soft-Tissue Window (WL=50, WW=350) for ligament/interspinous gap.
    - Ch 4: Posterior Arch Mask (values in [0.0, 1.0]).

    Parameters
    ----------
    ct_volume_or_slice : np.ndarray
        3D CT volume (Z, Y, X) or 2D sagittal slice (Z, Y).
    posterior_arch_mask : np.ndarray
        2D binary mask of posterior arch structures.
    spacing_xyz_mm : tuple
        Voxel spacing (sx, sy, sz) in mm for 3D, or (sy, sz) for 2D.
    body_mask : np.ndarray, optional
        Vertebral body mask to locate arch if arch mask is empty.
    slice_x_index : int, optional
        Left-Right slice index for 3D volume.
    target_size : tuple of int
        Output spatial shape (rows, cols), default (128, 128).
    margin_mm : float
        Margin (mm) around bounding box (default 6.0 mm).

    Returns
    -------
    np.ndarray
        Float32 array of shape (5, target_size[0], target_size[1]) in range [0.0, 1.0].
    """
    is_3d = ct_volume_or_slice.ndim == 3
    if is_3d:
        z_dim, y_dim, x_dim = ct_volume_or_slice.shape
        sx = float(spacing_xyz_mm[0]) if len(spacing_xyz_mm) >= 3 else 1.0
        sy = float(spacing_xyz_mm[1]) if len(spacing_xyz_mm) >= 3 else 1.0
        sz = float(spacing_xyz_mm[2]) if len(spacing_xyz_mm) >= 3 else 1.0
        mid_x = slice_x_index if slice_x_index is not None else x_dim // 2
        mid_x = int(np.clip(mid_x, 0, x_dim - 1))
        mid_sag_hu = ct_volume_or_slice[:, :, mid_x].astype(np.float32)
    else:
        mid_sag_hu = ct_volume_or_slice.astype(np.float32)
        sy = float(spacing_xyz_mm[0]) if len(spacing_xyz_mm) >= 2 else 1.0
        sz = float(spacing_xyz_mm[1]) if len(spacing_xyz_mm) >= 2 else 1.0
        sx = 1.0

    h_img, w_img = mid_sag_hu.shape
    arch_mask = np.asarray(posterior_arch_mask, dtype=bool)
    if arch_mask.shape != (h_img, w_img):
        raise ValueError(f"posterior_arch_mask shape {arch_mask.shape} does not match sagittal slice {(h_img, w_img)}.")

    occupied_rows, occupied_cols = np.where(arch_mask)
    if len(occupied_rows) > 0:
        r_min, r_max = int(occupied_rows.min()), int(occupied_rows.max())
        c_min, c_max = int(occupied_cols.min()), int(occupied_cols.max())
    elif body_mask is not None and np.any(body_mask):
        b_rows, b_cols = np.where(body_mask)
        r_min, r_max = int(b_rows.min()), int(b_rows.max())
        b_width = b_cols.max() - b_cols.min()
        c_min = int(b_cols.max())
        c_max = min(w_img - 1, c_min + int(1.5 * b_width))
    else:
        r_min, r_max = h_img // 4, 3 * h_img // 4
        c_min, c_max = w_img // 2, w_img - 1

    r_pad = max(2, int(round(margin_mm / max(sz, 0.1))))
    c_pad = max(2, int(round(margin_mm / max(sy, 0.1))))

    r_start = max(0, r_min - r_pad)
    r_end = min(h_img, r_max + r_pad + 1)
    c_start = max(0, c_min - c_pad)
    c_end = min(w_img, c_max + c_pad + 1)

    # 1. Ch 0: Mid-sagittal Bone Window
    ch0_full = apply_ct_window(mid_sag_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    ch0 = _resize_2d(ch0_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 2 & 3. Ch 1 & Ch 2: Parasagittal slices (-4mm and +4mm)
    if is_3d:
        dx_vox = max(1, int(round(4.0 / max(sx, 0.1))))
        x_left = int(np.clip(mid_x - dx_vox, 0, x_dim - 1))
        x_right = int(np.clip(mid_x + dx_vox, 0, x_dim - 1))
        ch1_full = apply_ct_window(ct_volume_or_slice[:, :, x_left], BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2_full = apply_ct_window(ct_volume_or_slice[:, :, x_right], BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch1 = _resize_2d(ch1_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)
        ch2 = _resize_2d(ch2_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)
    else:
        grad_r = np.abs(np.diff(ch0_full, axis=0, prepend=ch0_full[:1, :]))
        grad_c = np.abs(np.diff(ch0_full, axis=1, prepend=ch0_full[:, :1]))
        ch1 = _resize_2d(np.clip(grad_r * 5.0, 0.0, 1.0)[r_start:r_end, c_start:c_end], target_size, is_mask=False)
        ch2 = _resize_2d(np.clip(grad_c * 5.0, 0.0, 1.0)[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 4. Ch 3: Mid-sagittal Soft-Tissue Window (for interspinous ligament & edema)
    ch3_full = apply_ct_window(mid_sag_hu, SOFT_TISSUE_WINDOW_LEVEL, SOFT_TISSUE_WINDOW_WIDTH)
    ch3 = _resize_2d(ch3_full[r_start:r_end, c_start:c_end], target_size, is_mask=False)

    # 5. Ch 4: Posterior Arch Mask
    arch_float = arch_mask.astype(np.float32)
    ch4 = _resize_2d(arch_float[r_start:r_end, c_start:c_end], target_size, is_mask=True)

    roi_5d = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
    return np.clip(roi_5d, 0.0, 1.0)
