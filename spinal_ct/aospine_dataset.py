"""2.5D Dual-ROI Data Pipeline for AO Spine Deep Learning Models (Phase 2).

Provides:
- apply_ct_window: Hounsfield Unit (HU) windowing with standard presets.
- extract_body_canal_roi_25d: 5-channel ROI for Body + Spinal Canal (shape (5, 128, 128)).
- extract_posterior_arch_roi_25d: 5-channel ROI for Posterior Arch (shape (5, 128, 128)).
"""
from typing import Any
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# Standard Radiologic CT Window Presets
BONE_WINDOW_LEVEL = 400.0
BONE_WINDOW_WIDTH = 1800.0

SOFT_TISSUE_WINDOW_LEVEL = 50.0
SOFT_TISSUE_WINDOW_WIDTH = 350.0

# Standard Fixed Physical FOV for AO Spine ROI Extraction (0.625 mm/pixel at 128x128)
DEFAULT_PHYSICAL_FOV_MM = 80.0


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


def resample_fixed_fov_2d(
    arr_2d: np.ndarray,
    center_rc: tuple[float, float],
    fov_mm: float,
    spacing_zy_mm: tuple[float, float],
    target_size: tuple[int, int] = (128, 128),
    pad_value: float = -1024.0,
    is_mask: bool = False,
) -> np.ndarray:
    """Resample an arbitrary 2D array around center_rc to a fixed physical square FOV (fov_mm x fov_mm).

    Preserves 100% of physical size and anatomical aspect ratio (isotropic resolution fov_mm / target_size).
    Out-of-bound pixels are cleanly padded with pad_value (-1024.0 HU for CT, 0.0 for masks).

    Parameters
    ----------
    arr_2d : np.ndarray
        2D source array of shape (H, W).
    center_rc : tuple of float
        (center_row, center_col) in source pixel coordinates.
    fov_mm : float
        Physical width/height of the square FOV in millimeters.
    spacing_zy_mm : tuple of float
        (spacing_row_mm, spacing_col_mm) physical spacing per pixel.
    target_size : tuple of int
        Output shape (rows, cols), default (128, 128).
    pad_value : float
        Fill value for out-of-bounds coordinates (default -1024.0 HU).
    is_mask : bool
        If True, use nearest neighbor interpolation to preserve sharp mask labels;
        otherwise use bilinear interpolation.

    Returns
    -------
    np.ndarray
        Array of shape target_size in float32.
    """
    H, W = arr_2d.shape
    target_h, target_w = target_size
    sr, sc = float(spacing_zy_mm[0]), float(spacing_zy_mm[1])
    cr, cc = float(center_rc[0]), float(center_rc[1])

    grid_r = torch.linspace(
        -fov_mm / 2.0 + (fov_mm / (2.0 * target_h)),
        fov_mm / 2.0 - (fov_mm / (2.0 * target_h)),
        target_h,
        dtype=torch.float32,
    )
    grid_c = torch.linspace(
        -fov_mm / 2.0 + (fov_mm / (2.0 * target_w)),
        fov_mm / 2.0 - (fov_mm / (2.0 * target_w)),
        target_w,
        dtype=torch.float32,
    )
    gr, gc = torch.meshgrid(grid_r, grid_c, indexing='ij')

    r_src = cr + gr / max(sr, 1e-4)
    c_src = cc + gc / max(sc, 1e-4)

    norm_r = 2.0 * r_src / max(H - 1, 1) - 1.0 if H > 1 else torch.zeros_like(r_src)
    norm_c = 2.0 * c_src / max(W - 1, 1) - 1.0 if W > 1 else torch.zeros_like(c_src)

    grid = torch.stack([norm_c, norm_r], dim=-1).unsqueeze(0)  # (1, target_h, target_w, 2)

    t_in = torch.from_numpy(arr_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mode = 'nearest' if is_mask else 'bilinear'

    if pad_value == 0.0:
        sampled = F.grid_sample(t_in, grid, mode=mode, padding_mode='zeros', align_corners=True)
    else:
        shifted = t_in - pad_value
        sampled = F.grid_sample(shifted, grid, mode=mode, padding_mode='zeros', align_corners=True) + pad_value

    out = sampled.squeeze(0).squeeze(0).numpy().astype(np.float32)
    return out


def _resize_2d(arr: np.ndarray, target_size: tuple[int, int] = (128, 128), is_mask: bool = False) -> np.ndarray:
    """Resize 2D float32 array to target_size (rows, cols) using PIL (legacy helper)."""
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
    parasagittal_slices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None = None,
    fov_mm: float = DEFAULT_PHYSICAL_FOV_MM,
    posterior_shift_mm: float = 6.0,
) -> np.ndarray:
    """Extract 5-channel 2.5D Body + Canal ROI with Fixed Physical FOV (Isotropic 0.625 mm/px).

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
        Distance (mm) to extend reference mask posteriorly into spinal canal (default 12.0 mm).
    vertical_margin_mm : float
        Backward compatibility argument.
    parasagittal_slices : list or tuple of np.ndarray, optional
        Precomputed parasagittal slices for 2D inputs.
    fov_mm : float
        Fixed physical field-of-view in millimeters (default 80.0 mm).
    posterior_shift_mm : float
        Shift of window center posteriorly toward the spinal canal (default 6.0 mm).

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
    if len(occupied_rows) > 0:
        r_c = 0.5 * float(occupied_rows.min() + occupied_rows.max())
        c_c = 0.5 * float(occupied_cols.min() + occupied_cols.max())
        # Posterior shift to cover both vertebral body and spinal canal
        c_c += posterior_shift_mm / max(sy, 1e-4)
    else:
        r_c = h_img / 2.0
        c_c = w_img / 2.0

    # 1. Ch 0: Mid-sagittal Bone Window
    ch0_hu = resample_fixed_fov_2d(
        mid_sag_hu, (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
    )
    ch0 = apply_ct_window(ch0_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)

    # 2 & 3. Ch 1 & Ch 2: Parasagittal slices (-4mm and +4mm)
    if is_3d:
        dx_vox = max(1, int(round(4.0 / max(sx, 0.1))))
        x_left = int(np.clip(mid_x - dx_vox, 0, x_dim - 1))
        x_right = int(np.clip(mid_x + dx_vox, 0, x_dim - 1))
        ch1_hu = resample_fixed_fov_2d(
            ct_volume_or_slice[:, :, x_left], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch2_hu = resample_fixed_fov_2d(
            ct_volume_or_slice[:, :, x_right], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch1 = apply_ct_window(ch1_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2 = apply_ct_window(ch2_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    elif parasagittal_slices is not None and len(parasagittal_slices) >= 2:
        ch1_hu = resample_fixed_fov_2d(
            parasagittal_slices[0], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch2_hu = resample_fixed_fov_2d(
            parasagittal_slices[1], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch1 = apply_ct_window(ch1_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2 = apply_ct_window(ch2_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    else:
        # Fallback for 2D: Sobel gradient edges on Ch 0
        grad_r = np.abs(np.diff(ch0, axis=0, prepend=ch0[:1, :]))
        grad_c = np.abs(np.diff(ch0, axis=1, prepend=ch0[:, :1]))
        ch1 = np.clip(grad_r * 5.0, 0.0, 1.0).astype(np.float32)
        ch2 = np.clip(grad_c * 5.0, 0.0, 1.0).astype(np.float32)

    # 4. Ch 3: Axial slice at upper 1/3 (or Soft-tissue window fallback)
    if is_3d and len(occupied_rows) > 0:
        r_min, r_max = int(occupied_rows.min()), int(occupied_rows.max())
        z_axial = int(np.clip(round(r_min + 0.67 * (r_max - r_min)), 0, z_dim - 1))
        ax_hu = ct_volume_or_slice[z_axial, :, :].astype(np.float32)
        # In axial view: row is Y (anterior-posterior, spacing sy), col is X (left-right, spacing sx)
        ax_hu_res = resample_fixed_fov_2d(
            ax_hu, (c_c, float(mid_x)), fov_mm, (sy, sx), target_size, pad_value=-1024.0, is_mask=False
        )
        ch3 = apply_ct_window(ax_hu_res, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    else:
        # 2D fallback: Soft-tissue window of mid-sagittal patch
        ch3 = apply_ct_window(ch0_hu, SOFT_TISSUE_WINDOW_LEVEL, SOFT_TISSUE_WINDOW_WIDTH)

    # 5. Ch 4: Reference Mask: Body + Canal Reference Zone
    ref_mask = np.zeros((h_img, w_img), dtype=np.float32)
    ref_mask[b_mask] = 1.0
    if len(occupied_rows) > 0:
        c_max = int(occupied_cols.max())
        c_end_canal = min(w_img, c_max + max(4, int(round(canal_depth_mm / max(sy, 0.1)))) + 1)
        for r in range(int(occupied_rows.min()), int(occupied_rows.max()) + 1):
            ref_mask[r, c_max:c_end_canal] = 0.5  # 0.5 designates the canal intrusion zone

    ch4 = resample_fixed_fov_2d(
        ref_mask, (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=0.0, is_mask=True
    )

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
    parasagittal_slices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None = None,
    fov_mm: float = DEFAULT_PHYSICAL_FOV_MM,
) -> np.ndarray:
    """Extract 5-channel 2.5D Posterior Arch ROI with Fixed Physical FOV (Isotropic 0.625 mm/px).

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
        Backward compatibility argument.
    parasagittal_slices : list or tuple of np.ndarray, optional
        Precomputed parasagittal slices for 2D inputs.
    fov_mm : float
        Fixed physical field-of-view in millimeters (default 80.0 mm).

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
        r_c = 0.5 * float(occupied_rows.min() + occupied_rows.max())
        c_c = 0.5 * float(occupied_cols.min() + occupied_cols.max())
    elif body_mask is not None and np.any(body_mask):
        b_rows, b_cols = np.where(body_mask)
        r_c = 0.5 * float(b_rows.min() + b_rows.max())
        b_width_px = float(b_cols.max() - b_cols.min())
        # Center in posterior arch region (~15-20mm posterior to body)
        c_c = float(b_cols.max()) + 0.75 * b_width_px + (15.0 / max(sy, 1e-4))
    else:
        r_c = h_img / 2.0
        c_c = 0.75 * w_img

    # 1. Ch 0: Mid-sagittal Bone Window
    ch0_hu = resample_fixed_fov_2d(
        mid_sag_hu, (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
    )
    ch0 = apply_ct_window(ch0_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)

    # 2 & 3. Ch 1 & Ch 2: Parasagittal slices (-4mm and +4mm)
    if is_3d:
        dx_vox = max(1, int(round(4.0 / max(sx, 0.1))))
        x_left = int(np.clip(mid_x - dx_vox, 0, x_dim - 1))
        x_right = int(np.clip(mid_x + dx_vox, 0, x_dim - 1))
        ch1_hu = resample_fixed_fov_2d(
            ct_volume_or_slice[:, :, x_left], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch2_hu = resample_fixed_fov_2d(
            ct_volume_or_slice[:, :, x_right], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch1 = apply_ct_window(ch1_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2 = apply_ct_window(ch2_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    elif parasagittal_slices is not None and len(parasagittal_slices) >= 2:
        ch1_hu = resample_fixed_fov_2d(
            parasagittal_slices[0], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch2_hu = resample_fixed_fov_2d(
            parasagittal_slices[1], (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=-1024.0, is_mask=False
        )
        ch1 = apply_ct_window(ch1_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        ch2 = apply_ct_window(ch2_hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
    else:
        # Fallback for 2D: Sobel gradient edges
        grad_r = np.abs(np.diff(ch0, axis=0, prepend=ch0[:1, :]))
        grad_c = np.abs(np.diff(ch0, axis=1, prepend=ch0[:, :1]))
        ch1 = np.clip(grad_r * 5.0, 0.0, 1.0).astype(np.float32)
        ch2 = np.clip(grad_c * 5.0, 0.0, 1.0).astype(np.float32)

    # 4. Ch 3: Mid-sagittal Soft-Tissue Window (for interspinous ligament & edema)
    ch3 = apply_ct_window(ch0_hu, SOFT_TISSUE_WINDOW_LEVEL, SOFT_TISSUE_WINDOW_WIDTH)

    # 5. Ch 4: Posterior Arch Mask
    arch_float = arch_mask.astype(np.float32)
    ch4 = resample_fixed_fov_2d(
        arch_float, (r_c, c_c), fov_mm, (sz, sy), target_size, pad_value=0.0, is_mask=True
    )

    roi_5d = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
    return np.clip(roi_5d, 0.0, 1.0)
