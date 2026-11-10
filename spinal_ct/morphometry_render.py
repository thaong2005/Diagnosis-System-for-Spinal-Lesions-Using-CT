"""Shared overlays for body review, whole-spine side-by-side MPR, and exported QC."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, Polygon

from spinal_ct.blob_analysis import analyze_blob, normalize_region
from spinal_ct.localization import LEVELS
from spinal_ct.overlays import _boundary_anchor, _label_positions


def get_aospine_severity_color_and_label(
    level_name: str,
    info: dict[str, Any] | str | None = None,
    collapse_ratio: float | None = None,
) -> tuple[str, str]:
    """Return (color_hex, formatted_label) based on AO Spine classification and collapse degree.

    - Normal / A0 (collapse < 15%): Green (#4cd964), label: e.g. 'L1'
    - Mild (A1, or collapse 15% - 25%): Yellow (#ffd11a), label: e.g. 'L2 -23% | A1'
    - Moderate (A2, A3, or collapse 26% - 40%): Orange (#ff9933), label: e.g. 'L4 -38% | A3'
    - Severe / Unstable (A4, B1, B2, B3, C, or collapse > 40%): Bright Red (#ff4d4d), label: e.g. 'T11 -55% | A4'
    """
    if isinstance(info, str):
        info = {'predicted_class': info, 'collapse_ratio': collapse_ratio}

    if not info:
        return '#4cd964', level_name

    cls_name = str(info.get('predicted_class') or info.get('class') or info.get('aospine_class') or 'Normal').strip()

    # Calculate collapse percentage
    collapse_pct = None
    if 'collapse_pct' in info and info['collapse_pct'] is not None:
        collapse_pct = float(info['collapse_pct'])
    elif 'collapse_ratio' in info and info['collapse_ratio'] is not None:
        r = float(info['collapse_ratio'])
        collapse_pct = r if r > 1.0 else r * 100.0
    elif 'relative_collapse_ratio' in info and info['relative_collapse_ratio'] is not None:
        r = float(info['relative_collapse_ratio'])
        collapse_pct = max(0.0, (1.0 - r) * 100.0)

    # Determine severity tier
    if cls_name in ('A4', 'B1', 'B2', 'B3', 'C') or (collapse_pct is not None and collapse_pct > 40.0):
        color = '#ff4d4d'  # Bright Red
        if collapse_pct and collapse_pct >= 10.0:
            label = f"{level_name} -{collapse_pct:.0f}% | {cls_name}"
        else:
            label = f"{level_name} | {cls_name}"
    elif cls_name in ('A2', 'A3') or (collapse_pct is not None and 25.0 < collapse_pct <= 40.0):
        color = '#ff9933'  # Orange
        if collapse_pct and collapse_pct >= 10.0:
            label = f"{level_name} -{collapse_pct:.0f}% | {cls_name}"
        else:
            label = f"{level_name} | {cls_name}"
    elif cls_name in ('A1',) or (collapse_pct is not None and 15.0 <= collapse_pct <= 25.0):
        color = '#ffd11a'  # Yellow
        if collapse_pct and collapse_pct >= 10.0:
            label = f"{level_name} -{collapse_pct:.0f}% | {cls_name}"
        else:
            label = f"{level_name} | {cls_name}"
    else:
        color = '#4cd964'  # Green
        label = level_name

    return color, label


def _draw_clean_spine(ax, ct_slice: np.ndarray, spacing: tuple[float, float]) -> None:
    """Render a clean reference Sagittal CT slice without any overlays."""
    sy_sp, sz_sp = spacing
    h, w = ct_slice.shape
    extent = [-sy_sp / 2, (w - 0.5) * sy_sp, -sz_sp / 2, (h - 0.5) * sz_sp]
    ax.set_facecolor('#1e1e2e')
    ax.imshow(ct_slice, origin='lower', extent=extent, cmap='gray', vmin=-500, vmax=1300)
    ax.set_title('Clean Reference CT (Native)', color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')
    ax.set_aspect('equal')


def _draw_annotated_spine(
    ax,
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    spacing: tuple[float, float],
    vertebrae_info: dict[str | int, dict[str, Any]] | list[dict[str, Any]] | None = None,
) -> list[tuple]:
    """Render full-spine Sagittal MPR with sharp color-coded contours and anterior gutter labels."""
    sy_sp, sz_sp = spacing
    h, w = ct_slice.shape
    extent = [-sy_sp / 2, (w - 0.5) * sy_sp, -sz_sp / 2, (h - 0.5) * sz_sp]
    ax.set_facecolor('#1e1e2e')
    ax.imshow(ct_slice, origin='lower', extent=extent, cmap='gray', vmin=-500, vmax=1300)
    ax.set_title('Annotated Spine Overlay (AO Spine)', color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')
    ax.set_aspect('equal')

    # Normalize vertebrae info map
    info_map: dict[str, dict[str, Any]] = {}
    if isinstance(vertebrae_info, dict):
        for k, v in vertebrae_info.items():
            info_map[str(k).upper()] = v
    elif isinstance(vertebrae_info, list):
        for item in vertebrae_info:
            lvl = item.get('level')
            if lvl:
                info_map[str(lvl).upper()] = item

    unique_labels = sorted([int(x) for x in np.unique(mask_slice) if x > 0])
    anchors = []

    x_coords = np.arange(-1, w + 1) * sy_sp
    y_coords = np.arange(-1, h + 1) * sz_sp

    for label_id in unique_labels:
        if 1 <= label_id <= len(LEVELS):
            lvl_name = LEVELS[label_id - 1]
        else:
            lvl_name = f"V{label_id}"

        info = info_map.get(lvl_name.upper()) or info_map.get(str(label_id))
        color, label_text = get_aospine_severity_color_and_label(lvl_name, info)

        region = (mask_slice == label_id)
        if not region.any():
            continue

        # Draw sharp contour enclosing both vertebral body and posterior arch
        cs = ax.contour(
            x_coords, y_coords,
            np.pad(region.astype(np.uint8), 1),
            levels=[0.5],
            colors=[color],
            linewidths=1.2,
            antialiased=True,
        )
        cs.set_path_effects([
            patheffects.Stroke(linewidth=2.4, foreground='#101018', alpha=0.9),
            patheffects.Normal(),
        ])

        # Find anterior boundary anchor
        try:
            x_anc, y_anc = _boundary_anchor(region, spacing)
        except Exception:
            rows, cols = np.nonzero(region)
            x_anc = float(cols.min()) * sy_sp
            y_anc = float(rows.mean()) * sz_sp

        anchors.append((y_anc, x_anc, lvl_name, label_text, color, info))

        # Check for retropulsed bone fragment into spinal canal (delta_posterior_mm > 2.0)
        dp = 0.0
        if info:
            dp = float(info.get('delta_posterior_mm', 0.0))
        if dp > 2.0:
            rows, cols = np.nonzero(region)
            if len(cols):
                p_col = cols.max()
                p_rows = rows[cols >= max(0, p_col - 3)]
                p_row = int(np.median(p_rows)) if len(p_rows) else int(rows.mean())
                p_x, p_y = float(p_col) * sy_sp, float(p_row) * sz_sp
                ax.plot(p_x, p_y, 'x', color='#ff3333', markersize=8, markeredgewidth=2, zorder=6)
                ax.annotate(
                    f'Δ_post {dp:.1f}mm', (p_x, p_y),
                    color='#ff4d4d', xytext=(6, 6), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#2b0000', edgecolor='#ff4d4d', alpha=0.9),
                    zorder=7,
                )

    if not anchors:
        return []

    # Position anterior labels cleanly without overlapping
    anchors.sort(key=lambda a: a[0])
    left_x = -sy_sp / 2
    lower_y = -sz_sp / 2
    upper_y = (h - 0.5) * sz_sp
    gap = min(14.0, (upper_y - lower_y) / max(len(anchors), 1))
    y_positions = _label_positions([a[0] for a in anchors], lower_y, upper_y, gap)

    gutter = max(30.0, (w * sy_sp) * 0.22)
    ax.set_xlim(left_x - gutter, (w - 0.5) * sy_sp)

    for (y_anc, x_anc, lvl, txt, col, inf), y_pos in zip(anchors, y_positions):
        ax.annotate(
            txt,
            xy=(x_anc, y_anc),
            xytext=(left_x - 4.0, y_pos),
            ha='right', va='center',
            fontsize=8.5, fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#1e1e2e', edgecolor=col, alpha=0.95, linewidth=1.1),
            arrowprops=dict(arrowstyle='->', color=col, lw=0.9, shrinkA=3, shrinkB=2),
            zorder=8,
        )

    return anchors


def render_side_by_side_spine_mpr(
    ct_image: sitk.Image | np.ndarray,
    labels: sitk.Image | np.ndarray,
    vertebrae_info: dict[str | int, dict[str, Any]] | list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    dpi: int = 150,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate medical PACS-grade Side-by-Side Sagittal MPR comparison.

    Left View: Annotated Spine Overlay with AO Spine color coding & collapse metrics.
    Right View: Clean Native CT Reference (synchronized zoom/pan, sharex=True, sharey=True).

    Outputs (when output_dir is provided):
    - levels_mpr.png (Annotated overlay)
    - mpr.png (Clean reference CT)
    - comparison_mpr.png (Side-by-side combined)
    """
    if isinstance(vertebrae_info, (str, Path)) and output_dir is None:
        output_dir = vertebrae_info
        vertebrae_info = kwargs.get('vertebrae_info', None)
    if 'output_dir' in kwargs and output_dir is None:
        output_dir = kwargs['output_dir']
    # 1. Extract 2D sagittal slice and spacing
    if isinstance(ct_image, sitk.Image):
        ct_lps = sitk.DICOMOrient(ct_image, 'LPS')
        if isinstance(labels, sitk.Image):
            if labels.GetSize() != ct_image.GetSize():
                labels = sitk.Resample(labels, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            mask_lps = sitk.DICOMOrient(labels, 'LPS')
            mask_arr = sitk.GetArrayFromImage(mask_lps)
        else:
            mask_arr = np.asarray(labels)

        ct_arr = sitk.GetArrayFromImage(ct_lps)  # Shape (Z, Y, X)
        sp = ct_lps.GetSpacing()  # (sx, sy, sz)

        counts = (mask_arr > 0).sum(axis=(0, 1))
        x_idx = int(counts.argmax()) if counts.any() else ct_arr.shape[2] // 2
        ct_slice = ct_arr[:, :, x_idx].astype(float)
        mask_slice = mask_arr[:, :, x_idx]
        slice_spacing = (float(sp[1]), float(sp[2]))  # (sy, sz)
    elif isinstance(ct_image, np.ndarray):
        if ct_image.ndim == 3:
            mask_arr = np.asarray(labels) if labels is not None else np.zeros_like(ct_image)
            counts = (mask_arr > 0).sum(axis=(0, 1))
            x_idx = int(counts.argmax()) if len(counts) and counts.any() else ct_image.shape[2] // 2
            ct_slice = ct_image[:, :, x_idx].astype(float)
            mask_slice = mask_arr[:, :, x_idx]
        else:
            ct_slice = ct_image.astype(float)
            mask_slice = np.asarray(labels) if labels is not None else np.zeros_like(ct_slice)
        slice_spacing = spacing
    else:
        raise TypeError(f"Unsupported ct_image type: {type(ct_image)}")

    # 2. Render Side-by-Side Figure
    fig = Figure(figsize=(14, 10), facecolor='#1e1e2e', layout='constrained')
    FigureCanvasAgg(fig)
    ax_left, ax_right = fig.subplots(1, 2, sharex=True, sharey=True)

    _draw_annotated_spine(ax_left, ct_slice, mask_slice, slice_spacing, vertebrae_info)
    _draw_clean_spine(ax_right, ct_slice, slice_spacing)

    result: dict[str, Any] = {
        'figure': fig,
        'levels_mpr_path': None,
        'mpr_path': None,
        'comparison_mpr_path': None,
    }

    # 3. Export to files if output_dir provided
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        comp_path = out_dir / 'comparison_mpr.png'
        fig.savefig(comp_path, dpi=dpi, facecolor='#1e1e2e', bbox_inches='tight')
        result['comparison_mpr_path'] = comp_path

        # Single panel levels_mpr.png
        fig_left = Figure(figsize=(8, 10), facecolor='#1e1e2e', layout='constrained')
        FigureCanvasAgg(fig_left)
        ax_lev = fig_left.subplots()
        _draw_annotated_spine(ax_lev, ct_slice, mask_slice, slice_spacing, vertebrae_info)
        lev_path = out_dir / 'levels_mpr.png'
        fig_left.savefig(lev_path, dpi=dpi, facecolor='#1e1e2e', bbox_inches='tight')
        result['levels_mpr_path'] = lev_path

        # Single panel mpr.png
        fig_right = Figure(figsize=(8, 10), facecolor='#1e1e2e', layout='constrained')
        FigureCanvasAgg(fig_right)
        ax_cln = fig_right.subplots()
        _draw_clean_spine(ax_cln, ct_slice, slice_spacing)
        mpr_path = out_dir / 'mpr.png'
        fig_right.savefig(mpr_path, dpi=dpi, facecolor='#1e1e2e', bbox_inches='tight')
        result['mpr_path'] = mpr_path

    return result


def draw_body(
    ax,
    session,
    *,
    show_points: bool = False,
    show_blobs: bool = True,
    show_aospine: bool = True,
    show_mask: bool = True,
    show_landmarks: bool = True,
    dark_theme: bool = True,
) -> None:
    """Draw annotated vertebral body ROI on clinical dark background."""
    sx, sy = session.plane.spacing_xy_mm
    h, w = session.mask.shape
    extent = [-sx / 2, (w - 0.5) * sx, -sy / 2, (h - 0.5) * sy]
    lower, upper = np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]])

    bg_color = '#1e1e2e' if dark_theme else '#252525'
    ax.set_facecolor(bg_color)
    ax.imshow(session.ct, origin='lower', extent=extent, cmap='gray', vmin=-500, vmax=1500)

    if session.hint.any() and show_mask:
        ax.contour(session.hint, levels=[0.5], colors=['#79b7e3'], linewidths=0.7,
                   origin='lower', extent=extent, linestyles='dashed')

    if session.mask.any() and show_mask:
        x = np.arange(-1, w + 1) * sx
        y = np.arange(-1, h + 1) * sy
        ax.contour(x, y, np.pad(session.mask, 1), levels=[0.5], colors=['#3bf4b0'], linewidths=1.3)
        features = analyze_blob(session.mask, (sx, sy))
        ellipse = features['ellipse']
        ax.add_patch(Ellipse(features['centroid_xy_mm'], ellipse['major_length_mm'], ellipse['minor_length_mm'],
                             angle=ellipse['angle_deg'], fill=False, edgecolor='#ffb454', linewidth=1))
        ax.add_patch(Polygon(features['oriented_box']['corners_xy_mm'], fill=False, edgecolor='#f7e36d', linewidth=1))
        centre = np.array(features['centroid_xy_mm'])
        axes_vec = np.array(features['oriented_box']['axes_xy'])
        ellipse_radius = np.sqrt((axes_vec**2) @ (np.array([ellipse['major_length_mm'], ellipse['minor_length_mm']]) / 2)**2)
        corners = np.array(features['oriented_box']['corners_xy_mm'])
        lower = np.minimum(lower, np.minimum(corners.min(0), centre - ellipse_radius))
        upper = np.maximum(upper, np.maximum(corners.max(0), centre + ellipse_radius))

    # AO Spine Overlays: Posterior arch, canal line, retropulsed bone patch, spinous tip
    if show_aospine:
        x = np.arange(-1, w + 1) * sx
        y = np.arange(-1, h + 1) * sy
        if session.posterior_arch_mask is not None and session.posterior_arch_mask.any():
            ax.contour(x, y, np.pad(session.posterior_arch_mask, 1), levels=[0.5],
                       colors=['#00bcd4'], linewidths=0.9, linestyles='dotted')
        if session.spinous_tip and session.spinous_tip.get('tip_xy_mm'):
            tip_pt = session.spinous_tip['tip_xy_mm']
            ax.plot(tip_pt[0], tip_pt[1], '*', color='#00e5ff', markersize=9,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=6)
            ax.annotate('SP Tip', tip_pt, color='#00e5ff', xytext=(5, -5),
                        textcoords='offset points', fontsize=8, fontweight='bold')
        if session.retropulsion:
            ret = session.retropulsion
            canal_pts = ret.get('canal_line_points')
            if canal_pts and len(canal_pts) == 2:
                p_low, p_up = canal_pts
                ax.plot([p_low[0], p_up[0]], [p_low[1], p_up[1]], '--',
                        color='#26a69a', linewidth=1.5, alpha=0.85, zorder=4)
            delta_pw = ret.get('delta_posterior_mm', 0.0)
            bone_mask = ret.get('retropulsed_bone_mask')
            if delta_pw > 2.0 and bone_mask is not None and bone_mask.any():
                ax.contour(x, y, np.pad(bone_mask, 1), levels=[0.5],
                           colors=['#ff3333'], linewidths=1.8, zorder=5)
                max_pt = ret.get('max_retropulsion_point_xy_mm')
                if max_pt:
                    ax.plot(max_pt[0], max_pt[1], 'x', color='#ff3333', markersize=8, markeredgewidth=2, zorder=6)
                    ax.annotate(f'Δ_post {delta_pw:.1f}mm', max_pt, color='#ff3333', xytext=(6, 6),
                                textcoords='offset points', fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#2b0000', edgecolor='#ff3333', alpha=0.8))

    if show_landmarks:
        for i, point in enumerate(session.landmarks):
            ax.plot(*point, 'o', color='#f77aff', markersize=4)
            ax.annotate(str(i + 1), point, color='#f77aff', xytext=(4, 4), textcoords='offset points')

    blobs = (session.candidates or {}).get('candidates', []) if show_blobs else []
    for i, blob in enumerate(blobs):
        ax.add_patch(Circle(blob['centre_xy_mm'], blob['display_radius_mm'], fill=False,
                            edgecolor='#77dbff', alpha=0.6, linestyle=':'))
        ax.annotate(f'B{i+1}', blob['centre_xy_mm'], color='#77dbff', fontsize=8)

    if show_points and session.result:
        points = np.asarray(session.result['monte_carlo_area']['preview_points_xy_mm_hit'])
        ax.scatter(points[:, 0], points[:, 1], s=2, c=np.where(points[:, 2], '#3bf4b0', '#ff6060'), alpha=0.5)

    aospine_title = ''
    has_landmarks = (session.landmarks is not None and len(session.landmarks) == 6)
    has_override = bool(getattr(session, 'doctor_confirmed_class', None))
    if show_aospine and session.aospine and (has_landmarks or has_override):
        ao_cls = getattr(session, 'doctor_confirmed_class', None) or session.aospine.get('predicted_class', '')
        ao_conf = session.aospine.get('confidence', 0.0)
        ao_rank = session.aospine.get('rank', 1)
        ao_sec = f"+{session.aospine['secondary_subtype']}" if session.aospine.get('secondary_subtype') else ''
        aospine_title = f" | AO: {ao_cls}{ao_sec} ({ao_conf:.0%}) #{ao_rank}"
    elif show_aospine and session.aospine and not has_landmarks and not has_override:
        aospine_title = " | AO: Chờ đo mốc"

    title_text = f"{session.level} · {'Đã xác nhận' if session.reviewed else 'Vùng nháp'}{aospine_title}"
    title_color = '#e0e0e0' if dark_theme else '#233c4b'
    ax.set_title(title_text, color=title_color, fontsize=10, fontweight='bold')
    ax.set_xlim([lower[0], upper[0]])
    ax.set_ylim([lower[1], upper[1]])
    ax.set_aspect('equal')
    ax.axis('off')


def draw_clean_reference(ax, session, *, dark_theme: bool = True) -> None:
    """Render a clean native CT slice without any overlays for direct side-by-side comparison."""
    bg_color = '#1e1e2e' if dark_theme else '#252525'
    ax.set_facecolor(bg_color)
    if session is None or session.ct is None:
        ax.axis('off')
        return

    sx, sy = session.plane.spacing_xy_mm
    h, w = session.mask.shape if session.mask is not None else session.ct.shape
    extent = [-sx / 2, (w - 0.5) * sx, -sy / 2, (h - 0.5) * sy]

    ax.imshow(session.ct, origin='lower', extent=extent, cmap='gray', vmin=-500, vmax=1500)
    title_col = '#e0e0e0' if dark_theme else '#233c4b'
    ax.set_title('CT gốc tham chiếu (Clean Reference)', color=title_col, fontsize=10, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')


def draw_normalized(ax, session) -> None:
    """Retained for backward compatibility."""
    if not session.mask.any():
        ax.text(0.5, 0.5, 'Vẽ vùng thân đốt để chuẩn hóa', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    normalized = normalize_region(session.ct, session.mask, session.plane.spacing_xy_mm)
    ax.imshow(normalized['ct'], origin='lower', cmap='gray', vmin=-500, vmax=1500)
    ax.contour(normalized['mask'], levels=[0.5], colors=['#3bf4b0'], linewidths=1)
    ax.set(title='256 × 256 · giữ tỷ lệ', xlabel='Pixel', ylabel='Pixel')
    ax.set_aspect('equal')


def draw_convergence(ax, report: dict[str, Any] | None, *, dark_theme: bool = True) -> None:
    """Draw Monte Carlo area convergence curve on dark clinical background."""
    bg_color = '#1e1e2e' if dark_theme else '#ffffff'
    text_color = '#e0e0e0' if dark_theme else '#233c4b'
    ax.set_facecolor(bg_color)

    if not report or 'monte_carlo_area' not in report:
        ax.text(0.5, 0.5, 'Chưa có kết quả đo MC', ha='center', va='center',
                transform=ax.transAxes, color='#888899', fontsize=10)
        ax.axis('off')
        return

    curve = report['monte_carlo_area'].get('convergence', [])
    if not curve:
        ax.text(0.5, 0.5, 'Chưa có dữ liệu hội tụ MC', ha='center', va='center',
                transform=ax.transAxes, color='#888899', fontsize=10)
        ax.axis('off')
        return

    samples = [p['samples'] for p in curve]
    areas = [p['area_mm2'] for p in curve]
    ax.semilogx(samples, areas, 'o-', color='#3bf4b0', markersize=3.5, linewidth=1.2, label='MC')
    ax.axhline(report['features']['area_mm2'], color='#ff9933', linestyle='--', label='Đếm pixel')

    elbow = report['monte_carlo_area'].get('elbow')
    if elbow:
        el_x, el_y = elbow['samples'], elbow['area_mm2']
        ax.plot(el_x, el_y, 'o', color='#ff4d4d', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                label=f"Điểm Cùi chỏ ({el_y:.1f} mm²)")
        y_min, y_max = min(areas), max(areas)
        offset_y = -32 if (el_y - y_min) / (y_max - y_min + 1e-12) > 0.7 else 28
        va = 'top' if offset_y < 0 else 'bottom'
        ax.annotate(
            f"{el_y:.1f} mm²\n(N={el_x:,})",
            xy=(el_x, el_y), xytext=(0, offset_y), textcoords='offset points',
            ha='center', va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2b0000', edgecolor='#ff4d4d', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#ff4d4d', lw=1.5),
            fontsize=8, fontweight='bold', color='#ffb4b4',
        )

    ax.set_title('Hội tụ diện tích trên mask', color=text_color, fontsize=10, fontweight='bold')
    ax.set_xlabel('Số điểm', color=text_color, fontsize=9)
    ax.set_ylabel('Diện tích (mm²)', color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.legend(loc='upper right', fontsize=8,
              facecolor='#252538', edgecolor='#444455', labelcolor=text_color)
    ax.grid(True, linestyle=':', alpha=0.3, color='#555566')


def export_preview(session, report: dict[str, Any] | None, path: str | Path) -> None:
    """Export preview image with Annotated, Clean Reference CT, and Convergence."""
    figure = Figure(figsize=(15, 5), layout='constrained', facecolor='#1e1e2e')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    draw_body(axes[0], session, show_points=True, show_blobs=False, dark_theme=True)
    draw_clean_reference(axes[1], session, dark_theme=True)
    draw_convergence(axes[2], report, dark_theme=True)
    figure.savefig(path, dpi=130, facecolor='#1e1e2e')

    if (session.candidates or {}).get('candidates'):
        blob_figure = Figure(figsize=(7, 7), layout='constrained', facecolor='#1e1e2e')
        FigureCanvasAgg(blob_figure)
        draw_body(blob_figure.subplots(), session, dark_theme=True)
        blob_figure.savefig(Path(path).with_name('blob_candidates.png'), dpi=130, facecolor='#1e1e2e')
