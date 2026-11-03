"""Shared overlays for body review and exported QC (no pyplot global backend)."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Ellipse, Polygon, Circle
from spinal_ct.blob_analysis import analyze_blob, normalize_region


def draw_body(ax, session, *, show_points=False, show_blobs=True):
    sx, sy = session.plane.spacing_xy_mm
    h, w = session.mask.shape
    extent = [-sx/2, (w-.5)*sx, -sy/2, (h-.5)*sy]
    lower, upper = np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]])
    ax.set_facecolor('#252525')
    ax.imshow(session.ct, origin='lower', extent=extent, cmap='gray', vmin=-500, vmax=1500)
    if session.hint.any():
        ax.contour(session.hint, levels=[.5], colors=['#79b7e3'], linewidths=.7,
                   origin='lower', extent=extent, linestyles='dashed')
    if session.mask.any():
        # Pad so boundaries close even if the region touches the edge.
        x = np.arange(-1, w+1)*sx
        y = np.arange(-1, h+1)*sy
        ax.contour(x, y, np.pad(session.mask, 1), levels=[.5], colors=['#3bf4b0'], linewidths=1.3)
        features = analyze_blob(session.mask, (sx, sy))
        ellipse = features['ellipse']
        ax.add_patch(Ellipse(features['centroid_xy_mm'], ellipse['major_length_mm'], ellipse['minor_length_mm'],
                             angle=ellipse['angle_deg'], fill=False, edgecolor='#ffb454', linewidth=1))
        ax.add_patch(Polygon(features['oriented_box']['corners_xy_mm'], fill=False, edgecolor='#f7e36d', linewidth=1))
        centre = np.array(features['centroid_xy_mm'])
        axes = np.array(features['oriented_box']['axes_xy'])
        ellipse_radius = np.sqrt((axes**2) @ (np.array([ellipse['major_length_mm'], ellipse['minor_length_mm']])/2)**2)
        corners = np.array(features['oriented_box']['corners_xy_mm'])
        lower = np.minimum(lower, np.minimum(corners.min(0), centre-ellipse_radius))
        upper = np.maximum(upper, np.maximum(corners.max(0), centre+ellipse_radius))
    for i, point in enumerate(session.landmarks):
        ax.plot(*point, 'o', color='#f77aff', markersize=4)
        ax.annotate(str(i+1), point, color='#f77aff', xytext=(4, 4), textcoords='offset points')
    blobs = (session.candidates or {}).get('candidates', []) if show_blobs else []
    for i, blob in enumerate(blobs):
        ax.add_patch(Circle(blob['centre_xy_mm'], blob['display_radius_mm'], fill=False,
                            edgecolor='#77dbff', alpha=.6, linestyle=':'))
        ax.annotate(f'B{i+1}', blob['centre_xy_mm'], color='#77dbff', fontsize=8)
    if show_points and session.result:
        points = np.asarray(session.result['monte_carlo_area']['preview_points_xy_mm_hit'])
        ax.scatter(points[:, 0], points[:, 1], s=2, c=np.where(points[:, 2], '#3bf4b0', '#ff6060'), alpha=.5)
    ax.set(xlabel='Anterior → Posterior (mm)', ylabel='Inferior → Superior (mm)',
           title=f"{session.level} · {'Đã xác nhận' if session.reviewed else 'Vùng nháp'}",
           xlim=[lower[0], upper[0]], ylim=[lower[1], upper[1]])
    ax.set_aspect('equal')


def draw_normalized(ax, session):
    if not session.mask.any():
        ax.text(.5, .5, 'Vẽ vùng thân đốt để chuẩn hóa', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    normalized = normalize_region(session.ct, session.mask, session.plane.spacing_xy_mm)
    ax.imshow(normalized['ct'], origin='lower', cmap='gray', vmin=-500, vmax=1500)
    ax.contour(normalized['mask'], levels=[.5], colors=['#3bf4b0'], linewidths=1)
    ax.set(title='256 × 256 · giữ tỷ lệ', xlabel='Pixel', ylabel='Pixel')
    ax.set_aspect('equal')


def draw_convergence(ax, report):
    if not report or 'monte_carlo_area' not in report:
        ax.text(.5, .5, 'Chưa có kết quả đo MC', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    curve = report['monte_carlo_area'].get('convergence', [])
    if not curve:
        ax.text(.5, .5, 'Chưa có dữ liệu hội tụ MC', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    samples = [p['samples'] for p in curve]
    areas = [p['area_mm2'] for p in curve]
    ax.semilogx(samples, areas, 'o-', color='#1f77b4', markersize=3.5, linewidth=1.2, label='MC')
    ax.axhline(report['features']['area_mm2'], color='black', linestyle='--', label='Đếm pixel')
    elbow = report['monte_carlo_area'].get('elbow')
    if elbow:
        el_x, el_y = elbow['samples'], elbow['area_mm2']
        ax.plot(el_x, el_y, 'o', color='#ff7f0e', markersize=8, markeredgecolor='red', markeredgewidth=1.5,
                label=f"Điểm Cùi chỏ (Elbow point) ({el_y:.1f} mm²)")
        y_min, y_max = min(areas), max(areas)
        offset_y = -32 if (el_y - y_min) / (y_max - y_min + 1e-12) > 0.7 else 28
        va = 'top' if offset_y < 0 else 'bottom'
        ax.annotate(f"{el_y:.1f} mm²\n(N={el_x:,})",
                    xy=(el_x, el_y), xytext=(0, offset_y), textcoords='offset points',
                    ha='center', va=va,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#ff7f0e', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=8, fontweight='bold', color='#b30000')
    ax.set(title='Hội tụ diện tích trên mask', xlabel='Số điểm', ylabel='Diện tích (mm²)')
    ax.legend(loc='lower right' if elbow and elbow['samples'] < 5000 else 'best', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)


def export_preview(session, report, path):
    figure = Figure(figsize=(15, 5), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    draw_body(axes[0], session, show_points=True, show_blobs=False)
    draw_normalized(axes[1], session)
    draw_convergence(axes[2], report)
    figure.savefig(path, dpi=130)
    if (session.candidates or {}).get('candidates'):
        from pathlib import Path
        blob_figure = Figure(figsize=(7, 7), layout='constrained')
        FigureCanvasAgg(blob_figure)
        draw_body(blob_figure.subplots(), session)
        blob_figure.savefig(Path(path).with_name('blob_candidates.png'), dpi=130)
