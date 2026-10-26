"""Physical-space contours and slice-local names for categorical vertebral masks.

Rendering never smooths, dilates or otherwise changes the source labels.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from matplotlib import colormaps, patheffects
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

from spinal_ct.localization import LEVELS


def vertebral_palette() -> ListedColormap:
    return ListedColormap([(0, 0, 0, 0)] +
                          [colormaps['tab20'](i % 20) for i in range(len(LEVELS))])


def _label_positions(anchors, lower, upper, gap):
    """Separate ordered names, keeping every name within the visible vertical range."""
    gap = min(gap, (upper-lower)/max(len(anchors), 1))
    low, high = lower+gap/2, upper-gap/2
    positions = np.clip(np.asarray(anchors, dtype=float), low, high)
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i-1]+gap)
    if len(positions) and positions[-1] > high:
        positions[-1] = high
        for i in range(len(positions)-2, -1, -1):
            positions[i] = min(positions[i], positions[i+1]-gap)
    return positions


def _boundary_anchor(region, spacing):
    """Attach to a real left boundary on the largest visible component, not empty space."""
    components = sitk.GetArrayFromImage(sitk.ConnectedComponent(
        sitk.GetImageFromArray(region.astype(np.uint8))))
    counts = np.bincount(components.ravel())
    counts[0] = 0
    rows, columns = np.nonzero(components == counts.argmax())
    row = rows[np.argmin(np.abs(rows-np.median(rows)))]
    col = columns[rows == row].min()
    sx, sy = spacing
    return (float(col)-.5)*sx, float(row)*sy


def draw_vertebral_overlay(ax, plane, spacing, *, names=False, label_map=None):
    """Draw closed 0.5 contours at voxel boundaries; optionally reserve a name gutter.

    Coordinates are relative to the display grid's voxel centres in mm. Call
    after setting image limits. Names describe only labels present on this slice.
    Returns annotations to support picking/verification by callers.
    """
    sx, sy = spacing
    height, width = plane.shape
    palette = vertebral_palette()
    mapping = {i: level for i, level in enumerate(LEVELS, 1)}
    if label_map:
        mapping.update({int(key): value for key, value in label_map.items()})
    anchors = []
    for label_id in np.unique(plane):
        if not 1 <= label_id <= len(LEVELS):
            continue
        region = plane == label_id
        color = palette(int(label_id))
        # Outside padding closes a truncated mask exactly at the FOV edge.
        # Never interpolate the numeric IDs of different vertebrae together.
        contour = ax.contour(np.arange(-1, width+1)*sx,
                             np.arange(-1, height+1)*sy,
                             np.pad(region.astype(np.uint8), 1), levels=[.5],
                             colors=[color], linewidths=1.0, antialiased=True)
        contour.set_path_effects([
            patheffects.Stroke(linewidth=2.1, foreground='#071019', alpha=.8),
            patheffects.Normal(),
        ])
        if names:
            x, y = _boundary_anchor(region, spacing)
            anchors.append((y, x, int(label_id), color))
    if not anchors:
        return []
    anchors.sort()
    left, right = -sx/2, (width-.5)*sx
    lower, upper = sorted(ax.get_ylim())
    # Reserve enough *screen* space for names even on a tall CT in a short window.
    # Solve the equal-aspect scale with the gutter included, avoiding resize clipping.
    allocation = ax.get_position(original=True)
    available_width = allocation.width*ax.figure.bbox.width
    available_height = allocation.height*ax.figure.bbox.height
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(size=8.5, weight='bold')
    name_width = max(renderer.get_text_width_height_descent(mapping[a[2]], font, False)[0]
                     for a in anchors)
    gutter_pixels = name_width + 12*ax.figure.dpi/72
    scale = min(available_height/(upper-lower),
                max(available_width-gutter_pixels, 1)/(right-left))
    gutter = gutter_pixels/scale
    ax.set_xlim(left-gutter, right)
    ax.apply_aspect()
    # A point-size gap follows the actual panel height, including window resize.
    gap = 13 * ax.figure.dpi/72 * (upper-lower)/max(ax.bbox.height, 1)
    positions = _label_positions([a[0] for a in anchors], lower, upper, gap)
    annotations = []
    for (y, x, label_id, color), label_y in zip(anchors, positions):
        annotation = ax.annotate(mapping[label_id], xy=(x, y),
            xytext=(left-4*ax.figure.dpi/72/scale, label_y), ha='right', va='center',
            fontsize=8.5, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=.12', fc=ax.get_facecolor(), ec='none', alpha=.95),
            arrowprops=dict(arrowstyle='-', color=color, lw=.8,
                            shrinkA=3, shrinkB=1, connectionstyle='arc3'),
            annotation_clip=True, zorder=6)
        annotations.append(annotation)
    return annotations
