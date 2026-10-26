"""Verify geometric boundary placement and names without a model or patient CT."""
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path

from spinal_ct.overlays import draw_vertebral_overlay


class OverlayTests(unittest.TestCase):
    def axes(self, plane, spacing):
        fig = Figure(figsize=(4, 5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        h, w = plane.shape
        sx, sy = spacing
        ax.set_xlim(-sx/2, (w-.5)*sx)
        ax.set_ylim(-sy/2, (h-.5)*sy)
        ax.set_aspect('equal')
        return ax

    def test_border_contour_is_closed_at_half_voxel_with_anisotropic_spacing(self):
        plane = np.zeros((10, 12), np.uint8)
        plane[:4, :3] = 20
        before = plane.copy()
        ax = self.axes(plane, (.7, 2))
        names = draw_vertebral_overlay(ax, plane, (.7, 2), names=True)
        path = ax.collections[0].get_paths()[0]
        self.assertEqual(path.codes[-1], Path.CLOSEPOLY)
        np.testing.assert_allclose(path.vertices.min(axis=0), [-.35, -1])
        np.testing.assert_allclose(path.vertices.max(axis=0), [1.75, 7])
        self.assertEqual([item.get_text() for item in names], ['L1'])
        np.testing.assert_array_equal(plane, before)

    def test_names_anchor_to_visible_boundary_and_do_not_overlap_at_top(self):
        plane = np.zeros((50, 50), np.uint8)
        # All at the same height: centroids or independent y-offsets would overlap.
        for index, label_id in enumerate((18, 19, 20, 21)):
            plane[45:49, 8+index*10:14+index*10] = label_id
        # A disconnected speck must not attract L1's leader away from its body.
        plane[1, 0] = 20
        ax = self.axes(plane, (1, 1))
        names = draw_vertebral_overlay(ax, plane, (1, 1), names=True)
        self.assertEqual({a.get_text() for a in names}, {'T11', 'T12', 'L1', 'L2'})
        positions = sorted(a.get_position()[1] for a in names)
        self.assertTrue(all(y >= -.5 and y <= 49.5 for y in positions))
        self.assertGreater(min(np.diff(positions)), 2)
        for annotation in names:
            x, y = annotation.xy
            row, col = int(y), int(x+.5)
            label_id = {'T11': 18, 'T12': 19, 'L1': 20, 'L2': 21}[annotation.get_text()]
            self.assertEqual(plane[row, col], label_id)
            self.assertNotEqual(plane[row, col-1], label_id)
        self.assertGreater(next(a.xy[1] for a in names if a.get_text() == 'L1'), 40)
        ax.figure.canvas.draw()

    def test_empty_single_pixel_full_field_and_no_name_modes(self):
        for plane in (np.zeros((4, 4), np.uint8), np.full((4, 4), 25, np.uint8),
                      np.array([[20]], np.uint8)):
            with self.subTest(shape=plane.shape, total=int(plane.sum())):
                ax = self.axes(plane, (1, 1))
                original_xlim = ax.get_xlim()
                names = draw_vertebral_overlay(ax, plane, (1, 1), names=False)
                self.assertEqual(names, [])
                self.assertEqual(ax.get_xlim(), original_xlim)
                ax.figure.canvas.draw()

    def test_tall_ct_names_fit_inside_short_panel(self):
        plane = np.zeros((567, 512), np.uint8)
        for i, label_id in enumerate(range(13, 26)):
            plane[110+i*32:132+i*32, 230:290] = label_id
        ax = self.axes(plane, (.379, .7))
        ax.figure.set_size_inches(3, 3)
        names = draw_vertebral_overlay(ax, plane, (.379, .7), names=True)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [a.get_bbox_patch().get_window_extent(renderer) for a in names]
        self.assertEqual(len(boxes), 13)
        for i, box in enumerate(boxes):
            self.assertGreaterEqual(box.x0, ax.bbox.x0)
            self.assertLessEqual(box.x1, ax.bbox.x1)
            self.assertGreaterEqual(box.y0, ax.bbox.y0)
            self.assertLessEqual(box.y1, ax.bbox.y1)
            self.assertFalse(any(box.overlaps(other) for other in boxes[i+1:]))
