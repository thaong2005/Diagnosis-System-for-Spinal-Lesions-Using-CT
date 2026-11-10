"""Unit tests for AO Spine morphometry extensions in spinal_ct/morphometry.py."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from spinal_ct.morphometry import (
    extract_posterior_arch_mask,
    find_spinous_process_tip,
    measure_posterior_wall_retropulsion,
    compute_intervertebral_metrics,
    auto_segment_body,
    auto_detect_landmarks,
    BodyAutoParams,
    BodySession,
    project_point_between_planes,
)
from spinal_ct.body_region import BodyPlane
from spinal_ct.morphometry_render import (
    get_aospine_severity_color_and_label,
    render_side_by_side_spine_mpr,
)


class TestAOSpineMorphometry(unittest.TestCase):
    """Test suite covering AO Spine morphometric extraction and intervertebral metrics."""

    def setUp(self):
        self.spacing = (1.0, 1.0)  # 1 mm isotropic 2D sagittal

    def test_extract_posterior_arch_mask(self):
        """Test extraction of posterior arch mask separated by pedicle cut column."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 10:40] = True

        # Cut column at 25
        arch = extract_posterior_arch_mask(mask, cut_col=25)
        self.assertEqual(arch.shape, (50, 50))
        # col <= 25 should be False
        self.assertFalse(arch[:, :26].any())
        # col > 25 should retain original mask pixels
        self.assertTrue(arch[15:35, 26:40].all())

        # Degenerate cases
        empty_arch = extract_posterior_arch_mask(mask, cut_col=None)
        self.assertFalse(empty_arch.any())

        zero_arch = extract_posterior_arch_mask(mask, cut_col=0)
        self.assertFalse(zero_arch.any())

        out_arch = extract_posterior_arch_mask(mask, cut_col=50)
        self.assertFalse(out_arch.any())

    def test_find_spinous_process_tip(self):
        """Test detection of the spinous process tip (max col, min row) and quality score."""
        arch = np.zeros((60, 60), dtype=bool)
        # Draw a synthetic posterior arch sloping posteriorly and inferiorly (rows 10-30, cols 25-50)
        for c in range(25, 51):
            r_center = int(35 - (c - 25) * 0.8)
            arch[max(0, r_center - 2):min(60, r_center + 3), c] = True

        res = find_spinous_process_tip(arch, (1.0, 1.0))
        self.assertTrue(res['is_valid'])
        self.assertIsNotNone(res['tip_xy_mm'])
        self.assertIsNotNone(res['tip_rc'])
        # Max col is 50
        self.assertEqual(res['tip_rc'][1], 50)
        self.assertAlmostEqual(res['tip_xy_mm'][0], 50.0)
        self.assertGreater(res['area_mm2'], 0.0)
        self.assertGreaterEqual(res['q_seg'], 0.5)
        self.assertAlmostEqual(res['continuity'], 1.0)

        # Empty mask
        empty_res = find_spinous_process_tip(np.zeros((60, 60), dtype=bool), (1.0, 1.0))
        self.assertFalse(empty_res['is_valid'])
        self.assertIsNone(empty_res['tip_xy_mm'])
        self.assertEqual(empty_res['q_seg'], 0.0)

    def test_auto_segment_body_return_details(self):
        """Ensure auto_segment_body backward-compatibility and optional details dict."""
        ct = np.full((60, 60), -500.0)
        valid = np.ones((60, 60), dtype=bool)
        hint = np.zeros((60, 60), dtype=bool)
        # Vertebra body (10:30, 10:28) and posterior arch (10:30, 28:45)
        hint[15:35, 10:45] = True
        ct[hint] = 300.0
        # Narrow pedicle neck around column 28
        hint[22:28, 27:30] = True
        hint[15:22, 27:30] = False
        hint[28:35, 27:30] = False

        # 1. Standard call returns np.ndarray (100% backward compatible)
        body = auto_segment_body(ct, valid, hint, (1.0, 1.0))
        self.assertIsInstance(body, np.ndarray)
        self.assertTrue(body.any())

        # 2. Call with return_details=True returns dict
        details = auto_segment_body(ct, valid, hint, (1.0, 1.0), return_details=True)
        self.assertIsInstance(details, dict)
        self.assertIn('body_mask', details)
        self.assertIn('posterior_arch_mask', details)
        self.assertIn('cut_col', details)
        self.assertTrue(np.array_equal(body, details['body_mask']))

    def test_measure_posterior_wall_retropulsion_normal(self):
        """Test retropulsion measurement on normal intact vertebra with clear canal."""
        ct = np.full((100, 100), -100.0)
        # Current vertebra body: rows 35 to 65, cols 15 to 40
        ct[35:65, 15:40] = 350.0  # bone HU
        # Canal behind posterior wall (col > 40) is soft tissue / CSF (~30 HU)
        ct[35:65, 40:70] = 30.0

        landmarks = [
            [15.0, 65.0], [15.0, 35.0],  # AS, AI
            [27.5, 65.0], [27.5, 35.0],  # MS, MI
            [40.0, 65.0], [40.0, 35.0],  # PS, PI
        ]
        upper_pi = [40.0, 75.0]
        lower_ps = [40.0, 25.0]

        res = measure_posterior_wall_retropulsion(
            ct_slice=ct,
            spacing_mm=(1.0, 1.0),
            curr_vert_landmarks=landmarks,
            upper_vert_pi=upper_pi,
            lower_vert_ps=lower_ps,
        )

        self.assertEqual(res['delta_posterior_mm'], 0.0)
        self.assertEqual(res['canal_encroachment_ratio'], 0.0)
        self.assertEqual(res['retropulsed_pixel_count'], 0)
        self.assertIsNone(res['max_retropulsion_point_xy_mm'])
        self.assertEqual(res['reference_mode'], 'adjacent_vertebrae')

    def test_measure_posterior_wall_retropulsion_burst_a3_a4(self):
        """Test retropulsion measurement on burst fracture with retropulsed bone fragment."""
        ct = np.full((100, 100), -100.0)
        # Vertebra body: rows 35 to 65, cols 15 to 40
        ct[35:65, 15:40] = 350.0
        # Retropulsed bone fragment protruding into canal at row 50, cols 41 to 45 (4 mm protrusion)
        ct[48:52, 41:45] = 420.0  # HU >= 250

        landmarks = [
            [15.0, 65.0], [15.0, 35.0],  # AS, AI
            [27.5, 65.0], [27.5, 35.0],  # MS, MI
            [40.0, 65.0], [40.0, 35.0],  # PS, PI
        ]
        upper_pi = [40.0, 75.0]
        lower_ps = [40.0, 25.0]

        res = measure_posterior_wall_retropulsion(
            ct_slice=ct,
            spacing_mm=(1.0, 1.0),
            curr_vert_landmarks=landmarks,
            upper_vert_pi=upper_pi,
            lower_vert_ps=lower_ps,
            canal_depth_mm=15.0,
        )

        self.assertGreaterEqual(res['delta_posterior_mm'], 4.0)
        self.assertGreaterEqual(res['canal_encroachment_ratio'], 0.25)
        self.assertGreater(res['retropulsed_pixel_count'], 0)
        self.assertIsNotNone(res['max_retropulsion_point_xy_mm'])
        self.assertEqual(res['max_retropulsion_point_xy_mm'][0], 44.0)

        # Test fallback when adjacent vertebrae are omitted
        fallback_res = measure_posterior_wall_retropulsion(
            ct_slice=ct,
            spacing_mm=(1.0, 1.0),
            curr_vert_landmarks=landmarks,
            upper_vert_pi=None,
            lower_vert_ps=None,
        )
        self.assertEqual(fallback_res['reference_mode'], 'current_vertebra_fallback')
        self.assertGreaterEqual(fallback_res['delta_posterior_mm'], 4.0)

        # Test parasagittal slice with larger retropulsion (e.g. 6 mm)
        para_ct = ct.copy()
        para_ct[48:52, 41:47] = 450.0  # 6 mm protrusion on parasagittal
        para_res = measure_posterior_wall_retropulsion(
            ct_slice=ct,
            spacing_mm=(1.0, 1.0),
            curr_vert_landmarks=landmarks,
            upper_vert_pi=upper_pi,
            lower_vert_ps=lower_ps,
            parasagittal_slices=[para_ct],
        )
        self.assertGreaterEqual(para_res['delta_posterior_mm'], 6.0)

    def test_compute_intervertebral_metrics_normal_sequence(self):
        """Test intervertebral metrics on a normal, well-aligned 3-vertebra sequence."""
        # 3 vertebrae spaced evenly (T11 at y=70, T12 at y=45, L1 at y=20)
        vertebrae = [
            {
                'level': 'T11',
                'landmarks': [[15.0, 80.0], [15.0, 60.0], [25.0, 80.0], [25.0, 60.0], [35.0, 80.0], [35.0, 60.0]],
                'spinous_tip': [55.0, 65.0],
            },
            {
                'level': 'T12',
                'landmarks': [[15.0, 55.0], [15.0, 35.0], [25.0, 55.0], [25.0, 35.0], [35.0, 55.0], [35.0, 35.0]],
                'spinous_tip': [55.0, 40.0],
            },
            {
                'level': 'L1',
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 15.0],
            },
        ]

        metrics = compute_intervertebral_metrics(vertebrae)
        self.assertEqual(len(metrics), 3)

        t12 = metrics[1]
        self.assertAlmostEqual(t12['isp_ratio'], 1.0, delta=0.05)
        self.assertLess(t12['anterior_disc_angle_deg'], 5.0)
        self.assertAlmostEqual(t12['disc_opening_ratio'], 1.0, delta=0.1)
        self.assertLess(t12['translation_mm'], 0.5)
        self.assertAlmostEqual(t12['relative_collapse_ratio'], 1.0, delta=0.05)

    def test_compute_intervertebral_metrics_b2_distraction(self):
        """Test detection of posterior tension band disruption (AO Spine B2: ISP Ratio > 1.30)."""
        vertebrae = [
            {
                'level': 'T11',
                'landmarks': [[15.0, 80.0], [15.0, 60.0], [25.0, 80.0], [25.0, 60.0], [35.0, 80.0], [35.0, 60.0]],
                'spinous_tip': [55.0, 70.0],
            },
            {
                'level': 'T12',
                'landmarks': [[15.0, 55.0], [15.0, 35.0], [25.0, 55.0], [25.0, 35.0], [35.0, 55.0], [35.0, 35.0]],
                # Distracted spinous process tip widened by 50% vs baseline
                'spinous_tip': [55.0, 35.0],  # dist to T11 is 35 mm vs baseline 25 mm
            },
            {
                'level': 'L1',
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 10.0],  # dist to T12 is 25 mm
            },
        ]

        metrics = compute_intervertebral_metrics(vertebrae)
        t12 = metrics[1]
        self.assertGreaterEqual(t12['isp_ratio'], 1.35)

    def test_compute_intervertebral_metrics_b3_hyperextension(self):
        """Test detection of anterior disc gaping (AO Spine B3: disc_opening_ratio >= 2.0)."""
        # Upper vertebra T11 tilted into extension, opening anterior disc widely
        vertebrae = [
            {
                'level': 'T11',
                # Inferior endplate tilted up anteriorly (AI at y=64 vs PI at y=57)
                'landmarks': [[15.0, 84.0], [15.0, 64.0], [25.0, 80.0], [25.0, 60.0], [35.0, 77.0], [35.0, 57.0]],
                'spinous_tip': [55.0, 65.0],
            },
            {
                'level': 'T12',
                # Superior endplate flat (AS at y=52, PS at y=52)
                # d_ant = 64 - 52 = 12 mm; d_post = 57 - 52 = 5 mm -> ratio = 2.4
                'landmarks': [[15.0, 52.0], [15.0, 35.0], [25.0, 52.0], [25.0, 35.0], [35.0, 52.0], [35.0, 35.0]],
                'spinous_tip': [55.0, 40.0],
            },
            {
                'level': 'L1',
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 15.0],
            },
        ]

        metrics = compute_intervertebral_metrics(vertebrae)
        t12 = metrics[1]
        self.assertGreaterEqual(t12['anterior_disc_angle_deg'], 15.0)
        self.assertGreaterEqual(t12['disc_opening_ratio'], 2.0)

    def test_compute_intervertebral_metrics_c_translation(self):
        """Test detection of translation / dislocation (AO Spine C: translation_mm >= 3.0 mm)."""
        vertebrae = [
            {
                'level': 'T11',
                'landmarks': [[15.0, 80.0], [15.0, 60.0], [25.0, 80.0], [25.0, 60.0], [35.0, 80.0], [35.0, 60.0]],
                'spinous_tip': [55.0, 65.0],
            },
            {
                'level': 'T12',
                # Shifted posteriorly by 5.0 mm relative to spinal column
                'landmarks': [[20.0, 55.0], [20.0, 35.0], [30.0, 55.0], [30.0, 35.0], [40.0, 55.0], [40.0, 35.0]],
                'spinous_tip': [60.0, 40.0],
            },
            {
                'level': 'L1',
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 15.0],
            },
        ]

        metrics = compute_intervertebral_metrics(vertebrae)
        t12 = metrics[1]
        self.assertGreaterEqual(t12['translation_mm'], 4.5)

    def test_compute_intervertebral_metrics_a4_symmetric_collapse(self):
        """Test detection of complete symmetric collapse (AO Spine A4: relative_collapse_ratio <= 0.65)."""
        vertebrae = [
            {
                'level': 'T11',
                # Height = 20 mm
                'landmarks': [[15.0, 80.0], [15.0, 60.0], [25.0, 80.0], [25.0, 60.0], [35.0, 80.0], [35.0, 60.0]],
                'spinous_tip': [55.0, 65.0],
            },
            {
                'level': 'T12',
                # Symmetrically collapsed body (Ha=12, Hm=12, Hp=12 -> 40% loss, Ha/Hp=1.0)
                'landmarks': [[15.0, 47.0], [15.0, 35.0], [25.0, 47.0], [25.0, 35.0], [35.0, 47.0], [35.0, 35.0]],
                'spinous_tip': [55.0, 40.0],
            },
            {
                'level': 'L1',
                # Height = 20 mm
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 15.0],
            },
        ]

        metrics = compute_intervertebral_metrics(vertebrae)
        t12 = metrics[1]
        # While internal Ha/Hp is 1.0 (appearing normal), relative_collapse_ratio catches the severe collapse:
        self.assertAlmostEqual(t12['heights']['anterior_posterior_ratio'], 1.0, delta=0.05)
        self.assertLessEqual(t12['relative_collapse_ratio'], 0.65)

    def test_single_vertebra_and_reversed_order(self):
        """Test robustness on single-vertebra input and inverted caudal-to-cranial ordering."""
        single = [{
            'level': 'L1',
            'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
            'spinous_tip': [55.0, 15.0],
        }]
        res = compute_intervertebral_metrics(single)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['isp_ratio'], 1.0)
        self.assertEqual(res[0]['translation_mm'], 0.0)
        self.assertEqual(res[0]['relative_collapse_ratio'], 1.0)

        # Inverted ordering (caudal to cranial: L1 first, then T12, then T11)
        inverted = [
            {
                'level': 'L1',
                'landmarks': [[15.0, 30.0], [15.0, 10.0], [25.0, 30.0], [25.0, 10.0], [35.0, 30.0], [35.0, 10.0]],
                'spinous_tip': [55.0, 15.0],
            },
            {
                'level': 'T12',
                'landmarks': [[20.0, 55.0], [20.0, 35.0], [30.0, 55.0], [30.0, 35.0], [40.0, 55.0], [40.0, 35.0]],
                'spinous_tip': [60.0, 40.0],
            },
            {
                'level': 'T11',
                'landmarks': [[15.0, 80.0], [15.0, 60.0], [25.0, 80.0], [25.0, 60.0], [35.0, 80.0], [35.0, 60.0]],
                'spinous_tip': [55.0, 65.0],
            },
        ]
        res_inv = compute_intervertebral_metrics(inverted)
        self.assertEqual(len(res_inv), 3)
        # T12 is at index 1 and has translation >= 4.5
        self.assertGreaterEqual(res_inv[1]['translation_mm'], 4.5)

    def test_get_aospine_severity_color_and_label(self):
        """Test color coding and label generation according to AO Spine and collapse degree."""
        # 1. Normal
        c_norm, l_norm = get_aospine_severity_color_and_label('L1', 'Normal', 0.05)
        self.assertEqual(c_norm, '#4cd964')
        self.assertEqual(l_norm, 'L1')

        # 2. Mild A1 (15-25% collapse)
        c_a1, l_a1 = get_aospine_severity_color_and_label('L2', 'A1', 0.23)
        self.assertEqual(c_a1, '#ffd11a')
        self.assertEqual(l_a1, 'L2 -23% | A1')

        # 3. Moderate A3 (26-40% collapse)
        c_a3, l_a3 = get_aospine_severity_color_and_label('L4', 'A3', 0.38)
        self.assertEqual(c_a3, '#ff9933')
        self.assertEqual(l_a3, 'L4 -38% | A3')

        # 4. Severe A4 (> 40% collapse)
        c_a4, l_a4 = get_aospine_severity_color_and_label('T11', 'A4', 0.55)
        self.assertEqual(c_a4, '#ff4d4d')
        self.assertEqual(l_a4, 'T11 -55% | A4')

        # 5. Severe B2, C
        c_b2, l_b2 = get_aospine_severity_color_and_label('L3', 'B2', 0.10)
        self.assertEqual(c_b2, '#ff4d4d')
        c_c, l_c = get_aospine_severity_color_and_label('T12', 'C', 0.10)
        self.assertEqual(c_c, '#ff4d4d')

    def test_render_side_by_side_spine_mpr(self):
        """Test side-by-side MPR generation and export of levels_mpr, mpr, and comparison_mpr."""
        # Create synthetic 3D CT and mask
        ct_arr = np.random.randint(-200, 800, size=(20, 100, 100), dtype=np.int16)
        mask_arr = np.zeros((20, 100, 100), dtype=np.uint8)
        # Vertebra 1 (T12 = 14)
        mask_arr[8:12, 30:50, 40:60] = 14
        # Vertebra 2 (L1 = 15)
        mask_arr[8:12, 60:80, 40:60] = 15

        ct_img = sitk.GetImageFromArray(ct_arr)
        ct_img.SetSpacing((1.0, 1.0, 1.5))
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.SetSpacing((1.0, 1.0, 1.5))

        vertebrae_info = {
            'T12': {'aospine_class': 'Normal', 'collapse_ratio': 0.05},
            'L1': {'aospine_class': 'A3', 'collapse_ratio': 0.35, 'delta_posterior_mm': 3.2},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = render_side_by_side_spine_mpr(
                ct_img, mask_img, vertebrae_info, output_dir=tmp_dir, dpi=60
            )

            self.assertIn('levels_mpr_path', res)
            self.assertIn('mpr_path', res)
            self.assertIn('comparison_mpr_path', res)

            lev_path = Path(res['levels_mpr_path'])
            mpr_path = Path(res['mpr_path'])
            comp_path = Path(res['comparison_mpr_path'])

            self.assertTrue(lev_path.is_file())
            self.assertTrue(mpr_path.is_file())
            self.assertTrue(comp_path.is_file())

            self.assertGreater(lev_path.stat().st_size, 1000)
            self.assertGreater(mpr_path.stat().st_size, 1000)
            self.assertGreater(comp_path.stat().st_size, 1000)

    def test_project_point_between_planes(self):
        """Test projecting 2D coordinates between BodyPlanes with different 3D LPS origins."""
        p1 = BodyPlane(origin_lps_mm=(0.0, 0.0, 100.0), u_lps=(0.0, 1.0, 0.0), v_lps=(0.0, 0.0, 1.0),
                       spacing_xy_mm=(1.0, 1.0), size_xy=(80, 80))
        p2 = BodyPlane(origin_lps_mm=(0.0, 0.0, 70.0),  u_lps=(0.0, 1.0, 0.0), v_lps=(0.0, 0.0, 1.0),
                       spacing_xy_mm=(1.0, 1.0), size_xy=(80, 80))

        # Point (40.0, 20.0) in p1 -> 3D LPS (0.0, 40.0, 120.0) -> in p2: x=40.0, y=50.0
        pt_proj = project_point_between_planes([40.0, 20.0], p1, p2)
        self.assertEqual(pt_proj, [40.0, 50.0])

        # Self-projection (identity)
        self.assertEqual(project_point_between_planes([40.0, 20.0], p1, p1), [40.0, 20.0])

    def test_intervertebral_cross_plane_isp_and_canal_line(self):
        """Test ISP ratio and canal line orientation across sessions with different BodyPlane origins."""
        def make_ver(level, z_origin, sp_tip_y, up_d=0.0):
            pl = BodyPlane(origin_lps_mm=(0.0, 0.0, float(z_origin)), u_lps=(0.0, 1.0, 0.0), v_lps=(0.0, 0.0, 1.0),
                           spacing_xy_mm=(1.0, 1.0), size_xy=(80, 80))
            ct = np.full((80, 80), 100.0, dtype=np.float32)
            valid = np.ones((80, 80), dtype=bool)
            hint = np.ones((80, 80), dtype=bool)
            landmarks = [
                [15.0, 58.0], [15.0, 22.0],
                [30.0, 58.0 - float(up_d)], [30.0, 22.0],
                [45.0, 58.0], [45.0, 22.0],
            ]
            sess = BodySession('fp_test', level, 'Baseline', pl, ct, valid, hint, mask=hint.copy(),
                               landmarks=landmarks, age_group='adult')
            sess.spinous_tip = {'tip_xy_mm': [65.0, float(sp_tip_y)], 'q_seg': 0.90, 'is_valid': True}
            return sess

        s_t11 = make_ver('T11', 120.0, 30.0)
        s_t12 = make_ver('T12', 90.0, 30.0)
        s_l1  = make_ver('L1', 60.0, 15.0, up_d=3.0)

        sessions = {'T11': s_t11, 'T12': s_t12, 'L1': s_l1}
        for s in sessions.values():
            s.adjacent_sessions = sessions

        s_t12._ensure_retropulsion(force=True)
        canal_pts = s_t12.retropulsion.get('canal_line_points', [])
        self.assertEqual(len(canal_pts), 2)
        y_low, y_up = canal_pts[0][1], canal_pts[1][1]
        self.assertLess(y_low, 22.0)
        self.assertGreater(y_up, 58.0)

        s_l1.recompute_aospine_classification(sessions)
        isp_l1 = s_l1.aospine.get('raw_metrics', {}).get('isp_ratio', 1.0)
        self.assertAlmostEqual(isp_l1, 1.50, delta=0.05)

    def test_intervertebral_3d_translation_dislocation(self):
        """Test Type C translation detection when vertebra is dislocated in 3D LPS."""
        p1 = BodyPlane((0.0, 0.0, 120.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        p2_dislocated = BodyPlane((0.0, 6.0, 90.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        p3 = BodyPlane((0.0, 0.0, 60.0),  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        lm = [[15.0, 58.0], [15.0, 22.0], [30.0, 58.0], [30.0, 22.0], [45.0, 58.0], [45.0, 22.0]]
        v_trans_3d = [
            {'level': 'T11', 'landmarks': lm, 'spinous_tip': [65.0, 30.0], 'plane': p1},
            {'level': 'T12', 'landmarks': lm, 'spinous_tip': [65.0, 30.0], 'plane': p2_dislocated},
            {'level': 'L1',  'landmarks': lm, 'spinous_tip': [65.0, 30.0], 'plane': p3},
        ]
        res_tr = compute_intervertebral_metrics(v_trans_3d)
        trans_val = res_tr[1]['translation_mm']
        self.assertGreaterEqual(trans_val, 5.0)

    def test_auto_segment_preserves_posterior_body_and_suppresses_cortex_false_positive(self):
        """Verify that auto_segment_body does not truncate posterior body when canal is clear,
        posterior_arch_mask isolates the spinous process, and retropulsion measurement
        suppresses false-positive alerts on intact cortical bone."""
        shape = (80, 100)
        ct = np.full(shape, 50.0, dtype=np.float32)
        valid = np.ones(shape, dtype=bool)
        hint = np.zeros(shape, dtype=bool)

        # Vertebral body: cols 15..55 (rows 22..58 anteriorly, rows 30..50 posteriorly)
        hint[22:58, 15:46] = True
        hint[30:50, 45:56] = True
        # Spinous process: cols 70..88 (rows 25..45) separated by canal space (cols 56..69)
        hint[25:45, 70:89] = True
        ct[hint] = 400.0

        res_seg = auto_segment_body(ct, valid, hint, (1.0, 1.0), return_details=True)
        body_mask = res_seg['body_mask']
        arch_mask = res_seg['posterior_arch_mask']

        # 1. Body mask preserves posterior wall (reaches col 55, not truncated at col 44)
        max_col = int(np.nonzero(body_mask)[1].max())
        self.assertGreaterEqual(max_col, 53)

        # 2. Posterior arch mask isolates the spinous process (starts >= col 70)
        self.assertTrue(arch_mask.any())
        min_arch_col = int(np.nonzero(arch_mask)[1].min())
        self.assertGreaterEqual(min_arch_col, 70)

        # 3. Retropulsion measurement on intact flat vertebra does not false-alarm due to inset landmarks
        plane = BodyPlane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (100, 80))
        clean_body = np.zeros(shape, dtype=bool)
        clean_body[22:58, 15:55] = True
        ct_clean = np.full(shape, 50.0, dtype=np.float32)
        ct_clean[clean_body] = 400.0
        lm = auto_detect_landmarks(clean_body, plane)
        ret = measure_posterior_wall_retropulsion(ct_clean, (1.0, 1.0), lm)
        self.assertLess(ret['delta_posterior_mm'], 1.5)
        self.assertEqual(ret['delta_posterior_mm'], 0.0)


if __name__ == '__main__':
    unittest.main()

