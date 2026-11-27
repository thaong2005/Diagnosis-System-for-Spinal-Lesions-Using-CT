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
    is_vertebra_fov_truncated,
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

    def test_parasagittal_pedicle_suppression_on_flat_normal_vertebra(self):
        """Hotfix 3.2 Task 2: lateral pedicles on +/-4mm parasagittal slices do not create false burst."""
        shape = (80, 100)
        ct_mid = np.full(shape, 50.0, dtype=np.float32)
        body_mask = np.zeros(shape, dtype=bool)
        body_mask[22:58, 15:46] = True
        ct_mid[body_mask] = 400.0
        # Adjacent vertebra sitting just 1.5mm below within safety_margin_mm=2.0
        ct_mid[19:21, 15:52] = 400.0

        # Parasagittal slices at +/-4mm show normal anatomical pedicles in upper 1/3 (rows 46..56)
        para_ct = ct_mid.copy()
        para_ct[46:56, 46:53] = 450.0

        plane = BodyPlane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (100, 80))
        lm = auto_detect_landmarks(body_mask, plane)

        ret = measure_posterior_wall_retropulsion(
            ct_mid, (1.0, 1.0), lm,
            body_mask=body_mask,
            parasagittal_slices=[para_ct, para_ct],
        )
        self.assertLess(ret['delta_posterior_mm'], 1.5)

    def test_is_vertebra_fov_truncated_boundary_and_sandwiched(self):
        """Hotfix 3.2 Task 3: severe middle collapse with full valid FOV is NOT FOV truncated."""
        plane = BodyPlane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        ct = np.full((80, 80), 50.0, dtype=np.float32)

        # 1. Severely collapsed small vertebra in interior of valid FOV (area = 150 mm2 < 200)
        valid_full = np.ones((80, 80), dtype=bool)
        mask_small = np.zeros((80, 80), dtype=bool)
        mask_small[37:42, 20:50] = True  # 5x30 = 150 mm2 < 200, max_h ~ 5mm < 8mm
        lm_small = [[20.0, 41.0], [20.0, 37.0], [35.0, 41.0], [35.0, 37.0], [49.0, 41.0], [49.0, 37.0]]
        sess_mid = BodySession('fp', 'L1', 'Baseline', plane, ct, valid_full, mask_small.copy(),
                               mask=mask_small.copy(), landmarks=lm_small, age_group='adult')
        self.assertFalse(is_vertebra_fov_truncated(sess_mid))

        # 2. Actually touching CT invalid boundary (~valid) -> IS FOV truncated
        valid_clipped = np.ones((80, 80), dtype=bool)
        valid_clipped[:5, :] = False
        mask_edge = np.zeros((80, 80), dtype=bool)
        mask_edge[5:10, 20:50] = True  # row 5 touches ~eroded_valid (row 4 is False)
        lm_edge = [[20.0, 9.0], [20.0, 5.0], [35.0, 9.0], [35.0, 5.0], [49.0, 9.0], [49.0, 5.0]]
        sess_edge = BodySession('fp', 'T10', 'Baseline', plane, ct, valid_clipped, mask_edge.copy(),
                                mask=mask_edge.copy(), landmarks=lm_edge, age_group='adult')
        self.assertTrue(is_vertebra_fov_truncated(sess_edge))

        # 3. Sandwiched dict vertebra between valid neighbors is unflagged in compute_intervertebral_metrics
        lm_norm = [[20.0, 55.0], [20.0, 25.0], [35.0, 55.0], [35.0, 25.0], [50.0, 55.0], [50.0, 25.0]]
        inter = compute_intervertebral_metrics([
            {'level': 'T10', 'landmarks': lm_norm, 'heights': {'anterior_mm': 30.0, 'middle_mm': 30.0, 'posterior_mm': 30.0}, 'area_mm2': 600.0},
            {'level': 'T11', 'landmarks': lm_small, 'heights': {'anterior_mm': 5.0, 'middle_mm': 5.0, 'posterior_mm': 5.0}, 'area_mm2': 140.0},
            {'level': 'T12', 'landmarks': lm_norm, 'heights': {'anterior_mm': 30.0, 'middle_mm': 30.0, 'posterior_mm': 30.0}, 'area_mm2': 600.0},
        ])
        self.assertFalse(inter[1].get('is_fov_truncated', False))

    def test_hotfix33_endplate_stepoff_global_continuity_and_coronal_x(self):
        """Hotfix 3.3 Task 1.2: superior-endplate step-off + global continuity clamp for kyphotic B2 + coronal X Type C."""
        p_t11 = BodyPlane((0.0, 0.0, 120.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        p_t12 = BodyPlane((0.0, 0.0, 90.0),  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        p_l1  = BodyPlane((0.0, 0.0, 60.0),  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))

        lm_norm = [[15.0, 58.0], [15.0, 38.0], [30.0, 58.0], [30.0, 38.0], [45.0, 58.0], [45.0, 38.0]]
        # T12 is severely collapsed anteriorly (Ha/Hp < 0.65, rel_coll < 0.72), but posterior wall of T11 & L1 aligns straight at Y=45
        lm_wedged_b2 = [[15.0, 47.0], [15.0, 38.0], [30.0, 50.0], [30.0, 38.0], [48.0, 58.0], [48.0, 38.0]]
        v_b2 = [
            {'level': 'T11', 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_t11},
            {'level': 'T12', 'landmarks': lm_wedged_b2, 'spinous_tip': [65.0, 48.0], 'plane': p_t12},
            {'level': 'L1',  'landmarks': lm_norm, 'spinous_tip': [65.0, 30.0], 'plane': p_l1},
        ]
        res_b2 = compute_intervertebral_metrics(v_b2)
        self.assertLess(res_b2[1]['relative_collapse_ratio'], 0.72)
        self.assertLess(res_b2[1]['ha_hp_ratio'], 0.65)
        self.assertLess(res_b2[1]['translation_mm'], 3.5)

        # Coronal X shift >= 3.8 mm triggers translation_mm >= 3.8
        p_t12_lat = BodyPlane((5.2, 0.0, 90.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        v_coronal_c = [
            {'level': 'T11', 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_t11},
            {'level': 'T12', 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_t12_lat},
            {'level': 'L1',  'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_l1},
        ]
        res_cor = compute_intervertebral_metrics(v_coronal_c)
        self.assertGreaterEqual(res_cor[1]['coronal_step_mm'], 4.5)
        self.assertGreaterEqual(res_cor[1]['translation_mm'], 4.5)

        # Intact vertebra sitting above a severely tilted/wedged superior endplate on inferior vertebra must not trigger false step-off
        lm_tilted_inf = [[15.0, 40.0], [15.0, 22.0], [30.0, 45.0], [30.0, 22.0], [45.0, 58.0], [45.0, 22.0]]
        v_tilted_below = [
            {'level': 'T11', 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_t11},
            {'level': 'T12', 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': p_t12},
            {'level': 'L1',  'landmarks': lm_tilted_inf, 'spinous_tip': [65.0, 48.0], 'plane': p_l1},
        ]
        res_tilted = compute_intervertebral_metrics(v_tilted_below)
        self.assertLess(res_tilted[1]['translation_mm'], 1.0)

    def test_hotfix33_l4_l5_hinge_suppression_and_oof_injection_isolation(self):
        """Hotfix 3.3: intact L4/L5 hinge sets delta_posterior_mm=0 and explicit engine/bank bypasses singletons."""
        import spinal_ct.morphometry as morph

        plane = BodyPlane((0.0, 0.0, 60.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
        ct = np.full((80, 80), 40.0, dtype=np.float32)
        valid = np.ones((80, 80), dtype=bool)
        mask = np.zeros((80, 80), dtype=bool)
        mask[22:58, 15:46] = True
        ct[mask] = 400.0
        # Simulated osteophyte / bevel bulge posterior to L5 line while heights/endplates are well preserved
        ct[38:44, 46:50] = 400.0
        lm = [[15.0, 56.0], [15.0, 22.0], [30.0, 56.0], [30.0, 23.0], [45.0, 58.0], [45.0, 22.0]]

        sess = BodySession('fp_l5', 'L5', 'Baseline', plane, ct, valid, mask.copy(),
                           mask=mask.copy(), landmarks=lm, age_group='adult')
        sess._ensure_retropulsion(force=True)
        self.assertEqual(sess.retropulsion['delta_posterior_mm'], 0.0)

        # Singleton isolation test: injected custom engine & bank must NOT touch global singletons or _cached_*
        morph._INFERENCE_ENGINE = None
        morph._PROTOTYPE_BANK = None

        class _DummyEngine:
            def __init__(self):
                self.calls = 0
            def predict_batch(self, body_rois, arch_rois):
                self.calls += len(body_rois)
                return ([{'p_pw': 0.91, 'p_ep2': 0.12, 'p_post_tb': 0.05, 'p_ant_tb': 0.02, 'p_trans': 0.03, 'p_fracture': 0.88}] * len(body_rois), 0.001)

        class _DummyBank:
            def __init__(self):
                self.calls = 0
            def predict(self, feats, level=None):
                self.calls += 1
                return {'predicted_class': 'A3', 'p_anomaly': 0.82, 'class_probabilities': {'A3': 0.75, 'Normal': 0.15}}

        eng = _DummyEngine()
        bk = _DummyBank()
        from spinal_ct.morphometry import SpineSession
        spine = SpineSession(patient_id='OOF_TEST')
        spine.add_session(sess)
        spine.recompute_aospine_classification(inference_engine=eng, prototype_bank=bk)

        self.assertEqual(eng.calls, 1)
        self.assertEqual(bk.calls, 1)
        self.assertIsNone(morph._INFERENCE_ENGINE)
        self.assertIsNone(morph._PROTOTYPE_BANK)
        self.assertIsNone(sess._cached_dl_probs)
        self.assertIsNone(sess._cached_proto_result)

    def test_tilted_normal_vertebra_no_false_retropulsion(self):
        """Tilted normal vertebra (~18 deg lordosis) must not trigger false retropulsion from top/bottom band swap."""
        shape = (80, 80)
        ct = np.full(shape, 40.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        for r in range(20, 56):
            shift = (r - 20) * 0.33  # tan(18.26 deg) = 0.33
            c0 = int(round(14.0 + shift))
            c1 = int(round(42.0 + shift))
            mask[r, c0:c1 + 1] = True
            ct[r, c0:c1 + 1] = 400.0

        # Landmarks inset slightly inside cortical margin so mask-corner clamp executes
        lm_tilted = [
            [26.0, 55.0],
            [14.5, 20.0],
            [39.0, 55.0],
            [27.5, 20.0],
            [52.0, 55.0],
            [40.5, 20.0],
        ]
        res_clean = measure_posterior_wall_retropulsion(
            ct, (1.0, 1.0), lm_tilted, body_mask=mask,
        )
        self.assertLess(res_clean['delta_posterior_mm'], 1.0)

        # True central focal retropulsion (+5 mm behind tilted wall) still detected cleanly
        ct_frag = ct.copy()
        ct_frag[35:41, 49:55] = 450.0
        res_frag = measure_posterior_wall_retropulsion(
            ct_frag, (1.0, 1.0), lm_tilted, body_mask=mask,
        )
        self.assertGreaterEqual(res_frag['delta_posterior_mm'], 3.5)

    def test_smooth_physiological_lordosis_no_phantom_type_c(self):
        """25-deg smooth physiological lordosis curve (dev_curve > 7mm, step_raw ~ 0) stays < 1.0mm;
        true 5mm translation still yields >= 4.5mm."""
        R = 95.0
        thetas = np.deg2rad([23.0, 0.0, -23.0])  # T12 (+23 deg), L1 (0 deg apex), L2 (-23 deg)
        levels = ['T12', 'L1', 'L2']
        curved_verts = []
        for lvl, th in zip(levels, thetas):
            # Body center on smooth circular lordotic arc: dc/dth = R * e_si -> tangent & normal align with zero step-off
            cx = 30.0 + R * (np.cos(th) - 1.0)
            cy = 50.0 + R * np.sin(th)
            e_ap = np.array([np.cos(th), np.sin(th)])    # anterior -> posterior along tilted endplate
            e_si = np.array([-np.sin(th), np.cos(th)])   # inferior -> superior tangent to curve

            def _pt(ap_off: float, si_off: float) -> list[float]:
                p = np.array([cx, cy]) + ap_off * e_ap + si_off * e_si
                return [float(p[0]), float(p[1])]

            lm = [
                _pt(-14.0, +10.0), _pt(-14.0, -10.0),
                _pt(0.0, +10.0),   _pt(0.0, -10.0),
                _pt(+14.0, +10.0), _pt(+14.0, -10.0),
            ]
            curved_verts.append({
                'level': lvl,
                'landmarks': lm,
                'spinous_tip': _pt(+34.0, 0.0),
            })

        res_curve = compute_intervertebral_metrics(curved_verts)
        self.assertLess(res_curve[1]['translation_mm'], 1.0)

        # True 9.0 mm posterior dislocation on curved spine yields >= 4.5 mm
        dislocated_verts = [dict(v) for v in curved_verts]
        dislocated_verts[1]['landmarks'] = [[p[0] + 9.0, p[1]] for p in curved_verts[1]['landmarks']]
        dislocated_verts[1]['spinous_tip'] = [curved_verts[1]['spinous_tip'][0] + 9.0, curved_verts[1]['spinous_tip'][1]]
        res_dis = compute_intervertebral_metrics(dislocated_verts)
        self.assertGreaterEqual(res_dis[1]['translation_mm'], 4.5)

    def test_coronal_sequence_tilt_compensation_endpoints(self):
        """4-vertebra sequence tilted uniformly 10 deg in coronal X-Z must not fire false coronal_step_mm
        at sequence endpoints; < 3 vertebra chain must not self-trigger coronal_step_mm."""
        lm_norm = [[15.0, 58.0], [15.0, 22.0], [30.0, 58.0], [30.0, 22.0], [45.0, 58.0], [45.0, 22.0]]
        slope = float(np.tan(np.deg2rad(10.0)))
        verts = []
        for lvl, z_val in [('T11', 150.0), ('T12', 120.0), ('L1', 90.0), ('L2', 60.0)]:
            pl = BodyPlane((slope * z_val, 0.0, z_val), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (80, 80))
            verts.append({'level': lvl, 'landmarks': lm_norm, 'spinous_tip': [65.0, 48.0], 'plane': pl})

        res_tilted = compute_intervertebral_metrics(verts)
        for item in res_tilted:
            self.assertLess(item['coronal_step_mm'], 0.5)
            self.assertLess(item['translation_mm'], 1.0)

        # < 3 LPS vertebrae (only 2 levels): endpoint noise does not self-trigger coronal_step_mm
        res_two = compute_intervertebral_metrics(verts[:2])
        for item in res_two:
            self.assertEqual(item['coronal_step_mm'], 0.0)

    def test_auto_segment_trims_attached_pedicle_stub_even_with_separated_spinous(self):
        """Hotfix 4.3b: off-center attached pedicle stub (rows 40:54, cols 46:60) must be trimmed
        even when a spinous island (cols 72:88) triggers has_separated_posterior_arch."""
        shape = (80, 100)
        ct = np.full(shape, 40.0, dtype=np.float32)
        valid = np.ones(shape, dtype=bool)
        hint = np.zeros(shape, dtype=bool)

        # Vertebral body: rows 20:56, cols 15:46 (height = 36 mm, center = 37.5)
        hint[20:56, 15:46] = True
        # Off-center pedicle stub attached to upper posterior border: rows 40:54 (14 mm), cols 46:60
        hint[40:54, 46:60] = True
        # Separated spinous process island across canal: cols 72:88
        hint[25:45, 72:88] = True
        ct[hint] = 400.0

        res = auto_segment_body(ct, valid, hint, (1.0, 1.0), return_details=True)
        body_mask = res['body_mask']
        arch_mask = res['posterior_arch_mask']

        self.assertIsNotNone(res['cut_col'])
        max_col = int(np.nonzero(body_mask)[1].max())
        self.assertLessEqual(max_col, 48)
        # Spinous island preserved in posterior_arch_mask alongside trimmed pedicle tail
        self.assertTrue(arch_mask[:, 72:88].any())
        self.assertTrue(arch_mask[:, 46:60].any())


if __name__ == '__main__':
    unittest.main()

