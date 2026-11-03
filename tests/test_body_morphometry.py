"""Physical geometry, pixel-cell shapes, MC and saved-region integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from spinal_ct.body_region import BodyPlane, propose_plane, sample_plane, polygon_mask
from spinal_ct.blob_analysis import analyze_blob, normalize_region
from spinal_ct.area_monte_carlo import estimate_area
from spinal_ct.blob_detection import detect_blobs, assess_candidates
from spinal_ct.morphometry import (BodySession, load_session, save_session, measure_heights,
                                    BodyAutoParams, _fill_holes, _largest_anterior_component,
                                    auto_segment_body, auto_detect_landmarks)
from spinal_ct.review import ct_fingerprint
from tools.analyze_vertebral_body import main


def fixture():
    z, y, x = np.indices((36, 44, 24))
    source = sitk.GetImageFromArray((x+2*y+3*z).astype(np.float32))
    source.SetSpacing((.8, 1.2, 1.5))
    mask = np.zeros((36, 44, 24), np.uint8)
    mask[7:29, 8:36, 5:19] = 20
    labels = sitk.GetImageFromArray(mask)
    labels.CopyInformation(source)
    return source, labels


def session_fixture():
    source, labels = fixture()
    plane = propose_plane(source, labels, 20)
    ct, valid, hint = sample_plane(source, plane, labels, 20)
    session = BodySession(ct_fingerprint(source), 'L1', 'Baseline', plane, ct, valid, hint, age_group='adult')
    session.set_mask(polygon_mask([[12, 12], [30, 12], [30, 28], [12, 28]], ct.shape, plane.spacing_xy_mm))
    return source, labels, session


class BlobGeometryTests(unittest.TestCase):
    def test_rectangle_in_physical_units_and_pixel_cell_moments(self):
        mask = np.zeros((30, 50), bool)
        mask[5:25, 5:45] = True
        result = analyze_blob(mask, (.5, 2.))
        self.assertEqual(result['area_mm2'], 800)
        self.assertEqual(result['height_width_ratio'], 2)
        self.assertEqual(result['fill_ratio'], 0.5)
        self.assertEqual(result['oriented_box']['side_length_mm'], 40)
        self.assertEqual(result['solidity'], 1)
        self.assertAlmostEqual(result['ellipse']['major_length_mm'], 4*40/np.sqrt(12))
        self.assertAlmostEqual(result['ellipse']['minor_length_mm'], 4*20/np.sqrt(12))
        np.testing.assert_allclose(result['centroid_xy_mm'], [12.25, 29])

    def test_rotation_and_translation_invariance_with_spacing(self):
        mask = np.zeros((60, 80), bool)
        mask[15:45, 10:60] = True
        mask[30:45, 10:30] = False
        a = analyze_blob(mask, (.8, 1.4))
        b = analyze_blob(np.rot90(mask), (1.4, .8))
        c = analyze_blob(np.pad(mask, ((7, 5), (12, 2))), (.8, 1.4))
        for key in ('area_mm2', 'solidity', 'fill_ratio'):
            self.assertAlmostEqual(a[key], b[key], places=10)
            self.assertAlmostEqual(a[key], c[key], places=10)

    def test_disconnected_holes_and_boundary_preserved(self):
        mask = np.zeros((20, 30), bool)
        mask[:10, :10] = True
        mask[15:18, 25:28] = True
        mask[3:6, 3:6] = False
        before = mask.copy()
        result = analyze_blob(mask, (1, 1))
        self.assertEqual(result['area_mm2'], 100)
        self.assertEqual(result['component_count'], 2)
        self.assertIn('multiple_components', result['warnings'])
        self.assertIn('touches_plane_boundary', result['warnings'])
        self.assertLess(result['solidity'], 1)
        np.testing.assert_array_equal(mask, before)

    def test_single_cell_and_full_frame(self):
        for mask, expected_ratio in ((np.ones((1, 1), bool), 6/9),
                                     (np.ones((5, 9), bool), 270/324),
                                     (np.ones((6, 6), bool), 1.0)):
            spacing = (2, 2) if mask.shape == (6, 6) else (2, 3)
            result = analyze_blob(mask, spacing)
            self.assertEqual(result['area_mm2'], mask.size*spacing[0]*spacing[1])
            self.assertAlmostEqual(result['solidity'], 1)
            self.assertAlmostEqual(result['fill_ratio'], expected_ratio)

    def test_square_orientation_flag(self):
        result = analyze_blob(np.ones((10, 10), bool), (1, 1))
        self.assertFalse(result['ellipse']['orientation_reliable'])

    def test_full_frame_with_hole_uses_outer_hull(self):
        mask = np.ones((10, 20), bool)
        mask[3:7, 8:12] = False
        result = analyze_blob(mask, (1, 1))
        self.assertEqual(result['area_mm2'], 184)
        self.assertAlmostEqual(result['solidity'], 184/200)

    def test_invalid_masks_and_spacing(self):
        for mask, spacing in ((np.zeros((5, 5)), (1, 1)), (np.full((5, 5), 2), (1, 1)),
                              (np.ones((3, 4)), (0, 1)), (np.ones((3, 4)), (1, float('nan')))):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                analyze_blob(mask, spacing)

    def test_normalization_is_uniform_and_keeps_region_inside(self):
        mask = np.zeros((60, 90), bool)
        mask[10:30, 15:75] = True
        spacing = (.7, 2.)
        result = normalize_region(np.zeros(mask.shape), mask, spacing)
        transform = np.array(result['transform']['output_pixel_to_plane_mm'])
        scales = np.linalg.svd(transform[:2, :2], compute_uv=False)
        self.assertAlmostEqual(scales[0], scales[1])
        f = analyze_blob(mask, spacing)
        np.testing.assert_allclose(transform @ [127.5, 127.5, 1], [*f['centroid_xy_mm'], 1])
        inverse = np.linalg.inv(transform)
        corners = np.array(f['oriented_box']['corners_xy_mm'])
        out = np.column_stack((corners, np.ones(4))) @ inverse.T
        self.assertTrue((out[:, :2] > 2).all() and (out[:, :2] < 253).all())
        normalized = analyze_blob(result['mask'], (1, 1))
        self.assertAlmostEqual(normalized['ellipse']['axis_ratio'], f['ellipse']['axis_ratio'], delta=.05)
        self.assertEqual(result['mask'].shape, (256, 256))
        self.assertTrue((result['ct'][~result['mask']] == -1024).all())

    def test_monte_carlo_reference_reproducibility_and_multiple_seeds(self):
        rows, cols = np.indices((80, 100))
        mask = ((cols-50)/35)**2 + ((rows-40)/20)**2 <= 1
        a = estimate_area(mask, (.8, 1.3), samples=20_000, seed=9)
        self.assertEqual(a, estimate_area(mask, (.8, 1.3), samples=20_000, seed=9))
        self.assertEqual(a['convergence'][-1]['samples'], 20_000)
        self.assertIn('elbow', a)
        self.assertIn('samples', a['elbow'])
        self.assertIn('area_mm2', a['elbow'])
        errors = []
        for seed in range(8):
            result = estimate_area(mask, (.8, 1.3), samples=30_000, seed=seed)
            errors.append(result['error_vs_pixel_area_mm2'])
            self.assertLess(abs(errors[-1]), 6*result['standard_error_mm2'])
        self.assertLess(abs(np.mean(errors)), 10)

    def test_mc_all_hits_interval_and_invalid_params(self):
        mask = np.ones((20, 20), bool)
        result = estimate_area(mask, (2, 2), samples=100)
        self.assertEqual(result['area_mm2'], 1600)
        self.assertLess(result['sampling_interval95_mm2'][0], 1600)
        for kwargs in (dict(samples=2), dict(samples=100.5), dict(seed=-1), dict(seed=2.5)):
            with self.assertRaises(ValueError):
                estimate_area(mask, (1, 1), **kwargs)


class BodyPlaneTests(unittest.TestCase):
    def test_oblique_source_and_plane_sample_physical_linear_function(self):
        source, labels = fixture()
        angle = np.deg2rad(18)
        direction = [np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1]
        source.SetDirection(direction)
        labels.CopyInformation(source)
        plane = propose_plane(source, labels, 20, yaw_deg=12, tilt_deg=-8)
        ct, valid, hint = sample_plane(source, plane, labels, 20)
        self.assertTrue(hint.any())
        row, col = np.array(ct.shape)//2
        point = plane.point_lps([col*plane.spacing_xy_mm[0], row*plane.spacing_xy_mm[1]])
        x, y, z = source.TransformPhysicalPointToContinuousIndex(point)
        self.assertTrue(valid[row, col])
        self.assertAlmostEqual(ct[row, col], x+2*y+3*z, places=4)

    def test_missing_label_geometry_and_invalid_axes_rejected(self):
        source, labels = fixture()
        with self.assertRaises(ValueError):
            propose_plane(source, labels, 21)
        labels.SetOrigin((99, 0, 0))
        with self.assertRaises(ValueError):
            propose_plane(source, labels, 20)
        with self.assertRaises(ValueError):
            BodyPlane((0, 0, 0), (0, 1, 0), (0, 1, 0), (1, 1), (10, 10))

    def test_fov_mask_guard_and_polygon_guard(self):
        source, labels = fixture()
        plane = propose_plane(source, labels, 20, offset_mm=80)
        ct, valid, hint = sample_plane(source, plane, labels, 20)
        self.assertFalse(valid.any())
        session = BodySession(ct_fingerprint(source), 'L1', 'Baseline', plane, ct, valid, hint)
        with self.assertRaises(ValueError):
            session.set_mask(np.ones(ct.shape, bool))
        with self.assertRaises(ValueError):
            polygon_mask([[0, 0], [5, 5], [1000, 2]], (10, 10), (1, 1))


class SessionTests(unittest.TestCase):
    def test_review_required_and_edits_invalidate_result(self):
        _, _, session = session_fixture()
        with self.assertRaises(ValueError):
            session.measure(samples=100)
        session.approve()
        session.measure(samples=100)
        self.assertIsNotNone(session.result)
        session.set_landmarks([[14, 25]])
        self.assertFalse(session.reviewed)
        self.assertIsNone(session.result)
        with self.assertRaises(ValueError):
            session.approve()
        session.set_landmarks([])
        session.approve()
        session.mask[15, 15] = ~session.mask[15, 15]
        self.assertFalse(session.reviewed)

    def test_heights_order_and_no_ellipse_surrogate(self):
        _, _, session = session_fixture()
        points = [[12, 27], [12, 12], [20, 29], [20, 12], [28, 30], [28, 12]]
        session.set_landmarks(points)
        session.approve()
        result = session.measure(samples=100)
        self.assertEqual(result['heights']['anterior_mm'], 15)
        self.assertEqual(result['heights']['middle_mm'], 17)
        self.assertEqual(result['heights']['posterior_mm'], 18)
        self.assertAlmostEqual(result['heights']['anterior_posterior_ratio'], 15/18)
        with self.assertRaises(ValueError):
            measure_heights(points[::-1], session.plane)
        session.set_landmarks([])
        session.approve()
        self.assertIsNone(session.measure(samples=100)['heights'])

    def test_export_roundtrip_integrity_and_wrong_ct(self):
        source, _, session = session_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'result'
            session.approve()
            save_session(session, root, samples=1000)
            for name in ('region.json', 'measurements.json', 'preview.png', 'report.html', 'features.csv',
                         'normalized_ct.npy', 'normalized_mask.npy'):
                self.assertTrue((root/name).is_file(), name)
            loaded = load_session(root, source)
            self.assertTrue(loaded.reviewed)
            self.assertIsNone(loaded.result)
            np.testing.assert_array_equal(loaded.mask, session.mask)
            self.assertEqual(loaded.measure(samples=1000)['features'], session.result['features'])
            with self.assertRaises(ValueError):
                save_session(session, root, samples=100)
            with self.assertRaises(ValueError):
                load_session(root, source+1)
            np.save(root/'body_mask.npy', np.zeros(session.mask.shape, np.uint8))
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_session(root, source)

    def test_draft_and_modified_provenance(self):
        source, _, session = session_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'draft'
            save_session(session, root)
            self.assertFalse((root/'measurements.json').exists())
            self.assertFalse(load_session(root, source).reviewed)
            document = json.loads((root/'region.json').read_text())
            document['plane']['origin_lps_mm'][1] += 1
            (root/'region.json').write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_session(root, source)

    def test_export_failure_does_not_publish_partial_output(self):
        _, _, session = session_fixture()
        session.approve()
        with tempfile.TemporaryDirectory() as tmp, patch('spinal_ct.morphometry_render.export_preview', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                save_session(session, Path(tmp)/'result', samples=100)
            self.assertFalse((Path(tmp)/'result').exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_log_configuration_roundtrip(self):
        source, _, session = session_fixture()
        session.candidates = detect_blobs(session.ct, session.plane.spacing_xy_mm, session.valid)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'region'
            save_session(session, root)
            loaded = load_session(root, source)
            self.assertEqual(loaded.candidates, session.candidates)

    def test_legacy_sampling_restored_from_matching_measurement_report(self):
        source, _, session = session_fixture()
        session.approve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'legacy'
            save_session(session, root, samples=700, seed=1234)
            document = json.loads((root/'region.json').read_text())
            document.pop('area_sampling')
            (root/'region.json').write_text(json.dumps(document))
            loaded = load_session(root, source)
            self.assertEqual(loaded.sampling_config, dict(samples=700, seed=1234))
            self.assertIsNone(loaded.result)

    def test_draft_preserves_sampling_and_rejects_invalid_configuration(self):
        source, _, session = session_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'draft'
            save_session(session, root, samples=1500, seed=99)
            loaded = load_session(root, source)
            self.assertFalse(loaded.reviewed)
            self.assertEqual(loaded.sampling_config, dict(samples=1500, seed=99))
            with self.assertRaises(ValueError):
                save_session(session, Path(tmp)/'invalid', samples=1)
            self.assertFalse((Path(tmp)/'invalid').exists())

    def test_blob_evaluation_cli_and_duplicate_regions(self):
        from tools.evaluate_body_blobs import main as evaluate
        import sys
        source, _, session = session_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'ct.nii.gz'
            sitk.WriteImage(source, str(path))
            session.input_fingerprint = ct_fingerprint(sitk.ReadImage(str(path), sitk.sitkFloat32))
            session.approve()
            save_session(session, root/'region', samples=100)
            argv = ['evaluate', '--input', str(path), '--regions', str(root/'region'),
                    '--output', str(root/'evaluation.json'), '--tolerance-mm', '5']
            with patch.object(sys, 'argv', argv):
                self.assertEqual(evaluate(), 0)
            report = json.loads((root/'evaluation.json').read_text())
            self.assertEqual(report['regions'], 1)
            self.assertEqual(report['per_region'][0]['level'], 'L1')
            self.assertNotIn(str(root), json.dumps(report))
            duplicate = ['evaluate', '--input', str(path), '--regions', str(root/'region'), str(root/'region'),
                         '--output', str(root/'duplicate.json'), '--tolerance-mm', '5']
            with patch.object(sys, 'argv', duplicate):
                self.assertEqual(evaluate(), 1)
            self.assertFalse((root/'duplicate.json').exists())

    def test_cli_recomputes_without_overwriting(self):
        source, _, session = session_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'ct.nii.gz'
            sitk.WriteImage(source, str(path))
            # Fingerprint must reflect the decoded on-disk geometry.
            session.input_fingerprint = ct_fingerprint(sitk.ReadImage(str(path), sitk.sitkFloat32))
            session.approve()
            save_session(session, root/'region', samples=100)
            import sys
            argv = ['body', '--input', str(path), '--region', str(root/'region'), '--output', str(root/'recomputed'), '--samples', '500']
            with patch.object(sys, 'argv', argv):
                self.assertEqual(main(), 0)
            report = json.loads((root/'recomputed/measurements.json').read_text())
            self.assertEqual(report['monte_carlo_area']['samples'], 500)
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                main()


class LoGTests(unittest.TestCase):
    def test_candidate_matching_does_not_double_count_one_body(self):
        mask = np.ones((11, 11), bool)
        detection = dict(candidates=[dict(centre_xy_mm=[5, 5]), dict(centre_xy_mm=[6, 6]), dict(centre_xy_mm=[50, 50])])
        result = assess_candidates(detection, mask, (1, 1), tolerance_mm=2)
        self.assertTrue(result['matched'])
        self.assertEqual(result['extra_candidates'], 2)
        empty = assess_candidates(dict(candidates=[]), mask, (1, 1), tolerance_mm=2)
        self.assertFalse(empty['matched'])
        self.assertIsNone(empty['nearest_distance_mm'])

    def test_gaussian_centre_scale_and_physical_spacing(self):
        spacing = (.7, 1.4)
        row, col = np.indices((90, 130))
        x, y = col*spacing[0], row*spacing[1]
        sigma = 7.
        ct = -200+1000*np.exp(-((x-42)**2+(y-56)**2)/(2*sigma*sigma))
        result = detect_blobs(ct, spacing, min_sigma_mm=3, max_sigma_mm=12, scales=12)
        self.assertEqual(len(result['candidates']), 1)
        candidate = result['candidates'][0]
        np.testing.assert_allclose(candidate['centre_xy_mm'], [42, 56], atol=1.4)
        self.assertAlmostEqual(candidate['sigma_mm'], sigma, delta=1)

    def test_empty_constant_and_roi_constraint(self):
        ct = np.full((40, 40), 500.)
        self.assertEqual(detect_blobs(ct, (1, 1))['candidates'], [])
        ct[15:25, 15:25] = 1300
        self.assertEqual(detect_blobs(ct, (1, 1), np.zeros(ct.shape, bool))['candidates'], [])
        with self.assertRaises(ValueError):
            detect_blobs(ct, (3, 3), min_sigma_mm=1)


class AutoSegmentationHelperTests(unittest.TestCase):
    def test_body_auto_params_defaults_and_to_dict(self):
        from dataclasses import FrozenInstanceError
        params = BodyAutoParams()
        self.assertEqual(params.anterior_fraction, 0.65)
        self.assertEqual(params.neck_ratio, 0.60)
        self.assertEqual(params.ant_ref_span, 0.45)
        self.assertEqual(params.search_start, 0.35)
        self.assertEqual(params.search_end, 0.85)
        self.assertEqual(params.min_component_ratio, 0.15)
        self.assertEqual(params.inset_mm, 2.5)
        self.assertEqual(params.band_half_mm, 1.5)
        self.assertIsNone(params.min_hu)
        self.assertEqual(params.min_normal_ap_ratio, 0.80)
        self.assertEqual(params.max_normal_ap_ratio, 1.30)
        d = params.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['anterior_fraction'], 0.65)
        self.assertEqual(d['inset_mm'], 2.5)
        with self.assertRaises(FrozenInstanceError):
            params.anterior_fraction = 0.70
        with self.assertRaises(ValueError):
            BodyAutoParams(anterior_fraction=0.1)
        with self.assertRaises(ValueError):
            BodyAutoParams(neck_ratio=1.2)
        with self.assertRaises(ValueError):
            BodyAutoParams(inset_mm=-1.0)
        with self.assertRaises(ValueError):
            BodyAutoParams(band_half_mm=0.0)

    def test_fill_holes_internal_voids_and_immutability(self):
        # Empty mask
        empty = np.zeros((20, 20), bool)
        empty_filled = _fill_holes(empty)
        self.assertFalse(empty_filled.any())

        # Square ring with internal hole
        mask = np.zeros((30, 30), bool)
        mask[5:25, 5:25] = True
        mask[10:20, 10:20] = False  # internal void
        original = mask.copy()

        filled = _fill_holes(mask)
        # Check input immutability
        self.assertTrue(np.array_equal(mask, original))
        # Internal hole filled
        self.assertTrue(filled[10:20, 10:20].all())
        # Outer boundary unchanged
        self.assertTrue(filled[5:25, 5:25].all())
        self.assertFalse(filled[:5, :].any())
        self.assertFalse(filled[25:, :].any())

    def test_largest_anterior_component_selection_and_immutability(self):
        # Empty mask
        empty = np.zeros((40, 60), bool)
        self.assertFalse(_largest_anterior_component(empty).any())

        # Single component returns cleanly
        single = np.zeros((40, 60), bool)
        single[10:25, 10:25] = True
        self.assertTrue(np.array_equal(_largest_anterior_component(single), single))

        # Multi-component:
        # 1) Tiny noise: col 2..4, row 2..4 (area = 9 < 0.15 * 300)
        # 2) Anterior main body: col 10..24, row 10..29 (area = 15 * 20 = 300)
        # 3) Posterior arch: col 35..49, row 10..29 (area = 15 * 20 = 300)
        composite = np.zeros((40, 60), bool)
        composite[2:5, 2:5] = True
        composite[10:30, 10:25] = True
        composite[10:30, 35:50] = True
        original = composite.copy()

        chosen = _largest_anterior_component(composite, min_component_ratio=0.15)
        # Check input immutability
        self.assertTrue(np.array_equal(composite, original))
        # Must select anterior main body (cols 10..24), ignoring noise and posterior arch
        self.assertTrue(chosen[10:30, 10:25].all())
        self.assertFalse(chosen[2:5, 2:5].any())
        self.assertFalse(chosen[10:30, 35:50].any())


class AutoSegmentBodyTests(unittest.TestCase):
    def test_disconnected_posterior_arch_discarded(self):
        shape = (50, 70)
        ct = np.full(shape, 200.0)
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)
        # Vertebral body: anterior (cols 10..29, rows 10..30)
        hint[10:31, 10:30] = True
        # Posterior arch disconnected: (cols 45..55, rows 10..30)
        hint[10:31, 45:56] = True

        body = auto_segment_body(ct, valid, hint, (1.0, 1.0))
        self.assertTrue(body[10:31, 10:30].all())
        self.assertFalse(body[:, 45:56].any())

    def test_pedicle_bridge_neck_cut(self):
        shape = (50, 80)
        ct = np.full(shape, 200.0)
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)
        # Vertebral body: cols 10..30, rows 10..30 (height 20 px)
        hint[10:31, 10:31] = True
        # Narrow pedicle bridge: cols 31..38, rows 18..22 (height 4 px)
        hint[18:23, 31:39] = True
        # Spinous process: cols 39..60, rows 10..30 (height 20 px)
        hint[10:31, 39:61] = True

        body = auto_segment_body(ct, valid, hint, (1.0, 1.0))
        # Body is retained
        self.assertTrue(body[10:31, 10:30].all())
        # Posterior arch after bridge is trimmed off
        self.assertFalse(body[:, 40:].any())

    def test_anisotropic_spacing_and_min_hu(self):
        shape = (60, 60)
        ct = np.full(shape, 300.0)
        # Internal trabecular marrow with low HU
        ct[20:35, 20:35] = -50.0
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)
        hint[15:40, 15:40] = True

        params = BodyAutoParams(min_hu=100.0)
        body = auto_segment_body(ct, valid, hint, (0.7, 1.5), params=params)
        # The internal low-HU hole should be filled back in by _fill_holes
        self.assertTrue(body[20:35, 20:35].all())
        self.assertTrue(body[15:40, 15:40].all())

    def test_touches_plane_boundary_detected_by_analyze_blob(self):
        shape = (50, 50)
        ct = np.full(shape, 250.0)
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)
        # Touching boundary at row 0 and col 0
        hint[0:25, 0:25] = True

        body = auto_segment_body(ct, valid, hint, (1.0, 1.0))
        self.assertTrue(body[0:25, 0:25].all())
        features = analyze_blob(body, (1.0, 1.0))
        self.assertIn('touches_plane_boundary', features['warnings'])

    def test_invalid_inputs_and_edge_cases(self):
        shape = (30, 30)
        ct = np.zeros(shape)
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)

        # Shape mismatch
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid[:20, :], hint, (1.0, 1.0))
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint[:, :20], (1.0, 1.0))

        # Empty candidate
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint, (1.0, 1.0))

        # Too small region (< 3 pixels)
        hint_small = np.zeros(shape, bool)
        hint_small[5, 5] = True
        hint_small[5, 6] = True
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint_small, (1.0, 1.0))

        # Invalid params
        hint[10:20, 10:20] = True
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint, (1.0, 1.0),
                              params=BodyAutoParams(anterior_fraction=0.2))
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint, (1.0, 1.0),
                              params=BodyAutoParams(neck_ratio=1.5))
        with self.assertRaises(ValueError):
            auto_segment_body(ct, valid, hint, (1.0, 1.0),
                              params=BodyAutoParams(min_hu=float('nan')))

    def test_input_immutability(self):
        shape = (40, 50)
        ct = np.full(shape, 200.0)
        valid = np.ones(shape, bool)
        hint = np.zeros(shape, bool)
        hint[10:30, 10:30] = True

        orig_ct = ct.copy()
        orig_valid = valid.copy()
        orig_hint = hint.copy()

        _ = auto_segment_body(ct, valid, hint, (1.0, 1.0))
        self.assertTrue(np.array_equal(ct, orig_ct))
        self.assertTrue(np.array_equal(valid, orig_valid))
        self.assertTrue(np.array_equal(hint, orig_hint))


class AutoLandmarkTests(unittest.TestCase):
    def test_rectangle_equal_heights(self):
        shape = (60, 70)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(70, 60))
        mask = np.zeros(shape, bool)
        mask[20:41, 20:51] = True

        pts = auto_detect_landmarks(mask, plane)
        self.assertIsInstance(pts, list)
        self.assertEqual(len(pts), 6)
        heights = measure_heights(pts, plane)
        self.assertAlmostEqual(heights['anterior_mm'], 20.0, delta=0.5)
        self.assertAlmostEqual(heights['middle_mm'], 20.0, delta=0.5)
        self.assertAlmostEqual(heights['posterior_mm'], 20.0, delta=0.5)
        self.assertAlmostEqual(heights['anterior_posterior_ratio'], 1.0, delta=0.05)

    def test_staircase_edge_slanted_7deg(self):
        shape = (70, 80)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(80, 70))
        mask = np.zeros(shape, bool)
        slope = float(np.tan(np.deg2rad(7.0)))
        for c in range(20, 56):
            base_y = int(round(20.0 + slope * (c - 20)))
            mask[base_y:base_y + 22, c] = True

        pts = auto_detect_landmarks(mask, plane)
        heights = measure_heights(pts, plane)
        self.assertAlmostEqual(heights['anterior_posterior_ratio'], 1.0, delta=0.10)

    def test_wedge_fracture(self):
        shape = (60, 70)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(70, 60))
        mask = np.zeros(shape, bool)
        for c in range(20, 51):
            h_col = int(round(12.0 + 8.0 * (c - 20) / 30.0))
            mask[20:20 + h_col, c] = True

        pts = auto_detect_landmarks(mask, plane)
        heights = measure_heights(pts, plane)
        self.assertAlmostEqual(heights['anterior_posterior_ratio'], 0.60, delta=0.10)

    def test_biconcave_morphology(self):
        shape = (60, 70)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(70, 60))
        mask = np.zeros(shape, bool)
        for c in range(20, 51):
            dist_mid = abs(c - 35) / 15.0
            depress = int(round(3.0 * max(0.0, 1.0 - dist_mid ** 2)))
            mask[20 + depress:40 - depress, c] = True

        pts = auto_detect_landmarks(mask, plane)
        heights = measure_heights(pts, plane)
        self.assertLess(heights['middle_posterior_ratio'], 0.85)

    def test_anisotropic_spacing_physical_units(self):
        shape = (50, 60)
        sx, sy = 0.6, 1.2
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(sx, sy), size_xy=(60, 50))
        mask = np.zeros(shape, bool)
        mask[10:31, 15:46] = True

        pts = auto_detect_landmarks(mask, plane)
        heights = measure_heights(pts, plane)
        self.assertAlmostEqual(heights['anterior_mm'], 20 * sy, delta=0.5)
        self.assertAlmostEqual(heights['middle_mm'], 20 * sy, delta=0.5)
        self.assertAlmostEqual(heights['posterior_mm'], 20 * sy, delta=0.5)

    def test_empty_mask_rejected(self):
        shape = (40, 50)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(50, 40))
        empty = np.zeros(shape, bool)
        with self.assertRaises(ValueError):
            auto_detect_landmarks(empty, plane)

    def test_too_small_mask_rejected(self):
        shape = (40, 50)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(50, 40))
        narrow = np.zeros(shape, bool)
        narrow[20, 20:30] = True
        with self.assertRaises(ValueError):
            auto_detect_landmarks(narrow, plane)

        thin = np.zeros(shape, bool)
        thin[10:30, 20] = True
        with self.assertRaises(ValueError):
            auto_detect_landmarks(thin, plane)

    def test_multiple_components_rejected(self):
        shape = (40, 70)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(70, 40))
        multi = np.zeros(shape, bool)
        multi[10:30, 10:25] = True
        multi[10:30, 45:60] = True
        with self.assertRaises(ValueError):
            auto_detect_landmarks(multi, plane)

    def test_input_immutability(self):
        shape = (50, 60)
        plane = BodyPlane(origin_lps_mm=(0., 0., 0.), u_lps=(0., 1., 0.), v_lps=(0., 0., 1.),
                          spacing_xy_mm=(1.0, 1.0), size_xy=(60, 50))
        mask = np.zeros(shape, bool)
        mask[15:35, 15:45] = True
        orig = mask.copy()

        _ = auto_detect_landmarks(mask, plane)
        self.assertTrue(np.array_equal(mask, orig))


class AutoPopulateSessionTests(unittest.TestCase):
    def setUp(self):
        source, labels, base_session = session_fixture()
        self.source = source
        self.labels = labels
        self.plane = base_session.plane
        self.ct = base_session.ct
        self.valid = base_session.valid
        self.hint = base_session.hint
        self.fingerprint = ct_fingerprint(source)

        self.session = BodySession(
            input_fingerprint=self.fingerprint,
            level='L1',
            source_kind='Baseline',
            plane=self.plane,
            ct=self.ct,
            valid=self.valid,
            hint=self.hint,
            age_group='adult',
        )

    def test_auto_populate_populates_mask_landmarks_provenance_and_approve_flag(self):
        session = self.session
        self.assertFalse(session.reviewed)
        self.assertIsNone(session.auto_provenance)
        self.assertEqual(len(session.landmarks), 0)

        # Call auto_populate without approve
        params = BodyAutoParams(min_hu=100.0, inset_mm=2.5)
        session.auto_populate(params=params, approve=False)

        self.assertTrue(session.mask.any())
        self.assertEqual(len(session.landmarks), 6)
        self.assertFalse(session.reviewed)
        self.assertIsNotNone(session.auto_provenance)
        self.assertEqual(session.auto_provenance['method'], 'auto_v1')
        self.assertEqual(session.auto_provenance['params']['min_hu'], 100.0)
        self.assertEqual(session.auto_provenance['params']['inset_mm'], 2.5)

        # Call with approve=True
        session.auto_populate(params=params, approve=True)
        self.assertTrue(session.reviewed)
        self.assertIsNotNone(session.auto_provenance)

        # Measure result should include auto_provenance
        result = session.measure()
        self.assertEqual(result['auto_provenance'], session.auto_provenance)

    def test_manual_edits_clear_auto_provenance(self):
        session = self.session
        session.auto_populate(approve=True)
        self.assertIsNotNone(session.auto_provenance)

        # Modifying mask manually clears auto_provenance
        new_mask = session.mask.copy()
        new_mask[10, 10] = False
        session.set_mask(new_mask)
        self.assertIsNone(session.auto_provenance)

        # Auto populate again, then modifying landmarks clears auto_provenance
        session.auto_populate(approve=True)
        self.assertIsNotNone(session.auto_provenance)
        session.set_landmarks(session.landmarks[:5])
        self.assertIsNone(session.auto_provenance)

    def test_height_ratio_warnings_and_needs_review_status(self):
        session = self.session
        # Set a normal rectangle mask within session.valid and approve
        mask = polygon_mask([[12, 12], [30, 12], [30, 28], [12, 28]], session.ct.shape, session.plane.spacing_xy_mm)
        session.set_mask(mask)

        # Case 1: abnormal anterior_posterior_ratio (wedge compression: Ha=10, Hm=15, Hp=20 -> Ha/Hp=0.50)
        # cols: ant x=14.0, mid x=21.0, post x=28.0
        # ant: y=[15.0, 25.0] (height 10)
        # mid: y=[12.5, 27.5] (height 15)
        # post: y=[10.0, 30.0] (height 20)
        wedge_landmarks = [
            [14.0, 25.0], [14.0, 15.0],
            [21.0, 27.5], [21.0, 12.5],
            [28.0, 30.0], [28.0, 10.0],
        ]
        session.set_landmarks(wedge_landmarks)
        session.approve()
        result = session.measure()
        self.assertIn('abnormal_ap_height_ratio', result['warnings'])
        self.assertEqual(result['status'], 'needs_review')

        # Case 2: low middle_posterior_ratio (biconcave: Ha=20, Hm=12, Hp=20 -> Hm/Hp=0.60)
        biconcave_landmarks = [
            [14.0, 30.0], [14.0, 10.0],
            [21.0, 26.0], [21.0, 14.0],
            [28.0, 30.0], [28.0, 10.0],
        ]
        session.set_landmarks(biconcave_landmarks)
        session.approve()
        result = session.measure()
        self.assertIn('low_middle_posterior_ratio', result['warnings'])
        self.assertEqual(result['status'], 'needs_review')

    def test_save_and_load_session_roundtrips_auto_provenance_and_legacy_compatibility(self):
        session = self.session
        params = BodyAutoParams(min_hu=150.0)
        session.auto_populate(params=params, approve=True)
        prev_signature = session._signature()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'test_session'
            save_session(session, out_dir)

            # Check region.json content has auto_provenance
            doc = json.loads((out_dir / 'region.json').read_text(encoding='utf-8'))
            self.assertIn('auto_provenance', doc)
            self.assertEqual(doc['auto_provenance']['params']['min_hu'], 150.0)

            # Load session back and verify auto_provenance preserved
            loaded = load_session(out_dir, self.source, fingerprint=self.fingerprint)
            self.assertEqual(loaded.auto_provenance, session.auto_provenance)
            self.assertEqual(loaded._signature(), prev_signature)
            self.assertTrue(loaded.reviewed)

            # Check legacy session without auto_provenance field
            out_legacy = Path(tmpdir) / 'legacy_session'
            save_session(session, out_legacy)
            legacy_doc_path = out_legacy / 'region.json'
            legacy_doc = json.loads(legacy_doc_path.read_text(encoding='utf-8'))
            del legacy_doc['auto_provenance']
            legacy_doc_path.write_text(json.dumps(legacy_doc), encoding='utf-8')

            loaded_legacy = load_session(out_legacy, self.source, fingerprint=self.fingerprint)
            self.assertIsNone(loaded_legacy.auto_provenance)
            self.assertEqual(loaded_legacy._signature(), prev_signature)
            self.assertTrue(loaded_legacy.reviewed)


class AutoVsManualEvaluationCLITests(unittest.TestCase):
    def test_cli_evaluation_with_labels_success_and_bland_altman(self):
        from tools.evaluate_auto_vs_manual import main as eval_main
        import sys

        source, labels = fixture()
        plane = propose_plane(source, labels, 20)
        ct, valid, hint = sample_plane(source, plane, labels, 20)
        session = BodySession(ct_fingerprint(source), 'L1', 'Baseline', plane, ct, valid, hint, age_group='adult')

        # Create approved region with 6 landmarks
        session.auto_populate(approve=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ct_path = root / 'ct.nii.gz'
            labels_path = root / 'labels.nii.gz'
            reg_dir = root / 'manual_reg'
            out_json = root / 'eval_report.json'

            sitk.WriteImage(source, str(ct_path))
            sitk.WriteImage(labels, str(labels_path))
            # Session fingerprint must match decoded image
            session.input_fingerprint = ct_fingerprint(sitk.ReadImage(str(ct_path), sitk.sitkFloat32))
            session.approve()
            save_session(session, reg_dir)

            argv = [
                'evaluate_auto_vs_manual',
                '--input', str(ct_path),
                '--labels', str(labels_path),
                '--regions', str(reg_dir),
                '--output', str(out_json),
            ]
            with patch.object(sys, 'argv', argv):
                ret = eval_main()
            self.assertEqual(ret, 0)
            self.assertTrue(out_json.is_file())

            content = out_json.read_text(encoding='utf-8')
            report = json.loads(content)

            # Security / privacy assertion: no local paths
            self.assertNotIn(str(root), content)
            self.assertEqual(report['schema_version'], 1)
            self.assertEqual(report['task'], 'body_auto_vs_manual_evaluation')
            self.assertEqual(report['regions_evaluated'], 1)

            # Numerical tolerance checks
            summary = report['summary']
            # Since auto_populate was used to generate ground truth, Dice should be ~1.0 and error ~0.0
            self.assertAlmostEqual(summary['mean_dice'], 1.0, places=3)
            self.assertAlmostEqual(summary['mean_iou'], 1.0, places=3)
            self.assertAlmostEqual(summary['mean_landmark_error_mm'], 0.0, places=2)
            self.assertEqual(summary['total_landmarks_over_2mm'], 0)

            # Bland-Altman statistics check
            ba_ratio = summary['bland_altman']['anterior_posterior_ratio']
            self.assertAlmostEqual(ba_ratio['bias'], 0.0, places=3)
            self.assertAlmostEqual(ba_ratio['mae'], 0.0, places=3)
            self.assertAlmostEqual(ba_ratio['sd'], 0.0, places=3)

    def test_cli_evaluation_rejects_duplicate_regions_and_unconfirmed_regions(self):
        from tools.evaluate_auto_vs_manual import main as eval_main
        import sys

        source, labels = fixture()
        plane = propose_plane(source, labels, 20)
        ct, valid, hint = sample_plane(source, plane, labels, 20)
        session = BodySession(ct_fingerprint(source), 'L1', 'Baseline', plane, ct, valid, hint, age_group='adult')
        session.auto_populate(approve=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ct_path = root / 'ct.nii.gz'
            reg_dir = root / 'manual_reg'
            out_json = root / 'eval.json'

            sitk.WriteImage(source, str(ct_path))
            session.input_fingerprint = ct_fingerprint(sitk.ReadImage(str(ct_path), sitk.sitkFloat32))
            session.approve()
            save_session(session, reg_dir)

            # Duplicate region CLI invocation should fail (return 1)
            argv_dup = [
                'evaluate_auto_vs_manual',
                '--input', str(ct_path),
                '--regions', str(reg_dir), str(reg_dir),
                '--output', str(out_json),
            ]
            with patch.object(sys, 'argv', argv_dup):
                self.assertEqual(eval_main(), 1)
            self.assertFalse(out_json.exists())

            # Unconfirmed / draft region should fail
            draft_session = BodySession(ct_fingerprint(source), 'L1', 'Baseline', plane, ct, valid, hint, age_group='adult')
            draft_session.input_fingerprint = session.input_fingerprint
            draft_session.set_mask(session.mask)
            # Not approved
            draft_dir = root / 'draft_reg'
            save_session(draft_session, draft_dir)

            argv_draft = [
                'evaluate_auto_vs_manual',
                '--input', str(ct_path),
                '--regions', str(draft_dir),
                '--output', str(out_json),
            ]
            with patch.object(sys, 'argv', argv_draft):
                self.assertEqual(eval_main(), 1)
            self.assertFalse(out_json.exists())


if __name__ == '__main__':
    unittest.main()
