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
from spinal_ct.morphometry import BodySession, load_session, save_session, measure_heights
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


if __name__ == '__main__':
    unittest.main()
