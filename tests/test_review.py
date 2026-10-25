import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from spinal_ct.review import ct_fingerprint, display_image, load_review, report_html


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = sitk.GetImageFromArray(np.arange(120, dtype=np.float32).reshape(4, 5, 6))
        self.image.SetSpacing((.5, 1, 2))
        self.image.SetOrigin((10, 20, 30))
        self.mask = sitk.Cast(self.image > 60, sitk.sitkUInt8) * 20
        sitk.WriteImage(self.mask, str(self.root/'vertebrae.nii.gz'))
        self.report = dict(task='vertebral_localization', status='indeterminate', predictions=[],
                           warnings=['T8 touches boundary'], input_fingerprint=ct_fingerprint(self.image))
        self.save()

    def save(self):
        (self.root/'prediction.json').write_text(json.dumps(self.report), encoding='utf-8')

    def test_fingerprint_detects_different_ct_with_same_geometry(self):
        different = self.image + 1
        self.assertNotEqual(ct_fingerprint(self.image), ct_fingerprint(different))
        with self.assertRaisesRegex(ValueError, 'CT khác'):
            load_review(self.root, different, different, ct_fingerprint(different))

    def test_verified_and_legacy_results(self):
        review = load_review(self.root, self.image, self.image, ct_fingerprint(self.image))
        self.assertTrue(review['verified'])
        np.testing.assert_array_equal(review['masks']['Baseline'], sitk.GetArrayFromImage(self.mask))
        self.report.pop('input_fingerprint')
        self.save()
        legacy = load_review(self.root, self.image, self.image, ct_fingerprint(self.image))
        self.assertFalse(legacy['verified'])
        self.assertIn('chỉ khớp geometry', legacy['warnings'][0])

    def test_reject_geometry_and_invalid_mask(self):
        self.mask.SetOrigin((0, 0, 0))
        sitk.WriteImage(self.mask, str(self.root/'vertebrae.nii.gz'))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            load_review(self.root, self.image, self.image, ct_fingerprint(self.image))
        self.mask.CopyInformation(self.image)
        self.mask[0, 0, 0] = 99
        sitk.WriteImage(self.mask, str(self.root/'vertebrae.nii.gz'))
        with self.assertRaisesRegex(ValueError, 'không hợp lệ'):
            load_review(self.root, self.image, self.image, ct_fingerprint(self.image))

    def test_oblique_grid_and_original_unchanged(self):
        theta = np.pi/6
        direction = (np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1)
        self.image.SetDirection(direction)
        reference = display_image(self.image)
        self.assertEqual(reference.GetDirection(), tuple(np.eye(3).ravel()))
        self.assertEqual(self.image.GetDirection(), direction)
        for point in ((0, 0, 0), (5, 4, 3)):
            index = reference.TransformPhysicalPointToContinuousIndex(self.image.TransformIndexToPhysicalPoint(point))
            self.assertTrue(all(-.5 <= i <= n-.5 for i, n in zip(index, reference.GetSize())))

    def test_report_escapes_text_and_does_not_include_paths(self):
        self.report['source_path'] = 'PRIVATE_INPUT_PATH'
        content = report_html(self.report, ['<script>alert(1)</script>'], 'case_demo')
        self.assertNotIn('<script>', content)
        self.assertIn('&lt;script&gt;', content)
        self.assertNotIn('PRIVATE_INPUT_PATH', content)
