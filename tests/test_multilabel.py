import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from spinal_ct.localization import (LEVELS, load_backend_labels, summarize_labels,
                                    summarize_masks, export_binary_masks, validate_label_image)


class MultilabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = sitk.Image((8, 9, 30), sitk.sitkFloat32)
        self.image.SetSpacing((.7, 1.2, 2.1))
        self.image.SetOrigin((10, -20, 35))
        theta = .3
        self.image.SetDirection((np.cos(theta), -np.sin(theta), 0,
                                 np.sin(theta), np.cos(theta), 0, 0, 0, -1))

    def test_every_backend_id_maps_to_correct_anatomical_level(self):
        raw = sitk.Image(self.image.GetSize(), sitk.sitkUInt8)
        raw.CopyInformation(self.image)
        for i in range(25):
            raw[2:4, 3:6, i:i+1] = 50-i
        path = self.root/'backend.nii'
        sitk.WriteImage(raw, str(path))
        labels = load_backend_labels(self.image, path)
        report = summarize_labels(self.image, labels)
        self.assertEqual([p['level'] for p in report['predictions']], list(LEVELS))
        for i in range(25):
            self.assertEqual(labels[2, 3, i], i+1)
        self.assertEqual(labels.GetDirection(), self.image.GetDirection())
        # Compare against the existing binary-mask implementation, including FOV flags.
        export_binary_masks(labels, self.root/'masks')
        legacy, combined = summarize_masks(self.image, self.root/'masks')
        self.assertEqual(report['warnings'], legacy['warnings'])
        self.assertEqual(report['status'], legacy['status'])
        np.testing.assert_array_equal(sitk.GetArrayFromImage(labels), sitk.GetArrayFromImage(combined))
        for new, old in zip(report['predictions'], legacy['predictions']):
            for key in ('level', 'label_id', 'bbox_voxel_xyz', 'touches_boundary', 'volume_mm3'):
                self.assertEqual(new[key], old[key])
            np.testing.assert_allclose(new['centroid_lps_mm'], old['centroid_lps_mm'], atol=1e-10)
            for bound in ('min', 'max'):
                np.testing.assert_allclose(new['roi_lps_mm'][bound], old['roi_lps_mm'][bound], atol=1e-10)

    def test_rejects_nonvertebral_fractional_and_wrong_geometry_outputs(self):
        for value in (1, 25, 51, 31.5):
            with self.subTest(value=value):
                raw = sitk.Image(self.image.GetSize(), sitk.sitkFloat32)
                raw.CopyInformation(self.image)
                raw[0, 0, 0] = value
                path = self.root/'bad.nii'
                sitk.WriteImage(raw, str(path))
                with self.assertRaisesRegex(ValueError, 'Unexpected label'):
                    load_backend_labels(self.image, path)
        raw[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid vertebral label'):
            validate_label_image(self.image, raw)
        raw.SetOrigin((0, 0, 0))
        sitk.WriteImage(raw, str(path))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            load_backend_labels(self.image, path)
