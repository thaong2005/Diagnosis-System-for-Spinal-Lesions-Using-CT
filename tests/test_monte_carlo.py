from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from spinal_ct.localization import LEVELS
from spinal_ct.monte_carlo import aggregate_samples, perturb_hu


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.float32))
        self.image.SetSpacing((2., 3., 4.))
        self.image.SetOrigin((10., 20., 30.))
        self.image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
        self.level = LEVELS.index('L1')+1

    def write_samples(self, arrays):
        paths = []
        for i, values in enumerate(arrays):
            path = self.root/f'{i}.nii.gz'
            image = sitk.GetImageFromArray(values)
            image.CopyInformation(self.image)
            sitk.WriteImage(image, str(path))
            paths.append(path)
        return paths

    def test_seed_geometry_and_noise_moments(self):
        image = sitk.Image([60, 60, 30], sitk.sitkFloat32)
        image.SetSpacing((0.8, 0.8, 2.))
        original = sitk.GetArrayFromImage(image)
        a = perturb_hu(image, np.random.default_rng(42), 20.)
        b = perturb_hu(image, np.random.default_rng(42), 20.)
        c = perturb_hu(image, np.random.default_rng(43), 20.)
        arr = sitk.GetArrayFromImage(a)
        np.testing.assert_array_equal(arr, sitk.GetArrayFromImage(b))
        self.assertFalse(np.array_equal(arr, sitk.GetArrayFromImage(c)))
        np.testing.assert_array_equal(original, sitk.GetArrayFromImage(image))
        self.assertEqual(a.GetSpacing(), image.GetSpacing())
        self.assertAlmostEqual(float(arr.mean()), 0, delta=0.3)
        self.assertAlmostEqual(float(arr.std()), 20, delta=0.3)
        for bad in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                perturb_hu(image, np.random.default_rng(0), bad)

    def test_majority_disagreement_and_conditional_centroids(self):
        a = np.zeros((4,5,6), np.uint8)
        b, c = a.copy(), a.copy()
        a[1,2,1] = self.level
        b[1,2,2] = self.level
        # Third sample misses L1. Missing detections must contribute zero volume.
        paths = self.write_samples([a,b,c])
        report = aggregate_samples(self.image, paths, self.root/'mc')
        row = report['per_level'][self.level-1]
        self.assertAlmostEqual(row['detection_frequency'], 2/3)
        self.assertAlmostEqual(row['volume_mean_mm3'], 16)
        self.assertAlmostEqual(row['volume_std_mm3'], np.std([24,24,0],ddof=1))
        np.testing.assert_allclose(row['centroid_mean_lps_mm'], [7,14,34])
        np.testing.assert_allclose(row['centroid_std_lps_mm'], [np.sqrt(2),0,0])
        consensus = sitk.ReadImage(str(self.root/'mc/consensus.nii.gz'))
        self.assertEqual(consensus.GetDirection(), self.image.GetDirection())
        self.assertEqual(int(sitk.GetArrayFromImage(consensus).sum()), 0)
        d = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root/'mc/disagreement.nii.gz')))
        self.assertAlmostEqual(float(d[1,2,1]), 1/3, places=6)
        absent = report['per_level'][0]
        self.assertIsNone(absent['centroid_std_lps_mm'])
        self.assertIsNone(absent['centroid_mean_lps_mm'])
        self.assertFalse(list((self.root/'mc').glob('.votes-*')))

    def test_competing_labels_tie_is_not_assigned(self):
        a = np.zeros((4,5,6), np.uint8)
        a[1,2,1] = self.level
        b = a.copy(); b[1,2,1] = self.level+1
        report = aggregate_samples(self.image, self.write_samples([a,b]), self.root/'mc')
        self.assertEqual(report['no_majority_voxels'], 1)
        out = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root/'mc/consensus.nii.gz')))
        self.assertEqual(out[1,2,1], 0)

    def test_stable_samples_and_single_detection_std(self):
        a = np.zeros((4,5,6), np.uint8); a[1,2,1] = self.level
        report = aggregate_samples(self.image, self.write_samples([a,a,a]), self.root/'stable')
        row = report['per_level'][self.level-1]
        self.assertEqual(row['volume_std_mm3'], 0)
        self.assertEqual(row['centroid_std_lps_mm'], [0,0,0])
        out = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root/'stable/consensus.nii.gz')))
        np.testing.assert_array_equal(out, a)
        empty = np.zeros_like(a)
        report = aggregate_samples(self.image, self.write_samples([a,empty]), self.root/'single')
        self.assertIsNone(report['per_level'][self.level-1]['centroid_std_lps_mm'])

    def test_invalid_label_and_geometry_rejected(self):
        a = np.zeros((4,5,6), np.uint8)
        a[1,2,1] = 255
        with self.assertRaisesRegex(ValueError, 'Invalid vertebral label'):
            aggregate_samples(self.image, self.write_samples([a,a]), self.root/'bad-label')
        a[:] = 0
        paths = self.write_samples([a,a])
        bad = sitk.ReadImage(str(paths[1])); bad.SetOrigin((0,0,0))
        sitk.WriteImage(bad, str(paths[1]))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            aggregate_samples(self.image, paths, self.root/'bad-geometry')

    def test_chunked_votes_equal_direct_oracle_for_two_through_64_samples(self):
        rng = np.random.default_rng(87)
        for count in (2, 3, 4, 64):
            with self.subTest(samples=count):
                arrays = rng.integers(0, 26, size=(count, 4, 5, 6), dtype=np.uint8)
                arrays[:, 0] = 20  # Stable foreground, including 64 votes without overflow.
                arrays[:, 1] = 0   # Stable background counts in disagreement.
                paths = self.write_samples(arrays)
                folder = self.root/f'oracle-{count}'
                with patch('spinal_ct.monte_carlo.VOTE_CHUNK_VOXELS', 17):
                    report = aggregate_samples(self.image, paths, folder)
                votes = np.stack([(arrays == label).sum(axis=0) for label in range(26)])
                maximum = votes.max(axis=0)
                expected = np.where(maximum > count/2, votes.argmax(axis=0), 0)
                actual = sitk.GetArrayFromImage(sitk.ReadImage(str(folder/'consensus.nii.gz')))
                np.testing.assert_array_equal(actual, expected)
                disagreement = sitk.GetArrayFromImage(sitk.ReadImage(str(folder/'disagreement.nii.gz')))
                np.testing.assert_array_equal(disagreement, 1-maximum.astype(np.float32)/count)
                self.assertEqual(report['no_majority_voxels'], np.count_nonzero(maximum <= count/2))
                for label, row in enumerate(report['per_level'], 1):
                    sizes, centres = [], []
                    for sample in arrays:
                        xyz = np.argwhere(sample == label)[:, ::-1]
                        sizes.append(len(xyz)*24.)
                        if len(xyz):
                            centres.append(self.image.TransformContinuousIndexToPhysicalPoint(tuple(xyz.mean(axis=0))))
                    self.assertAlmostEqual(row['volume_mean_mm3'], np.mean(sizes))
                    self.assertAlmostEqual(row['volume_std_mm3'], np.std(sizes, ddof=1))
                    self.assertEqual(row['detected_samples'], len(centres))
                    if centres:
                        np.testing.assert_allclose(row['centroid_mean_lps_mm'], np.mean(centres, axis=0), atol=1e-10)
                    if len(centres) > 1:
                        np.testing.assert_allclose(row['centroid_std_lps_mm'], np.std(centres, axis=0, ddof=1), atol=1e-10)

    def test_cancellation_cleans_memory_maps(self):
        array = np.zeros((4, 5, 6), np.uint8)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ValueError('cancelled')
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            aggregate_samples(self.image, self.write_samples([array, array]), self.root/'cancelled',
                              check_cancelled=cancel)
        self.assertFalse(list((self.root/'cancelled').glob('.votes-*')))

    def test_statistics_use_original_ct_geometry_not_rounded_nifti_header(self):
        self.image.SetSpacing((.37890625, .37890625, .7))
        self.image.SetOrigin((10.123456789, -20.123456789, 700.123456789))
        array = np.zeros((4, 5, 6), np.uint8)
        array[1:4, 2:4, 1:5] = self.level
        paths = self.write_samples([array, array])
        report = aggregate_samples(self.image, paths, self.root/'precision')
        expected = self.image.TransformContinuousIndexToPhysicalPoint((2.5, 2.5, 2.))
        np.testing.assert_allclose(report['per_level'][self.level-1]['centroid_mean_lps_mm'],
                                   expected, rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
