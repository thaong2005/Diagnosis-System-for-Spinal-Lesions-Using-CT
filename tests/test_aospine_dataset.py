"""Unit tests for Phase 2 AO Spine 2.5D Dual-ROI Dataset and Split Pipeline."""
import unittest
import numpy as np

from spinal_ct.aospine_dataset import (
    apply_ct_window,
    extract_body_canal_roi_25d,
    extract_posterior_arch_roi_25d,
    BONE_WINDOW_LEVEL,
    BONE_WINDOW_WIDTH,
    SOFT_TISSUE_WINDOW_LEVEL,
    SOFT_TISSUE_WINDOW_WIDTH,
)
from scripts.prepare_aospine_dataset import (
    map_aospine_to_concepts,
    filter_clean_normal_vertebrae,
    create_patient_stratified_splits,
)


class TestAOSpineDataset(unittest.TestCase):
    """Test CT windowing and 2.5D Dual-ROI tensor extraction."""

    def test_apply_ct_window_bone(self):
        """Test bone window mapping (-500 to +1300 HU -> [0.0, 1.0])."""
        hu = np.array([-1000.0, -500.0, 400.0, 1300.0, 2000.0], dtype=np.float32)
        norm = apply_ct_window(hu, BONE_WINDOW_LEVEL, BONE_WINDOW_WIDTH)
        self.assertEqual(norm.dtype, np.float32)
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[1], 0.0)
        self.assertAlmostEqual(norm[2], 0.5)
        self.assertAlmostEqual(norm[3], 1.0)
        self.assertAlmostEqual(norm[4], 1.0)

    def test_apply_ct_window_soft_tissue(self):
        """Test soft tissue window mapping (-125 to +225 HU -> [0.0, 1.0])."""
        hu = np.array([-200.0, -125.0, 50.0, 225.0, 500.0], dtype=np.float32)
        norm = apply_ct_window(hu, SOFT_TISSUE_WINDOW_LEVEL, SOFT_TISSUE_WINDOW_WIDTH)
        self.assertEqual(norm.dtype, np.float32)
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[1], 0.0)
        self.assertAlmostEqual(norm[2], 0.5)
        self.assertAlmostEqual(norm[3], 1.0)
        self.assertAlmostEqual(norm[4], 1.0)

    def test_apply_ct_window_invalid_width(self):
        """Test error when window width is non-positive."""
        with self.assertRaises(ValueError):
            apply_ct_window(np.array([100.0]), wl=400.0, ww=0.0)

    def test_extract_body_canal_roi_25d_from_2d(self):
        """Test 5-channel ROI extraction from 2D slice."""
        ct_2d = np.full((100, 120), -100.0, dtype=np.float32)
        # Vertebral body: rows 30 to 60, cols 20 to 50
        ct_2d[30:60, 20:50] = 350.0
        # Spinal canal behind posterior wall: rows 30 to 60, cols 50 to 75
        ct_2d[30:60, 50:75] = 20.0

        mask_2d = np.zeros((100, 120), dtype=bool)
        mask_2d[30:60, 20:50] = True

        spacing = (1.0, 1.0)
        roi_5d = extract_body_canal_roi_25d(ct_2d, mask_2d, spacing, target_size=(128, 128))

        self.assertEqual(roi_5d.shape, (5, 128, 128))
        self.assertEqual(roi_5d.dtype, np.float32)
        self.assertGreaterEqual(float(roi_5d.min()), 0.0)
        self.assertLessEqual(float(roi_5d.max()), 1.0)

    def test_extract_body_canal_roi_25d_from_3d(self):
        """Test 5-channel ROI extraction from 3D volume with parasagittal and axial channels."""
        # 3D volume (Z=80, Y=100, X=60)
        vol = np.full((80, 100, 60), -200.0, dtype=np.float32)
        # Vertebral body centered at X=30, rows 25 to 55, cols 20 to 50
        vol[25:55, 20:50, 20:40] = 400.0

        mask_2d = np.zeros((80, 100), dtype=bool)
        mask_2d[25:55, 20:50] = True

        spacing_xyz = (1.0, 1.0, 1.0)
        roi_5d = extract_body_canal_roi_25d(vol, mask_2d, spacing_xyz, slice_x_index=30, target_size=(128, 128))

        self.assertEqual(roi_5d.shape, (5, 128, 128))
        self.assertEqual(roi_5d.dtype, np.float32)
        self.assertGreaterEqual(float(roi_5d.min()), 0.0)
        self.assertLessEqual(float(roi_5d.max()), 1.0)

    def test_extract_posterior_arch_roi_25d_from_3d(self):
        """Test 5-channel Posterior Arch ROI extraction covering spinous process and tension band."""
        vol = np.full((80, 100, 60), -200.0, dtype=np.float32)
        # Posterior arch: rows 20 to 60, cols 50 to 90
        vol[20:60, 50:90, 25:35] = 300.0

        arch_mask = np.zeros((80, 100), dtype=bool)
        arch_mask[20:60, 50:90] = True

        spacing_xyz = (1.0, 1.0, 1.0)
        roi_5d = extract_posterior_arch_roi_25d(vol, arch_mask, spacing_xyz, slice_x_index=30, target_size=(128, 128))

        self.assertEqual(roi_5d.shape, (5, 128, 128))
        self.assertEqual(roi_5d.dtype, np.float32)
        self.assertGreaterEqual(float(roi_5d.min()), 0.0)
        self.assertLessEqual(float(roi_5d.max()), 1.0)


class TestAOSpineConceptMapping(unittest.TestCase):
    """Test multi-task concept mapping and mask validity."""

    def test_map_normal(self):
        res = map_aospine_to_concepts('Normal')
        self.assertEqual(res['is_fractured'], 0)
        self.assertEqual(res['y_concepts'], [0, 0, 0, 0])
        self.assertEqual(res['concept_mask'], [1, 1, 1, 1])

    def test_map_a1_compression(self):
        res = map_aospine_to_concepts('A1')
        self.assertEqual(res['is_fractured'], 1)
        self.assertEqual(res['y_concepts'], [0, 0, 0, 0])
        self.assertEqual(res['concept_mask'], [1, 1, 1, 1])

    def test_map_a3_burst(self):
        res = map_aospine_to_concepts('A3')
        self.assertEqual(res['is_fractured'], 1)
        self.assertEqual(res['y_concepts'], [1, 0, 0, 0])
        self.assertEqual(res['concept_mask'], [1, 1, 1, 1])

    def test_map_a4_burst_complete(self):
        res = map_aospine_to_concepts('A4')
        self.assertEqual(res['is_fractured'], 1)
        self.assertEqual(res['y_concepts'], [1, 1, 0, 0])
        self.assertEqual(res['concept_mask'], [1, 1, 1, 1])

    def test_map_b2_distraction(self):
        # B2 with intact posterior wall
        res = map_aospine_to_concepts('B2', {'delta_posterior_mm': 0.0})
        self.assertEqual(res['is_fractured'], 1)
        self.assertEqual(res['y_concepts'], [0, 0, 1, 0])
        self.assertEqual(res['concept_mask'], [1, 0, 1, 1])

        # B2 with retropulsed fragment (secondary A3 burst)
        res_burst = map_aospine_to_concepts('B2', {'delta_posterior_mm': 3.5})
        self.assertEqual(res_burst['y_concepts'], [1, 0, 1, 0])
        self.assertEqual(res_burst['concept_mask'], [1, 0, 1, 1])

    def test_map_c_translation(self):
        res = map_aospine_to_concepts('C')
        self.assertEqual(res['is_fractured'], 1)
        self.assertEqual(res['y_concepts'][2:], [1, 1])
        self.assertEqual(res['concept_mask'][2:], [1, 1])


class TestAOSpineSplitsAndFiltering(unittest.TestCase):
    """Test clean normal filtering and 5-fold patient stratification."""

    def test_filter_clean_normal_vertebrae(self):
        """Test that ambiguous levels adjacent to fracture are discarded from normal pool."""
        records = [
            # Ground truth fracture
            {'patient_id': 'p1', 'level': 'L1', 'aospine_class': 'A3', 'heights': {'anterior_posterior_ratio': 0.65}},
            # Clean T12 adjacent to L1 (Ha/Hp = 0.92 >= 0.85) -> kept
            {'patient_id': 'p1', 'level': 'T12', 'aospine_class': 'Normal', 'heights': {'anterior_posterior_ratio': 0.92, 'middle_posterior_ratio': 0.90}},
            # Ambiguous L2 adjacent to L1 (Ha/Hp = 0.80 < 0.85) -> filtered out!
            {'patient_id': 'p1', 'level': 'L2', 'aospine_class': 'Normal', 'heights': {'anterior_posterior_ratio': 0.80, 'middle_posterior_ratio': 0.82}},
            # Distant L4 (level distance = 3) with mild 0.82 -> kept because not adjacent
            {'patient_id': 'p1', 'level': 'L4', 'aospine_class': 'Normal', 'heights': {'anterior_posterior_ratio': 0.82, 'middle_posterior_ratio': 0.86}},
        ]
        filtered = filter_clean_normal_vertebrae(records)
        levels = [r['level'] for r in filtered]
        self.assertIn('L1', levels)
        self.assertIn('T12', levels)
        self.assertNotIn('L2', levels)  # Ambiguous adjacent filtered out
        self.assertIn('L4', levels)

    def test_create_patient_stratified_splits_zero_leakage(self):
        """Test 100% patient isolation across all 5 folds (zero leakage)."""
        # Create dataset with 20 patients across multiple classes
        classes = ['A4'] * 5 + ['A3'] * 5 + ['B2'] * 4 + ['C'] * 4 + ['A1'] * 1 + ['B3'] * 1
        records = []
        for i, c in enumerate(classes):
            pid = f"pat_{i:02d}"
            # Multiple levels per patient
            records.append({'patient_id': pid, 'sample_id': f"{pid}_fx", 'level': 'L1', 'aospine_class': c})
            records.append({'patient_id': pid, 'sample_id': f"{pid}_n1", 'level': 'T12', 'aospine_class': 'Normal'})
            records.append({'patient_id': pid, 'sample_id': f"{pid}_n2", 'level': 'L2', 'aospine_class': 'Normal'})

        splits = create_patient_stratified_splits(records, n_splits=5, seed=42)
        self.assertTrue(splits['zero_leakage_verified'])
        self.assertEqual(splits['n_splits'], 5)
        self.assertEqual(len(splits['folds']), 5)

        for fold in splits['folds']:
            train_pids = set(fold['train_patient_ids'])
            val_pids = set(fold['val_patient_ids'])
            rare_pids = set(splits['rare_class_challenge_set'])
            # Zero patient leakage
            self.assertEqual(train_pids & val_pids, set())
            # Zero leakage of rare challenge patients into training set
            self.assertEqual(train_pids & rare_pids, set())
            self.assertEqual(len(train_pids) + len(val_pids) + len(rare_pids), splits['total_patients'])

            # All samples of a patient must be together
            train_sids = set(fold['train_sample_ids'])
            val_sids = set(fold['val_sample_ids'])
            self.assertEqual(train_sids & val_sids, set())


if __name__ == '__main__':
    unittest.main()
