"""Unit tests for Phase 2 AO Spine 2.5D Dual-ROI Dataset and Split Pipeline."""
import unittest
from pathlib import Path
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
from spinal_ct.prototype_bank import (
    extract_roi_prototype_features,
    AOSpinePrototypeBank,
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

    def test_extract_body_canal_roi_25d_preserves_aspect_ratio(self):
        """Test that Fixed Physical FOV preserves true geometric aspect ratio of vertebrae without stretching."""
        # 2D CT slice (H=120, W=120) with 1.0 mm/pixel spacing
        ct_2d = np.full((120, 120), -100.0, dtype=np.float32)
        mask_2d = np.zeros((120, 120), dtype=bool)

        # Create a rectangular vertebral body with severe collapse:
        # Height = 20 mm (rows 50 to 69), Width = 40 mm (cols 30 to 69)
        # Ground-truth aspect ratio = 20.0 / 40.0 = 0.50
        mask_2d[50:70, 30:70] = True
        ct_2d[50:70, 30:70] = 400.0

        spacing = (1.0, 1.0)
        roi_5d = extract_body_canal_roi_25d(
            ct_2d, mask_2d, spacing, target_size=(128, 128), fov_mm=80.0, posterior_shift_mm=0.0
        )

        # Verify vertebral body on Mask Channel (Ch 4, value >= 0.95)
        body_pixels = (roi_5d[4] >= 0.95)
        self.assertTrue(body_pixels.any(), "Vertebral body mask must be present on Ch 4")
        occ_r, occ_c = np.where(body_pixels)
        h_roi = float(occ_r.max() - occ_r.min() + 1)
        w_roi = float(occ_c.max() - occ_c.min() + 1)
        observed_aspect_ratio = h_roi / w_roi

        # Expected: exactly 32 px / 64 px = 0.50. Under old bounding box resize, this was ~1.0
        self.assertAlmostEqual(observed_aspect_ratio, 0.50, delta=0.06)
        self.assertLess(observed_aspect_ratio, 0.70, "Aspect ratio must NOT be stretched toward 1.0")

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


class TestAOSpinePrototypeBank(unittest.TestCase):
    """Test Prototype Bank fold isolation, zero patient leakage verification, and classification."""

    def setUp(self):
        # Synthetic 5-channel body and arch ROIs
        self.body_roi = np.zeros((5, 128, 128), dtype=np.float32)
        # Vertebral body mask (Ch 4 >= 0.95)
        self.body_roi[4, 40:80, 40:80] = 1.0
        # Bone intensity (Ch 0)
        self.body_roi[0, 40:80, 40:80] = 0.65
        # Soft tissue (Ch 3)
        self.body_roi[3, 40:80, 40:80] = 0.45

        self.arch_roi = np.zeros((5, 128, 128), dtype=np.float32)
        self.arch_roi[4, 30:60, 70:110] = 1.0
        self.arch_roi[0, 30:60, 70:110] = 0.55
        self.arch_roi[3, 30:60, 70:110] = 0.35

    def test_extract_roi_prototype_features(self):
        """Test feature extraction generates valid 16-D unit-normalized vector."""
        feats = extract_roi_prototype_features(self.body_roi, self.arch_roi)
        self.assertEqual(feats.shape, (16,))
        self.assertEqual(feats.dtype, np.float32)
        self.assertFalse(np.isnan(feats).any(), "Features must not contain NaN")
        self.assertFalse(np.isinf(feats).any(), "Features must not contain Inf")
        norm = np.linalg.norm(feats)
        self.assertAlmostEqual(float(norm), 1.0, places=4, msg="Feature vector must be L2-normalized")

    def test_prototype_bank_fold_isolation_and_zero_leakage(self):
        """Test Prototype Bank trains on fold train samples and evaluates on val samples with zero leakage."""
        # Setup synthetic dataset with 10 patients across 3 classes
        samples = []
        classes = ['Normal', 'A3', 'A4', 'B2', 'C']
        for i in range(10):
            pid = f"patient_{i:02d}"
            c = classes[i % len(classes)]
            # 2 samples per patient
            f1 = extract_roi_prototype_features(self.body_roi, self.arch_roi)
            # Add slight perturbation for distinct samples
            f2 = f1 + np.random.RandomState(i).normal(0, 0.05, f1.shape).astype(np.float32)
            f2 /= np.linalg.norm(f2)

            samples.append({
                'sample_id': f"{pid}_T12",
                'patient_id': pid,
                'aospine_class': 'Normal',
                'prototype_features': f1.tolist(),
            })
            samples.append({
                'sample_id': f"{pid}_L1",
                'patient_id': pid,
                'aospine_class': c,
                'prototype_features': f2.tolist(),
            })

        train_patients = {'patient_00', 'patient_01', 'patient_02', 'patient_03', 'patient_04', 'patient_05', 'patient_06'}
        val_patients = {'patient_07', 'patient_08', 'patient_09'}

        train_samples = [s['sample_id'] for s in samples if s['patient_id'] in train_patients]
        val_samples = [s['sample_id'] for s in samples if s['patient_id'] in val_patients]

        bank = AOSpinePrototypeBank()
        bank.build_from_fold(
            samples=samples,
            train_sample_ids=train_samples,
            train_patient_ids=train_patients,
            fold_id=0,
        )

        self.assertGreater(len(bank.prototypes), 0, "Prototypes must be computed")
        self.assertEqual(bank.trained_patient_ids, train_patients)

        # Evaluate on validation set
        eval_res = bank.evaluate_fold_val(
            samples=samples,
            val_sample_ids=val_samples,
            val_patient_ids=val_patients,
        )

        self.assertTrue(eval_res['zero_leakage_verified'])
        self.assertEqual(eval_res['total_val_samples'], len(val_samples))
        self.assertGreaterEqual(eval_res['accuracy'], 0.0)

    def test_prototype_bank_leakage_detection_raises_error(self):
        """Test Prototype Bank immediately raises RuntimeError if patient leakage occurs."""
        samples = [
            {'sample_id': 'p1_L1', 'patient_id': 'p1', 'aospine_class': 'A3', 'prototype_features': [1.0] + [0.0]*15},
            {'sample_id': 'p2_L1', 'patient_id': 'p2', 'aospine_class': 'Normal', 'prototype_features': [0.0, 1.0] + [0.0]*14},
        ]

        bank = AOSpinePrototypeBank()
        # Case A: Training set includes a sample from an unauthorized patient
        with self.assertRaises(RuntimeError):
            bank.build_from_fold(
                samples=samples,
                train_sample_ids=['p1_L1', 'p2_L1'],
                train_patient_ids=['p1'],  # p2 is unauthorized in train!
            )

        # Case B: Val set shares a patient with train set
        bank.build_from_fold(
            samples=samples,
            train_sample_ids=['p1_L1'],
            train_patient_ids=['p1'],
        )
        with self.assertRaises(RuntimeError):
            bank.evaluate_fold_val(
                samples=samples,
                val_sample_ids=['p1_L1'],
                val_patient_ids=['p1'],  # p1 is in train! Leaked!
            )

    def test_extract_roi_prototype_features_with_morph_metrics(self):
        """Test optional morph_metrics concatenation expands vector to 22-D with valid values."""
        m_meta = {
            'ha_hp_ratio': 0.65,
            'hm_hp_ratio': 0.70,
            'relative_collapse_ratio': 0.65,
            'delta_posterior_mm': 3.5,
            'isp_ratio': 1.8,
            'translation_mm': 4.2,
        }
        feats = extract_roi_prototype_features(self.body_roi, self.arch_roi, morph_metrics=m_meta)
        self.assertEqual(feats.shape, (22,))
        self.assertEqual(feats.dtype, np.float32)
        self.assertFalse(np.isnan(feats).any())
        norm = np.linalg.norm(feats)
        self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_normal_prototypes_by_level_and_fallback(self):
        """Test level-wise normal prototypes separate T10..L5 and fallback when levels are missing."""
        samples = []
        # Populate levels T12 and L1 with multiple normals
        for i in range(4):
            pid = f"norm_pat_{i}"
            f_vec = [0.1 * i] * 16
            samples.append({'sample_id': f"{pid}_T12", 'patient_id': pid, 'level': 'T12', 'aospine_class': 'Normal', 'prototype_features': f_vec})
            samples.append({'sample_id': f"{pid}_L1", 'patient_id': pid, 'level': 'L1', 'aospine_class': 'Normal', 'prototype_features': f_vec})

        train_sids = [s['sample_id'] for s in samples]
        train_pids = {s['patient_id'] for s in samples}

        bank = AOSpinePrototypeBank()
        bank.build_from_fold(samples, train_sids, train_pids)

        # T12 and L1 should have count = 4
        self.assertEqual(bank.normal_prototypes_by_level['T12']['count'], 4)
        self.assertEqual(bank.normal_prototypes_by_level['L1']['count'], 4)

        # T10 has 0 samples, should fallback cleanly to global normal prototype
        self.assertEqual(bank.normal_prototypes_by_level['T10']['count'], 0)
        self.assertIsNotNone(bank.normal_prototypes_by_level['T10']['mean'])

        # Query T10: should use global normal prototype without error
        res_t10 = bank.compute_normal_anomaly_score([0.2] * 16, level='T10')
        self.assertEqual(res_t10['reference_level'], 'global')
        self.assertFalse(res_t10['level_matched'])

        # Query T12: should match level T12
        res_t12 = bank.compute_normal_anomaly_score([0.2] * 16, level='T12')
        self.assertEqual(res_t12['reference_level'], 'T12')
        self.assertTrue(res_t12['level_matched'])

    def test_compute_normal_anomaly_score(self):
        """Test compute_normal_anomaly_score outputs low p_anomaly for normal and high for fracture."""
        norm_feat = np.array([0.5, 0.1, 0.6, 0.02, 0.0, 0.0, 0.65, 0.98, 0.95, 0.05, 0.45, 0.20, 0.05, 0.08, 0.55, 0.08], dtype=np.float32)
        norm_feat /= np.linalg.norm(norm_feat)

        # Severe burst fracture features (high canal intrusion, collapsed height, asymmetric)
        frac_feat = np.array([0.7, 0.35, 0.85, 0.25, 0.45, 0.65, 0.35, 0.50, 0.55, 0.35, 0.30, 0.45, 0.20, 0.05, 0.70, 0.25], dtype=np.float32)
        frac_feat /= np.linalg.norm(frac_feat)

        samples = []
        for i in range(6):
            pid = f"pat_{i}"
            # Add slight noise to normal features
            noise = np.random.RandomState(i).normal(0, 0.01, 16).astype(np.float32)
            n_vec = (norm_feat + noise) / np.linalg.norm(norm_feat + noise)
            samples.append({'sample_id': f"{pid}_L1", 'patient_id': pid, 'level': 'L1', 'aospine_class': 'Normal', 'prototype_features': n_vec.tolist()})

        # Add 1 fracture in training
        samples.append({'sample_id': 'pat_fx_L1', 'patient_id': 'pat_fx', 'level': 'L1', 'aospine_class': 'A3', 'prototype_features': frac_feat.tolist()})

        train_sids = [s['sample_id'] for s in samples]
        train_pids = {s['patient_id'] for s in samples}

        bank = AOSpinePrototypeBank()
        bank.build_from_fold(samples, train_sids, train_pids)

        score_norm = bank.compute_normal_anomaly_score(norm_feat, level='L1')
        score_frac = bank.compute_normal_anomaly_score(frac_feat, level='L1')

        self.assertLess(score_norm['p_anomaly'], 0.45, "Normal vertebra must have low p_anomaly")
        self.assertGreater(score_frac['p_anomaly'], 0.60, "Severe burst fracture must have high p_anomaly")
        self.assertGreater(score_frac['anomaly_distance'], score_norm['anomaly_distance'])

    def test_match_lesion_prototypes_safe_missing_classes(self):
        """Test match_lesion_prototypes handles missing lesion classes (e.g. A2, B3) safely without NaN."""
        samples = [
            {'sample_id': 'p1_L1', 'patient_id': 'p1', 'level': 'L1', 'aospine_class': 'Normal', 'prototype_features': [0.1] * 16},
            {'sample_id': 'p2_L1', 'patient_id': 'p2', 'level': 'L1', 'aospine_class': 'A3', 'prototype_features': [0.5] * 16},
        ]
        bank = AOSpinePrototypeBank()
        bank.build_from_fold(samples, ['p1_L1', 'p2_L1'], {'p1', 'p2'})

        # Classes without samples (A1, A2, A4, B2, B3, C)
        self.assertFalse(bank.lesion_prototypes_by_class['A2']['has_samples'])
        self.assertEqual(bank.lesion_prototypes_by_class['A2']['sample_count'], 0)
        self.assertIsNone(bank.lesion_prototypes_by_class['A2']['centroid'])

        # Match query: should predict available class A3 safely without NaN
        res = bank.match_lesion_prototypes(np.array([0.4] * 16, dtype=np.float32))
        self.assertEqual(res['predicted_lesion_class'], 'A3')
        self.assertIn('A3', res['similarities'])
        self.assertNotIn('A2', res['similarities'])
        self.assertFalse(np.isnan(res['confidence']))

    def test_save_and_load_preserves_inference(self):
        """Test save() and load() roundtrip preserves exact inference results and state."""
        import tempfile
        samples = [
            {'sample_id': 'p1_T12', 'patient_id': 'p1', 'level': 'T12', 'aospine_class': 'Normal', 'prototype_features': [0.2] * 16},
            {'sample_id': 'p2_L1', 'patient_id': 'p2', 'level': 'L1', 'aospine_class': 'A3', 'prototype_features': [0.6] * 16},
            {'sample_id': 'p3_L2', 'patient_id': 'p3', 'level': 'L2', 'aospine_class': 'B2', 'prototype_features': [0.8] * 16},
        ]
        bank = AOSpinePrototypeBank()
        bank.build_from_fold(samples, ['p1_T12', 'p2_L1', 'p3_L2'], {'p1', 'p2', 'p3'}, fold_id=2)

        q = np.array([0.55] * 16, dtype=np.float32)
        pred_orig = bank.predict(q, level='L1')
        anom_orig = bank.compute_normal_anomaly_score(q, level='L1')

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / 'bank_fold2.json'
            bank.save(json_file)
            self.assertTrue(json_file.is_file())

            loaded_bank = AOSpinePrototypeBank().load(json_file)
            self.assertEqual(loaded_bank.fold_id, 2)
            self.assertEqual(loaded_bank.feature_dim, 16)
            self.assertEqual(loaded_bank.trained_patient_ids, {'p1', 'p2', 'p3'})

            pred_loaded = loaded_bank.predict(q, level='L1')
            anom_loaded = loaded_bank.compute_normal_anomaly_score(q, level='L1')

            self.assertEqual(pred_orig['predicted_class'], pred_loaded['predicted_class'])
            self.assertAlmostEqual(pred_orig['confidence'], pred_loaded['confidence'], places=4)
            self.assertAlmostEqual(anom_orig['p_anomaly'], anom_loaded['p_anomaly'], places=4)
            self.assertAlmostEqual(anom_orig['anomaly_distance'], anom_loaded['anomaly_distance'], places=4)


class TestGroundTruthAndSeriesSelection(unittest.TestCase):
    """Phase 4.1 regression tests for CSV cleanup, level parsing, flat C-prefix dirs, and ROI extraction."""

    def test_csv_nbsp_and_complex_levels_matching(self):
        import tempfile
        from scripts.evaluate_aospine_baseline import (
            load_ground_truth,
            normalize_level_name,
            parse_levels,
            discover_patient_cases,
            strip_accents,
        )

        self.assertEqual(normalize_level_name('Gãy L1\xa0'), 'L1')
        self.assertEqual(normalize_level_name('D11-D12'), 'T11-T12')
        self.assertEqual(parse_levels('Gãy L1'), ['L1'])
        self.assertEqual(parse_levels('D11-D12'), ['T11', 'T12'])
        self.assertEqual(parse_levels('L3L4'), ['L3', 'L4'])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_file = root / 'gt.csv'
            csv_file.write_text(
                "STT,Tên,Năm sinh,Đốt gãy,ID,phân loại,kèm theo\n"
                "6,Phạm Sỹ Thế ,1996,L2,1240010187\xa0,A4,\n"
                "24,Võ Văn Nhi,1995,Gãy L1,1240060146,A4,\n"
                "15,Lê Quang Huy,1992,D11-D12,1240069906,C,\n"
                "51,Lê Nữ Thu Giang,2002,L3L4,1240090611\xa0,C,\n",
                encoding='utf-8-sig',
            )
            gt = load_ground_truth(csv_file)
            self.assertEqual(gt[strip_accents('pham sy the')]['true_level'], 'L2')
            self.assertEqual(gt[strip_accents('vo van nhi')]['true_level'], 'L1')
            self.assertEqual(gt['1240010187']['name'], 'Phạm Sỹ Thế')
            self.assertEqual(parse_levels(gt[strip_accents('le nu thu giang')]['true_level']), ['L3', 'L4'])

            # Flat directory whose name starts with 'C' and contains a DICOM subfolder must NOT be split as category 'C'
            data_dir = root / 'raw'
            (data_dir / 'A3' / 'duong kim hoa').mkdir(parents=True)
            (data_dir / 'Chau Doc' / 'DICOM').mkdir(parents=True)
            found = discover_patient_cases(data_dir)
            names = [p.name for p in found]
            self.assertIn('duong kim hoa', names)
            self.assertIn('Chau Doc', names)
            self.assertNotIn('DICOM', names)


if __name__ == '__main__':
    unittest.main()
