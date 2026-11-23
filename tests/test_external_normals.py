"""Unit tests for External Normal Vertebra Ingestion and Anti-Noise Pipeline (Step 2.4)."""
import unittest
import tempfile
import json
import shutil
from pathlib import Path
import numpy as np

from spinal_ct.prototype_bank import (
    AOSpinePrototypeBank,
    extract_roi_prototype_features,
    SPINE_LEVELS,
)
from scripts.prepare_external_normals import (
    compute_mid_sagittal_heights,
)


class TestExternalNormals(unittest.TestCase):
    """Test anti-noise filtering and multi-scanner prototype bank distribution pooling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_mid_sagittal_heights(self):
        """Test geometric height ratio computation on synthetic 2D mask."""
        # Create a rectangular vertebra mask: 30 rows (Z) x 40 cols (Y)
        mask = np.zeros((60, 60), dtype=bool)
        mask[15:45, 10:50] = True
        ha_hp, hm_hp, h_tot, h_post = compute_mid_sagittal_heights(mask, sz_mm=1.0)
        self.assertAlmostEqual(ha_hp, 1.0, delta=0.05)
        self.assertAlmostEqual(hm_hp, 1.0, delta=0.05)
        self.assertAlmostEqual(h_post, 30.0, delta=1.0)

        # Create a wedged vertebra mask (anterior compressed to 15 rows)
        wedged = np.zeros((60, 60), dtype=bool)
        # Left (ant) is 15 rows, right (post) is 30 rows
        for col in range(10, 50):
            # linear interpolation from 15 to 30 height
            frac = (col - 10) / 40.0
            h = int(15 + frac * 15)
            wedged[30 - h // 2: 30 + h // 2, col] = True

        w_ha_hp, w_hm_hp, _, _ = compute_mid_sagittal_heights(wedged, sz_mm=1.0)
        self.assertLess(w_ha_hp, 0.70)

    def test_ingest_external_normals_weighted_pooling(self):
        """Test weighted pooled mean and variance calculation across scanners."""
        bank = AOSpinePrototypeBank()

        # Build initial bank from synthetic train records
        train_samples = []
        for i in range(10):
            train_samples.append({
                'sample_id': f"local_norm_{i}",
                'patient_id': f"pid_{i}",
                'level': 'T12',
                'aospine_class': 'Normal',
                'prototype_features': np.random.normal(loc=0.1, scale=0.05, size=16).tolist(),
            })
        for i in range(5):
            train_samples.append({
                'sample_id': f"local_frac_{i}",
                'patient_id': f"pid_f_{i}",
                'level': 'L1',
                'aospine_class': 'A3',
                'prototype_features': np.random.normal(loc=0.5, scale=0.1, size=16).tolist(),
            })

        train_sids = [s['sample_id'] for s in train_samples]
        train_pids = [s['patient_id'] for s in train_samples]
        bank.build_from_fold(
            samples=train_samples,
            train_sample_ids=train_sids,
            train_patient_ids=train_pids,
            fold_id=0,
        )

        initial_t12_mean = bank.normal_prototypes_by_level['T12']['mean'].copy()
        initial_t12_std = bank.normal_prototypes_by_level['T12']['std'].copy()
        self.assertEqual(bank.external_ingested_count, 0)

        # Prepare external manifest with synthetic clean normals
        ext_dir = self.tmp_path / "external_rois"
        ext_dir.mkdir()
        ext_samples = []
        for i in range(20):
            ext_samples.append({
                'sample_id': f"ext_s{i:04d}_T12",
                'patient_id': f"totalseg_s{i:04d}",
                'level': 'T12',
                'aospine_class': 'Normal',
                'prototype_features': np.random.normal(loc=0.3, scale=0.08, size=16).tolist(),
            })
        for i in range(15):
            ext_samples.append({
                'sample_id': f"ext_s{i:04d}_L1",
                'patient_id': f"totalseg_s{i:04d}",
                'level': 'L1',
                'aospine_class': 'Normal',
                'prototype_features': np.random.normal(loc=0.25, scale=0.08, size=16).tolist(),
            })

        manifest = {
            'total_clean_normals': len(ext_samples),
            'samples': ext_samples,
        }
        (ext_dir / "manifest.json").write_text(json.dumps(manifest), encoding='utf-8')

        # Ingest external normals with weight w = 0.4
        ingested = bank.ingest_external_normals(ext_dir, external_weight=0.4)
        self.assertEqual(ingested, len(ext_samples))
        self.assertEqual(bank.external_ingested_count, len(ext_samples))

        # Check updated mean & std
        updated_t12_mean = bank.normal_prototypes_by_level['T12']['mean']
        updated_t12_std = bank.normal_prototypes_by_level['T12']['std']

        # Mean shifted towards external distribution
        self.assertFalse(np.allclose(initial_t12_mean, updated_t12_mean))
        # Variance floor strictly respected (>= 0.25)
        self.assertTrue(np.all(updated_t12_std >= 0.25))
        # No NaNs or infinities
        self.assertFalse(np.isnan(updated_t12_mean).any())
        self.assertFalse(np.isnan(updated_t12_std).any())

        # Test Save & Load roundtrip
        save_file = self.tmp_path / "bank.json"
        bank.save(save_file)

        loaded_bank = AOSpinePrototypeBank().load(save_file)
        self.assertEqual(loaded_bank.external_ingested_count, len(ext_samples))
        np.testing.assert_allclose(loaded_bank.normal_prototypes_by_level['T12']['mean'], updated_t12_mean)
        np.testing.assert_allclose(loaded_bank.normal_prototypes_by_level['T12']['std'], updated_t12_std)

    def test_ingest_verse_normals_alias(self):
        """Test ingest_verse_normals backward compatibility alias."""
        bank = AOSpinePrototypeBank()
        # Empty dir returns 0
        empty_dir = self.tmp_path / "empty"
        empty_dir.mkdir()
        # Should raise RuntimeError if called before bank is built
        with self.assertRaises(RuntimeError):
            bank.ingest_verse_normals(empty_dir)


if __name__ == '__main__':
    unittest.main()
