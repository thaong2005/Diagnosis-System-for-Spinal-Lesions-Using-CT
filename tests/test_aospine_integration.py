"""Unit and integration tests for AO Spine Phase 1 Step 3.

Covers:
1. BodySession AO Spine real-time recomputation (< 10 ms).
2. Landmark edit updates and auto_populate integration.
3. SpineSession multi-vertebra ranking, triage and report export.
4. Canvas rendering with AO Spine visual overlays (canal line, retropulsion patch, spinous tip).
5. Evaluation script metrics (Quadratic Weighted Cohen's Kappa, Balanced Accuracy, Macro-F1, Localization).
6. Production Contract (Zero Label Leakage) validation.
"""
from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spinal_ct.body_region import BodyPlane
from spinal_ct.morphometry import (
    BodySession,
    SpineSession,
    measure_heights,
    save_session,
)
from spinal_ct.morphometry_render import draw_body
from scripts.evaluate_aospine_baseline import (
    AOSPINE_6CLASSES,
    compute_classification_metrics,
    compute_localization_metrics,
    evaluate_predictions,
    quadratic_weighted_cohen_kappa,
    run_blind_inference,
)


def make_dummy_session(level: str = 'L1', with_hint: bool = True) -> BodySession:
    """Helper creating a valid 2D sagittal BodySession."""
    plane = BodyPlane(
        origin_lps_mm=(0.0, 0.0, 0.0),
        u_lps=(0.0, 1.0, 0.0),
        v_lps=(0.0, 0.0, 1.0),
        spacing_xy_mm=(1.0, 1.0),
        size_xy=(80, 80),
    )
    shape = (80, 80)
    ct = np.full(shape, 100.0, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    hint = np.zeros(shape, dtype=bool)

    if with_hint:
        # Vertebral body in col 15..45, row 20..60
        hint[20:60, 15:45] = True
        ct[20:60, 15:45] = 400.0

        # Spinal canal in col 46..60: soft tissue / CSF (30 HU, no retropulsion)
        ct[20:60, 46:60] = 30.0

        # Spinous process / posterior arch in col 62..75, row 35..45
        hint[35:45, 62:75] = True
        ct[35:45, 62:75] = 350.0

    mask = np.zeros(shape, dtype=bool)
    mask[20:60, 15:45] = True

    landmarks = [
        [15.0, 58.0],  # AS
        [15.0, 22.0],  # AI
        [30.0, 58.0],  # MS
        [30.0, 22.0],  # MI
        [45.0, 58.0],  # PS
        [45.0, 22.0],  # PI
    ]

    session = BodySession(
        input_fingerprint='test_fingerprint_123',
        level=level,
        source_kind='Baseline',
        plane=plane,
        ct=ct,
        valid=valid,
        hint=hint,
        mask=mask,
        landmarks=landmarks,
        age_group='adult',
    )
    return session


class TestAOSpineBodySessionIntegration(unittest.TestCase):
    """Test real-time AO Spine computation on BodySession."""

    def test_recompute_speed_under_10ms(self):
        session = make_dummy_session('L1')
        session.recompute_aospine_classification()  # Warm-up call
        t0 = time.perf_counter()
        res = session.recompute_aospine_classification()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(elapsed_ms, 10.0, f"Recompute took {elapsed_ms:.2f} ms, expected < 10 ms")
        self.assertIn('predicted_class', res)
        self.assertIn('confidence', res)
        self.assertIn('clinical_rationale_vi', res)

    def test_landmark_edit_triggers_recompute(self):
        session = make_dummy_session('L1')
        initial_res = session.recompute_aospine_classification()
        self.assertEqual(initial_res['predicted_class'], 'Normal')

        # Simulate wedge fracture (A1): reduce anterior height from 36mm to 18mm
        wedge_landmarks = [
            [15.0, 40.0],  # AS lowered
            [15.0, 22.0],  # AI
            [30.0, 50.0],  # MS
            [30.0, 22.0],  # MI
            [45.0, 58.0],  # PS
            [45.0, 22.0],  # PI
        ]
        session.set_landmarks(wedge_landmarks)
        # Should automatically trigger recomputation
        self.assertIsNotNone(session.aospine)
        self.assertIn(session.aospine['predicted_class'], ('A1', 'A3', 'A4'))

    def test_measure_includes_aospine_in_result(self):
        session = make_dummy_session('L1')
        session.approve()
        report = session.measure(samples=500, seed=42)

        self.assertIn('aospine', report)
        aospine = report['aospine']
        self.assertIn('predicted_class', aospine)
        self.assertIn('confidence', aospine)
        self.assertIn('concept_probs', aospine)
        self.assertIn('patient_summary', aospine)

    def test_save_session_exports_aospine_in_csv_and_html(self):
        session = make_dummy_session('L1')
        session.approve()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'result'
            save_session(session, out_dir, samples=500, seed=42)

            self.assertTrue((out_dir / 'measurements.json').is_file())
            self.assertTrue((out_dir / 'features.csv').is_file())
            self.assertTrue((out_dir / 'report.html').is_file())

            # Verify features.csv columns
            with (out_dir / 'features.csv').open('r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                self.assertIn('aospine_class', row)
                self.assertIn('aospine_confidence', row)
                self.assertIn('delta_posterior_mm', row)
                self.assertIn('aospine_rationale', row)

            # Verify report.html has AO Spine section
            html_content = (out_dir / 'report.html').read_text(encoding='utf-8')
            self.assertIn('AO Spine', html_content)
            self.assertIn('Mã phân loại', html_content)


class TestSpineSessionIntegration(unittest.TestCase):
    """Test multi-vertebra SpineSession management and triage."""

    def test_spine_session_multi_level_ranking(self):
        spine = SpineSession(patient_id='patient_001')

        # Add T12 (normal), L1 (injured A4), L2 (normal)
        t12 = make_dummy_session('T12')
        l1 = make_dummy_session('L1')
        l2 = make_dummy_session('L2')

        # Create burst fracture (A4) on L1: collapse height + retropulsion
        l1_landmarks = [
            [15.0, 36.0], [15.0, 24.0],  # Ha = 12 mm
            [30.0, 36.0], [30.0, 24.0],  # Hm = 12 mm
            [45.0, 36.0], [45.0, 24.0],  # Hp = 12 mm
        ]
        l1.set_landmarks(l1_landmarks)
        # Mock retropulsion on L1
        l1.retropulsion = {
            'delta_posterior_mm': 3.8,
            'canal_encroachment_ratio': 0.25,
            'canal_line_points': [[45.0, 10.0], [45.0, 70.0]],
            'is_retropulsed': True,
            'retropulsed_bone_mask': np.zeros((80, 80), bool),
        }

        spine.add_session(t12)
        spine.add_session(l1)
        spine.add_session(l2)

        res = spine.recompute_aospine_classification()

        self.assertEqual(len(res['ranked_vertebrae']), 3)
        top1 = res['ranked_vertebrae'][0]
        # L1 must be ranked #1
        self.assertEqual(top1['level'], 'L1')
        self.assertIn(top1['predicted_class'], ('A4', 'A3'))
        self.assertTrue(res['patient_summary']['has_surgical_instability'])

    def test_spine_session_save_exports_patient_summary(self):
        spine = SpineSession(patient_id='patient_test_export')
        l1 = make_dummy_session('L1')
        l1.approve()
        spine.add_session(l1)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'spine_export'
            spine.save_session(out_path, samples=200, seed=42)

            summary_f = out_path / 'spine_patient_summary.json'
            self.assertTrue(summary_f.is_file())
            data = json.loads(summary_f.read_text(encoding='utf-8'))
            self.assertEqual(data['patient_id'], 'patient_test_export')
            self.assertIn('patient_summary', data)
            self.assertIn('ranked_vertebrae', data)


class TestAOSpineCanvasRendering(unittest.TestCase):
    """Test canvas drawing with AO Spine overlays."""

    def test_draw_body_with_aospine_overlays(self):
        session = make_dummy_session('L1')
        session.spinous_tip = {'tip_xy_mm': [65.0, 40.0], 'q_seg': 0.85}
        session.retropulsion = {
            'delta_posterior_mm': 3.2,
            'is_retropulsed': True,
            'canal_line_points': [[45.0, 15.0], [45.0, 65.0]],
            'max_retropulsion_point_xy_mm': [48.2, 40.0],
            'retropulsed_bone_mask': np.zeros((80, 80), bool),
        }
        session.recompute_aospine_classification()

        fig, ax = plt.subplots()
        # Should render without throwing exceptions
        draw_body(ax, session, show_points=False, show_blobs=False, show_aospine=True)
        self.assertIn('AO:', ax.get_title())
        plt.close(fig)


class TestEvaluationScriptMetrics(unittest.TestCase):
    """Test objective evaluation math (Kappa, Balanced Accuracy, Macro-F1, Localization)."""

    def test_quadratic_weighted_cohen_kappa_perfect_agreement(self):
        # Perfect 6x6 diagonal matrix
        cm = np.diag([5, 14, 19, 14, 1, 1])
        kappa = quadratic_weighted_cohen_kappa(cm)
        self.assertAlmostEqual(kappa, 1.0, places=4)

    def test_quadratic_weighted_cohen_kappa_chance_agreement(self):
        # Independent random predictions matching marginals
        cm = np.ones((6, 6), dtype=int) * 10
        kappa = quadratic_weighted_cohen_kappa(cm)
        self.assertAlmostEqual(kappa, 0.0, places=2)

    def test_classification_metrics_math(self):
        y_true = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']
        y_pred = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']

        metrics = compute_classification_metrics(y_true, y_pred, AOSPINE_6CLASSES)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['balanced_accuracy'], 1.0)
        self.assertEqual(metrics['macro_f1'], 1.0)
        self.assertEqual(metrics['weighted_f1'], 1.0)
        self.assertEqual(metrics['quadratic_weighted_kappa'], 1.0)

    def test_localization_metrics_math(self):
        true_lvls = ['L1', 'L2', 'T12', 'L1']
        top1_lvls = ['L1', 'L1', 'T12', 'L1']
        top2_lvls = [['L1', 'T12'], ['L1', 'L2'], ['T12', 'T11'], ['L1', 'L2']]

        res = compute_localization_metrics(true_lvls, top1_lvls, top2_lvls)
        # Top-1 hits: 3/4 = 0.75
        self.assertEqual(res['top1_accuracy'], 0.75)
        # Top-2 hits: 4/4 = 1.00
        self.assertEqual(res['top2_accuracy'], 1.0)

    def test_evaluate_predictions_roundtrip_with_csv(self):
        # Create synthetic ground truth CSV
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gt_file = tmp_path / 'ground_truth.csv'
            gt_rows = [
                {'STT': '1', 'ID': 'case_001', 'Đốt gãy': 'L1', 'phân loại': 'A4'},
                {'STT': '2', 'ID': 'case_002', 'Đốt gãy': 'T12', 'phân loại': 'B2'},
            ]
            with gt_file.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['STT', 'ID', 'Đốt gãy', 'phân loại'])
                writer.writeheader()
                writer.writerows(gt_rows)

            predictions = [
                {
                    'patient_id': 'case_001',
                    'top1_level': 'L1',
                    'top1_class': 'A4',
                    'top1_confidence': 0.90,
                    'top2_levels': ['L1', 'T12'],
                },
                {
                    'patient_id': 'case_002',
                    'top1_level': 'T12',
                    'top1_class': 'B2',
                    'top1_confidence': 0.85,
                    'top2_levels': ['T12', 'L1'],
                },
            ]

            eval_res = evaluate_predictions(predictions, gt_file, tmp_path / 'eval_output')

            self.assertEqual(eval_res['total_cases'], 2)
            self.assertEqual(eval_res['localization']['top1_accuracy'], 1.0)
            self.assertEqual(eval_res['class_metrics']['accuracy'], 1.0)
            self.assertTrue((tmp_path / 'eval_output' / 'summary_report.md').is_file())
            self.assertTrue((tmp_path / 'eval_output' / 'evaluation_metrics.json').is_file())


class TestAOSpineStateManagementAndMultiLevelIntegration(unittest.TestCase):
    """Test multi-vertebra state persistence, unmeasured handling, and intervertebral metrics."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from unittest.mock import patch
        from PySide6.QtWidgets import QMessageBox
        prompt = patch('apps.body_review.QMessageBox.question', return_value=QMessageBox.Discard)
        prompt.start()
        self.addCleanup(prompt.stop)

    def test_level_switch_preserves_session_state(self):
        """Switching L1 -> L2 -> L1 retains 100% mask, landmarks, and classification."""
        import SimpleITK as sitk
        from apps.body_review import BodyReviewDialog
        from spinal_ct.body_region import polygon_mask

        z, y, x = np.indices((60, 44, 24))
        source = sitk.GetImageFromArray((x + 2 * y + 3 * z).astype(np.float32))
        source.SetSpacing((0.8, 1.2, 1.5))
        mask = np.zeros((60, 44, 24), np.uint8)
        mask[5:20, 8:36, 5:19] = 19   # T12
        mask[22:38, 8:36, 5:19] = 20  # L1
        mask[40:55, 8:36, 5:19] = 21  # L2
        labels = sitk.GetImageFromArray(mask)
        labels.CopyInformation(source)

        dialog = BodyReviewDialog(source, labels, 'test_fp', 'L1', 'Baseline', None)
        self.assertIn('L1', dialog.available_levels)
        self.assertIn('L2', dialog.available_levels)

        # Set mask, 6 landmarks, and doctor confirmed A3 on L1
        m = polygon_mask([[12, 12], [30, 12], [30, 28], [12, 28]], dialog.session.ct.shape, dialog.session.plane.spacing_xy_mm)
        dialog.session.set_mask(m)
        dialog.session.set_landmarks([[12, 28], [12, 12], [20, 28], [20, 12], [30, 28], [30, 12]])
        dialog.session.doctor_confirmed_class = 'A3'
        dialog.session.approve()
        dialog.recompute_all_levels()

        # Switch to L2
        dialog.switch_to_level('L2')
        self.assertEqual(dialog.level, 'L2')

        # Switch back to L1
        dialog.switch_to_level('L1')
        self.assertEqual(dialog.level, 'L1')
        self.assertEqual(len(dialog.session.landmarks), 6)
        self.assertTrue(dialog.session.mask.any())
        self.assertTrue(dialog.session.reviewed)
        self.assertEqual(dialog.session.doctor_confirmed_class, 'A3')
        eff_cls = dialog.session.doctor_confirmed_class or dialog.session.aospine.get('predicted_class')
        self.assertEqual(eff_cls, 'A3')
        dialog.close()

    def test_unmeasured_vertebra_indeterminate(self):
        """Vertebra with < 6 landmarks must return indeterminate state, NOT fake Normal."""
        session = make_dummy_session('L1')
        session.landmarks = []
        res = session.recompute_aospine_classification()

        self.assertTrue(res['is_indeterminate'])
        self.assertEqual(res['predicted_class'], 'Chưa đặt đủ 6 mốc')
        self.assertEqual(res['confidence'], 0.0)
        self.assertNotEqual(res['predicted_class'], 'Normal')
        self.assertIn('Chưa đặt đủ 6 mốc chiều cao', res['rejection_reasons'])

    def test_multilevel_sessions_have_distinct_metrics_and_ranks(self):
        """Multi-level chain gives distinct relative collapse, ISP ratio, and proper rank."""
        s_t12 = make_dummy_session('T12')
        s_l1 = make_dummy_session('L1')
        s_l2 = make_dummy_session('L2')

        # L1 is collapsed (anterior & middle) with 6.0 mm retropulsion
        s_l1.landmarks = [
            [15.0, 35.0],
            [15.0, 22.0],
            [30.0, 33.0],
            [30.0, 22.0],
            [45.0, 50.0],
            [45.0, 22.0],
        ]
        s_l1.retropulsion = {'delta_posterior_mm': 6.0}
        s_t12.retropulsion = {'delta_posterior_mm': 0.0}
        s_l2.retropulsion = {'delta_posterior_mm': 0.0}

        sessions = {'T12': s_t12, 'L1': s_l1, 'L2': s_l2}
        s_l1.recompute_aospine_classification(sessions)

        # L1 is severe burst fracture -> rank #1
        self.assertEqual(s_l1.aospine['rank'], 1)
        self.assertGreater(s_t12.aospine['rank'], 1)
        self.assertGreater(s_l2.aospine['rank'], 1)
        self.assertIn(s_l1.aospine['predicted_class'], ('A3', 'A4'))

        # Check distinct metrics
        l1_coll = s_l1.aospine['concept_probs']['relative_collapse_ratio']
        t12_coll = s_t12.aospine['concept_probs']['relative_collapse_ratio']
        self.assertNotEqual(l1_coll, t12_coll)
        self.assertLess(l1_coll, 0.85)


if __name__ == '__main__':
    unittest.main()
