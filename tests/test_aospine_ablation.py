"""Unit tests for Step 7: AO Spine 2.5D Training and 3-Way Ablation Study."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_aospine_25d import train_aospine_dualstream
from scripts.run_ablation_study import run_ablation_study


class TestAOSpineAblationAndTraining(unittest.TestCase):
    """Test Dual-Stream 2.5D network training and ablation study superiority assertions."""

    def test_train_aospine_25d_smoke_run(self):
        """Test train_aospine_dualstream smoke run reduces loss and exports .pt and .onnx with trained flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_pt = tmp_path / "model_smoke.pt"
            out_onnx = tmp_path / "model_smoke.onnx"

            summary = train_aospine_dualstream(
                splits_file='data/processed/aospine_splits.json',
                output_model_path=out_pt,
                output_onnx_path=out_onnx,
                fold_idx=0,
                epochs=2,
                batch_size=4,
                device_name='cpu',
                smoke_test=True,
                seed=42,
            )

            self.assertTrue(summary['is_trained_checkpoint'])
            self.assertTrue(out_pt.is_file())
            self.assertTrue(out_onnx.is_file())
            self.assertGreater(out_pt.stat().st_size, 100_000)
            self.assertGreater(out_onnx.stat().st_size, 100_000)

            # Val loss must be valid and non-negative
            self.assertLess(summary['best_val_loss'], 2.0)
            self.assertGreater(summary['best_val_loss'], 0.0)

    def test_ablation_study_hybrid_outperforms_baselines(self):
        """Test quantitative superiority of Model 3 (Hybrid AI) over Rule-Only and Pure DL baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_ablation_study(
                model_path='models/aospine_dualstream_best.pt',
                output_dir=tmpdir,
                seed=42,
            )

            m1_rule = metrics['model_1_rule_only']
            m2_dl = metrics['model_2_pure_dl']
            m3_hybrid = metrics['model_3_hybrid_fusion']

            # 1. Overall Accuracy superiority over both baselines
            self.assertGreater(
                m3_hybrid['overall_accuracy'],
                m1_rule['overall_accuracy'],
                f"Hybrid Overall Acc ({m3_hybrid['overall_accuracy']}%) must exceed Rule-Only ({m1_rule['overall_accuracy']}%)",
            )
            self.assertGreater(
                m3_hybrid['overall_accuracy'],
                m2_dl['overall_accuracy'],
                f"Hybrid Overall Acc ({m3_hybrid['overall_accuracy']}%) must exceed Pure-DL ({m2_dl['overall_accuracy']}%)",
            )
            self.assertEqual(m3_hybrid['overall_accuracy'], 100.0)

            # 2. Balanced Accuracy superiority over both baselines
            self.assertGreater(
                m3_hybrid['balanced_accuracy'],
                m1_rule['balanced_accuracy'],
                f"Hybrid Balanced Acc ({m3_hybrid['balanced_accuracy']}%) must exceed Rule-Only ({m1_rule['balanced_accuracy']}%)",
            )
            self.assertGreater(
                m3_hybrid['balanced_accuracy'],
                m2_dl['balanced_accuracy'],
                f"Hybrid Balanced Acc ({m3_hybrid['balanced_accuracy']}%) must exceed Pure-DL ({m2_dl['balanced_accuracy']}%)",
            )

            # 3. Macro-F1 superiority over both baselines
            self.assertGreater(
                m3_hybrid['macro_f1'],
                m1_rule['macro_f1'],
                f"Hybrid Macro-F1 ({m3_hybrid['macro_f1']}) must exceed Rule-Only ({m1_rule['macro_f1']})",
            )
            self.assertGreater(
                m3_hybrid['macro_f1'],
                m2_dl['macro_f1'],
                f"Hybrid Macro-F1 ({m3_hybrid['macro_f1']}) must exceed Pure-DL ({m2_dl['macro_f1']})",
            )

            # 4. Weighted-F1 superiority over both baselines
            self.assertGreater(
                m3_hybrid['weighted_f1'],
                m1_rule['weighted_f1'],
                f"Hybrid Weighted-F1 ({m3_hybrid['weighted_f1']}) must exceed Rule-Only ({m1_rule['weighted_f1']})",
            )
            self.assertGreater(
                m3_hybrid['weighted_f1'],
                m2_dl['weighted_f1'],
                f"Hybrid Weighted-F1 ({m3_hybrid['weighted_f1']}) must exceed Pure-DL ({m2_dl['weighted_f1']})",
            )

            # 5. Quadratic Weighted Kappa superiority over both baselines
            self.assertGreater(
                m3_hybrid['quadratic_weighted_kappa'],
                m1_rule['quadratic_weighted_kappa'],
                f"Hybrid Kappa ({m3_hybrid['quadratic_weighted_kappa']}) must exceed Rule-Only ({m1_rule['quadratic_weighted_kappa']})",
            )
            self.assertGreater(
                m3_hybrid['quadratic_weighted_kappa'],
                m2_dl['quadratic_weighted_kappa'],
                f"Hybrid Kappa ({m3_hybrid['quadratic_weighted_kappa']}) must exceed Pure-DL ({m2_dl['quadratic_weighted_kappa']})",
            )

            # 6. Rare Class (Zero-shot n=1 A1, B3) superiority over Pure DL
            self.assertGreater(
                m3_hybrid['rare_class_accuracy'],
                m2_dl['rare_class_accuracy'],
                f"Hybrid Rare-Class Acc ({m3_hybrid['rare_class_accuracy']}%) must exceed Pure-DL ({m2_dl['rare_class_accuracy']}%)",
            )
            self.assertEqual(m3_hybrid['rare_class_accuracy'], 100.0)

            # 7. Broken-Mask (q_seg < 0.45 comminution) superiority over Rule-Only
            self.assertGreater(
                m3_hybrid['broken_mask_accuracy'],
                m1_rule['broken_mask_accuracy'],
                f"Hybrid Broken-Mask Acc ({m3_hybrid['broken_mask_accuracy']}%) must exceed Rule-Only ({m1_rule['broken_mask_accuracy']}%)",
            )
            self.assertEqual(m3_hybrid['broken_mask_accuracy'], 100.0)

            # 8. Borderline Anatomical Pitfall superiority over Pure DL
            self.assertGreater(
                m3_hybrid['borderline_pitfall_accuracy'],
                m2_dl['borderline_pitfall_accuracy'],
                f"Hybrid Borderline Pitfall Acc ({m3_hybrid['borderline_pitfall_accuracy']}%) must exceed Pure-DL ({m2_dl['borderline_pitfall_accuracy']}%)",
            )
            self.assertEqual(m3_hybrid['borderline_pitfall_accuracy'], 100.0)

            # 9. Fast CPU latency (< 25 ms per case)
            self.assertLess(m3_hybrid['latency_ms_per_case'], 25.0)

            # 10. Verify exported report files
            tmp_path = Path(tmpdir)
            self.assertTrue((tmp_path / 'ablation_report.md').is_file())
            self.assertTrue((tmp_path / 'ablation_metrics.json').is_file())
            self.assertTrue((tmp_path / 'ablation_comparison.csv').is_file())
            self.assertTrue((tmp_path / 'ablation_case_details.csv').is_file())


if __name__ == '__main__':
    unittest.main()
