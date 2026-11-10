"""Unit tests for AO Spine fuzzy decision tree and ranking in spinal_ct/aospine_rules.py."""
import unittest

from spinal_ct.aospine_rules import (
    fuzzy_sigmoid,
    compute_anatomical_rule_probabilities,
    evaluate_aospine_decision_tree,
    rank_and_classify_spine,
)


class TestAOSpineRules(unittest.TestCase):
    """Test suite covering AO Spine fuzzy clinical reasoning and full-spine ranking."""

    def test_fuzzy_sigmoid_properties(self):
        """Test standard, inverted, edge cases and clipping bounds of fuzzy sigmoid."""
        # Midpoint: x == threshold -> p == 0.50
        self.assertAlmostEqual(fuzzy_sigmoid(3.5, threshold=3.5, temperature=0.8), 0.50, delta=1e-3)
        # Normal direction: x > threshold -> p > 0.50
        self.assertGreater(fuzzy_sigmoid(5.0, threshold=3.5, temperature=0.8), 0.80)
        # Normal direction: x < threshold -> p < 0.50
        self.assertLess(fuzzy_sigmoid(1.0, threshold=3.5, temperature=0.8), 0.10)

        # Inverted direction (for collapse/height loss ratios): x < threshold -> p > 0.50
        self.assertGreater(fuzzy_sigmoid(0.60, threshold=0.78, temperature=0.06, invert=True), 0.90)
        self.assertLess(fuzzy_sigmoid(1.00, threshold=0.78, temperature=0.06, invert=True), 0.05)

        # Clipping bounds to avoid zero-probability collapse
        self.assertGreaterEqual(fuzzy_sigmoid(-1000.0, threshold=0.0, temperature=1.0), 0.01)
        self.assertLessEqual(fuzzy_sigmoid(1000.0, threshold=0.0, temperature=1.0), 0.99)

        # Temperature validation
        with self.assertRaises(ValueError):
            fuzzy_sigmoid(1.0, threshold=1.0, temperature=0.0)

    def test_case_normal(self):
        """Test intact vertebra with physiological metrics -> Predicted Normal with high confidence."""
        metrics = {
            'level': 'L2',
            'ha_hp_ratio': 1.02,
            'hm_hp_ratio': 0.98,
            'relative_collapse_ratio': 1.01,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.00,
            'anterior_disc_angle_deg': 2.0,
            'disc_opening_ratio': 1.05,
            'translation_mm': 0.2,
            'q_seg': 0.92,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'Normal')
        self.assertGreater(res['confidence'], 0.80)
        self.assertFalse(res['is_indeterminate'])
        self.assertFalse(res['modifier_m1'])
        self.assertIn('Đốt sống bình thường', res['clinical_rationale_vi'])

    def test_case_a1_wedge_compression(self):
        """Test A1 wedge compression: anterior loss Ha/Hp = 0.72, posterior wall intact (0.2 mm)."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.72,
            'hm_hp_ratio': 0.82,
            'relative_collapse_ratio': 0.88,
            'delta_posterior_mm': 0.2,
            'isp_ratio': 1.02,
            'anterior_disc_angle_deg': 3.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.3,
            'q_seg': 0.88,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A1')
        self.assertGreater(res['confidence'], 0.50)
        self.assertFalse(res['is_indeterminate'])
        self.assertIn('Gãy lún chêm A1', res['clinical_rationale_vi'])

    def test_case_a3_incomplete_burst(self):
        """Test A3 incomplete burst: posterior wall retropulsion 3.2 mm, single endplate involved."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.78,
            'hm_hp_ratio': 0.75,
            'relative_collapse_ratio': 0.85,  # not complete collapse (<0.78)
            'delta_posterior_mm': 3.2,
            'isp_ratio': 1.05,
            'anterior_disc_angle_deg': 4.0,
            'disc_opening_ratio': 1.1,
            'translation_mm': 0.4,
            'q_seg': 0.85,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A3')
        self.assertGreater(res['confidence'], 0.50)
        self.assertIn('Gãy vỡ nhiều mảnh không hoàn toàn A3', res['clinical_rationale_vi'])

    def test_case_a4_complete_burst(self):
        """Test A4 complete burst: severe collapse (0.62) with PW 4.5 mm, internal Ha/Hp normal (0.95)."""
        metrics = {
            'level': 'T12',
            'ha_hp_ratio': 0.95,  # internal ratio symmetric, masking burst internally
            'hm_hp_ratio': 0.93,
            'relative_collapse_ratio': 0.62,  # severe symmetric collapse vs adjacent
            'delta_posterior_mm': 4.5,
            'isp_ratio': 1.08,
            'anterior_disc_angle_deg': 2.5,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.5,
            'q_seg': 0.80,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A4')
        self.assertGreater(res['confidence'], 0.50)
        self.assertIn('Gãy vỡ nhiều mảnh hoàn toàn A4', res['clinical_rationale_vi'])

    def test_case_b2_combined_with_a4(self):
        """Test B2 posterior tension band disruption combined with A4 vertebral burst."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.90,
            'hm_hp_ratio': 0.85,
            'relative_collapse_ratio': 0.65,
            'delta_posterior_mm': 3.8,
            'isp_ratio': 1.42,  # Distracted spinous process (ISP ratio > 1.30)
            'anterior_disc_angle_deg': 3.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.8,
            'q_seg': 0.82,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'B2')
        self.assertEqual(res['secondary_subtype'], 'A4')
        self.assertIn('B2 (kèm gãy thân đốt kiểu A4)', res['clinical_rationale_vi'])

    def test_case_b3_hyperextension(self):
        """Test B3 anterior tension band disruption: gaping anterior disc (19.5 deg, ratio 2.4)."""
        metrics = {
            'level': 'T11',
            'ha_hp_ratio': 1.05,
            'hm_hp_ratio': 1.00,
            'relative_collapse_ratio': 0.98,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.02,
            'anterior_disc_angle_deg': 19.5,
            'disc_opening_ratio': 2.4,
            'translation_mm': 0.5,
            'q_seg': 0.90,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'B3')
        self.assertGreater(res['confidence'], 0.60)
        self.assertTrue(res['modifier_m2'])
        self.assertIn('Tổn thương dải căng trước loại B3', res['clinical_rationale_vi'])

    def test_case_c_translation(self):
        """Test Type C translation/dislocation: lateral/AP slip 5.2 mm (threshold 3.5 mm)."""
        metrics = {
            'level': 'T12',
            'ha_hp_ratio': 0.90,
            'hm_hp_ratio': 0.88,
            'relative_collapse_ratio': 0.92,
            'delta_posterior_mm': 1.5,
            'isp_ratio': 1.10,
            'anterior_disc_angle_deg': 4.0,
            'disc_opening_ratio': 1.1,
            'translation_mm': 5.2,
            'q_seg': 0.85,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'C')
        self.assertGreater(res['confidence'], 0.70)
        self.assertIn('Gãy trật cột sống loại C', res['clinical_rationale_vi'])

    def test_modifier_m1_trigger(self):
        """Test trigger of Modifier M1 flag when ISP ratio falls into grey zone (1.15 to 1.30)."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.75,
            'hm_hp_ratio': 0.72,
            'relative_collapse_ratio': 0.85,
            'delta_posterior_mm': 2.5,
            'isp_ratio': 1.22,  # 1.15 < 1.22 <= 1.30 -> M1 warning
            'anterior_disc_angle_deg': 3.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.3,
            'q_seg': 0.85,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A3')
        self.assertTrue(res['modifier_m1'])
        self.assertTrue(any('M1:' in mod for mod in res['modifiers']))

    def test_indeterminate_rejection(self):
        """Test rejection / indeterminate state when segmentation quality q_seg < 0.40."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.70,
            'hm_hp_ratio': 0.70,
            'relative_collapse_ratio': 0.70,
            'delta_posterior_mm': 3.0,
            'isp_ratio': 1.0,
            'anterior_disc_angle_deg': 0.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.0,
            'q_seg': 0.25,  # Severely fragmented mask below threshold 0.40
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertTrue(res['is_indeterminate'])
        self.assertTrue(any('q_seg' in r for r in res['rejection_reasons']))

    def test_rank_and_classify_spine_t10_l5(self):
        """Test full-spine ranking on 8 vertebrae (T10-L5) with single severe L1 burst fracture."""
        spine = []
        levels = ['T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
        for lvl in levels:
            if lvl == 'L1':
                # Severe burst fracture A4
                spine.append({
                    'level': lvl,
                    'ha_hp_ratio': 0.88,
                    'hm_hp_ratio': 0.85,
                    'relative_collapse_ratio': 0.65,
                    'delta_posterior_mm': 4.0,
                    'isp_ratio': 1.05,
                    'anterior_disc_angle_deg': 2.0,
                    'disc_opening_ratio': 1.0,
                    'translation_mm': 0.4,
                    'q_seg': 0.85,
                })
            else:
                # Intact vertebrae
                spine.append({
                    'level': lvl,
                    'ha_hp_ratio': 1.0,
                    'hm_hp_ratio': 1.0,
                    'relative_collapse_ratio': 1.0,
                    'delta_posterior_mm': 0.0,
                    'isp_ratio': 1.0,
                    'anterior_disc_angle_deg': 2.0,
                    'disc_opening_ratio': 1.0,
                    'translation_mm': 0.2,
                    'q_seg': 0.90,
                })

        summary = rank_and_classify_spine(spine)
        ranked = summary['ranked_vertebrae']
        pat = summary['patient_summary']

        self.assertEqual(len(ranked), 8)
        # Top-1 must be L1
        self.assertEqual(ranked[0]['level'], 'L1')
        self.assertEqual(ranked[0]['predicted_class'], 'A4')
        self.assertGreater(ranked[0]['lesion_severity_score'], 0.60)

        # Other vertebrae must have low severity (< 0.15)
        for v in ranked[1:]:
            self.assertLess(v['lesion_severity_score'], 0.15)
            self.assertEqual(v['predicted_class'], 'Normal')

        # Patient summary verification
        self.assertEqual(pat['primary_lesion_level'], 'L1')
        self.assertEqual(pat['primary_aospine_class'], 'A4')
        self.assertTrue(pat['has_surgical_instability'])
        self.assertEqual(pat['top2_levels'][0], 'L1')

    def test_case_severe_a1_no_a2_leakage(self):
        """Test severe wedge compression A1 does not leak into A2 when no coronal split is present."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.58,  # severe anterior wedging
            'hm_hp_ratio': 0.72,
            'relative_collapse_ratio': 0.70,  # severe collapse < 0.78
            'delta_posterior_mm': 0.1,  # posterior wall completely intact
            'isp_ratio': 1.0,
            'anterior_disc_angle_deg': 2.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.0,
            'q_seg': 0.90,
            # No coronal_split_mm provided
        }
        res = evaluate_aospine_decision_tree(metrics)
        # Must strictly predict A1, with zero leakage to A2
        self.assertEqual(res['predicted_class'], 'A1')
        self.assertGreater(res['class_probabilities']['A1'], 0.70)
        self.assertEqual(res['class_probabilities'].get('A2', 0.0), 0.0)

    def test_case_a2_split_when_explicit_metric_present(self):
        """Test A2 pincer/split classification when explicit coronal split metric is present (> 2.0 mm)."""
        metrics = {
            'level': 'L2',
            'ha_hp_ratio': 0.90,
            'hm_hp_ratio': 0.88,
            'relative_collapse_ratio': 0.92,
            'delta_posterior_mm': 0.3,  # posterior wall intact
            'coronal_split_mm': 4.0,  # explicit vertical split through vertebral body
            'isp_ratio': 1.02,
            'anterior_disc_angle_deg': 2.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.2,
            'q_seg': 0.90,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A2')
        self.assertGreater(res['class_probabilities']['A2'], 0.60)
        self.assertIn('Gãy chẻ/tách A2', res['clinical_rationale_vi'])

    def test_case_severe_a3_asymmetric_collapse_not_a4(self):
        """Test severe asymmetric burst A3 (relative_collapse < 0.78) is not misclassified as A4."""
        metrics = {
            'level': 'L1',
            'ha_hp_ratio': 0.60,  # severe asymmetric anterior wedge burst
            'hm_hp_ratio': 0.70,
            'relative_collapse_ratio': 0.72,  # < 0.78 overall collapse due to massive upper endplate crater
            'hp_relative_ratio': 0.95,  # posterior height intact
            'delta_posterior_mm': 3.5,  # retropulsion into canal
            'isp_ratio': 1.05,
            'anterior_disc_angle_deg': 4.0,
            'disc_opening_ratio': 1.0,
            'translation_mm': 0.3,
            'q_seg': 0.88,
        }
        res = evaluate_aospine_decision_tree(metrics)
        self.assertEqual(res['predicted_class'], 'A3')
        self.assertLess(res['concept_probs']['p_ep2'], 0.40)
        self.assertGreater(res['class_probabilities']['A3'], 0.50)


if __name__ == '__main__':
    unittest.main()

