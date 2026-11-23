"""Unit tests for Phase 2 & 3: Adaptive Rule Fusion Layer and Biomechanical Guards."""
import unittest

from spinal_ct.aospine_fusion import (
    compute_adaptive_fusion_weights,
    fuse_vertebra_concept_probabilities,
    fuse_and_classify_spine,
)


class TestAOSpineFusion(unittest.TestCase):
    """Test Adaptive Rule Fusion Layer and Biomechanical Guards."""

    def test_adaptive_weights_by_segmentation_quality(self):
        """Test reciprocal reliability: high q_seg favours rules, low q_seg favours DL."""
        # Crisp mask (q_seg = 0.85) -> rule-heavy alpha ~ 0.35
        weights_crisp = compute_adaptive_fusion_weights(q_seg=0.85)
        self.assertAlmostEqual(weights_crisp['pw'], 0.35, places=2)

        # Broken mask (q_seg = 0.30) -> visual-heavy alpha ~ 0.80
        weights_broken = compute_adaptive_fusion_weights(q_seg=0.30)
        self.assertAlmostEqual(weights_broken['pw'], 0.80, places=2)

        # Fallback when model not trained -> alpha = 0.0
        weights_untrained = compute_adaptive_fusion_weights(q_seg=0.90, dl_model_trained=False)
        self.assertEqual(weights_untrained['pw'], 0.0)

    def test_rare_class_a1_rescue_test(self):
        """Test Rare Class n=1 A1 rescue: Intact Wall Guard prevents noisy CNN from causing false A3/A4."""
        # Clinical A1 vertebra: Ha/Hp = 0.62 (wedge deformity), posterior wall pristine (delta_pw = 0.2mm)
        metrics = {
            'level': 'L1',
            'q_seg': 0.85,
            'delta_posterior_mm': 0.2,
            'isp_ratio': 1.02,
            'translation_mm': 0.0,
            'relative_collapse_ratio': 0.88,
            'heights': {
                'anterior_posterior_ratio': 0.62,
                'middle_posterior_ratio': 0.75,
            },
            'ha_hp_ratio': 0.62,
        }

        # Simulated CNN output with high false-positive posterior wall activation
        noisy_dl = {
            'p_pw': 0.75,  # CNN false alarm!
            'p_ep2': 0.10,
            'p_post_tb': 0.05,
            'p_ant_tb': 0.01,
            'p_trans': 0.05,
            'p_fracture': 0.88,
        }

        result = fuse_and_classify_spine([metrics], dl_concept_probs_list=[noisy_dl], dl_model_trained=True)
        top1 = result['ranked_vertebrae'][0]

        # The Intact Wall Guard MUST trigger and cap S_pw <= 0.35, preventing false A3/A4
        diag = top1['fusion_diagnostics']
        self.assertLessEqual(diag['fused_probs']['p_pw'], 0.35)
        self.assertTrue(any('Intact Wall Guard' in g for g in diag['triggered_guards']))

        # Final prediction MUST stay A1 (Wedge Compression)!
        self.assertEqual(top1['predicted_class'], 'A1')

    def test_broken_mask_a4_rescue_test(self):
        """Test Comminuted / Broken Mask A4 rescue: High visual CNN alpha (0.80) rescues geometric rule miss."""
        # Comminuted burst fracture where segmentation mask is fragmented (q_seg = 0.30)
        # leading rule geometry to underestimate retropulsion (delta_pw = 0.5mm, P_Rule = 0.20)
        metrics = {
            'level': 'T12',
            'q_seg': 0.30,
            'delta_posterior_mm': 0.5,
            'isp_ratio': 1.05,
            'translation_mm': 0.0,
            'relative_collapse_ratio': 0.90,  # Rule thinks height is okay
            'heights': {
                'anterior_posterior_ratio': 0.85,
                'middle_posterior_ratio': 0.85,
            },
            'ha_hp_ratio': 0.85,
        }

        # DL CNN visual features correctly identify severe burst and comminution
        dl_burst = {
            'p_pw': 0.92,
            'p_ep2': 0.88,
            'p_post_tb': 0.05,
            'p_ant_tb': 0.01,
            'p_trans': 0.05,
            'p_fracture': 0.95,
        }

        result = fuse_and_classify_spine([metrics], dl_concept_probs_list=[dl_burst], dl_model_trained=True)
        top1 = result['ranked_vertebrae'][0]

        diag = top1['fusion_diagnostics']
        # Alpha should shift to DL (alpha ~ 0.80)
        self.assertAlmostEqual(diag['alpha_weights']['pw'], 0.80, places=2)
        # Fused posterior wall probability should be high
        self.assertGreater(diag['fused_probs']['p_pw'], 0.65)
        self.assertGreater(diag['fused_probs']['p_ep2'], 0.65)

        # Final prediction successfully rescued to A4!
        self.assertEqual(top1['predicted_class'], 'A4')

    def test_boundary_guards_coverage(self):
        """Test Bone Fragment, Distraction, and Translation floor guards."""
        # Bone Fragment Floor Guard (delta_pw >= 2.8)
        m_burst = {'level': 'L1', 'q_seg': 0.80, 'delta_posterior_mm': 3.2, 'isp_ratio': 1.0, 'translation_mm': 0.0}
        dl_low = {'p_pw': 0.10, 'p_ep2': 0.10, 'p_post_tb': 0.10, 'p_ant_tb': 0.01, 'p_trans': 0.10, 'p_fracture': 0.10}
        fused, diag = fuse_vertebra_concept_probabilities(m_burst, dl_low)
        self.assertGreaterEqual(fused['p_pw'], 0.65)
        self.assertTrue(any('Bone Fragment Floor Guard' in g for g in diag['triggered_guards']))

        # Distraction Floor Guard (isp_ratio >= 1.45)
        m_dist = {'level': 'L2', 'q_seg': 0.80, 'delta_posterior_mm': 0.0, 'isp_ratio': 1.45, 'translation_mm': 0.0}
        fused_d, diag_d = fuse_vertebra_concept_probabilities(m_dist, dl_low)
        self.assertGreaterEqual(fused_d['p_post_tb'], 0.65)
        self.assertTrue(any('Distraction Floor Guard' in g for g in diag_d['triggered_guards']))

        # Translation Floor Guard (trans >= 3.8mm for L1)
        m_trans = {'level': 'L1', 'q_seg': 0.80, 'delta_posterior_mm': 0.0, 'isp_ratio': 1.0, 'translation_mm': 4.2}
        fused_t, diag_t = fuse_vertebra_concept_probabilities(m_trans, dl_low)
        self.assertGreaterEqual(fused_t['p_trans'], 0.65)
        self.assertTrue(any('Translation Floor Guard' in g for g in diag_t['triggered_guards']))

    def test_b3_hyperextension_preservation(self):
        """Test B3 hyperextension disruption is preserved even if DL model defaults p_ant_tb=0.01."""
        m_b3 = {'level': 'T11', 'q_seg': 0.40, 'anterior_disc_angle_deg': 21.0, 'disc_opening_ratio': 2.5}
        dl_dummy = {'p_pw': 0.05, 'p_ep2': 0.05, 'p_post_tb': 0.05, 'p_ant_tb': 0.01, 'p_trans': 0.05, 'p_fracture': 0.20}
        res_b3 = fuse_and_classify_spine([m_b3], [dl_dummy], dl_model_trained=True)
        pred_b3 = res_b3['ranked_vertebrae'][0]['predicted_class']
        self.assertEqual(pred_b3, 'B3')

    def test_bone_fragment_floor_prevents_normal_drop(self):
        """Test Bone Fragment Floor Guard raises s_frac so clear burst does not drop to Normal."""
        m_burst = {'level': 'L1', 'q_seg': 0.80, 'delta_posterior_mm': 3.2, 'isp_ratio': 1.0, 'translation_mm': 0.0}
        dl_low = {'p_pw': 0.10, 'p_ep2': 0.10, 'p_post_tb': 0.05, 'p_ant_tb': 0.01, 'p_trans': 0.05, 'p_fracture': 0.05}
        res_burst = fuse_and_classify_spine([m_burst], [dl_low], dl_model_trained=True)
        pred_burst = res_burst['ranked_vertebrae'][0]['predicted_class']
        self.assertIn(pred_burst, ('A3', 'A4'))

    def test_classic_a1_wedge_guard_prevents_false_b2(self):
        """Test Classic A1 Wedge Guard prevents false B2 distraction override on mild wedge metrics."""
        m_a1 = {
            'level': 'L1',
            'q_seg': 0.88,
            'ha_hp_ratio': 0.784,
            'hm_hp_ratio': 0.814,
            'relative_collapse_ratio': 0.953,
            'delta_posterior_mm': 0.3,
            'isp_ratio': 1.080,
            'translation_mm': 0.94,
            'anterior_disc_angle_deg': 12.73,
            'disc_opening_ratio': 1.0,
            'lower_endplate_deformity_mm': 0.0,
        }
        res_fallback = fuse_and_classify_spine([m_a1], dl_concept_probs_list=None, dl_model_trained=False)
        eval_fb = res_fallback['ranked_vertebrae'][0]
        self.assertEqual(eval_fb['predicted_class'], 'A1')
        self.assertLess(eval_fb['concept_probs']['p_post_tb'], 0.25)

    def test_proto_b2_not_blocked_by_guard_6(self):
        """Test is_proto_b2 is not blocked by Guard 6 (Intact Tension Band Guard)."""
        metrics = {
            'level': 'L1',
            'q_seg': 0.85,
            'isp_ratio': 1.05,
            'delta_posterior_mm': 0.0,
            'translation_mm': 0.5,
            'relative_collapse_ratio': 0.95,
            'ha_hp_ratio': 0.92,
            'hm_hp_ratio': 0.90,
        }
        prototype_result = {
            'p_anomaly': 0.78,
            'predicted_class': 'B2',
            'class_probabilities': {'B2': 0.80, 'A4': 0.10, 'A3': 0.05, 'Normal': 0.05},
        }
        fused_probs, diag = fuse_vertebra_concept_probabilities(
            metrics,
            dl_probs=None,
            dl_model_trained=False,
            prototype_result=prototype_result,
        )
        self.assertFalse(any('Intact Tension Band Guard' in g for g in diag['triggered_guards']))
        self.assertGreaterEqual(fused_probs['p_post_tb'], 0.70)
        self.assertTrue(any('Prototype B2 Support' in g for g in diag['triggered_guards']))

    def test_burst_collapse_not_forced_to_type_c_by_guard_4(self):
        """Test is_burst_collapse is not forced into Type C by Guard 4."""
        metrics = {
            'level': 'L1',
            'q_seg': 0.80,
            'delta_posterior_mm': 3.5,
            'relative_collapse_ratio': 0.65,
            'translation_mm': 4.2,
            'isp_ratio': 1.0,
            'ha_hp_ratio': 0.60,
            'hm_hp_ratio': 0.62,
        }
        prototype_result = {
            'p_anomaly': 0.85,
            'predicted_class': 'A4',
            'class_probabilities': {'A4': 0.82, 'A3': 0.10, 'Normal': 0.02},
        }
        fused_probs, diag = fuse_vertebra_concept_probabilities(
            metrics,
            dl_probs=None,
            dl_model_trained=False,
            prototype_result=prototype_result,
        )
        self.assertFalse(any('Translation Floor Guard' in g for g in diag['triggered_guards']))
        self.assertLessEqual(fused_probs['p_trans'], 0.45)
        self.assertTrue(any('Burst Collapse Guard' in g for g in diag['triggered_guards']))

        res = fuse_and_classify_spine(
            [metrics],
            dl_concept_probs_list=None,
            dl_model_trained=False,
            prototype_results_list=[prototype_result],
        )
        pred_cls = res['ranked_vertebrae'][0]['predicted_class']
        self.assertIn(pred_cls, ('A4', 'A3'))
        self.assertNotEqual(pred_cls, 'C')


if __name__ == '__main__':
    unittest.main()
