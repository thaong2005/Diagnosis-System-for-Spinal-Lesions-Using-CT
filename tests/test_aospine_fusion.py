"""Unit tests for Phase 2 & 3: Adaptive Rule Fusion Layer and Biomechanical Guards."""
import unittest

from spinal_ct.aospine_fusion import (
    compute_adaptive_fusion_weights,
    fuse_vertebra_concept_probabilities,
    fuse_and_classify_spine,
)
from spinal_ct.aospine_rules import evaluate_aospine_decision_tree


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

    def test_guard3_intact_neighbor_does_not_force_b2(self):
        """Hotfix 3.2 Guard 3: intact adjacent vertebra with moderate isp_ratio stays Normal."""
        intact_l2 = {
            'level': 'L2',
            'q_seg': 0.90,
            'ha_hp_ratio': 0.95,
            'hm_hp_ratio': 0.95,
            'relative_collapse_ratio': 1.05,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.53,
            'translation_mm': 1.2,
        }
        fused, diag = fuse_vertebra_concept_probabilities(intact_l2, dl_probs=None, dl_model_trained=False)
        self.assertFalse(any('Distraction Floor Guard' in g for g in diag['triggered_guards']))
        self.assertLess(fused['p_post_tb'], 0.65)
        self.assertLess(fused['p_fracture'], 0.25)

        # Standalone extreme widening (>= 1.65) or multimodal B2 confirmation still triggers Guard 3
        proto_b2 = {'p_anomaly': 0.85, 'predicted_class': 'B2', 'class_probabilities': {'B2': 0.80}}
        fused_proto, diag_proto = fuse_vertebra_concept_probabilities(
            intact_l2, dl_probs=None, dl_model_trained=False, prototype_result=proto_b2
        )
        self.assertTrue(any('Distraction Floor Guard' in g for g in diag_proto['triggered_guards']))
        self.assertGreaterEqual(fused_proto['p_post_tb'], 0.65)

    def test_guard4_intact_lower_lumbar_physiological_slip_not_type_c(self):
        """Hotfix 3.2 Guard 4: intact L4/L5 with 6-7mm degenerative slip is protected from Type C."""
        intact_l5 = {
            'level': 'L5',
            'q_seg': 0.90,
            'ha_hp_ratio': 1.0,
            'hm_hp_ratio': 1.0,
            'relative_collapse_ratio': 1.05,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.0,
            'translation_mm': 7.1,
        }
        fused, diag = fuse_vertebra_concept_probabilities(intact_l5, dl_probs=None, dl_model_trained=False)
        self.assertLessEqual(fused['p_trans'], 0.45)
        self.assertTrue(any('Lower Lumbar Degenerative Slip Guard' in g for g in diag['triggered_guards']))

        res = fuse_and_classify_spine([intact_l5], dl_concept_probs_list=None, dl_model_trained=False)
        self.assertNotEqual(res['ranked_vertebrae'][0]['predicted_class'], 'C')

        # True L5 dislocation (>= 7.5 mm) still classifies as C
        true_c_l5 = dict(intact_l5, translation_mm=8.5, isp_ratio=1.80)
        res_c = fuse_and_classify_spine([true_c_l5], dl_concept_probs_list=None, dl_model_trained=False)
        self.assertEqual(res_c['ranked_vertebrae'][0]['predicted_class'], 'C')

    def test_guard5_severe_burst_not_overridden_to_b3_and_intact_neighbor_unfloored(self):
        """Hotfix 3.2 Guard 5: severe A4 burst with secondary wedging is not overridden to B3."""
        burst_l1 = {
            'level': 'L1',
            'q_seg': 0.90,
            'ha_hp_ratio': 0.57,
            'hm_hp_ratio': 0.71,
            'relative_collapse_ratio': 0.62,
            'delta_posterior_mm': 5.5,
            'isp_ratio': 1.18,
            'translation_mm': 1.4,
            'anterior_disc_angle_deg': 18.25,
            'disc_opening_ratio': 3.24,
        }
        res = fuse_and_classify_spine([burst_l1], dl_concept_probs_list=None, dl_model_trained=False)
        top1 = res['ranked_vertebrae'][0]
        self.assertIn(top1['predicted_class'], ('A4', 'A3'))
        self.assertNotEqual(top1['predicted_class'], 'B3')
        self.assertTrue(any('Burst Kyphosis B3 Suppression' in g for g in top1['fusion_diagnostics']['triggered_guards']))

        # Intact neighbor exposed to large adjacent disc angle must not get synthetic s_frac=0.65
        intact_t11 = {
            'level': 'T11',
            'q_seg': 0.90,
            'ha_hp_ratio': 1.0,
            'hm_hp_ratio': 0.90,
            'relative_collapse_ratio': 0.95,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.0,
            'translation_mm': 1.2,
            'anterior_disc_angle_deg': 20.0,
            'disc_opening_ratio': 2.4,
        }
        fused_t11, _ = fuse_vertebra_concept_probabilities(intact_t11, dl_probs=None, dl_model_trained=False)
        self.assertLess(fused_t11['p_fracture'], 0.25)

    def test_decision_tree_p_abnormal_prevents_normal_and_preserves_raw_metrics(self):
        """Hotfix 3.2 rules: high aggregate p_abnormal beats Normal argmax; Normal preserves raw_metrics."""
        diffuse_probs = {
            'p_trans': 0.10,
            'p_ant_tb': 0.25,
            'p_post_tb': 0.25,
            'p_pw': 0.35,
            'p_ep2': 0.35,
            'p_fracture': 0.72,
            'p_split': 0.0,
        }
        res_abn = evaluate_aospine_decision_tree({'level': 'L1', 'q_seg': 0.85}, fused_concept_probs=diffuse_probs)
        self.assertNotEqual(res_abn['predicted_class'], 'Normal')

        res_norm = evaluate_aospine_decision_tree({
            'level': 'L2',
            'q_seg': 0.90,
            'ha_hp_ratio': 0.96,
            'hm_hp_ratio': 0.95,
            'relative_collapse_ratio': 0.98,
            'delta_posterior_mm': 0.85,
            'isp_ratio': 1.05,
            'translation_mm': 1.1,
        })
        self.assertEqual(res_norm['predicted_class'], 'Normal')
        self.assertAlmostEqual(res_norm['raw_metrics']['delta_posterior_mm'], 0.85, places=2)
        self.assertAlmostEqual(res_norm['raw_metrics']['isp_ratio'], 1.05, places=3)

    def test_hotfix33_is_proto_b2_tightened_on_clear_burst(self):
        """Hotfix 3.3: marginal prototype B2 with clear posterior wall burst (delta_pw >= 2.8)
        and narrow ISP (< 1.30) must NOT trigger Guard 8 / override A3/A4 into B2."""
        burst_l1 = {
            'level': 'L1',
            'q_seg': 0.90,
            'delta_posterior_mm': 5.8,
            'isp_ratio': 1.14,
            'translation_mm': 0.5,
            'relative_collapse_ratio': 0.75,
            'ha_hp_ratio': 0.85,
            'hm_hp_ratio': 0.78,
            'upper_endplate_deformity_mm': 2.2,
            'lower_endplate_deformity_mm': 0.6,
        }
        # Prototype bank slightly favors B2 over A3 (margin = 0.06 < 0.15)
        marginal_proto_b2 = {
            'p_anomaly': 0.72,
            'predicted_class': 'B2',
            'class_probabilities': {'B2': 0.32, 'A3': 0.26, 'A4': 0.20, 'Normal': 0.22},
        }
        dl_ambig = {
            'p_pw': 0.80,
            'p_ep2': 0.25,
            'p_post_tb': 0.60,
            'p_ant_tb': 0.05,
            'p_trans': 0.10,
            'p_fracture': 0.85,
        }
        res = fuse_and_classify_spine(
            [burst_l1],
            dl_concept_probs_list=[dl_ambig],
            dl_model_trained=True,
            prototype_results_list=[marginal_proto_b2],
        )
        top1 = res['ranked_vertebrae'][0]
        self.assertNotIn('Prototype B2 Support', ' '.join(top1['fusion_diagnostics']['triggered_guards']))
        self.assertEqual(top1['predicted_class'], 'A3')

        # Conversely, strong margin (>= 0.15) AND CNN confirmation (dl_tb >= 0.55) does trigger Guard 8
        strong_proto_b2 = {
            'p_anomaly': 0.85,
            'predicted_class': 'B2',
            'class_probabilities': {'B2': 0.52, 'A3': 0.20, 'A4': 0.15, 'Normal': 0.13},
        }
        fused_strong, diag_strong = fuse_vertebra_concept_probabilities(
            burst_l1,
            dl_probs=dl_ambig,
            dl_model_trained=True,
            prototype_result=strong_proto_b2,
        )
        self.assertTrue(any('Prototype B2 Support' in g for g in diag_strong['triggered_guards']))
        self.assertGreaterEqual(fused_strong['p_post_tb'], 0.70)

    def test_principle1_multimodal_normal_consensus_preserves_true_c(self):
        """Principle 1: intact heights + Normal prototype + low CNN suppresses edge noise across T10-L5
        without suppressing true Type C dislocation (ha_hp out of bounds / trans >= thresh)."""
        # Physiological L5 with spurious borderline delta_pw and moderate lordotic translation (< 5.5mm)
        normal_l5 = {
            'level': 'L5',
            'q_seg': 0.90,
            'relative_collapse_ratio': 0.96,
            'ha_hp_ratio': 1.02,
            'hm_hp_ratio': 0.98,
            'delta_posterior_mm': 5.2,
            'isp_ratio': 1.20,
            'translation_mm': 4.4,  # < 5.5 L4/L5 trans_thresh
            'anterior_disc_angle_deg': 26.0,
            'disc_opening_ratio': 2.2,
        }
        proto_norm = {'is_anomaly': False, 'p_anomaly': 0.22, 'predicted_class': 'Normal', 'class_probabilities': {'Normal': 0.85}}
        dl_low = {'p_fracture': 0.18, 'p_pw': 0.15, 'p_post_tb': 0.16, 'p_ant_tb': 0.12, 'p_trans': 0.14, 'p_ep2': 0.10}

        fused_n, diag_n = fuse_vertebra_concept_probabilities(
            normal_l5, dl_probs=dl_low, dl_model_trained=True, prototype_result=proto_norm
        )
        self.assertTrue(any('Multimodal Normal Consensus' in g for g in diag_n['triggered_guards']))
        self.assertLessEqual(fused_n['p_pw'], 0.35)
        self.assertLessEqual(fused_n['p_trans'], 0.35)
        self.assertLessEqual(fused_n['p_post_tb'], 0.35)
        self.assertLessEqual(fused_n['p_fracture'], 0.35)

        res_n = fuse_and_classify_spine([normal_l5], [dl_low], dl_model_trained=True, prototype_results_list=[proto_norm])
        self.assertEqual(res_n['ranked_vertebrae'][0]['predicted_class'], 'Normal')

        # True Type C (e.g. T12 wedge-dislocation: ha_hp=2.03, trans_mm=6.6, proto=Normal) is NOT suppressed
        true_c_t12 = {
            'level': 'T12',
            'q_seg': 0.90,
            'relative_collapse_ratio': 1.01,
            'ha_hp_ratio': 2.03,
            'hm_hp_ratio': 1.15,
            'delta_posterior_mm': 4.7,
            'isp_ratio': 1.00,
            'translation_mm': 6.6,
        }
        res_c = fuse_and_classify_spine([true_c_t12], [dl_low], dl_model_trained=True, prototype_results_list=[proto_norm])
        top_c = res_c['ranked_vertebrae'][0]
        self.assertFalse(any('Multimodal Normal Consensus' in g for g in top_c['fusion_diagnostics']['triggered_guards']))
        self.assertEqual(top_c['predicted_class'], 'C')

    def test_principle2_burst_vs_b2_consensus_at_guard3(self):
        """Principle 2: clear burst (delta_pw >= 2.8) with isp_ratio=1.48 and proto=A3 stays A3
        instead of B2; true proto=B2 at the same metrics still classifies as B2."""
        burst_l1 = {
            'level': 'L1',
            'q_seg': 0.90,
            'delta_posterior_mm': 5.8,
            'isp_ratio': 1.48,
            'translation_mm': 0.8,
            'relative_collapse_ratio': 0.73,
            'ha_hp_ratio': 0.72,
            'hm_hp_ratio': 0.76,
            'upper_endplate_deformity_mm': 2.5,
            'lower_endplate_deformity_mm': 0.6,
        }
        dl_burst_ambig = {
            'p_pw': 0.85,
            'p_ep2': 0.18,
            'p_post_tb': 0.65,
            'p_ant_tb': 0.05,
            'p_trans': 0.08,
            'p_fracture': 0.90,
        }
        proto_a3 = {
            'is_anomaly': True,
            'p_anomaly': 0.85,
            'predicted_class': 'A3',
            'class_probabilities': {'A3': 0.70, 'A4': 0.15, 'B2': 0.05, 'Normal': 0.10},
        }
        res_a3 = fuse_and_classify_spine([burst_l1], [dl_burst_ambig], dl_model_trained=True, prototype_results_list=[proto_a3])
        top_a3 = res_a3['ranked_vertebrae'][0]
        self.assertFalse(any('Distraction Floor Guard' in g for g in top_a3['fusion_diagnostics']['triggered_guards']))
        self.assertEqual(top_a3['predicted_class'], 'A3')

        # Conversely, true proto=B2 at the exact same burst+ISP metrics activates B2
        proto_b2 = {
            'is_anomaly': True,
            'p_anomaly': 0.85,
            'predicted_class': 'B2',
            'class_probabilities': {'B2': 0.75, 'A3': 0.10, 'A4': 0.10, 'Normal': 0.05},
        }
        res_b2 = fuse_and_classify_spine([burst_l1], [dl_burst_ambig], dl_model_trained=True, prototype_results_list=[proto_b2])
        top_b2 = res_b2['ranked_vertebrae'][0]
        self.assertTrue(any('Distraction Floor Guard' in g for g in top_b2['fusion_diagnostics']['triggered_guards']))
        self.assertEqual(top_b2['predicted_class'], 'B2')

    def test_hotfix43a_multimodal_normal_caps_s_ant_tb(self):
        """Hotfix 4.3a: intact T11 with large anterior disc angle (22 deg, opening 2.5)
        satisfying Multimodal Normal Consensus caps p_ant_tb <= 0.25 and stays Normal (not B3)."""
        intact_t11 = {
            'level': 'T11',
            'q_seg': 0.90,
            'relative_collapse_ratio': 1.12,
            'ha_hp_ratio': 0.99,
            'hm_hp_ratio': 0.86,
            'delta_posterior_mm': 0.0,
            'isp_ratio': 1.00,
            'translation_mm': 0.8,
            'anterior_disc_angle_deg': 22.0,
            'disc_opening_ratio': 2.5,
            'upper_endplate_deformity_mm': 2.5,
            'lower_endplate_deformity_mm': 0.5,
        }
        proto_norm = {'is_anomaly': False, 'p_anomaly': 0.15, 'predicted_class': 'Normal', 'class_probabilities': {'Normal': 0.90}}
        # Simulates p_ant_tb default low (<= 0.02) where initial blend falls back to p_rule['p_ant_tb']
        dl_norm = {'p_fracture': 0.09, 'p_pw': 0.05, 'p_post_tb': 0.09, 'p_ant_tb': 0.01, 'p_trans': 0.06, 'p_ep2': 0.06}

        fused, diag = fuse_vertebra_concept_probabilities(
            intact_t11, dl_probs=dl_norm, dl_model_trained=True, prototype_result=proto_norm
        )
        self.assertLessEqual(fused['p_ant_tb'], 0.25)
        self.assertLessEqual(fused['p_frac' if 'p_frac' in fused else 'p_fracture'], 0.35)
        self.assertTrue(any('S_ant_tb' in g for g in diag['triggered_guards']))

        res = fuse_and_classify_spine([intact_t11], [dl_norm], dl_model_trained=True, prototype_results_list=[proto_norm])
        self.assertEqual(res['ranked_vertebrae'][0]['predicted_class'], 'Normal')


if __name__ == '__main__':
    unittest.main()
