"""Unit tests for Phase 2 & 3: Dual-Stream 2.5D Concept CNN and Adaptive Rule Fusion Layer."""
import time
import unittest
import numpy as np
import torch

from spinal_ct.aospine_model import (
    DualStreamAOSpineNet,
    MaskedMultiConceptLoss,
    BatchedCPUInferenceEngine,
    export_to_onnx,
)
from spinal_ct.aospine_fusion import (
    compute_adaptive_fusion_weights,
    fuse_vertebra_concept_probabilities,
    fuse_and_classify_spine,
)


class TestAOSpineModel(unittest.TestCase):
    """Test DualStreamAOSpineNet architecture, masked loss, and CPU inference speed."""

    def setUp(self):
        self.model = DualStreamAOSpineNet(in_channels=5, feature_dim=128)
        self.model.eval()

    def test_forward_pass_shape_and_speed_cpu(self):
        """Test B=8 forward pass runs on CPU in < 0.25 seconds with outputs in [0.01, 0.99]."""
        batch_size = 8
        dummy_body = torch.randn(batch_size, 5, 128, 128, dtype=torch.float32)
        dummy_arch = torch.randn(batch_size, 5, 128, 128, dtype=torch.float32)

        t0 = time.perf_counter()
        probs_list = self.model.predict_concept_probabilities(dummy_body, dummy_arch)
        latency = time.perf_counter() - t0

        self.assertLess(latency, 0.25, f"B=8 inference latency too high: {latency*1000:.1f}ms (limit 250ms)")
        self.assertEqual(len(probs_list), batch_size)

        for probs in probs_list:
            for k in ['p_pw', 'p_ep2', 'p_post_tb', 'p_ant_tb', 'p_trans', 'p_fracture']:
                self.assertIn(k, probs)
                val = probs[k]
                self.assertGreaterEqual(val, 0.01)
                self.assertLessEqual(val, 0.99)

    def test_masked_loss_gradient_isolation(self):
        """Test MaskedMultiConceptLoss isolates unobserved concepts (zero gradients for masked heads)."""
        trainable_model = DualStreamAOSpineNet(in_channels=5, feature_dim=128)
        trainable_model.train()

        b_roi = torch.randn(2, 5, 128, 128)
        a_roi = torch.randn(2, 5, 128, 128)

        out = trainable_model(b_roi, a_roi)
        loss_fn = MaskedMultiConceptLoss()

        # Concept order: [pw, ep2, tb, trans]
        # Supervise pw and tb only; mask out ep2 and trans (m_ep2 = 0, m_trans = 0)
        y = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

        loss = loss_fn(out, y, mask)
        loss.backward()

        # head_pw and head_tb MUST have gradients
        self.assertIsNotNone(trainable_model.head_pw[0].weight.grad)
        self.assertGreater(float(trainable_model.head_pw[0].weight.grad.abs().sum()), 0.0)

        self.assertIsNotNone(trainable_model.head_tb[0].weight.grad)
        self.assertGreater(float(trainable_model.head_tb[0].weight.grad.abs().sum()), 0.0)

        # head_ep2 and head_trans MUST receive zero gradient because they were masked out
        ep2_grad = trainable_model.head_ep2[0].weight.grad
        self.assertTrue(ep2_grad is None or bool(torch.all(ep2_grad == 0.0)))

        trans_grad = trainable_model.head_trans[0].weight.grad
        self.assertTrue(trans_grad is None or bool(torch.all(trans_grad == 0.0)))

    def test_batched_cpu_inference_engine(self):
        """Test BatchedCPUInferenceEngine processes list of numpy ROIs."""
        engine = BatchedCPUInferenceEngine(self.model)
        body_list = [np.random.randn(5, 128, 128).astype(np.float32) for _ in range(6)]
        arch_list = [np.random.randn(5, 128, 128).astype(np.float32) for _ in range(6)]

        probs, latency = engine.predict_batch(body_list, arch_list)
        self.assertEqual(len(probs), 6)
        self.assertLess(latency, 0.25)

    def test_is_trained_checkpoint_flag(self):
        """Test DualStreamAOSpineNet tracks is_trained_checkpoint attribute."""
        net = DualStreamAOSpineNet()
        self.assertTrue(hasattr(net, 'is_trained_checkpoint'))
        self.assertFalse(net.is_trained_checkpoint)

    def test_export_to_onnx_wrapper_clean_outputs(self):
        """Test export_to_onnx exports exactly 5 concept heads via _ONNXExportWrapper."""
        import tempfile
        import onnx
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model.onnx"
            res_path = export_to_onnx(self.model, out_path, batch_size=2)
            self.assertTrue(res_path.is_file())
            onnx_model = onnx.load(str(res_path))
            output_names = [o.name for o in onnx_model.graph.output]
            expected = ['logits_pw', 'logits_ep2', 'logits_fracture', 'logits_tb', 'logits_trans']
            self.assertEqual(output_names, expected)


class TestAOSpineFusion(unittest.TestCase):
    """Test Adaptive Rule Fusion Layer and Anatomical Boundary Guards."""

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

        # Simulated noisy DL model output with high posterior wall suspicion due to calcification/bone spur
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

        # Distraction Floor Guard (isp_ratio >= 1.35)
        m_dist = {'level': 'L2', 'q_seg': 0.80, 'delta_posterior_mm': 0.0, 'isp_ratio': 1.45, 'translation_mm': 0.0}
        fused_d, diag_d = fuse_vertebra_concept_probabilities(m_dist, dl_low)
        self.assertGreaterEqual(fused_d['p_post_tb'], 0.65)
        self.assertTrue(any('Distraction Floor Guard' in g for g in diag_d['triggered_guards']))

        # Translation Floor Guard (trans >= 3.5mm)
        m_trans = {'level': 'L3', 'q_seg': 0.80, 'delta_posterior_mm': 0.0, 'isp_ratio': 1.0, 'translation_mm': 4.2}
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


if __name__ == '__main__':
    unittest.main()
