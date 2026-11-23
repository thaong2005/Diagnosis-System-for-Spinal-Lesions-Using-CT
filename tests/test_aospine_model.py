"""Unit tests for Phase 2 DualStreamAOSpineNet concept CNN, masked loss, and ONNX export."""
import time
import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
import torch

from spinal_ct.aospine_model import (
    DualStreamAOSpineNet,
    MaskedMultiConceptLoss,
    BatchedCPUInferenceEngine,
    export_to_onnx,
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

        # Warm-up pass for cold-start JIT overhead on Windows CPU
        engine.predict_batch(body_list[:1], arch_list[:1])

        probs, latency = engine.predict_batch(body_list, arch_list)
        self.assertEqual(len(probs), 6)
        self.assertLess(latency, 1.5)

    def test_is_trained_checkpoint_flag(self):
        """Test DualStreamAOSpineNet tracks is_trained_checkpoint attribute."""
        net = DualStreamAOSpineNet()
        self.assertTrue(hasattr(net, 'is_trained_checkpoint'))
        self.assertFalse(net.is_trained_checkpoint)

    def test_export_to_onnx_wrapper_clean_outputs(self):
        """Test export_to_onnx exports exactly 5 concept heads via _ONNXExportWrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model.onnx"
            res_path = export_to_onnx(self.model, out_path, batch_size=2)
            self.assertTrue(res_path.is_file())
            onnx_model = onnx.load(str(res_path))
            output_names = [o.name for o in onnx_model.graph.output]
            expected = ['logits_pw', 'logits_ep2', 'logits_fracture', 'logits_tb', 'logits_trans']
            self.assertEqual(output_names, expected)


if __name__ == '__main__':
    unittest.main()
