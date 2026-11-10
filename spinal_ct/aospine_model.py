"""Dual-Stream 2.5D Concept Network for AO Spine Classification (Phase 2 & 3).

Optimized for pure offline CPU execution on desktop/workstation environments.
Uses lightweight depthwise-separable convolutional blocks and GroupNorm
to extract anatomical features and predict multi-task concept probabilities
without any external weights or network calls.
"""
from __future__ import annotations

import time
from typing import Any, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONCEPT_NAMES = ['p_pw', 'p_ep2', 'p_post_tb', 'p_ant_tb', 'p_trans', 'p_fracture']


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-Separable Convolution with GroupNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Use GroupNorm with max 8 groups (or divisor of out_channels)
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(x)
        return self.act(x)


class StreamEncoder25D(nn.Module):
    """Lightweight 2.5D ROI Encoder producing a 128-dimensional anatomical representation."""

    def __init__(self, in_channels: int = 5, feature_dim: int = 128):
        super().__init__()
        # Initial stem: (B, 5, 128, 128) -> (B, 32, 64, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        # Stage 1: (B, 32, 64, 64) -> (B, 64, 32, 32)
        self.stage1 = DepthwiseSeparableConv2d(32, 64, stride=2)
        # Stage 2: (B, 64, 32, 32) -> (B, 96, 16, 16)
        self.stage2 = DepthwiseSeparableConv2d(64, 96, stride=2)
        # Stage 3: (B, 96, 16, 16) -> (B, feature_dim, 8, 8)
        self.stage3 = DepthwiseSeparableConv2d(96, feature_dim, stride=2)

        # Global average pool + projector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class DualStreamAOSpineNet(nn.Module):
    """Dual-Stream 2.5D Anatomical Concept Network.

    Streams:
    - body_encoder: processes body_canal_roi (5, 128, 128) -> f_body (128-d)
    - arch_encoder: processes posterior_arch_roi (5, 128, 128) -> f_arch (128-d)

    Heads:
    - head_pw(f_body) -> Logits for posterior wall compromise P_DL(pw)
    - head_ep2(f_body) -> Logits for both endplates comminution P_DL(ep2)
    - head_fracture(f_body) -> Logits for vertebral body fracture P_DL(fracture)
    - head_tb(cat([f_body, f_arch])) -> Logits for tension band disruption P_DL(tb)
    - head_trans(cat([f_body, f_arch])) -> Logits for translation / dislocation P_DL(trans)
    """

    def __init__(self, in_channels: int = 5, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.body_encoder = StreamEncoder25D(in_channels=in_channels, feature_dim=feature_dim)
        self.arch_encoder = StreamEncoder25D(in_channels=in_channels, feature_dim=feature_dim)

        # Concept Heads
        self.head_pw = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_ep2 = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_fracture = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_tb = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_trans = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.is_trained_checkpoint: bool = False

    def load_checkpoint(self, checkpoint_path: str | Path, map_location: str = 'cpu') -> None:
        """Load model state_dict from a trained checkpoint file."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.is_trained_checkpoint = True

    def forward(
        self,
        body_roi: torch.Tensor,
        arch_roi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass extracting anatomical features and concept logits.

        Parameters
        ----------
        body_roi : torch.Tensor
            Batch of Body & Canal 2.5D ROIs of shape (B, 5, 128, 128).
        arch_roi : torch.Tensor
            Batch of Posterior Arch 2.5D ROIs of shape (B, 5, 128, 128).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing feature vectors and raw logits.
        """
        f_body = self.body_encoder(body_roi)
        f_arch = self.arch_encoder(arch_roi)
        f_joint = torch.cat([f_body, f_arch], dim=-1)

        logits_pw = self.head_pw(f_body)
        logits_ep2 = self.head_ep2(f_body)
        logits_fracture = self.head_fracture(f_body)
        logits_tb = self.head_tb(f_joint)
        logits_trans = self.head_trans(f_joint)

        return {
            'logits_pw': logits_pw,
            'logits_ep2': logits_ep2,
            'logits_fracture': logits_fracture,
            'logits_tb': logits_tb,
            'logits_trans': logits_trans,
            'f_body': f_body,
            'f_arch': f_arch,
        }

    @torch.no_grad()
    def predict_concept_probabilities(
        self,
        body_roi: torch.Tensor | np.ndarray,
        arch_roi: torch.Tensor | np.ndarray,
    ) -> list[dict[str, float]] | dict[str, float]:
        """Predict calibrated anatomical concept probabilities clipped to [0.01, 0.99].

        Accepts either torch Tensors or numpy arrays of shape (5, 128, 128) or (B, 5, 128, 128).
        """
        self.eval()

        was_single = False
        if isinstance(body_roi, np.ndarray):
            b_t = torch.from_numpy(body_roi.astype(np.float32))
        else:
            b_t = body_roi.float()

        if isinstance(arch_roi, np.ndarray):
            a_t = torch.from_numpy(arch_roi.astype(np.float32))
        else:
            a_t = arch_roi.float()

        if b_t.ndim == 3:
            b_t = b_t.unsqueeze(0)
            a_t = a_t.unsqueeze(0)
            was_single = True

        device = next(self.parameters()).device
        b_t = b_t.to(device)
        a_t = a_t.to(device)

        out = self.forward(b_t, a_t)

        p_pw = torch.clamp(torch.sigmoid(out['logits_pw']).squeeze(-1), 0.01, 0.99).cpu().numpy()
        p_ep2 = torch.clamp(torch.sigmoid(out['logits_ep2']).squeeze(-1), 0.01, 0.99).cpu().numpy()
        p_frac = torch.clamp(torch.sigmoid(out['logits_fracture']).squeeze(-1), 0.01, 0.99).cpu().numpy()
        p_tb = torch.clamp(torch.sigmoid(out['logits_tb']).squeeze(-1), 0.01, 0.99).cpu().numpy()
        p_trans = torch.clamp(torch.sigmoid(out['logits_trans']).squeeze(-1), 0.01, 0.99).cpu().numpy()

        # Handle batch or single
        b_size = b_t.shape[0]
        p_pw = np.atleast_1d(p_pw)
        p_ep2 = np.atleast_1d(p_ep2)
        p_frac = np.atleast_1d(p_frac)
        p_tb = np.atleast_1d(p_tb)
        p_trans = np.atleast_1d(p_trans)

        results = []
        for i in range(b_size):
            tb_val = float(round(float(p_tb[i]), 4))
            results.append({
                'p_pw': float(round(float(p_pw[i]), 4)),
                'p_ep2': float(round(float(p_ep2[i]), 4)),
                'p_post_tb': tb_val,
                'p_ant_tb': 0.01,  # Anterior tension band evaluated geometrically / secondary
                'p_trans': float(round(float(p_trans[i]), 4)),
                'p_fracture': float(round(float(p_frac[i]), 4)),
            })

        if was_single:
            return results[0]
        return results


class MaskedMultiConceptLoss(nn.Module):
    """Multi-task binary loss masked by anatomical concept supervision validity.

    Supervises:
    - [p_pw, p_ep2, p_tb, p_trans] using concept_mask [m_pw, m_ep2, m_tb, m_trans]
    - Optional is_fractured supervision for head_fracture.

    Elements where concept_mask == 0 produce 0 loss and receive no gradient backpropagation.
    """

    def __init__(self, fracture_weight: float = 0.5):
        super().__init__()
        self.fracture_weight = fracture_weight

    def forward(
        self,
        logits_dict: dict[str, torch.Tensor],
        y_concepts: torch.Tensor,
        concept_mask: torch.Tensor,
        is_fractured: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked multi-concept cross entropy.

        Parameters
        ----------
        logits_dict : dict
            Model outputs containing 'logits_pw', 'logits_ep2', 'logits_tb', 'logits_trans', 'logits_fracture'.
        y_concepts : torch.Tensor
            Binary targets of shape (B, 4) corresponding to [pw, ep2, tb, trans].
        concept_mask : torch.Tensor
            Binary validity mask of shape (B, 4).
        is_fractured : torch.Tensor, optional
            Binary target of shape (B,) or (B, 1) indicating fracture presence.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        # Stack logits: (B, 4)
        logits_stack = torch.cat([
            logits_dict['logits_pw'],
            logits_dict['logits_ep2'],
            logits_dict['logits_tb'],
            logits_dict['logits_trans'],
        ], dim=-1)

        y_float = y_concepts.float()
        mask_float = concept_mask.float()

        # BCE with logits: (B, 4)
        bce = F.binary_cross_entropy_with_logits(logits_stack, y_float, reduction='none')
        masked_bce = bce * mask_float

        mask_sum = mask_float.sum()
        if mask_sum > 0:
            loss_concepts = masked_bce.sum() / mask_sum
        else:
            loss_concepts = torch.tensor(0.0, device=logits_stack.device, requires_grad=True)

        if is_fractured is not None:
            frac_target = is_fractured.float()
            if frac_target.ndim == 1:
                frac_target = frac_target.unsqueeze(-1)
            loss_frac = F.binary_cross_entropy_with_logits(
                logits_dict['logits_fracture'],
                frac_target,
            )
            return loss_concepts + self.fracture_weight * loss_frac

        return loss_concepts


class BatchedCPUInferenceEngine:
    """High-throughput inference engine optimized for batched spine CT scans on CPU."""

    def __init__(self, model: DualStreamAOSpineNet | None = None):
        if model is None:
            model = DualStreamAOSpineNet()
        self.model = model
        self.model.eval()

    def predict_batch(
        self,
        body_rois: Sequence[np.ndarray] | np.ndarray,
        arch_rois: Sequence[np.ndarray] | np.ndarray,
    ) -> tuple[list[dict[str, float]], float]:
        """Run batched inference across all vertebrae in session (e.g. B=8 for T10-L5).

        Returns
        -------
        list of dict, float
            List of concept probability dicts and execution latency in seconds.
        """
        t0 = time.perf_counter()

        if isinstance(body_rois, (list, tuple)):
            body_batch = np.stack(body_rois, axis=0)
        else:
            body_batch = body_rois

        if isinstance(arch_rois, (list, tuple)):
            arch_batch = np.stack(arch_rois, axis=0)
        else:
            arch_batch = arch_rois

        probs = self.model.predict_concept_probabilities(body_batch, arch_batch)
        if isinstance(probs, dict):
            probs = [probs]

        latency = time.perf_counter() - t0
        return probs, latency


class _ONNXExportWrapper(nn.Module):
    """Wrapper that exposes only the 5 concept logits as a clean tuple for ONNX export."""

    def __init__(self, model: DualStreamAOSpineNet):
        super().__init__()
        self.model = model

    def forward(
        self,
        body_roi: torch.Tensor,
        arch_roi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(body_roi, arch_roi)
        return (
            out['logits_pw'],
            out['logits_ep2'],
            out['logits_fracture'],
            out['logits_tb'],
            out['logits_trans'],
        )


def export_to_onnx(
    model: DualStreamAOSpineNet,
    onnx_path: str | Path,
    batch_size: int = 8,
) -> Path:
    """Export DualStreamAOSpineNet to ONNX format with dynamic batch dimensions.

    Parameters
    ----------
    model : DualStreamAOSpineNet
        Trained or initialized PyTorch model.
    onnx_path : str or Path
        Target filepath for .onnx file.
    batch_size : int
        Sample batch size for tracing.

    Returns
    -------
    Path
        Path to exported ONNX model.
    """
    out_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _ONNXExportWrapper(model)
    wrapper.eval()
    dummy_body = torch.randn(batch_size, 5, 128, 128, dtype=torch.float32)
    dummy_arch = torch.randn(batch_size, 5, 128, 128, dtype=torch.float32)

    input_names = ['body_roi', 'arch_roi']
    output_names = ['logits_pw', 'logits_ep2', 'logits_fracture', 'logits_tb', 'logits_trans']

    dynamic_axes = {
        'body_roi': {0: 'batch_size'},
        'arch_roi': {0: 'batch_size'},
        'logits_pw': {0: 'batch_size'},
        'logits_ep2': {0: 'batch_size'},
        'logits_fracture': {0: 'batch_size'},
        'logits_tb': {0: 'batch_size'},
        'logits_trans': {0: 'batch_size'},
    }

    torch.onnx.export(
        wrapper,
        (dummy_body, dummy_arch),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )
    return out_path
