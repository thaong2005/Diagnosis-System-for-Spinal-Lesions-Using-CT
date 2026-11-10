"""Adaptive Rule Fusion Layer for AO Spine Classification (Phase 3).

Dynamically balances deep visual features (DualStreamAOSpineNet) and
precise physical geometric rules (spinal_ct.morphometry) based on segmentation
quality (q_seg) and clinical Anatomical Boundary Guards.

Contract:
- Reciprocal Reliability:
  - High-quality mask (q_seg >= 0.75): favors geometry (alpha_dl ~ 0.35).
  - Comminuted / broken mask (q_seg < 0.50): favors visual features (alpha_dl ~ 0.80).
- Anatomical Boundary Guards:
  - Intact Wall Guard: Prevents false positive A3/A4 on A1 when delta_pw < 1.0mm.
  - Bone Fragment Floor Guard: Enforces burst suspicion when delta_pw >= 2.8mm.
  - Distraction Floor Guard: Enforces B2 suspicion when isp_ratio >= 1.35.
  - Translation Floor Guard: Enforces C suspicion when translation_mm >= 3.5mm.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from spinal_ct.aospine_rules import (
    compute_anatomical_rule_probabilities,
    rank_and_classify_spine,
)


def compute_adaptive_fusion_weights(
    q_seg: float,
    morph_metrics: dict[str, Any] | None = None,
    dl_model_trained: bool = True,
) -> dict[str, float]:
    """Compute adaptive alpha weights for fusing Deep Learning and Geometric Rules.

    S_k = alpha_k * P_DL(k) + (1 - alpha_k) * P_Rule(k)

    Parameters
    ----------
    q_seg : float
        Segmentation quality score in [0.0, 1.0].
    morph_metrics : dict, optional
        Vertebra measurements dictionary.
    dl_model_trained : bool
        If False, returns all alphas = 0.0 (pure rule mode fallback).

    Returns
    -------
    dict[str, float]
        Dictionary mapping concept keys to alpha weights in [0.15, 0.85] (or 0.0 if not trained).
    """
    if not dl_model_trained:
        return {
            'pw': 0.0,
            'ep2': 0.0,
            'post_tb': 0.0,
            'ant_tb': 0.0,
            'trans': 0.0,
            'fracture': 0.0,
        }

    q = float(np.clip(q_seg, 0.0, 1.0))

    # Reciprocal Reliability mapping:
    # q_seg >= 0.75 -> alpha = 0.35 (trust crisp geometry)
    # q_seg <= 0.40 -> alpha = 0.80 (trust visual CNN when mask is comminuted/broken)
    if q >= 0.75:
        base_alpha = 0.35
    elif q <= 0.40:
        base_alpha = 0.80
    else:
        # Linear interpolation between 0.40 and 0.75
        ratio = (q - 0.40) / (0.75 - 0.40)
        base_alpha = 0.80 - ratio * (0.80 - 0.35)

    base_alpha = float(np.clip(base_alpha, 0.15, 0.85))

    return {
        'pw': round(base_alpha, 4),
        'ep2': round(base_alpha, 4),
        'post_tb': round(base_alpha, 4),
        'ant_tb': round(base_alpha, 4),
        'trans': round(base_alpha, 4),
        'fracture': round(base_alpha, 4),
    }


def fuse_vertebra_concept_probabilities(
    metrics: dict[str, Any],
    dl_probs: dict[str, float] | None = None,
    dl_model_trained: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fuse physical rule probabilities with DL visual concept probabilities for one vertebra.

    Applies Anatomical Boundary Guards:
    - Intact Wall Guard: q_seg >= 0.70, delta_pw < 1.0mm, P_Rule(pw) < 0.15 -> S_pw <= 0.35
    - Bone Fragment Floor Guard: delta_pw >= 2.8mm -> S_pw >= 0.65
    - Distraction Floor Guard: isp_ratio >= 1.35 -> S_post_tb >= 0.65
    - Translation Floor Guard: translation_mm >= 3.5mm -> S_trans >= 0.65

    Parameters
    ----------
    metrics : dict[str, Any]
        Physical measurements dictionary from spinal_ct.morphometry.
    dl_probs : dict[str, float], optional
        Probabilities predicted by DualStreamAOSpineNet.
    dl_model_trained : bool
        Whether DL model is trained and active.

    Returns
    -------
    tuple of (dict, dict)
        (fused_concept_probs, fusion_diagnostics)
    """
    p_rule = compute_anatomical_rule_probabilities(metrics)
    q_seg = float(metrics.get('q_seg', 1.0))

    if dl_probs is None or not dl_model_trained:
        # Full fallback to pure geometric rule
        fused = dict(p_rule)
        diagnostics = {
            'alpha_weights': {k: 0.0 for k in ('pw', 'ep2', 'post_tb', 'ant_tb', 'trans', 'fracture')},
            'raw_p_rule': p_rule,
            'raw_p_dl': {},
            'fused_probs': fused,
            'triggered_guards': [],
            'q_seg': q_seg,
        }
        return fused, diagnostics

    alphas = compute_adaptive_fusion_weights(q_seg, morph_metrics=metrics, dl_model_trained=True)

    # Initial convex combination: S_k = alpha_k * P_DL + (1 - alpha_k) * P_Rule
    s_pw = alphas['pw'] * dl_probs.get('p_pw', 0.5) + (1.0 - alphas['pw']) * p_rule['p_pw']
    s_ep2 = alphas['ep2'] * dl_probs.get('p_ep2', 0.5) + (1.0 - alphas['ep2']) * p_rule['p_ep2']
    s_post_tb = alphas['post_tb'] * dl_probs.get('p_post_tb', 0.5) + (1.0 - alphas['post_tb']) * p_rule['p_post_tb']

    # For anterior tension band (Type B3): do not suppress high geometric evidence if DL has default/dummy low probability
    p_dl_ant = dl_probs.get('p_ant_tb', 0.01)
    if p_dl_ant <= 0.02 and p_rule['p_ant_tb'] > 0.10:
        s_ant_tb = p_rule['p_ant_tb']
    else:
        s_ant_tb = alphas['ant_tb'] * p_dl_ant + (1.0 - alphas['ant_tb']) * p_rule['p_ant_tb']

    s_trans = alphas['trans'] * dl_probs.get('p_trans', 0.5) + (1.0 - alphas['trans']) * p_rule['p_trans']
    s_frac = alphas['fracture'] * dl_probs.get('p_fracture', 0.5) + (1.0 - alphas['fracture']) * p_rule['p_fracture']

    triggered_guards: list[str] = []

    # Physical measurements for guards
    delta_pw = float(metrics.get('delta_posterior_mm', 0.0))
    isp_ratio = float(metrics.get('isp_ratio', 1.0))
    trans_mm = float(metrics.get('translation_mm', 0.0))
    disc_deg = float(metrics.get('anterior_disc_angle_deg', 0.0))
    disc_ratio = float(metrics.get('disc_opening_ratio', 1.0))

    # Guard 1: Intact Wall Guard (Rescues pure A1 from CNN false alarms)
    if q_seg >= 0.70 and delta_pw < 1.0 and p_rule['p_pw'] < 0.15:
        s_pw = min(s_pw, 0.35)
        if dl_probs.get('p_pw', 0.0) >= 0.40 or s_pw >= 0.35:
            triggered_guards.append(
                f"Intact Wall Guard (q_seg={q_seg:.2f}, delta_pw={delta_pw:.1f}mm: S_pw constrained to {s_pw:.4f} <= 0.35)"
            )

    # Guard 2: Bone Fragment Floor Guard (Burst fracture with clear retropulsion)
    if delta_pw >= 2.8:
        s_pw = max(s_pw, 0.65)
        s_frac = max(s_frac, 0.70)
        triggered_guards.append(
            f"Bone Fragment Floor Guard (delta_pw={delta_pw:.1f}mm: S_pw and S_frac floored to >= 0.65)"
        )

    # Guard 3: Distraction Floor Guard (Interspinous widening B2)
    if isp_ratio >= 1.35:
        s_post_tb = max(s_post_tb, 0.65)
        s_frac = max(s_frac, 0.65)
        triggered_guards.append(
            f"Distraction Floor Guard (isp_ratio={isp_ratio:.2f}: S_post_tb and S_frac floored to 0.65)"
        )

    # Guard 4: Translation Floor Guard (Dislocation C)
    if trans_mm >= 3.5:
        s_trans = max(s_trans, 0.65)
        s_frac = max(s_frac, 0.65)
        triggered_guards.append(
            f"Translation Floor Guard (translation_mm={trans_mm:.1f}mm: S_trans and S_frac floored to 0.65)"
        )

    # Guard 5: Hyperextension / Anterior Tension Band Floor Guard (Type B3)
    if (disc_deg >= 15.0 and disc_ratio >= 1.8) or p_rule['p_ant_tb'] >= 0.65:
        s_ant_tb = max(s_ant_tb, 0.65)
        s_frac = max(s_frac, 0.65)
        triggered_guards.append(
            f"Anterior Tension Band Floor Guard (angle={disc_deg:.1f}deg, ratio={disc_ratio:.2f}: S_ant_tb and S_frac floored to 0.65)"
        )

    # Guard 6: Intact Tension Band Guard (Prevents CNN vertical laminar crack noise triggering false B2)
    if q_seg >= 0.75 and isp_ratio < 1.12 and p_rule['p_post_tb'] < 0.15:
        s_post_tb = min(s_post_tb, 0.35)
        if dl_probs.get('p_post_tb', 0.0) >= 0.40 or s_post_tb >= 0.35:
            triggered_guards.append(
                f"Intact Tension Band Guard (q_seg={q_seg:.2f}, isp_ratio={isp_ratio:.2f}: S_post_tb constrained to {s_post_tb:.4f} <= 0.35)"
            )

    fused_probs = {
        'p_pw': float(round(float(np.clip(s_pw, 0.01, 0.99)), 4)),
        'p_ep2': float(round(float(np.clip(s_ep2, 0.01, 0.99)), 4)),
        'p_post_tb': float(round(float(np.clip(s_post_tb, 0.01, 0.99)), 4)),
        'p_ant_tb': float(round(float(np.clip(s_ant_tb, 0.01, 0.99)), 4)),
        'p_trans': float(round(float(np.clip(s_trans, 0.01, 0.99)), 4)),
        'p_fracture': float(round(float(np.clip(s_frac, 0.01, 0.99)), 4)),
        'p_split': p_rule.get('p_split', 0.0),
    }

    diagnostics = {
        'alpha_weights': alphas,
        'raw_p_rule': p_rule,
        'raw_p_dl': dl_probs,
        'fused_probs': fused_probs,
        'triggered_guards': triggered_guards,
        'q_seg': q_seg,
    }

    return fused_probs, diagnostics


def fuse_and_classify_spine(
    vertebrae_metrics: list[dict[str, Any]],
    dl_concept_probs_list: list[dict[str, float]] | None = None,
    dl_model_trained: bool = True,
) -> dict[str, Any]:
    """Fuse physical rules and DL visual features across spine and produce ranked diagnosis.

    Parameters
    ----------
    vertebrae_metrics : list of dict
        Physical measurements for each vertebra in cranial-to-caudal order.
    dl_concept_probs_list : list of dict, optional
        Precomputed DL concept probabilities matching vertebrae_metrics.
    dl_model_trained : bool
        Whether DL concept predictions should be dynamically fused.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - 'ranked_vertebrae': list of vertebrae classified and sorted by severity,
          each equipped with 'fusion_diagnostics'.
        - 'patient_summary': high-level diagnosis and triage summary.
    """
    if not vertebrae_metrics:
        return {'ranked_vertebrae': [], 'patient_summary': {}}

    fused_probs_list: list[dict[str, float]] = []
    diagnostics_by_level: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(vertebrae_metrics):
        dl_p = None
        if dl_concept_probs_list and idx < len(dl_concept_probs_list):
            dl_p = dl_concept_probs_list[idx]

        fused_p, diag = fuse_vertebra_concept_probabilities(
            item,
            dl_probs=dl_p,
            dl_model_trained=dl_model_trained,
        )
        fused_probs_list.append(fused_p)
        level_key = str(item.get('level', f"V_{idx}"))
        diagnostics_by_level[level_key] = diag

    # Call rank_and_classify_spine with fused probabilities
    result = rank_and_classify_spine(
        vertebrae_metrics,
        fused_concept_probs_list=fused_probs_list,
    )

    # Attach fusion_diagnostics to each ranked vertebra
    for ev in result.get('ranked_vertebrae', []):
        lvl = str(ev.get('level', ''))
        ev['fusion_diagnostics'] = diagnostics_by_level.get(lvl, {})

    return result
