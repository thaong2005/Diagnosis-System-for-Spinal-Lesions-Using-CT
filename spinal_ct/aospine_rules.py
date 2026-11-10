"""AO Spine fuzzy clinical reasoning, probabilistic decision tree, and vertebra ranking."""
from typing import Any
import numpy as np

from spinal_ct.localization import LEVELS

# Standard international AO Spine classes and severity weights
AOSPINE_CLASSES = ('C', 'B3', 'B2', 'A4', 'A3', 'A2', 'A1', 'A0', 'Normal')

SEVERITY_WEIGHTS = {
    'C': 1.00,
    'B3': 0.90,
    'B2': 0.85,
    'A4': 0.70,
    'A3': 0.55,
    'A2': 0.45,
    'A1': 0.35,
    'A0': 0.15,
    'Normal': 0.02,
}


def fuzzy_sigmoid(x: float, threshold: float, temperature: float, invert: bool = False) -> float:
    """Map a physical measurement to a calibrated probability in [0.01, 0.99].

    Parameters
    ----------
    x : float
        Measured physical or morphometric value.
    threshold : float
        Clinical cutoff value where sigmoid probability is exactly 0.50.
    temperature : float
        Slope/scale parameter controlling transition sharpness (must be > 0).
    invert : bool
        If True, smaller x yields higher probability (for collapse/height loss).

    Returns
    -------
    float
        Probability clipped to [0.01, 0.99] to prevent zero-probability collapse.
    """
    if temperature <= 0:
        raise ValueError('temperature must be strictly positive.')
    if not np.isfinite(x) or not np.isfinite(threshold):
        return 0.01 if not invert else 0.99

    diff = (x - threshold) / temperature
    if invert:
        diff = -diff

    diff_clipped = float(np.clip(diff, -50.0, 50.0))
    p = 1.0 / (1.0 + np.exp(-diff_clipped))
    return float(np.clip(p, 0.01, 0.99))


def compute_anatomical_rule_probabilities(metrics: dict[str, Any]) -> dict[str, float]:
    """Compute calibrated probabilities for 6 core anatomical concepts from physical metrics.

    Concepts:
    - p_trans: Vertebral body displacement/translation (Type C)
    - p_ant_tb: Anterior tension band disruption / hyperextension (Type B3)
    - p_post_tb: Posterior tension band disruption / interspinous distraction (Type B2)
    - p_pw: Posterior wall fragment retropulsion into spinal canal (Type A3/A4)
    - p_ep2: Both endplates comminution / complete vertebral collapse (A4 vs A3/A1)
    - p_fracture: Presence of vertebral body fracture / height loss

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary of measurements from spinal_ct.morphometry.

    Returns
    -------
    dict[str, float]
        Calibrated probabilities for each anatomical concept in [0.01, 0.99].
    """
    # 0. Morphological intactness analysis
    h_info = metrics.get('heights') or {}
    ha_hp = float(metrics.get('ha_hp_ratio', h_info.get('anterior_posterior_ratio', 1.0)))
    hm_hp = float(metrics.get('hm_hp_ratio', h_info.get('middle_posterior_ratio', 1.0)))
    relative_collapse_ratio = float(metrics.get('relative_collapse_ratio', 1.0))
    upper_deformity = metrics.get('upper_endplate_deformity_mm')
    lower_deformity = metrics.get('lower_endplate_deformity_mm')
    coronal_split_mm = float(metrics.get('coronal_split_mm', 0.0))
    up_def_val = float(upper_deformity) if upper_deformity is not None else 0.0
    lo_def_val = float(lower_deformity) if lower_deformity is not None else 0.0

    has_endplate_data = (upper_deformity is not None and lower_deformity is not None)
    if has_endplate_data:
        is_morphologically_intact = (
            ha_hp >= 0.80 and hm_hp >= 0.80 and relative_collapse_ratio >= 0.85
            and up_def_val < 2.0 and lo_def_val < 2.0 and coronal_split_mm <= 2.0
        )
    else:
        is_morphologically_intact = (
            ha_hp >= 0.85 and hm_hp >= 0.85 and relative_collapse_ratio >= 0.88
            and float(metrics.get('delta_posterior_mm', 0.0)) < 2.0 and coronal_split_mm <= 2.0
        )

    # 1. Translation / dislocation (Type C)
    translation_mm = float(metrics.get('translation_mm', 0.0))
    p_trans = fuzzy_sigmoid(translation_mm, threshold=3.5, temperature=0.8)
    if is_morphologically_intact and translation_mm < 3.5:
        p_trans = min(p_trans, 0.01)

    # 2. Anterior tension band (Type B3): geometric mean of opening angle and height ratio
    anterior_disc_angle_deg = float(metrics.get('anterior_disc_angle_deg', 0.0))
    disc_opening_ratio = float(metrics.get('disc_opening_ratio', 1.0))
    p_angle = fuzzy_sigmoid(anterior_disc_angle_deg, threshold=17.5, temperature=1.5)
    p_open = fuzzy_sigmoid(disc_opening_ratio, threshold=2.0, temperature=0.35)
    p_ant_tb = float(np.clip(np.sqrt(p_angle * p_open), 0.01, 0.99))
    if is_morphologically_intact and anterior_disc_angle_deg < 15.0:
        p_ant_tb = min(p_ant_tb, 0.01)

    # 3. Posterior tension band (Type B2): interspinous distance ratio (ISP Ratio)
    isp_ratio = float(metrics.get('isp_ratio', 1.0))
    p_post_tb = fuzzy_sigmoid(isp_ratio, threshold=1.30, temperature=0.08)

    # 4. Posterior wall retropulsion (Type A3/A4)
    delta_posterior_mm = float(metrics.get('delta_posterior_mm', 0.0))
    if is_morphologically_intact:
        p_pw = 0.01
    else:
        p_pw = fuzzy_sigmoid(delta_posterior_mm, threshold=2.0, temperature=0.5)

    # 5. Both endplates / symmetric collapse (A4 vs A3/A1)
    p_collapse = fuzzy_sigmoid(relative_collapse_ratio, threshold=0.78, temperature=0.06, invert=True)

    both_endplates_fractured = metrics.get('both_endplates_fractured')

    if both_endplates_fractured is not None:
        p_ep2 = max(p_collapse, 0.85) if bool(both_endplates_fractured) else min(p_collapse, 0.25)
    elif upper_deformity is not None and lower_deformity is not None:
        p_up = fuzzy_sigmoid(float(upper_deformity), threshold=2.0, temperature=0.5)
        p_lo = fuzzy_sigmoid(float(lower_deformity), threshold=2.0, temperature=0.5)
        p_ep_geom = float(np.sqrt(p_up * p_lo))
        if relative_collapse_ratio < 0.65 and ha_hp >= 0.75:
            # Severe symmetric body collapse (pancake crush): both endplates crushed together
            p_ep2 = max(p_ep_geom, p_collapse)
        elif upper_deformity >= 1.8 and lower_deformity < 1.5:
            # Asymmetric single-endplate burst (A3)
            p_ep2 = min(p_ep_geom, 0.25)
        elif lower_deformity >= 1.8 and upper_deformity < 1.5:
            # Asymmetric single-endplate burst (A3)
            p_ep2 = min(p_ep_geom, 0.25)
        else:
            p_ep2 = p_ep_geom
    else:
        # Height symmetry analysis: distinguish A4 symmetric burst from A3 asymmetric wedge burst
        hp_rel = float(metrics.get('hp_relative_ratio', metrics.get('posterior_height_collapse_ratio', 1.0)))

        # In severe A3, an isolated single endplate crush can drop relative_collapse_ratio < 0.78,
        # but ha_hp_ratio is severely wedged (< 0.65) while posterior height Hp remains largely intact (hp_rel > 0.88).
        if relative_collapse_ratio < 0.78 and ha_hp < 0.65 and hp_rel > 0.88:
            # Asymmetric single-endplate wedge burst (A3): reduce p_ep2 to prevent false A4
            p_ep2 = float(np.clip(p_collapse * 0.30, 0.01, 0.35))
        elif relative_collapse_ratio < 0.78:
            # Symmetric or bilateral endplate burst (A4)
            p_ep2 = max(p_collapse, 0.78)
        else:
            p_ep2 = p_collapse

    # 6. Coronal split indicator (Type A2)
    p_split_input = metrics.get('p_split')
    if p_split_input is not None:
        p_split = float(np.clip(p_split_input, 0.0, 1.0))
    elif coronal_split_mm > 2.0:
        p_split = fuzzy_sigmoid(coronal_split_mm, threshold=2.5, temperature=0.5)
    else:
        p_split = 0.0

    # 7. Fracture presence: maximum of internal Genant drop, relative collapse, retropulsion, or coronal split
    if is_morphologically_intact:
        p_fracture = 0.01
    else:
        genant_ratio = min(ha_hp, hm_hp)
        p_genant = fuzzy_sigmoid(genant_ratio, threshold=0.82, temperature=0.05, invert=True)
        p_rel_frac = fuzzy_sigmoid(relative_collapse_ratio, threshold=0.84, temperature=0.05, invert=True)
        p_ep_max = fuzzy_sigmoid(max(up_def_val, lo_def_val), threshold=2.0, temperature=0.4)
        p_fracture = float(np.clip(max(p_genant, p_rel_frac, p_pw, p_split, p_ep_max), 0.01, 0.99))

    # Isolated B2 suppression: an intact vertebral body without translation cannot have isolated B2 disruption
    if is_morphologically_intact or (p_fracture < 0.40 and translation_mm < 2.0):
        p_post_tb = min(p_post_tb, 0.20)

    return dict(
        p_trans=round(p_trans, 4),
        p_ant_tb=round(p_ant_tb, 4),
        p_post_tb=round(p_post_tb, 4),
        p_pw=round(p_pw, 4),
        p_ep2=round(p_ep2, 4),
        p_fracture=round(p_fracture, 4),
        p_split=round(p_split, 4),
    )


def evaluate_aospine_decision_tree(
    metrics: dict[str, Any],
    fused_concept_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate AO Spine hierarchical decision tree from anatomical probabilities.

    Applies the international AO Spine priority order (C > B > A > Normal),
    the Combined Morphology rule for B2, Modifier M1/M2 flags, and indeterminate rejection.

    Parameters
    ----------
    metrics : dict[str, Any]
        Physical metrics dictionary for the vertebra.
    fused_concept_probs : dict[str, float] | None
        Optional externally fused probabilities (for Phase 2/3 DL fusion).
        If None, computed automatically via compute_anatomical_rule_probabilities.

    Returns
    -------
    dict[str, Any]
        Evaluation result including predicted_class, confidence, secondary_subtype,
        class_probabilities, modifiers, is_indeterminate, and clinical_rationale_vi.
    """
    if metrics.get('is_fov_truncated') or metrics.get('predicted_class') == 'Ngoài vùng khảo sát (Chạm biên CT)':
        level_name = str(metrics.get('level', 'Đốt sống'))
        clean_raw_metrics = dict(
            delta_posterior_mm=0.0,
            isp_ratio=1.0,
            translation_mm=0.0,
            relative_collapse_ratio=1.0,
            anterior_disc_angle_deg=0.0,
            ha_hp_ratio=1.0,
            hm_hp_ratio=1.0,
        )
        return dict(
            level=level_name,
            predicted_class='Ngoài vùng khảo sát (Chạm biên CT)',
            confidence=1.0,
            secondary_subtype=None,
            class_probabilities={c: (1.0 if c == 'Normal' else 0.0) for c in ('Normal', 'A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'C')},
            concept_probs={k: 0.0 for k in ('p_trans', 'p_ant_tb', 'p_post_tb', 'p_pw', 'p_ep2', 'p_fracture', 'p_split')},
            raw_metrics=clean_raw_metrics,
            modifier_m1=False,
            modifier_m2=False,
            modifiers=[],
            is_indeterminate=True,
            rejection_reasons=['Đốt sống bị cắt cụt do chạm mép biên trường chụp (FOV Truncated)'],
            clinical_rationale_vi=f"{level_name} - Ngoài vùng khảo sát (Chạm biên CT): Đốt sống bị cắt cụt ở biên ảnh CT, không đủ cơ sở đánh giá biến dạng.",
            lesion_severity_score=0.0,
        )

    if metrics.get('level') == 'S1':
        level_name = 'S1'
        clean_raw_metrics = dict(
            delta_posterior_mm=0.0,
            isp_ratio=1.0,
            translation_mm=0.0,
            relative_collapse_ratio=1.0,
            anterior_disc_angle_deg=0.0,
            ha_hp_ratio=1.0,
            hm_hp_ratio=1.0,
        )
        return dict(
            level=level_name,
            predicted_class='S1 (Tham chiếu)',
            confidence=1.0,
            secondary_subtype=None,
            class_probabilities={c: (1.0 if c == 'Normal' else 0.0) for c in ('Normal', 'A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'C')},
            concept_probs={k: 0.0 for k in ('p_trans', 'p_ant_tb', 'p_post_tb', 'p_pw', 'p_ep2', 'p_fracture', 'p_split')},
            raw_metrics=clean_raw_metrics,
            modifier_m1=False,
            modifier_m2=False,
            modifiers=[],
            is_indeterminate=False,
            rejection_reasons=[],
            clinical_rationale_vi="S1 - Xương cùng (Sacrum): Dùng làm mốc giải phẫu tham chiếu cho tầng thắt lưng L5.",
            lesion_severity_score=0.0,
        )

    probs = fused_concept_probs or compute_anatomical_rule_probabilities(metrics)

    s_trans = probs['p_trans']
    s_ant_tb = probs['p_ant_tb']
    s_post_tb = probs['p_post_tb']
    s_pw = probs['p_pw']
    s_ep2 = probs['p_ep2']
    s_frac = probs['p_fracture']
    s_split = probs.get('p_split', 0.0)

    # Priority routing C > B > A > Normal
    s_bc = max(s_trans, s_ant_tb, s_post_tb)

    raw_scores = {
        'C': s_trans,
        'B3': (1.0 - s_trans) * s_ant_tb,
        'B2': (1.0 - s_trans) * (1.0 - s_ant_tb) * s_post_tb if (s_frac >= 0.40 or s_trans >= 0.40) else 0.0,
        'A4': (1.0 - s_bc) * s_frac * s_pw * s_ep2,
        'A3': (1.0 - s_bc) * s_frac * s_pw * (1.0 - s_ep2),
        'A0': (1.0 - s_bc) * (1.0 - s_frac) * 0.05,
        'Normal': (1.0 - s_bc) * (1.0 - s_frac),
    }

    # A1 vs A2 routing: only activate A2 branch when explicit coronal split is present (> 2.0 mm)
    if s_split > 0.0:
        raw_scores['A2'] = (1.0 - s_bc) * s_frac * (1.0 - s_pw) * s_split
        raw_scores['A1'] = (1.0 - s_bc) * s_frac * (1.0 - s_pw) * (1.0 - s_split)
    else:
        raw_scores['A2'] = 0.0
        raw_scores['A1'] = (1.0 - s_bc) * s_frac * (1.0 - s_pw)

    total_raw = sum(raw_scores.values())
    class_probs = {k: float(v / max(total_raw, 1e-6)) for k, v in raw_scores.items()}

    predicted_class = max(class_probs, key=class_probs.get)
    confidence = float(class_probs[predicted_class])

    # Combined Morphology Rule for B2
    # If posterior tension band is disrupted and vertebral body has fracture:
    # AO Spine dictates primary class B2 with secondary A subtype
    secondary_subtype = None
    if s_trans < 0.50 and s_frac >= 0.40 and (s_post_tb >= 0.55 or (predicted_class == 'B2' and s_frac >= 0.50)):
        predicted_class = 'B2'
        confidence = float(max(confidence, class_probs.get('B2', 0.50)))
        if s_pw >= 0.50 and s_ep2 >= 0.50:
            secondary_subtype = 'A4'
        elif s_pw >= 0.50:
            secondary_subtype = 'A3'
        elif s_split >= 0.50:
            secondary_subtype = 'A2'
        elif s_frac >= 0.50:
            secondary_subtype = 'A1'

    # Modifiers
    isp_ratio = float(metrics.get('isp_ratio', 1.0))
    modifier_m1 = bool(1.15 < isp_ratio <= 1.30 and predicted_class not in ('B2', 'C', 'Normal'))
    modifier_m2 = bool(predicted_class == 'B3' or metrics.get('modifier_m2', False))

    modifiers: list[str] = []
    if modifier_m1:
        modifiers.append('M1: Nghi ngờ tổn thương dải căng sau trên CT (ISP ratio nằm trong vùng nghi ngờ 1.15-1.30)')
    if modifier_m2:
        modifiers.append('M2: Cột sống cứng khớp / Bệnh lý DISH đi kèm')

    # Rejection Flag / Indeterminate State
    q_seg = float(metrics.get('q_seg', 1.0))
    if isinstance(metrics.get('spinous_tip'), dict):
        q_seg = float(metrics['spinous_tip'].get('q_seg', q_seg))

    rejection_reasons: list[str] = []
    if q_seg < 0.40:
        rejection_reasons.append(f'Chất lượng phân đoạn mask thấp (q_seg={q_seg:.2f} < 0.40)')
    if confidence < 0.45:
        rejection_reasons.append(f'Độ tin cậy phân loại chưa đủ cao (confidence={confidence:.2f} < 0.45)')

    is_indeterminate = len(rejection_reasons) > 0

    # Clinical Rationale Generation (Tiếng Việt súc tích)
    level_name = str(metrics.get('level', 'Đốt sống'))
    trans_mm = float(metrics.get('translation_mm', 0.0))
    disc_deg = float(metrics.get('anterior_disc_angle_deg', 0.0))
    disc_ratio = float(metrics.get('disc_opening_ratio', 1.0))
    delta_pw = float(metrics.get('delta_posterior_mm', 0.0))
    rel_coll = float(metrics.get('relative_collapse_ratio', 1.0))
    h_info = metrics.get('heights') or {}
    ha_hp = float(metrics.get('ha_hp_ratio', h_info.get('anterior_posterior_ratio', 1.0)))

    if predicted_class == 'C':
        rationale = (f"{level_name} - Gãy trật cột sống loại C: Trượt/di lệch thân đốt {trans_mm:.1f} mm "
                     f"(vượt ngưỡng an toàn 3.5 mm), mất vững hoàn toàn 3 cột.")
    elif predicted_class == 'B3':
        rationale = (f"{level_name} - Tổn thương dải căng trước loại B3 (Hyperextension): "
                     f"Toác góc đĩa đệm trước {disc_deg:.1f}° (ngưỡng 15°) và tỷ số mở rộng đĩa đệm {disc_ratio:.2f} lần.")
    elif predicted_class == 'B2':
        if secondary_subtype:
            rationale = (f"{level_name} - Tổn thương dải căng sau B2 (kèm gãy thân đốt kiểu {secondary_subtype}): "
                         f"Toác khoảng liên gai sau (tỷ lệ {isp_ratio:.2f} lần, vượt ngưỡng 1.30) kèm tổn thương thân đốt.")
        else:
            rationale = (f"{level_name} - Tổn thương dải căng sau B2: Toác khoảng liên gai sau "
                         f"(tỷ lệ {isp_ratio:.2f} lần, vượt ngưỡng 1.30).")
    elif predicted_class == 'A4':
        pct_coll = max(0.0, (1.0 - rel_coll) * 100.0)
        pct_kept = min(100.0, rel_coll * 100.0)
        rationale = (f"{level_name} - Gãy vỡ nhiều mảnh hoàn toàn A4: Tường sau lồi {delta_pw:.1f} mm "
                     f"vào ống sống, sập lún toàn thân (xẹp lún -{pct_coll:.0f}%, bảo tồn {pct_kept:.0f}% chiều cao), tổn thương cả 2 mâm đốt.")
    elif predicted_class == 'A3':
        pct_coll = max(0.0, (1.0 - rel_coll) * 100.0)
        pct_kept = min(100.0, rel_coll * 100.0)
        rationale = (f"{level_name} - Gãy vỡ nhiều mảnh không hoàn toàn A3: Tường sau lồi {delta_pw:.1f} mm "
                     f"vào ống sống, tổn thương 1 mâm đốt (xẹp lún -{pct_coll:.0f}%, bảo tồn {pct_kept:.0f}% chiều cao).")
    elif predicted_class == 'A2':
        coronal_split = float(metrics.get('coronal_split_mm', 0.0))
        rationale = (f"{level_name} - Gãy chẻ/tách A2: Đường nứt chẻ dọc thân đốt "
                     f"({coronal_split:.1f} mm), tường sau nguyên vẹn.")
    elif predicted_class == 'A1':
        pct_ha_hp = ha_hp * 100.0
        rationale = (f"{level_name} - Gãy lún chêm A1: Chiều cao trước giảm (chiều cao trước/sau: {pct_ha_hp:.0f}%), "
                     f"tường sau nguyên vẹn (độ lồi {delta_pw:.1f} mm).")
    elif predicted_class == 'A0':
        rationale = f"{level_name} - Gãy phụ không đáng kể A0: Không ảnh hưởng thân đốt sống."
    else:
        rationale = (f"{level_name} - Đốt sống bình thường: Chiều cao thân đốt (trước/sau: {ha_hp*100:.0f}%), "
                     f"tường sau và dải căng sau đều trong giới hạn sinh lý.")

    if predicted_class == 'Normal':
        delta_pw = 0.0
        isp_ratio = min(isp_ratio, 1.0)
        if trans_mm < 3.5:
            trans_mm = 0.0

    raw_metrics = dict(
        delta_posterior_mm=round(delta_pw, 2),
        isp_ratio=round(isp_ratio, 3),
        translation_mm=round(trans_mm, 2),
        relative_collapse_ratio=round(rel_coll, 3),
        anterior_disc_angle_deg=round(disc_deg, 2),
        ha_hp_ratio=round(ha_hp, 3),
        hm_hp_ratio=round(float(h_info.get('middle_posterior_ratio', 1.0)), 3),
    )

    c_probs = dict(probs)
    c_probs['relative_collapse_ratio'] = round(rel_coll, 3)
    c_probs['isp_ratio'] = round(isp_ratio, 3)
    c_probs['translation_mm'] = round(trans_mm, 2)
    c_probs['delta_posterior_mm'] = round(delta_pw, 2)

    return dict(
        level=level_name,
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        secondary_subtype=secondary_subtype,
        class_probabilities={k: round(v, 4) for k, v in class_probs.items()},
        concept_probs=c_probs,
        raw_metrics=raw_metrics,
        modifier_m1=modifier_m1,
        modifier_m2=modifier_m2,
        modifiers=modifiers,
        is_indeterminate=is_indeterminate,
        rejection_reasons=rejection_reasons,
        clinical_rationale_vi=rationale,
    )


def rank_and_classify_spine(
    vertebrae_metrics: list[dict[str, Any]],
    fused_concept_probs_list: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate and rank a sequence of vertebrae (e.g. T10 to L5) by lesion severity.

    Implements smart intervertebral pairing resolution: when interspinous widening
    or disc gaping occurs between adjacent vertebrae (i, i+1), the lesion is
    attributed to the vertebra exhibiting greater vertebral body comminution/fracture.

    Parameters
    ----------
    vertebrae_metrics : list[dict[str, Any]]
        List of vertebra morphometry dictionaries along the spine.
    fused_concept_probs_list : list[dict[str, float]] | None
        Optional list of fused concept probability dicts matching vertebrae_metrics.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'ranked_vertebrae': list of evaluated vertebrae sorted descending by severity
        - 'patient_summary': high-level diagnosis and surgical triage summary
    """
    if not vertebrae_metrics:
        return dict(ranked_vertebrae=[], patient_summary={})

    n = len(vertebrae_metrics)
    evaluations: list[dict[str, Any]] = []

    for idx, item in enumerate(vertebrae_metrics):
        fused = fused_concept_probs_list[idx] if fused_concept_probs_list else None
        res = evaluate_aospine_decision_tree(item, fused_concept_probs=fused)
        evaluations.append(res)

    # Smart intervertebral pairing resolution for B2/B3/C
    # Avoid duplicate B2/B3 diagnosis on pristine adjacent vertebra
    for i in range(n):
        curr_eval = evaluations[i]
        if curr_eval['predicted_class'] in ('B2', 'B3'):
            curr_frac = curr_eval['concept_probs']['p_fracture']
            curr_pw = curr_eval['concept_probs']['p_pw']

            # Check neighbors
            for neighbor_idx in (i - 1, i + 1):
                if 0 <= neighbor_idx < n:
                    neigh_eval = evaluations[neighbor_idx]
                    neigh_frac = neigh_eval['concept_probs']['p_fracture']
                    neigh_pw = neigh_eval['concept_probs']['p_pw']

                    # If neighbor has pristine body while current vertebra is fractured:
                    if curr_frac >= 0.50 and neigh_frac < 0.25 and neigh_pw < 0.25:
                        if neigh_eval['predicted_class'] in ('B2', 'B3'):
                            # Demote neighbor to Normal with adjacent note
                            neigh_eval['predicted_class'] = 'Normal'
                            neigh_eval['confidence'] = round(float(neigh_eval['class_probabilities'].get('Normal', 0.70)), 4)
                            neigh_eval['clinical_rationale_vi'] = (
                                f"{neigh_eval['level']} - Đốt lành tiếp giáp khe tổn thương dải căng của {curr_eval['level']}."
                            )

    # Calculate lesion severity score
    for idx, ev in enumerate(evaluations):
        cls_name = ev['predicted_class']
        if cls_name in ('Normal', 'A0', 'Ngoài vùng khảo sát (Chạm biên CT)', 'S1 (Tham chiếu)') or ev.get('is_indeterminate'):
            ev['lesion_severity_score'] = 0.0
            continue
        base_w = SEVERITY_WEIGHTS.get(cls_name, 0.05)
        p_abnormal = 1.0 - ev['class_probabilities'].get('Normal', 0.0)
        conf = ev['confidence']

        # Extra weight if bone fragment retropulsed into canal
        pw_bonus = min(0.15, float(vertebrae_metrics[idx].get('delta_posterior_mm', 0.0)) / 30.0)

        score = base_w * p_abnormal * (0.50 + 0.50 * conf) + pw_bonus
        ev['lesion_severity_score'] = round(float(np.clip(score, 0.0, 1.0)), 4)

    # Sort descending by severity score, placing excluded/reference levels at the end
    def _rank_sort_key(ev: dict[str, Any]):
        cls = ev.get('predicted_class', '')
        is_excluded = (
            cls in ('Ngoài vùng khảo sát (Chạm biên CT)', 'S1 (Tham chiếu)', 'Chưa đặt đủ 6 mốc')
            or ev.get('is_fov_truncated', False)
            or ev.get('is_indeterminate', False)
        )
        lvl = ev.get('level', '')
        lvl_idx = LEVELS.index(lvl) if lvl in LEVELS else 999
        return (1 if is_excluded else 0, -ev.get('lesion_severity_score', 0.0), lvl_idx)

    ranked = sorted(evaluations, key=_rank_sort_key)

    top1 = ranked[0]
    top2_levels = [v['level'] for v in ranked[:2]]

    patient_summary = dict(
        primary_lesion_level=top1['level'],
        primary_aospine_class=top1['predicted_class'],
        confidence=top1['confidence'],
        severity_score=top1['lesion_severity_score'],
        secondary_subtype=top1.get('secondary_subtype'),
        top2_levels=top2_levels,
        has_surgical_instability=bool(top1['predicted_class'] in ('C', 'B3', 'B2', 'A4')),
        modifiers=top1['modifiers'],
        is_indeterminate=top1['is_indeterminate'],
        clinical_rationale_vi=top1['clinical_rationale_vi'],
    )

    return dict(
        ranked_vertebrae=ranked,
        patient_summary=patient_summary,
    )

