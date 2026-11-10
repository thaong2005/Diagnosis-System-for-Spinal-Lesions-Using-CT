"""Prepare AO Spine 2.5D Dual-ROI Dataset and 5-Fold Stratified Splits (Phase 2).

Adheres to:
- Production Contract: Absolute Zero Patient Leakage between Train and Val across all folds.
- Concept Mapping: Multi-task binary vector [y_pw, y_ep2, y_tb, y_trans] with validity masks.
- Clean Normal Filtering: Excludes ambiguous adjacent vertebrae from normal pool.
"""
from typing import Any
from pathlib import Path
import argparse
import json
import re
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

LEVELS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
          'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
          'L1', 'L2', 'L3', 'L4', 'L5']

CONCEPT_NAMES = ['posterior_wall', 'both_endplates', 'tension_band', 'translation']


def map_aospine_to_concepts(
    aospine_class: str,
    morph_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map an AO Spine clinical class label to multi-task binary concept targets and supervision masks.

    Concepts:
    - y_pw: Posterior wall compromise / retropulsion into canal (1 = compromised, 0 = intact)
    - y_ep2: Both endplates comminuted / complete collapse (1 = both, 0 = single/intact)
    - y_tb: Tension band disruption (posterior B2 or anterior B3) (1 = disrupted, 0 = intact)
    - y_trans: Vertebral body translation / dislocation (Type C) (1 = displaced, 0 = aligned)

    Masks (1 = supervised, 0 = unobserved / masked out in loss calculation):
    - Normal / A0: y=[0, 0, 0, 0], mask=[1, 1, 1, 1], is_fractured=0
    - A1: y=[0, 0, 0, 0], mask=[1, 1, 1, 1], is_fractured=1
    - A2: y=[0, 0, 0, 0], mask=[1, 1, 1, 1], is_fractured=1
    - A3: y=[1, 0, 0, 0], mask=[1, 1, 1, 1], is_fractured=1
    - A4: y=[1, 1, 0, 0], mask=[1, 1, 1, 1], is_fractured=1
    - B2 / B3: y=[y_pw, 0, 1, 0], mask=[m_pw, 0, 1, 1], is_fractured=1
    - C: y=[0, 0, 1, 1], mask=[0, 0, 1, 1], is_fractured=1

    Parameters
    ----------
    aospine_class : str
        AO Spine class label (e.g. 'A1', 'A3', 'A4', 'B2', 'B3', 'C', 'Normal', 'A0').
    morph_metrics : dict, optional
        Physical metrics (e.g. delta_posterior_mm) to resolve secondary features in B/C types.

    Returns
    -------
    dict
        Target dictionary with 'aospine_class', 'is_fractured', 'y_concepts', 'concept_mask', 'concept_names'.
    """
    cls_upper = str(aospine_class).strip().upper()
    metrics = morph_metrics or {}

    dp = float(metrics.get('delta_posterior_mm', 0.0))

    if cls_upper in ('NORMAL', 'A0', 'NONE'):
        y_concepts = [0, 0, 0, 0]
        concept_mask = [1, 1, 1, 1]
        is_fractured = 0
        canonical_class = 'Normal'
    elif cls_upper == 'A1':
        y_concepts = [0, 0, 0, 0]
        concept_mask = [1, 1, 1, 1]
        is_fractured = 1
        canonical_class = 'A1'
    elif cls_upper == 'A2':
        y_concepts = [0, 0, 0, 0]
        concept_mask = [1, 1, 1, 1]
        is_fractured = 1
        canonical_class = 'A2'
    elif cls_upper == 'A3':
        y_concepts = [1, 0, 0, 0]
        concept_mask = [1, 1, 1, 1]
        is_fractured = 1
        canonical_class = 'A3'
    elif cls_upper == 'A4':
        y_concepts = [1, 1, 0, 0]
        concept_mask = [1, 1, 1, 1]
        is_fractured = 1
        canonical_class = 'A4'
    elif cls_upper in ('B1', 'B2', 'B3'):
        # Tension band disrupted
        y_tb = 1
        m_tb = 1
        y_trans = 0
        m_trans = 1
        # Posterior wall status
        if dp >= 2.0:
            y_pw, m_pw = 1, 1
        elif 'delta_posterior_mm' in metrics:
            y_pw, m_pw = 0, 1
        else:
            y_pw, m_pw = 0, 0  # Unobserved
        # Endplate status masked out
        y_ep2, m_ep2 = 0, 0
        y_concepts = [y_pw, y_ep2, y_tb, y_trans]
        concept_mask = [m_pw, m_ep2, m_tb, m_trans]
        is_fractured = 1
        canonical_class = cls_upper
    elif cls_upper == 'C':
        # Translation and tension band disrupted
        y_trans = 1
        m_trans = 1
        y_tb = 1
        m_tb = 1
        if dp >= 2.0:
            y_pw, m_pw = 1, 1
        else:
            y_pw, m_pw = 0, 0
        y_ep2, m_ep2 = 0, 0
        y_concepts = [y_pw, y_ep2, y_tb, y_trans]
        concept_mask = [m_pw, m_ep2, m_tb, m_trans]
        is_fractured = 1
        canonical_class = 'C'
    else:
        # Unknown/indeterminate
        y_concepts = [0, 0, 0, 0]
        concept_mask = [0, 0, 0, 0]
        is_fractured = 0
        canonical_class = cls_upper

    return {
        'aospine_class': canonical_class,
        'is_fractured': is_fractured,
        'y_concepts': y_concepts,
        'concept_mask': concept_mask,
        'concept_names': list(CONCEPT_NAMES),
    }


def filter_clean_normal_vertebrae(
    vertebra_records: list[dict[str, Any]],
    fractured_levels_by_patient: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Filter candidate normal vertebrae to remove ambiguous borderline levels adjacent to fractures.

    Criteria:
    - If a vertebra is explicitly labeled as fractured, it is retained.
    - If a vertebra is an unlabeled control:
      - If adjacent to a fracture (level distance <= 1) and shows ambiguous deformity
        (min(Ha/Hp, Hm/Hp) < 0.85 or translation_mm >= 2.0 or delta_posterior_mm >= 1.5),
        it is excluded to prevent label noise in the negative class pool.
      - Otherwise, it is kept as verified clean normal.

    Parameters
    ----------
    vertebra_records : list of dict
        List of vertebra dictionaries with 'patient_id', 'level', 'heights', and optional 'aospine_class'.
    fractured_fractures_by_patient : dict, optional
        Mapping of patient_id -> list of fractured vertebral levels (e.g. {'p1': ['L1']}).

    Returns
    -------
    list of dict
        Filtered list of clean vertebra records.
    """
    frac_map = fractured_levels_by_patient or {}

    # Build internal fracture map if not provided
    if not frac_map:
        for r in vertebra_records:
            pid = str(r.get('patient_id', ''))
            cls_name = str(r.get('aospine_class', '')).upper()
            lvl = str(r.get('level', '')).upper()
            if cls_name not in ('', 'NORMAL', 'A0', 'NONE', 'UNKNOWN') and lvl in LEVELS:
                frac_map.setdefault(pid, []).append(lvl)

    filtered = []
    for r in vertebra_records:
        pid = str(r.get('patient_id', ''))
        cls_name = str(r.get('aospine_class', '')).upper()
        lvl = str(r.get('level', '')).upper()

        # Keep explicit fractures
        if cls_name not in ('', 'NORMAL', 'A0', 'NONE', 'UNKNOWN'):
            filtered.append(r)
            continue

        # Check clean normal criteria
        p_fracs = frac_map.get(pid, [])
        is_adjacent_to_frac = False
        if lvl in LEVELS:
            curr_idx = LEVELS.index(lvl)
            for f_lvl in p_fracs:
                if f_lvl in LEVELS and abs(LEVELS.index(f_lvl) - curr_idx) <= 1:
                    is_adjacent_to_frac = True
                    break

        h_info = r.get('heights') or {}
        ha_hp = float(r.get('ha_hp_ratio', h_info.get('anterior_posterior_ratio', 1.0)))
        hm_hp = float(r.get('hm_hp_ratio', h_info.get('middle_posterior_ratio', 1.0)))
        trans = float(r.get('translation_mm', 0.0))
        dp = float(r.get('delta_posterior_mm', 0.0))

        min_ratio = min(ha_hp, hm_hp)

        # Discard ambiguous adjacent levels
        if is_adjacent_to_frac and (min_ratio < 0.85 or trans >= 2.0 or dp >= 1.5):
            continue

        # Discard severely deformed vertebrae even if far from fracture if unlabeled
        if min_ratio < 0.75 or trans >= 3.0 or dp >= 2.5:
            continue

        filtered.append(r)

    return filtered


def create_patient_stratified_splits(
    records: list[dict[str, Any]],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate 5-fold stratified splits ensuring absolute zero patient leakage.

    All samples belonging to a patient are strictly assigned to either Train or Val.

    Parameters
    ----------
    records : list of dict
        List of records, each containing 'patient_id', 'sample_id', and 'aospine_class'.
    n_splits : int
        Number of folds (default 5).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    dict
        Split structure with 5 folds, sample lists, and verification metrics.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    # 1. Group records by patient and identify patient primary class
    patients: dict[str, dict[str, Any]] = {}
    for r in records:
        pid = str(r['patient_id'])
        cls_name = str(r.get('aospine_class', 'Normal'))
        if pid not in patients:
            patients[pid] = {
                'patient_id': pid,
                'sample_ids': [],
                'classes': [],
            }
        sid = r.get('sample_id', f"{pid}_{r.get('level', len(patients[pid]['sample_ids']))}")
        patients[pid]['sample_ids'].append(sid)
        patients[pid]['classes'].append(cls_name)

    # Patient primary classification (highest severity fracture or Normal)
    severity_order = {'C': 6, 'B3': 5, 'B2': 4, 'A4': 3, 'A3': 2, 'A2': 1, 'A1': 1, 'A0': 0, 'Normal': 0}

    patient_list = []
    for pid, pdata in patients.items():
        primary_cls = max(pdata['classes'], key=lambda c: severity_order.get(c, 0))
        patient_list.append({
            'patient_id': pid,
            'primary_class': primary_cls,
            'sample_ids': pdata['sample_ids'],
        })

    rng = np.random.RandomState(seed)

    # 2. Separate frequent classes (>= n_splits) and rare classes (< n_splits)
    class_groups: dict[str, list[dict]] = {}
    for p in patient_list:
        class_groups.setdefault(p['primary_class'], []).append(p)

    folds_patients: list[list[str]] = [[] for _ in range(n_splits)]
    rare_challenge_patients: list[str] = []

    # Sort classes for determinism
    for c in sorted(class_groups.keys()):
        p_group = class_groups[c]
        rng.shuffle(p_group)

        if len(p_group) < 2:
            # Super-rare class (n=1, e.g. A1, B3): held out strictly for zero-shot generalization test
            rare_challenge_patients.extend([p['patient_id'] for p in p_group])
        else:
            # Frequent/replicated class: distribute evenly across folds
            for idx, p in enumerate(p_group):
                folds_patients[idx % n_splits].append(p['patient_id'])

    all_pids = set(patients.keys())

    # 3. Construct 5 folds
    folds_data = []
    for fold_idx in range(n_splits):
        val_pids = set(folds_patients[fold_idx])
        train_pids = (all_pids - val_pids) - set(rare_challenge_patients)

        # Strict validation: 0% patient leakage
        leakage = train_pids & val_pids
        if leakage:
            raise RuntimeError(f"Data leakage detected in fold {fold_idx}: {leakage}")

        train_samples = []
        for pid in train_pids:
            train_samples.extend(patients[pid]['sample_ids'])

        val_samples = []
        for pid in val_pids:
            val_samples.extend(patients[pid]['sample_ids'])

        # Class distributions
        train_cls_counts: dict[str, int] = {}
        for sid in train_samples:
            matching = [r for r in records if r.get('sample_id') == sid or f"{r['patient_id']}_{r.get('level')}" == sid]
            if matching:
                c = matching[0].get('aospine_class', 'Normal')
                train_cls_counts[c] = train_cls_counts.get(c, 0) + 1

        val_cls_counts: dict[str, int] = {}
        for sid in val_samples:
            matching = [r for r in records if r.get('sample_id') == sid or f"{r['patient_id']}_{r.get('level')}" == sid]
            if matching:
                c = matching[0].get('aospine_class', 'Normal')
                val_cls_counts[c] = val_cls_counts.get(c, 0) + 1

        folds_data.append({
            'fold_id': fold_idx,
            'train_patient_ids': sorted(list(train_pids)),
            'val_patient_ids': sorted(list(val_pids)),
            'challenge_val_patient_ids': sorted(list(rare_challenge_patients)),
            'train_sample_ids': sorted(train_samples),
            'val_sample_ids': sorted(val_samples),
            'train_sample_count': len(train_samples),
            'val_sample_count': len(val_samples),
            'train_class_counts': train_cls_counts,
            'val_class_counts': val_cls_counts,
        })

    return {
        'schema_version': 1,
        'n_splits': n_splits,
        'random_seed': seed,
        'total_patients': len(patients),
        'total_samples': len(records),
        'zero_leakage_verified': True,
        'rare_class_challenge_set': sorted(rare_challenge_patients),
        'rare_class_challenge_patients': sorted(rare_challenge_patients),
        'folds': folds_data,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare AO Spine 2.5D Dataset and 5-Fold Stratified Splits.")
    parser.add_argument('--output', type=str, default='data/processed/aospine_splits.json', help="Output splits JSON.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating 5-fold stratified splits with zero patient leakage...")
    gt_file = Path('data/raw/data ngay 27.09.2026.xlsx')
    if gt_file.is_file():
        try:
            from scripts.evaluate_aospine_baseline import load_ground_truth, strip_accents, parse_levels
            gt = load_ground_truth(gt_file)
            unique_cases = [v for k, v in gt.items() if k.startswith('case_')]
            records = []
            for c in unique_cases:
                pid = c.get('patient_id') or strip_accents(c.get('name', '')) or f"case_{c.get('stt', len(records)+1)}"
                parsed = parse_levels(c.get('true_level', 'L1'))
                lvl = parsed[0] if (parsed and parsed[0] in LEVELS) else 'L1'
                cls_name = c.get('true_class', 'Normal')
                records.append({
                    'patient_id': pid,
                    'name': c.get('name', ''),
                    'level': lvl,
                    'aospine_class': cls_name,
                    'sample_id': f"{pid}_{lvl}",
                })
            print(f"Loaded {len(records)} ground-truth records from {gt_file}")
        except Exception as e:
            print(f"Warning: Failed to load {gt_file}: {e}. Falling back to sample records.")
            sample_records = [
                {'patient_id': f'pat_{i:02d}', 'level': 'L1', 'aospine_class': c, 'sample_id': f'pat_{i:02d}_L1'}
                for i, c in enumerate(['A4', 'A4', 'A4', 'A4', 'A3', 'A3', 'A3', 'A3', 'B2', 'B2', 'B2', 'B2', 'C', 'C', 'C', 'A1', 'B3'])
            ]
            records = sample_records
    else:
        sample_records = [
            {'patient_id': f'pat_{i:02d}', 'level': 'L1', 'aospine_class': c, 'sample_id': f'pat_{i:02d}_L1'}
            for i, c in enumerate(['A4', 'A4', 'A4', 'A4', 'A3', 'A3', 'A3', 'A3', 'B2', 'B2', 'B2', 'B2', 'C', 'C', 'C', 'A1', 'B3'])
        ]
        records = sample_records

    splits = create_patient_stratified_splits(records, n_splits=5, seed=args.seed)
    out_file.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Splits exported to {out_file}. Zero leakage: {splits['zero_leakage_verified']}, Rare challenge: {splits['rare_class_challenge_set']}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
