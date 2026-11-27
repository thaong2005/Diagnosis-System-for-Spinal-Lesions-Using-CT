"""Objective evaluation script for AO Spine fracture detection and classification baseline.

Strictly adheres to the Production Contract (Zero Label Leakage):
- Phase 1 (Blind Inference): Evaluates CT scans and vertebra morphometry across T10-L5
  without accessing or reading any ground truth annotations.
- Phase 2 (Ground Truth Matching): Compares predictions against clinical ground truth
  to compute Top-1 & Top-2 Localization Accuracy, Group-level (A vs B vs C) metrics,
  and 6-class detailed metrics (Balanced Accuracy, Macro-F1, Weighted F1, Quadratic
  Weighted Cohen's Kappa, 6x6 Confusion Matrix).
- Exports error analysis and summary reports in Markdown and CSV formats.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# Standard 6-class target order sorted by biomechanical severity
AOSPINE_6CLASSES = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']
AOSPINE_GROUPS = ['A', 'B', 'C']


def strip_accents(text: str) -> str:
    """Normalize Vietnamese diacritics and symbols for robust patient matching."""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', str(text))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return re.sub(r'[^a-z0-9]', '', text.lower())


def normalize_level_name(level_str: str) -> str:
    """Normalize level strings like 'D12' to 'T12', 'Gãy L1' to 'L1', 'L01' to 'L1'."""
    s = str(level_str).replace('\xa0', ' ').strip().upper()
    s = re.sub(r'^G[ÃA]Y\s+', '', s)
    s = re.sub(r'D(\d+)', r'T\1', s)
    s = re.sub(r'([TL])0+(\d+)', r'\1\2', s)
    return s


def parse_levels(level_str: str) -> list[str]:
    """Parse complex vertebra level strings like 'T11-D12', 'T12-L1', 'L3L4', 'GÃY L1'."""
    s = str(level_str).strip().upper()
    s = re.sub(r'D(\d+)', r'T\1', s)
    found = re.findall(r'[TL]\d+', s)
    res = [re.sub(r'([TL])0+(\d+)', r'\1\2', x) for x in found]
    if not res:
        norm = normalize_level_name(level_str)
        if norm:
            res = [norm]
    return res


def normalize_class_name(class_str: str) -> str:
    """Normalize class strings like 'b2 (kèm a4)' to 'B2'."""
    s = str(class_str).strip().upper()
    for c in ('B3', 'B2', 'B1', 'A4', 'A3', 'A2', 'A1', 'A0', 'C', 'NORMAL'):
        if s.startswith(c):
            return c
    return s


def load_ground_truth(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load ground truth annotations from Excel (.xlsx) or CSV format.

    Returns mapping of normalized patient identifier / ID / slug to
    dict containing patient details: patient_id, name, true_level, true_class, raw_row.
    """
    path = Path(path)
    if not path.is_file():
        if path.suffix.lower() == '.xlsx' and path.with_suffix('.csv').is_file():
            path = path.with_suffix('.csv')
        elif path.suffix.lower() == '.csv' and path.with_suffix('.xlsx').is_file():
            path = path.with_suffix('.xlsx')
        else:
            raise FileNotFoundError(f"Ground truth file not found: {path}")

    records: list[dict[str, Any]] = []

    if path.suffix.lower() == '.xlsx':
        with zipfile.ZipFile(path) as z:
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                sst = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for e in sst.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    t_parts = [t.text for t in e.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text]
                    strings.append(''.join(t_parts))
            sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
            rows_xml = list(sheet.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'))
            if not rows_xml:
                return {}

            for row_el in rows_xml[1:]:
                row_dict: dict[str, str] = {}
                for c in row_el.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                    ref = c.get('r', '')
                    col = re.match(r'([A-Z]+)', ref).group(1) if ref else ''
                    t = c.get('t')
                    v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    val = v.text if v is not None else ''
                    if t == 's' and val.isdigit() and int(val) < len(strings):
                        val = strings[int(val)]
                    row_dict[col] = val.strip()

                stt = row_dict.get('A', '')
                name = row_dict.get('B', '')
                year = row_dict.get('C', '')
                level = row_dict.get('D', '')
                pid = row_dict.get('E', '')
                cls_val = row_dict.get('F', '')
                notes = row_dict.get('G', '')

                # Handle shifted rows
                if not level and 'L' in notes:
                    level = notes
                if cls_val in ('A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C', 'NORMAL'):
                    pass
                elif name in ('A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C', 'NORMAL'):
                    # Swapped name and class
                    name, cls_val = cls_val, name

                if level and cls_val:
                    records.append({
                        'stt': stt,
                        'name': name,
                        'year': year,
                        'true_level': normalize_level_name(level),
                        'true_class': normalize_class_name(cls_val),
                        'patient_id': pid,
                        'notes': notes,
                    })

    elif path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                def _clean(val: Any) -> str:
                    return re.sub(r'\s+', ' ', str(val or '').replace('\xa0', ' ')).strip()

                stt = _clean(row.get('STT') or row.get('stt'))
                name = _clean(row.get('Tên') or row.get('Ten') or row.get('name'))
                level = _clean(row.get('Đốt gãy') or row.get('Dot gay') or row.get('level') or row.get('true_level'))
                cls_val = _clean(row.get('phân loại') or row.get('phan loai') or row.get('class') or row.get('true_class'))
                pid = _clean(row.get('ID') or row.get('id') or row.get('patient_id'))
                if level and cls_val:
                    records.append({
                        'stt': stt,
                        'name': name,
                        'true_level': normalize_level_name(level),
                        'true_class': normalize_class_name(cls_val),
                        'patient_id': pid,
                        'raw_row': row,
                    })

    gt_map: dict[str, dict[str, Any]] = {}
    for idx, r in enumerate(records, 1):
        key = r['patient_id'] or f"case_{idx:03d}"
        gt_map[key] = r
        gt_map[f"case_{idx:03d}"] = r
        if r.get('stt'):
            gt_map[f"stt_{r['stt']}"] = r
        if r['name']:
            clean_name = re.sub(r'\s+', '_', r['name'].strip().lower())
            gt_map[clean_name] = r
            slug = strip_accents(r['name'])
            if slug:
                gt_map[slug] = r

    return gt_map


def quadratic_weighted_cohen_kappa(cm: np.ndarray) -> float:
    """Calculate Quadratic Weighted Cohen's Kappa for ordinal multi-class agreement.

    Parameters
    ----------
    cm : np.ndarray
        K x K confusion matrix where rows are true classes and columns are predictions.

    Returns
    -------
    float
        Kappa coefficient in [-1.0, 1.0]. 1.0 indicates perfect agreement.
    """
    k = cm.shape[0]
    if k <= 1 or cm.sum() == 0:
        return 1.0

    n = float(cm.sum())
    i_idx, j_idx = np.indices((k, k))
    weights = ((i_idx - j_idx) ** 2) / ((k - 1) ** 2)

    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / n

    weighted_observed = float(np.sum(weights * cm))
    weighted_expected = float(np.sum(weights * expected))

    if weighted_expected == 0.0:
        return 1.0 if weighted_observed == 0.0 else 0.0

    kappa = 1.0 - (weighted_observed / weighted_expected)
    return float(np.clip(kappa, -1.0, 1.0))


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_order: list[str]
) -> dict[str, Any]:
    """Calculate multi-class accuracy, balanced accuracy, macro-F1, weighted-F1,

    quadratic weighted kappa, and confusion matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have matching length.")

    n = len(y_true)
    k = len(class_order)
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        t_idx = class_to_idx.get(t)
        p_idx = class_to_idx.get(p)
        if t_idx is not None and p_idx is not None:
            cm[t_idx, p_idx] += 1

    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for i, c in enumerate(class_order):
        tp = float(sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c))
        fp = float(sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c))
        fn = float(sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c))
        supp = int(sum(1 for t in y_true if t == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': supp,
        }

        if supp > 0:
            recalls.append(recall)
        f1s.append(f1)
        supports.append(supp)

    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = float(total_correct / max(n, 1))
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    total_supp = sum(supports)
    weighted_f1 = (
        float(sum(f * s for f, s in zip(f1s, supports)) / total_supp)
        if total_supp > 0 else 0.0
    )

    qw_kappa = quadratic_weighted_cohen_kappa(cm)

    return {
        'total_samples': n,
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'quadratic_weighted_kappa': round(qw_kappa, 4),
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
        'classes': class_order,
    }


def compute_localization_metrics(
    true_levels: list[str],
    pred_top1_levels: list[str],
    pred_top2_levels: list[list[str]]
) -> dict[str, float]:
    """Calculate Top-1 and Top-2 vertebra localization accuracy across spine."""
    n = len(true_levels)
    if n == 0:
        return {'top1_accuracy': 0.0, 'top2_accuracy': 0.0, 'total_cases': 0, 'top1_hits': 0, 'top2_hits': 0}

    top1_hits = 0
    top2_hits = 0
    for t_str, p1, p2 in zip(true_levels, pred_top1_levels, pred_top2_levels):
        target_lvls = parse_levels(t_str)
        if p1 in target_lvls:
            top1_hits += 1
        if any(lvl in target_lvls for lvl in p2):
            top2_hits += 1

    return {
        'top1_accuracy': round(float(top1_hits / n), 4),
        'top2_accuracy': round(float(top2_hits / n), 4),
        'total_cases': n,
        'top1_hits': top1_hits,
        'top2_hits': top2_hits,
    }


def to_group_class(cls_name: str) -> str:
    """Map detailed AO Spine sub-class to main group (A, B, C)."""
    cls_upper = normalize_class_name(cls_name)
    if cls_upper.startswith('A'):
        return 'A'
    if cls_upper.startswith('B'):
        return 'B'
    if cls_upper.startswith('C'):
        return 'C'
    return 'Normal'


def discover_patient_cases(data_dir: Path) -> list[Path]:
    """Find patient directories inside data_dir, supporting flat or category-nested layouts."""
    case_dirs: list[Path] = []
    category_names = {'A0', 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C', 'NO FRACTURE', 'NO_FRACTURE', 'NORMAL'}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        sub_dirs = [s for s in d.iterdir() if s.is_dir() and not s.name.startswith('.')]
        is_category = d.name.strip().upper() in category_names
        if is_category and sub_dirs and not (d / 'exam').exists() and not (d / 'csc').exists():
            for s in sorted(sub_dirs):
                case_dirs.append(s)
        else:
            case_dirs.append(d)
    return case_dirs


def measure_retropulsion_from_mask(
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float],
    landmarks: list[list[float]] | dict | None = None,
) -> float:
    """Measure posterior body protrusion into spinal canal relative to posterior corners."""
    if not isinstance(body_mask, np.ndarray) or not body_mask.any():
        return 0.0
    sx, sy = spacing_mm

    rows = np.flatnonzero(body_mask.any(axis=1))
    if len(rows) < 5:
        return 0.0

    r_min, r_max = int(rows[0]), int(rows[-1])
    h_rows = r_max - r_min
    if h_rows < 4:
        return 0.0

    # Upper 25% and lower 25% rows of the vertebral body
    r_lo_end = r_min + max(1, int(round(0.25 * h_rows)))
    r_hi_start = r_max - max(1, int(round(0.25 * h_rows)))

    lo_cols = [int(np.flatnonzero(body_mask[r, :])[-1]) for r in range(r_min, r_lo_end + 1) if body_mask[r, :].any()]
    hi_cols = [int(np.flatnonzero(body_mask[r, :])[-1]) for r in range(r_hi_start, r_max + 1) if body_mask[r, :].any()]

    if not lo_cols or not hi_cols:
        return 0.0

    # True posterior cortical corners in physical coordinates
    c_inf = float(np.median(lo_cols))
    r_inf = float((r_min + r_lo_end) / 2.0)
    p_inf = np.array([c_inf * sx, r_inf * sy])

    c_sup = float(np.median(hi_cols))
    r_sup = float((r_hi_start + r_max) / 2.0)
    p_sup = np.array([c_sup * sx, r_sup * sy])

    dx = p_sup[0] - p_inf[0]
    dy = p_sup[1] - p_inf[1]
    line_len = float(np.hypot(dx, dy))
    if line_len < 1e-4:
        return 0.0

    # Measure protrusion beyond the reference line connecting the two cortical corners
    max_protrusion = 0.0
    for r in range(r_min, r_max + 1):
        cols = np.flatnonzero(body_mask[r, :])
        if len(cols) == 0:
            continue
        c_post = cols[-1]
        x_mm, y_mm = c_post * sx, r * sy
        d = ((x_mm - p_inf[0]) * dy - (y_mm - p_inf[1]) * dx) / line_len
        if d > max_protrusion:
            max_protrusion = d
    return round(float(max_protrusion), 3)


def run_blind_inference(
    data_dir: Path | str,
    output_pred_file: Path | str | None = None,
    limit: int | None = None,
    use_prototypes: bool = False,
    use_dl: bool = False,
    oof_splits_path: Path | str = 'data/processed/aospine_splits.json',
    models_dir: Path | str = 'models',
    manifest_path: Path | str = 'data/processed/aospine_rois/manifest.json',
    base_predictions: Path | str | None = None,
    external_normals_dir: Path | str | None = None,
    external_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Phase 1: Blind Inference on patient CT scans (T10-L5) without ground truth access.

    Respects the Zero Label Leakage and Zero Patient Leakage contract strictly.
    Only computes metrics for vertebrae with true measurements/segmentations.
    Never mixes fake/synthetic dummy landmarks into real measurement chains.
    Supports Out-Of-Fold (OOF) evaluation for both Prototype Bank and DL models.
    """
    import SimpleITK as sitk
    from spinal_ct.localization import LEVELS
    from spinal_ct.body_region import BodyPlane
    from spinal_ct.aospine_rules import rank_and_classify_spine
    from spinal_ct.aospine_fusion import fuse_and_classify_spine
    from spinal_ct.morphometry import (
        compute_intervertebral_metrics,
        measure_posterior_wall_retropulsion,
        find_spinous_process_tip,
        extract_posterior_arch_mask,
        auto_segment_body,
        auto_detect_landmarks,
        measure_heights,
    )

    data_dir = Path(data_dir)
    predictions: list[dict[str, Any]] = []

    case_dirs = discover_patient_cases(data_dir)
    if limit:
        case_dirs = case_dirs[:limit]

    spine_levels = ['T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
    out_root = Path('data/output')
    out_dirs = [d for d in out_root.iterdir() if d.is_dir()] if out_root.is_dir() else []
    proc_root = Path('data/processed')
    proc_dirs = [d for d in proc_root.iterdir() if d.is_dir()] if proc_root.is_dir() else []

    # Lazy cache for DL models and Prototype banks across folds
    dl_models_by_fold: dict[int, Any] = {}
    prototype_banks_by_fold: dict[int, Any] = {}

    def _get_dl_model(fold_id: int):
        if fold_id not in dl_models_by_fold:
            from spinal_ct.aospine_model import DualStreamAOSpineNet
            net = DualStreamAOSpineNet(in_channels=5, feature_dim=128)
            f_path = Path(models_dir) / f"aospine_dualstream_fold{fold_id}.pt"
            if not f_path.is_file():
                f_path = Path(models_dir) / "aospine_dualstream_best.pt"
            net.load_checkpoint(f_path)
            net.eval()
            dl_models_by_fold[fold_id] = net
        return dl_models_by_fold[fold_id]

    def _get_prototype_bank(fold_id: int):
        if fold_id not in prototype_banks_by_fold:
            from spinal_ct.prototype_bank import AOSpinePrototypeBank
            from scripts.prepare_aospine_dataset import filter_clean_normal_vertebrae
            bank = AOSpinePrototypeBank()
            manifest_data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            clean_samples = filter_clean_normal_vertebrae(manifest_data.get('samples', []))
            splits_data = json.loads(Path(oof_splits_path).read_text(encoding='utf-8'))
            f_info = splits_data['folds'][fold_id]
            bank.build_from_fold(
                samples=clean_samples,
                train_sample_ids=f_info['train_sample_ids'],
                train_patient_ids=f_info['train_patient_ids'],
                roi_dir=Path(manifest_path).parent,
                fold_id=fold_id,
            )
            if external_normals_dir and Path(external_normals_dir).exists():
                bank.ingest_external_normals(external_normals_dir, external_weight=external_weight)
            prototype_banks_by_fold[fold_id] = bank
        return prototype_banks_by_fold[fold_id]

    if base_predictions and Path(base_predictions).is_file():
        base_preds = json.loads(Path(base_predictions).read_text(encoding='utf-8'))
        predictions = []
        for bp in base_preds:
            if not bp.get('has_data', False) or not bp.get('ranked_vertebrae'):
                predictions.append(bp)
                continue

            patient_id = bp['patient_id']
            slug = strip_accents(patient_id)
            clean_slug = slug.replace(' ', '').lower()

            target_folds = []
            if (use_prototypes or use_dl) and Path(oof_splits_path).is_file():
                splits_data = json.loads(Path(oof_splits_path).read_text(encoding='utf-8'))
                for f_idx, f_info in enumerate(splits_data.get('folds', [])):
                    val_pids = [strip_accents(p).replace(' ', '').lower() for p in f_info.get('val_patient_ids', [])]
                    if clean_slug in val_pids:
                        target_folds = [f_idx]
                        break
                if not target_folds:
                    target_folds = list(range(len(splits_data.get('folds', []))))

            inter = []
            for v in bp['ranked_vertebrae']:
                raw_m = dict(v.get('raw_metrics', {}))
                raw_m['level'] = v.get('level', '')
                raw_m['heights'] = {
                    'anterior_posterior_ratio': raw_m.get('ha_hp_ratio', 1.0),
                    'middle_posterior_ratio': raw_m.get('hm_hp_ratio', 1.0),
                }
                raw_m['q_seg'] = 0.90
                inter.append(raw_m)

            inter.sort(key=lambda x: LEVELS.index(x['level']) if x['level'] in LEVELS else 999)

            proto_results_list = None
            dl_probs_list = None

            if use_prototypes or use_dl:
                proto_results_list = []
                dl_probs_list = []

                for item in inter:
                    lvl_name = str(item.get('level', '')).upper()
                    b_file = Path(f'data/processed/aospine_rois/{clean_slug}_{lvl_name}_body.npy')
                    a_file = Path(f'data/processed/aospine_rois/{clean_slug}_{lvl_name}_arch.npy')

                    p_res = None
                    dl_p = None

                    if b_file.is_file() and a_file.is_file() and target_folds:
                        body_roi = np.load(b_file).astype(np.float32)
                        arch_roi = np.load(a_file).astype(np.float32)

                        if use_prototypes:
                            from spinal_ct.prototype_bank import extract_roi_prototype_features
                            feats = extract_roi_prototype_features(body_roi, arch_roi, morph_metrics=None)
                            f_preds = []
                            for fid in target_folds:
                                b = _get_prototype_bank(fid)
                                f_preds.append(b.predict(feats, level=lvl_name))

                            avg_p_anom = float(np.mean([p['p_anomaly'] for p in f_preds]))
                            all_c = set()
                            for p in f_preds:
                                all_c.update(p.get('confidence_distribution', {}).keys())
                                all_c.update(p.get('class_probabilities', {}).keys())
                            avg_c_probs = {}
                            for c in all_c:
                                avg_c_probs[c] = float(np.mean([p.get('class_probabilities', {}).get(c, p.get('confidence_distribution', {}).get(c, 0.0)) for p in f_preds]))
                            best_c = max(avg_c_probs.keys(), key=lambda c: avg_c_probs[c]) if avg_c_probs else 'Normal'
                            p_res = {
                                'p_anomaly': round(avg_p_anom, 4),
                                'predicted_class': best_c,
                                'class_probabilities': avg_c_probs,
                                'confidence': round(avg_c_probs.get(best_c, 0.5), 4),
                            }

                        if use_dl:
                            import torch
                            body_t = torch.from_numpy(body_roi).unsqueeze(0)
                            arch_t = torch.from_numpy(arch_roi).unsqueeze(0)
                            f_dl = []
                            for fid in target_folds:
                                net = _get_dl_model(fid)
                                with torch.no_grad():
                                    res = net.predict_concept_probabilities(body_t, arch_t)[0]
                                    f_dl.append(res)
                            avg_dl = {}
                            for k in f_dl[0].keys():
                                avg_dl[k] = float(round(float(np.mean([res[k] for res in f_dl])), 4))
                            dl_p = avg_dl

                    proto_results_list.append(p_res or {})
                    dl_probs_list.append(dl_p or {})

            ranked = fuse_and_classify_spine(
                inter,
                dl_concept_probs_list=dl_probs_list if use_dl else None,
                dl_model_trained=use_dl,
                prototype_results_list=proto_results_list if use_prototypes else None,
            )

            top1 = ranked['ranked_vertebrae'][0]
            top2 = [v['level'] for v in ranked['ranked_vertebrae'][:2]]

            pred_entry = {
                'patient_id': patient_id,
                'has_data': True,
                'top1_level': top1['level'],
                'top1_class': top1['predicted_class'],
                'top1_confidence': top1['confidence'],
                'top2_levels': top2,
                'patient_summary': ranked['patient_summary'],
                'ranked_vertebrae': ranked['ranked_vertebrae'],
            }
            predictions.append(pred_entry)

        if output_pred_file:
            out_f = Path(output_pred_file)
            out_f.parent.mkdir(parents=True, exist_ok=True)
            out_f.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')

        return predictions

    for idx, c_dir in enumerate(case_dirs, 1):
        patient_id = c_dir.name
        slug = strip_accents(patient_id)

        # Resolve Out-Of-Fold target folds for this patient
        target_folds: list[int] = []
        if (use_prototypes or use_dl) and Path(oof_splits_path).is_file():
            splits_data = json.loads(Path(oof_splits_path).read_text(encoding='utf-8'))
            clean_slug = slug.replace(' ', '').lower()
            for f_idx, f_info in enumerate(splits_data.get('folds', [])):
                val_pids = [strip_accents(p).replace(' ', '').lower() for p in f_info.get('val_patient_ids', [])]
                if clean_slug in val_pids:
                    target_folds = [f_idx]
                    break
            if not target_folds:
                # Rare zero-shot challenge patient or unassigned: average across all 5 folds
                target_folds = list(range(len(splits_data.get('folds', []))))

        candidate_dirs = [c_dir]
        for od in out_dirs:
            if strip_accents(od.name) == slug:
                candidate_dirs.append(od)

        vertebrae_data: list[dict[str, Any]] = []

        # 1. Search for actual session measurements in do_than_dot or level subdirs
        sessions_by_level: dict[str, Path] = {}
        for cd in candidate_dirs:
            # Check lvl/measurements.json
            for lvl in spine_levels:
                mf = cd / lvl / 'measurements.json'
                if mf.is_file():
                    sessions_by_level[lvl] = cd / lvl
            # Check do_than_dot
            dtd = cd / 'do_than_dot'
            if dtd.is_dir():
                for sub in sorted(dtd.iterdir()):
                    if sub.is_dir():
                        d_level = ''
                        meas_f = sub / 'measurements.json'
                        reg_f = sub / 'region.json'
                        if reg_f.is_file():
                            try:
                                d_level = json.loads(reg_f.read_text(encoding='utf-8')).get('level', '')
                            except Exception:
                                pass
                        if not d_level and meas_f.is_file():
                            try:
                                d_level = json.loads(meas_f.read_text(encoding='utf-8')).get('level', '')
                            except Exception:
                                pass
                        lvl_match = re.search(r'([TL]\d+)', sub.name, re.IGNORECASE)
                        folder_lvl = lvl_match.group(1).upper() if lvl_match else ''
                        lvl = normalize_level_name(d_level or folder_lvl)
                        if lvl in spine_levels:
                            sessions_by_level[lvl] = sub

        for lvl, s_dir in sessions_by_level.items():
            ct_path = s_dir / 'normalized_ct.npy'
            mask_path = s_dir / 'normalized_mask.npy'
            reg_path = s_dir / 'region.json'
            meas_path = s_dir / 'measurements.json'

            if not reg_path.is_file() and not meas_path.is_file():
                continue

            reg = json.loads(reg_path.read_text(encoding='utf-8')) if reg_path.is_file() else {}
            meas = json.loads(meas_path.read_text(encoding='utf-8')) if meas_path.is_file() else {}
            plane_info = reg.get('plane') or meas.get('plane') or {}
            sp = tuple(plane_info.get('spacing_xy_mm', (0.37890625, 0.37890625)))
            lm = reg.get('landmarks_xy_mm') or meas.get('landmarks_xy_mm') or []

            # Recompute retropulsion from body_mask or true CT slice
            dp = 0.0
            body_mask_path = s_dir / 'body_mask.npy'
            if body_mask_path.is_file() and lm and len(lm) >= 6:
                try:
                    bm = np.load(body_mask_path)
                    dp = measure_retropulsion_from_mask(bm, sp, lm)
                except Exception:
                    pass
            if not dp and ct_path.is_file() and lm and len(lm) >= 6:
                try:
                    ct_slice = np.load(ct_path)
                    if ct_slice.shape == tuple(plane_info.get('size_xy', (0, 0))[::-1]):
                        ret = measure_posterior_wall_retropulsion(ct_slice, sp, lm)
                        dp = float(ret.get('delta_posterior_mm', 0.0))
                except Exception:
                    pass
            if not dp:
                raw_m = meas.get('aospine', {}).get('raw_metrics', {})
                dp = float(raw_m.get('delta_posterior_mm',
                           meas.get('aospine', {}).get('concept_probs', {}).get('delta_posterior_mm', 0.0)))

            # Recompute spinous tip if mask available
            tip = None
            if mask_path.is_file():
                try:
                    mask = np.load(mask_path)
                    cols = np.flatnonzero(mask.any(axis=0))
                    if len(cols) >= 6:
                        cut_col = int(cols[0] + 0.45 * (cols[-1] - cols[0]))
                        arch = extract_posterior_arch_mask(mask, cut_col)
                        tip = find_spinous_process_tip(arch, sp)
                except Exception:
                    pass

            h = meas.get('heights')
            if not h and lm and len(lm) >= 6:
                ha = float(np.linalg.norm(np.array(lm[0]) - np.array(lm[1])))
                hm = float(np.linalg.norm(np.array(lm[2]) - np.array(lm[3])))
                hp = float(np.linalg.norm(np.array(lm[4]) - np.array(lm[5])))
                h = {
                    'anterior_mm': ha,
                    'middle_mm': hm,
                    'posterior_mm': hp,
                    'anterior_posterior_ratio': float(ha / max(hp, 1e-4)),
                    'middle_posterior_ratio': float(hm / max(hp, 1e-4)),
                }

            plane_obj = None
            if plane_info and 'origin_lps_mm' in plane_info:
                try:
                    plane_obj = BodyPlane(
                        origin_lps_mm=tuple(plane_info['origin_lps_mm']),
                        u_lps=tuple(plane_info['u_lps']),
                        v_lps=tuple(plane_info['v_lps']),
                        spacing_xy_mm=tuple(plane_info['spacing_xy_mm']),
                        size_xy=tuple(plane_info['size_xy']),
                    )
                except Exception:
                    plane_obj = None

            lm_lps = None
            if plane_obj and lm and len(lm) >= 6:
                try:
                    lm_lps = [plane_obj.point_lps(pt) for pt in lm[:6]]
                except Exception:
                    lm_lps = None

            sp_lps = None
            if plane_obj and tip and isinstance(tip, dict) and tip.get('tip_xy_mm'):
                try:
                    sp_lps = plane_obj.point_lps(tip['tip_xy_mm'])
                except Exception:
                    sp_lps = None

            if lm and len(lm) >= 6 and h:
                vertebrae_data.append({
                    'level': lvl,
                    'landmarks': lm,
                    'landmarks_lps': lm_lps,
                    'spinous_tip_lps': sp_lps,
                    'plane': plane_obj,
                    'heights': h,
                    'delta_posterior_mm': dp,
                    'spinous_tip': tip,
                    'q_seg': float(tip.get('q_seg', 1.0)) if tip else 1.0,
                })

        # 2. Check if case has 3D volume and segmentation (like data/processed/case_053 or data/output/<patient>/dinh_vi_cot_song)
        if len(vertebrae_data) < 3:
            # 2a. Check processed directories
            cand_p_dirs = []
            for pd in proc_dirs:
                if (slug in strip_accents(pd.name)
                        or (slug == 'huynhtriet' and pd.name == 'case_053')
                        or patient_id == pd.name):
                    cand_p_dirs.append(pd)

            for pd in cand_p_dirs:
                vol_file = pd / 'ct_hu.npy'
                mask_file = pd / 'localization_mc/vertebrae.nii.gz'
                if vol_file.is_file() and mask_file.is_file():
                    try:
                        ct_vol = np.load(vol_file, mmap_mode='r')
                        mask_img = sitk.ReadImage(str(mask_file))
                        mask_vol = sitk.GetArrayViewFromImage(mask_img)

                        draft_file = pd / 'annotation_sagittal_draft.json'
                        slice_x = 212
                        if draft_file.is_file():
                            d_info = json.loads(draft_file.read_text(encoding='utf-8'))
                            slice_x = d_info.get('slice_x_index', 212)

                        ct_2d = ct_vol[:, :, slice_x].astype(np.float32)
                        mask_2d = mask_vol[:, :, slice_x]
                        spacing_mm = (0.296875, 1.0)

                        class SPlane:
                            def __init__(self, sp, sz):
                                self.spacing_xy_mm = sp
                                self.size_xy = sz

                        splane = SPlane(spacing_mm, (mask_2d.shape[1], mask_2d.shape[0]))

                        temp_v = []
                        for lvl in spine_levels:
                            if lvl in LEVELS:
                                lvl_idx = LEVELS.index(lvl) + 1
                                v_m = (mask_2d == lvl_idx)
                                if v_m.any() and v_m.sum() >= 50:
                                    det = auto_segment_body(ct_2d, np.ones_like(v_m, bool), v_m, spacing_mm, return_details=True)
                                    b_m = det['body_mask']
                                    arch_m = det.get('posterior_arch_mask')
                                    lm_v = auto_detect_landmarks(b_m, splane)
                                    h_v = measure_heights(lm_v, splane)
                                    ret_v = measure_posterior_wall_retropulsion(
                                        ct_2d, spacing_mm, lm_v,
                                        exclude_mask=arch_m,
                                        search_depth_mm=6.0,
                                        max_gap_mm=2.5,
                                    )
                                    dp_mask = measure_retropulsion_from_mask(b_m, spacing_mm, lm_v)
                                    dp_ct = float(ret_v.get('delta_posterior_mm', 0.0))
                                    dp_v = min(dp_ct, dp_mask) if dp_mask > 0 else 0.0

                                    tip_v = None
                                    if arch_m is not None and arch_m.any():
                                        tip_v = find_spinous_process_tip(arch_m, spacing_mm)
                                    if tip_v is None or tip_v.get('tip_xy_mm') is None or float(tip_v.get('q_seg', 0.0)) < 0.50:
                                        # Fallback to 3D mask: most posterior-inferior voxel
                                        v_3d = (mask_vol == lvl_idx)
                                        if v_3d.any():
                                            z_pts, y_pts, _ = np.where(v_3d)
                                            max_y = int(np.max(y_pts))
                                            min_z = int(np.min(z_pts[y_pts >= max_y - 2]))
                                            tip_v = {
                                                'tip_xy_mm': [float(max_y * spacing_mm[0]), float(min_z * spacing_mm[1])],
                                                'q_seg': 0.80,
                                            }

                                    # Compute true 3D LPS coordinates for landmarks and spinous tip
                                    lm_lps = []
                                    for pt in lm_v[:6]:
                                        c_pt = int(np.clip(round(pt[0] / spacing_mm[0]), 0, mask_2d.shape[1] - 1))
                                        r_pt = int(np.clip(round(pt[1] / spacing_mm[1]), 0, mask_2d.shape[0] - 1))
                                        lm_lps.append(list(mask_img.TransformIndexToPhysicalPoint((int(slice_x), c_pt, r_pt))))

                                    sp_lps = None
                                    if tip_v and tip_v.get('tip_xy_mm'):
                                        c_tip = int(np.clip(round(tip_v['tip_xy_mm'][0] / spacing_mm[0]), 0, mask_2d.shape[1] - 1))
                                        r_tip = int(np.clip(round(tip_v['tip_xy_mm'][1] / spacing_mm[1]), 0, mask_2d.shape[0] - 1))
                                        sp_lps = list(mask_img.TransformIndexToPhysicalPoint((int(slice_x), c_tip, r_tip)))

                                    temp_v.append({
                                        'level': lvl,
                                        'landmarks': lm_v,
                                        'landmarks_lps': lm_lps,
                                        'spinous_tip_lps': sp_lps,
                                        'heights': h_v,
                                        'delta_posterior_mm': float(dp_v),
                                        'spinous_tip': tip_v,
                                        'q_seg': float(tip_v.get('q_seg', 1.0)) if tip_v else 1.0,
                                    })
                        if len(temp_v) >= len(vertebrae_data):
                            vertebrae_data = temp_v
                    except Exception:
                        pass

            # 2b. Check output directories with dinh_vi_cot_song segmentation and raw DICOM
            if len(vertebrae_data) < 3:
                for cd in candidate_dirs:
                    mask_files = list(cd.glob('dinh_vi_cot_song/**/vertebrae.nii.gz'))
                    if not mask_files:
                        continue

                    # Geometry-Matched Selection: prioritize newest and largest mask files
                    mask_files.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)

                    # Gather all valid DICOM series in c_dir
                    reader = sitk.ImageSeriesReader()
                    dcm_candidates: list[list[str]] = []
                    for sub in c_dir.rglob('*'):
                        if sub.is_dir():
                            try:
                                files = reader.GetGDCMSeriesFileNames(str(sub))
                                if files:
                                    dcm_candidates.append(files)
                            except Exception:
                                pass
                    dcm_candidates.sort(key=lambda s: len(s), reverse=True)

                    matched_mask_img = None
                    matched_mask_arr = None
                    matched_dcm = None

                    for mf in mask_files:
                        try:
                            m_img = sitk.ReadImage(str(mf))
                            m_arr = sitk.GetArrayFromImage(m_img)
                            sz = m_arr.shape[0]

                            for dcm_series in dcm_candidates:
                                if len(dcm_series) == sz:
                                    matched_mask_img = m_img
                                    matched_mask_arr = m_arr
                                    matched_dcm = dcm_series
                                    break
                            if matched_mask_img is not None:
                                break
                        except Exception:
                            continue

                    if matched_mask_img is not None and matched_dcm is not None:
                        try:
                            mask_img = matched_mask_img
                            mask_arr = matched_mask_arr
                            reader.SetFileNames(matched_dcm)
                            ct_img = reader.Execute()
                            ct_arr = sitk.GetArrayFromImage(ct_img)

                            sp_xyz = mask_img.GetSpacing()
                            spacing_2d = (sp_xyz[1], sp_xyz[2])

                            tl_indices = [LEVELS.index(lvl) + 1 for lvl in spine_levels if lvl in LEVELS]
                            tl_voxels = np.isin(mask_arr, tl_indices)
                            if tl_voxels.any():
                                _, _, x_pts = np.where(tl_voxels)
                                mid_x = int(np.median(x_pts))
                            else:
                                mid_x = mask_arr.shape[2] // 2

                            ct_2d = ct_arr[:, :, mid_x].astype(np.float32)
                            mask_2d = mask_arr[:, :, mid_x]

                            class SPlaneDcm:
                                def __init__(self, sp, sz):
                                    self.spacing_xy_mm = sp
                                    self.size_xy = sz

                            splane_dcm = SPlaneDcm(spacing_2d, (mask_2d.shape[1], mask_2d.shape[0]))
                            temp_v = []
                            for lvl in spine_levels:
                                lvl_idx = LEVELS.index(lvl) + 1
                                v_m = (mask_2d == lvl_idx)
                                if v_m.any() and v_m.sum() >= 30:
                                    det = auto_segment_body(ct_2d, np.ones_like(v_m, bool), v_m, spacing_2d, return_details=True)
                                    b_m = det['body_mask']
                                    arch_m = det.get('posterior_arch_mask')
                                    lm_v = auto_detect_landmarks(b_m, splane_dcm)
                                    h_v = measure_heights(lm_v, splane_dcm)
                                    ret_v = measure_posterior_wall_retropulsion(
                                        ct_2d, spacing_2d, lm_v,
                                        exclude_mask=arch_m,
                                        search_depth_mm=6.0,
                                        max_gap_mm=2.5,
                                    )
                                    dp_mask = measure_retropulsion_from_mask(b_m, spacing_2d, lm_v)
                                    dp_ct = float(ret_v.get('delta_posterior_mm', 0.0))
                                    dp_v = min(dp_ct, dp_mask) if dp_mask > 0 else 0.0

                                    tip_v = None
                                    if arch_m is not None and arch_m.any():
                                        tip_v = find_spinous_process_tip(arch_m, spacing_2d)
                                    if tip_v is None or tip_v.get('tip_xy_mm') is None or float(tip_v.get('q_seg', 0.0)) < 0.50:
                                        # Fallback to 3D mask: most posterior-inferior voxel
                                        v_3d = (mask_arr == lvl_idx)
                                        if v_3d.any():
                                            z_pts, y_pts, _ = np.where(v_3d)
                                            max_y = int(np.max(y_pts))
                                            min_z = int(np.min(z_pts[y_pts >= max_y - 2]))
                                            tip_v = {
                                                'tip_xy_mm': [float(max_y * spacing_2d[0]), float(min_z * spacing_2d[1])],
                                                'q_seg': 0.80,
                                            }

                                    # Compute true 3D LPS coordinates for landmarks and spinous tip
                                    lm_lps = []
                                    for pt in lm_v[:6]:
                                        c_pt = int(np.clip(round(pt[0] / spacing_2d[0]), 0, mask_2d.shape[1] - 1))
                                        r_pt = int(np.clip(round(pt[1] / spacing_2d[1]), 0, mask_2d.shape[0] - 1))
                                        lm_lps.append(list(mask_img.TransformIndexToPhysicalPoint((int(mid_x), c_pt, r_pt))))

                                    sp_lps = None
                                    if tip_v and tip_v.get('tip_xy_mm'):
                                        c_tip = int(np.clip(round(tip_v['tip_xy_mm'][0] / spacing_2d[0]), 0, mask_2d.shape[1] - 1))
                                        r_tip = int(np.clip(round(tip_v['tip_xy_mm'][1] / spacing_2d[1]), 0, mask_2d.shape[0] - 1))
                                        sp_lps = list(mask_img.TransformIndexToPhysicalPoint((int(mid_x), c_tip, r_tip)))

                                    temp_v.append({
                                        'level': lvl,
                                        'landmarks': lm_v,
                                        'landmarks_lps': lm_lps,
                                        'spinous_tip_lps': sp_lps,
                                        'heights': h_v,
                                        'delta_posterior_mm': float(dp_v),
                                        'spinous_tip': tip_v,
                                        'q_seg': float(tip_v.get('q_seg', 1.0)) if tip_v else 1.0,
                                    })
                            if len(temp_v) >= len(vertebrae_data):
                                vertebrae_data = temp_v
                        except Exception:
                            pass

        # 3. Evaluate vertebrae if real data exists; DO NOT mix dummy vertebrae!
        if vertebrae_data:
            # Sort in anatomical cranial-to-caudal order
            vertebrae_data.sort(key=lambda x: LEVELS.index(x['level']) if x['level'] in LEVELS else 999)
            inter = compute_intervertebral_metrics(vertebrae_data)

            proto_results_list = None
            dl_probs_list = None

            if use_prototypes or use_dl:
                proto_results_list = []
                dl_probs_list = []
                clean_slug = slug.replace(' ', '').lower()

                for item in inter:
                    lvl_name = str(item.get('level', '')).upper()
                    b_file = Path(f'data/processed/aospine_rois/{clean_slug}_{lvl_name}_body.npy')
                    a_file = Path(f'data/processed/aospine_rois/{clean_slug}_{lvl_name}_arch.npy')

                    p_res = None
                    dl_p = None

                    if b_file.is_file() and a_file.is_file() and target_folds:
                        body_roi = np.load(b_file).astype(np.float32)
                        arch_roi = np.load(a_file).astype(np.float32)

                        if use_prototypes:
                            from spinal_ct.prototype_bank import extract_roi_prototype_features
                            feats = extract_roi_prototype_features(body_roi, arch_roi, morph_metrics=None)
                            f_preds = []
                            for fid in target_folds:
                                b = _get_prototype_bank(fid)
                                f_preds.append(b.predict(feats, level=lvl_name))

                            avg_p_anom = float(np.mean([p['p_anomaly'] for p in f_preds]))
                            all_c = set()
                            for p in f_preds:
                                all_c.update(p.get('confidence_distribution', {}).keys())
                                all_c.update(p.get('class_probabilities', {}).keys())
                            avg_c_probs = {}
                            for c in all_c:
                                avg_c_probs[c] = float(np.mean([p.get('class_probabilities', {}).get(c, p.get('confidence_distribution', {}).get(c, 0.0)) for p in f_preds]))
                            best_c = max(avg_c_probs.keys(), key=lambda c: avg_c_probs[c]) if avg_c_probs else 'Normal'
                            p_res = {
                                'p_anomaly': round(avg_p_anom, 4),
                                'predicted_class': best_c,
                                'class_probabilities': avg_c_probs,
                                'confidence': round(avg_c_probs.get(best_c, 0.5), 4),
                            }

                        if use_dl:
                            import torch
                            body_t = torch.from_numpy(body_roi).unsqueeze(0)
                            arch_t = torch.from_numpy(arch_roi).unsqueeze(0)
                            f_dl = []
                            for fid in target_folds:
                                net = _get_dl_model(fid)
                                with torch.no_grad():
                                    res = net.predict_concept_probabilities(body_t, arch_t)[0]
                                    f_dl.append(res)
                            avg_dl = {}
                            for k in f_dl[0].keys():
                                avg_dl[k] = float(round(float(np.mean([res[k] for res in f_dl])), 4))
                            dl_p = avg_dl

                    proto_results_list.append(p_res or {})
                    dl_probs_list.append(dl_p or {})

            ranked = fuse_and_classify_spine(
                inter,
                dl_concept_probs_list=dl_probs_list if use_dl else None,
                dl_model_trained=use_dl,
                prototype_results_list=proto_results_list if use_prototypes else None,
            )

            top1 = ranked['ranked_vertebrae'][0]
            top2 = [v['level'] for v in ranked['ranked_vertebrae'][:2]]

            pred_entry = {
                'patient_id': patient_id,
                'has_data': True,
                'top1_level': top1['level'],
                'top1_class': top1['predicted_class'],
                'top1_confidence': top1['confidence'],
                'top2_levels': top2,
                'patient_summary': ranked['patient_summary'],
                'ranked_vertebrae': ranked['ranked_vertebrae'],
            }
        else:
            # Case has NO segmentation or measurement data available yet (e.g. unsegmented raw CT)
            pred_entry = {
                'patient_id': patient_id,
                'has_data': False,
                'top1_level': 'None',
                'top1_class': 'Indeterminate',
                'top1_confidence': 0.0,
                'top2_levels': [],
                'patient_summary': {
                    'status': 'unsegmented',
                    'clinical_rationale_vi': 'Chưa có dữ liệu phân đoạn đốt sống hoặc phiên đo mốc thân đốt.',
                },
                'ranked_vertebrae': [],
            }

        predictions.append(pred_entry)

    if output_pred_file:
        out_f = Path(output_pred_file)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')

    return predictions


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truth_path: Path | str,
    output_dir: Path | str
) -> dict[str, Any]:
    """Phase 2: Match predictions with Ground Truth and generate evaluation reports."""
    gt_map = load_ground_truth(ground_truth_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_pairs: list[dict[str, Any]] = []

    for pred in predictions:
        pid = pred.get('patient_id', '')
        clean_pid = re.sub(r'\s+', '_', pid.strip().lower())
        slug_pid = strip_accents(pid)

        gt_item = (
            gt_map.get(pid)
            or gt_map.get(clean_pid)
            or (gt_map.get(slug_pid) if slug_pid else None)
        )

        # Fuzzy match by STT or patient name
        if not gt_item and slug_pid:
            for k, v in gt_map.items():
                k_slug = strip_accents(v.get('name', ''))
                if k_slug and (slug_pid == k_slug or slug_pid in k_slug or k_slug in slug_pid):
                    gt_item = v
                    break

        if gt_item:
            matched_pairs.append({
                'patient_id': pid,
                'has_data': pred.get('has_data', True),
                'true_level': gt_item['true_level'],
                'true_class': gt_item['true_class'],
                'pred_top1_level': pred['top1_level'],
                'pred_top1_class': pred['top1_class'],
                'pred_top1_conf': pred['top1_confidence'],
                'pred_top2_levels': pred.get('top2_levels', [pred['top1_level']]),
                'patient_summary': pred.get('patient_summary', {}),
                'ranked_vertebrae': pred.get('ranked_vertebrae', []),
            })

    if not matched_pairs:
        raise ValueError("No matching cases found between predictions and ground truth.")

    true_levels = [m['true_level'] for m in matched_pairs]
    pred_top1_levels = [m['pred_top1_level'] for m in matched_pairs]
    pred_top2_levels = [m['pred_top2_levels'] for m in matched_pairs]

    true_classes = [m['true_class'] for m in matched_pairs]
    pred_classes = [m['pred_top1_class'] for m in matched_pairs]

    true_groups = [to_group_class(c) for c in true_classes]
    pred_groups = [to_group_class(c) for c in pred_classes]

    # Metrics computation
    loc_metrics = compute_localization_metrics(true_levels, pred_top1_levels, pred_top2_levels)
    group_metrics = compute_classification_metrics(true_groups, pred_groups, AOSPINE_GROUPS)
    class_metrics = compute_classification_metrics(true_classes, pred_classes, AOSPINE_6CLASSES)

    # Compile Error Analysis Table
    error_cases: list[dict[str, Any]] = []
    cases_with_data = sum(1 for m in matched_pairs if m.get('has_data', True) and m['pred_top1_level'] != 'None')
    cases_without_data = len(matched_pairs) - cases_with_data

    for idx, m in enumerate(matched_pairs, 1):
        target_lvls = parse_levels(m['true_level'])
        is_loc_error = m['pred_top1_level'] not in target_lvls
        is_cls_error = m['true_class'] != m['pred_top1_class']
        if is_loc_error or is_cls_error:
            summary = m.get('patient_summary') or {}
            ranked_v = m.get('ranked_vertebrae') or []
            top_v = ranked_v[0] if ranked_v else {}
            raw_m = top_v.get('raw_metrics') or {}

            if not m.get('has_data', True) or m['pred_top1_level'] == 'None':
                err_type = 'Chưa phân đoạn (Chưa có dữ liệu)'
                dp_str = "N/A"
                isp_str = "N/A"
                trans_str = "N/A"
                r_coll_str = "N/A"
            else:
                err_type = ('Định vị + Phân loại' if (is_loc_error and is_cls_error)
                              else ('Sai định vị' if is_loc_error else 'Sai phân loại'))
                dp_str = f"{float(raw_m.get('delta_posterior_mm', 0.0)):.1f}"
                isp_str = f"{float(raw_m.get('isp_ratio', 1.0)):.2f}"
                trans_str = f"{float(raw_m.get('translation_mm', 0.0)):.1f}"
                r_coll_str = f"{float(raw_m.get('relative_collapse_ratio', 1.0)):.2f}"

            error_cases.append({
                'STT': idx,
                'Patient_ID': m['patient_id'],
                'True_Level': m['true_level'],
                'True_Class': m['true_class'],
                'Pred_Top1_Level': m['pred_top1_level'],
                'Pred_Top1_Class': m['pred_top1_class'],
                'Confidence': f"{m['pred_top1_conf']:.1%}",
                'Pred_Top2_Levels': ', '.join(m['pred_top2_levels']) if m['pred_top2_levels'] else 'None',
                'Error_Type': err_type,
                'Delta_PW_mm': dp_str,
                'ISP_ratio': isp_str,
                'Translation_mm': trans_str,
                'R_collapse': r_coll_str,
                'Modifiers': '; '.join(summary.get('modifiers', [])),
                'Rationale': summary.get('clinical_rationale_vi', ''),
            })

    # Export Error Analysis CSV
    if error_cases:
        with (out_dir / 'error_analysis.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(error_cases[0].keys()))
            writer.writeheader()
            writer.writerows(error_cases)

    # Export Error Analysis Markdown
    err_md_rows = []
    for err in error_cases:
        err_md_rows.append(
            f"| {err['STT']} | {err['Patient_ID']} | **{err['True_Level']}** | **{err['True_Class']}** | "
            f"{err['Pred_Top1_Level']} | {err['Pred_Top1_Class']} | {err['Confidence']} | {err['Error_Type']} | "
            f"{err['Delta_PW_mm']} | {err['ISP_ratio']} | {err['R_collapse']} |"
        )

    err_md = (
        "# Bảng Phân Tích Các Ca Sai Lệch (Error Analysis Table)\n\n"
        f"**Tổng số ca lỗi:** {len(error_cases)} / {len(matched_pairs)} ({len(error_cases)/len(matched_pairs):.1%})\n\n"
        "| STT | Mã Bệnh Nhân | True Level | True Class | Pred Top-1 Level | Pred Top-1 Class | Tin cậy | Loại Sai Sót | Δ_post (mm) | ISP | R_collapse |\n"
        "|:---:|---|:---:|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|\n"
        + '\n'.join(err_md_rows) + '\n'
    )
    (out_dir / 'error_analysis.md').write_text(err_md, encoding='utf-8')

    # Export Summary Markdown
    cm_6x6 = class_metrics['confusion_matrix']
    cm_rows = []
    for idx, row in enumerate(cm_6x6):
        cls_t = AOSPINE_6CLASSES[idx]
        cells = ' | '.join(str(val) for val in row)
        cm_rows.append(f"| **{cls_t}** | {cells} |")

    summary_md = f"""# Báo Cáo Đánh Giá Khách Quan Hệ Thống AO Spine Baseline (T10 - L5)

**Thời gian đánh giá:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}  
**Tổng số ca bệnh nhân đối chiếu:** {len(matched_pairs)} ca  
**Production Contract:** Tuân thủ 100% Zero Label Leakage (Pha suy luận độc lập tuyệt đối với ground truth).

---

## 1. Khả năng Định vị Đốt Tổn thương (Vertebra Localization)
- **Top-1 Localization Accuracy:** **{loc_metrics['top1_accuracy']:.1%}** ({loc_metrics['top1_hits']}/{loc_metrics['total_cases']})
- **Top-2 Localization Accuracy:** **{loc_metrics['top2_accuracy']:.1%}** ({loc_metrics['top2_hits']}/{loc_metrics['total_cases']})

---

## 2. Phân loại Cấp độ Nhóm (Group-level: Type A vs Type B vs Type C)
- **Group Accuracy:** **{group_metrics['accuracy']:.1%}**
- **Group Macro-F1:** **{group_metrics['macro_f1']:.4f}**
- **Group Weighted-F1:** **{group_metrics['weighted_f1']:.4f}**

---

## 3. Phân loại 6 Lớp Chi tiết (A1, A3, A4, B2, B3, C)
- **Overall Accuracy:** **{class_metrics['accuracy']:.1%}**
- **Balanced Accuracy:** **{class_metrics['balanced_accuracy']:.1%}**
- **Macro-F1 Score:** **{class_metrics['macro_f1']:.4f}**
- **Weighted F1 Score:** **{class_metrics['weighted_f1']:.4f}**
- **Quadratic Weighted Cohen's Kappa (κ_w):** **{class_metrics['quadratic_weighted_kappa']:.4f}**

### Ma trận nhầm lẫn 6×6 (Confusion Matrix)
| True \\ Pred | {' | '.join(AOSPINE_6CLASSES)} |
|:---:|{'|'.join([':---:'] * len(AOSPINE_6CLASSES))}|
{chr(10).join(cm_rows)}

---

## 4. Chi tiết Từng Lớp Tổn thương
| Mã Lớp | Số ca thực tế (Support) | Precision | Recall (Độ nhạy) | F1-Score |
|:---:|:---:|:---:|:---:|:---:|
"""
    for c in AOSPINE_6CLASSES:
        p_c = class_metrics['per_class_metrics'].get(c, {})
        summary_md += f"| **{c}** | {p_c.get('support', 0)} | {p_c.get('precision', 0.0):.1%} | {p_c.get('recall', 0.0):.1%} | {p_c.get('f1_score', 0.0):.4f} |\n"

    summary_md += f"""
---

## 5. Hiện trạng Phân đoạn & Dữ liệu Đối chiếu (Data Availability)
- **Số ca có dữ liệu mốc/phân đoạn thực tế:** {cases_with_data} / {len(matched_pairs)} ({cases_with_data/len(matched_pairs):.1%})
- **Số ca CT DICOM thô chưa có phân đoạn:** {cases_without_data} / {len(matched_pairs)} ({cases_without_data/len(matched_pairs):.1%})
- **Đặc tả kiểm chứng Zero Label Leakage:** Hệ thống xử lý hoàn toàn mù thông tin ground truth; tuyệt đối không gán mốc giả mạo cho các đốt chưa phân đoạn. Với các ca chưa có mốc, hệ thống đánh dấu trung thực `Chưa phân đoạn (Indeterminate)`.
"""

    (out_dir / 'summary_report.md').write_text(summary_md, encoding='utf-8')

    results = {
        'total_cases': len(matched_pairs),
        'cases_with_data': cases_with_data,
        'cases_without_data': cases_without_data,
        'localization': loc_metrics,
        'group_metrics': group_metrics,
        'class_metrics': class_metrics,
        'error_cases_count': len(error_cases),
    }

    (out_dir / 'evaluation_metrics.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AO Spine detection and classification baseline.")
    parser.add_argument('--data-dir', type=str, default='data/raw', help="CT and segmentation scans directory.")
    parser.add_argument('--ground-truth', type=str, default='data/raw/data ngay 27.09.2026.xlsx', help="Ground truth Excel/CSV.")
    parser.add_argument('--output-dir', type=str, default='data/output/aospine_evaluation', help="Results output folder.")
    parser.add_argument('--predictions', type=str, default=None, help="Existing predictions JSON (skips Phase 1).")
    parser.add_argument('--save-predictions', type=str, default=None, help="Path to save Phase 1 predictions.")
    parser.add_argument('--sample-limit', type=int, default=None, help="Limit number of cases for quick verification.")
    parser.add_argument('--use-prototypes', action='store_true', help="Enable 2-step prototype bank.")
    parser.add_argument('--use-dl', action='store_true', help="Enable DualStreamAOSpineNet visual concepts.")
    parser.add_argument('--oof-splits', type=str, default='data/processed/aospine_splits.json', help="OOF splits file.")
    parser.add_argument('--models-dir', type=str, default='models', help="Models directory.")
    parser.add_argument('--prototype-manifest', type=str, default='data/processed/aospine_rois/manifest.json', help="Manifest file.")
    parser.add_argument('--base-predictions', type=str, default=None, help="Base predictions JSON to avoid re-extracting geometry.")
    parser.add_argument('--external-normals-dir', type=str, default=None, help="Directory containing external normal ROIs/manifest.")
    parser.add_argument('--external-weight', type=float, default=0.5, help="Weight w for external normal variance pooling.")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_file = Path(args.save_predictions) if args.save_predictions else out_dir / 'predictions.json'

    if args.predictions and Path(args.predictions).is_file():
        print(f"[Phase 1 Skipped] Loading precomputed predictions: {args.predictions}")
        predictions = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    else:
        print(f"[Phase 1: Blind Inference] Processing cases from {args.data_dir} (Zero Label Leakage)...")
        predictions = run_blind_inference(
            args.data_dir,
            output_pred_file=pred_file,
            limit=args.sample_limit,
            use_prototypes=args.use_prototypes,
            use_dl=args.use_dl,
            oof_splits_path=args.oof_splits,
            models_dir=args.models_dir,
            manifest_path=args.prototype_manifest,
            base_predictions=args.base_predictions,
            external_normals_dir=args.external_normals_dir,
            external_weight=args.external_weight,
        )
        print(f"Phase 1 complete: {len(predictions)} cases processed. Predictions saved to {pred_file}")

    print(f"[Phase 2: Ground Truth Matching] Comparing with {args.ground_truth}...")
    try:
        results = evaluate_predictions(predictions, args.ground_truth, out_dir)
        print("\n" + "=" * 60)
        print(f"AO Spine Baseline Evaluation Summary (N = {results['total_cases']})")
        print("=" * 60)
        print(f"Top-1 Localization Accuracy: {results['localization']['top1_accuracy']:.1%}")
        print(f"Top-2 Localization Accuracy: {results['localization']['top2_accuracy']:.1%}")
        print(f"Group Accuracy (A vs B vs C): {results['group_metrics']['accuracy']:.1%}")
        print(f"Group Macro-F1:              {results['group_metrics']['macro_f1']:.4f}")
        print(f"6-Class Accuracy:             {results['class_metrics']['accuracy']:.1%}")
        print(f"6-Class Balanced Accuracy:    {results['class_metrics']['balanced_accuracy']:.1%}")
        print(f"6-Class Macro-F1:             {results['class_metrics']['macro_f1']:.4f}")
        print(f"Quadratic Weighted Kappa:     {results['class_metrics']['quadratic_weighted_kappa']:.4f}")
        print(f"Error Cases Logged:           {results['error_cases_count']}")
        print(f"Reports saved to:             {out_dir}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
