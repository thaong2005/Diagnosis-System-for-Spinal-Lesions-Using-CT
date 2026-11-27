#!/usr/bin/env python3
"""Batch diagnosis pipeline for Spinal CT raw cases using Three-Pillar AI Fusion.

Processes raw patient CT scans across categories (A1, A3, A4, B2, B3, C):
1. For each patient, selects optimal axial CT series (matching mask geometry if available).
2. Performs 3D vertebral morphometry across target levels T10-L5.
3. Executes Three-Pillar AI Fusion:
   - Pillar 1: Morphological rules & reflex arcs (aospine_rules.py).
   - Pillar 2: 2-Step Prototype Bank (w=0.3, prototype_bank.py).
   - Pillar 3: Dual-Stream 2.5D CNN (aospine_model.py).
   - Adaptive Fusion with 8 Biomechanical Guards (aospine_fusion.py).
4. Matches predictions against ground truth CSV (data ngay 27.09.2026.csv).
5. Exports structured predictions and comprehensive comparative metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any

# Add workspace root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import list_ct_series, load_ct, slugify
from spinal_ct.localization import LEVELS, AOSPINE_TARGET_LEVELS
from spinal_ct.body_region import (
    propose_plane,
    sample_plane,
    sample_parasagittal_slices,
    same_geometry,
)
from spinal_ct.morphometry import (
    BodySession,
    SpineSession,
    ct_fingerprint,
    auto_segment_body,
    auto_detect_landmarks,
    measure_heights,
    measure_posterior_wall_retropulsion,
    compute_intervertebral_metrics,
)
from spinal_ct.aospine_fusion import fuse_and_classify_spine
from spinal_ct.aospine_rules import rank_and_classify_spine

AOSPINE_6CLASSES = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']
AOSPINE_GROUPS = ['A', 'B', 'C']


def strip_accents(text: str) -> str:
    """Normalize Vietnamese diacritics and symbols for robust matching."""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', str(text))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return re.sub(r'[^a-z0-9]', '', text.lower())


def normalize_level_name(level_str: str) -> str:
    """Normalize level strings like 'D12' to 'T12', 'L01' to 'L1', 'Gãy L1' to 'L1'."""
    s = str(level_str).replace('\xa0', ' ').strip().upper()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^G[ÃA]Y\s+', '', s)
    s = re.sub(r'^D(\d+)', r'T\1', s)
    s = re.sub(r'([TL])0+(\d+)', r'\1\2', s)
    return s


def parse_levels(level_str: str) -> list[str]:
    """Parse complex vertebra level strings like 'T11-D12', 'T12-L1', 'L3L4', 'Gãy L1'."""
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


def to_group(cls_name: str) -> str:
    cls_upper = normalize_class_name(cls_name)
    if cls_upper.startswith('A'):
        return 'A'
    if cls_upper.startswith('B'):
        return 'B'
    if cls_upper.startswith('C'):
        return 'C'
    return 'Normal'


def select_best_series(p_dir: Path, mask_path: Path | None = None) -> dict[str, Any] | None:
    """Select the best axial CT series, prioritizing exact geometry match with mask."""
    choices = list_ct_series(p_dir)
    if not choices:
        return None

    # Priority 0: Match exact input_fingerprint from prediction.json if available
    if mask_path and mask_path.is_file():
        pred_json = mask_path.parent / 'prediction.json'
        if pred_json.is_file():
            try:
                pj_data = json.loads(pred_json.read_text(encoding='utf-8'))
                expected_fp = pj_data.get('input_fingerprint')
                if expected_fp:
                    for c in choices:
                        if c.get('axial'):
                            try:
                                img = load_ct(p_dir, c['uid'])
                                if ct_fingerprint(img) == expected_fp:
                                    return c
                            except Exception:
                                pass
            except Exception:
                pass

    # Priority 1: Match exact number of slices from existing mask if available
    if mask_path and mask_path.is_file():
        try:
            m_img = sitk.ReadImage(str(mask_path))
            m_slices = m_img.GetSize()[2]
            matching = [c for c in choices if c.get('axial') and c.get('slices') == m_slices]
            if matching:
                # Prefer bone kernel if multiple match slice count
                for k in ['BONE', 'SPINE', 'DETAIL', 'STANDARD']:
                    for c in matching:
                        if k in c.get('kernel', '').upper():
                            return c
                return matching[0]
        except Exception:
            pass

    # Priority 2: Standard selection (valid axial, >= 30 slices, bone kernel)
    valid = [c for c in choices if c.get('axial') and c.get('geometry_error') is None and c.get('slices', 0) >= 30]
    if not valid:
        fallback = [c for c in choices if c.get('axial') and c.get('slices', 0) >= 20]
        if not fallback:
            return None
        valid = fallback

    max_sl = max(c['slices'] for c in valid)
    top = [c for c in valid if c['slices'] == max_sl]

    for k in ['BONE', 'SPINE', 'DETAIL', 'STANDARD']:
        for c in top:
            if k in c.get('kernel', '').upper():
                return c

    return top[0]


def measure_retropulsion_from_mask(
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float],
    landmarks: list[list[float]] | dict,
) -> float:
    """Measure posterior body protrusion into spinal canal relative to posterior corners."""
    sx, sy = spacing_mm
    if isinstance(landmarks, dict):
        ps = landmarks.get('posterior_superior')
        pi = landmarks.get('posterior_inferior')
    elif isinstance(landmarks, (list, tuple)) and len(landmarks) >= 6:
        ps, pi = landmarks[4], landmarks[5]
    else:
        return 0.0
    if ps is None or pi is None:
        return 0.0

    dx = ps[0] - pi[0]
    dy = ps[1] - pi[1]
    line_len = float(np.hypot(dx, dy))
    if line_len < 1e-4:
        return 0.0
    r_min = max(0, int(np.floor(min(pi[1], ps[1]) / sy)))
    r_max = min(body_mask.shape[0] - 1, int(np.ceil(max(pi[1], ps[1]) / sy)))
    if r_max <= r_min:
        return 0.0

    max_protrusion = 0.0
    for r in range(r_min, r_max + 1):
        cols = np.flatnonzero(body_mask[r, :])
        if len(cols) == 0:
            continue
        c_post = cols[-1]
        x_mm, y_mm = c_post * sx, r * sy
        d = ((x_mm - pi[0]) * dy - (y_mm - pi[1]) * dx) / line_len
        if d > max_protrusion:
            max_protrusion = d
    return round(float(max_protrusion), 3)


def run_morphometry_and_classification(
    mask_path: Path,
    ct_image: sitk.Image,
    patient_id: str = 'patient',
    spine_levels: tuple[str, ...] = AOSPINE_TARGET_LEVELS,
    *,
    inference_engine: Any = None,
    prototype_bank: Any = None,
) -> dict[str, Any]:
    """Extract vertebral morphometry and execute Production Three-Pillar AI Fusion.

    Uses 3D SpineSession + BodySession with parasagittal slices (-4mm, 0mm, +4mm)
    feeding Pillar 1 (Rules), Pillar 2 (Prototype Bank w=0.3), Pillar 3 (2.5D CNN),
    and Biomechanical Guards. When `inference_engine` / `prototype_bank` are passed
    explicitly (e.g. 5-Fold OOF evaluation), global singletons and cached fold
    predictions are bypassed strictly.
    """
    mask_img = sitk.ReadImage(str(mask_path))
    fp = ct_fingerprint(ct_image)

    # Approach A: True 3D SpineSession when geometries align
    if same_geometry(ct_image, mask_img):
        spine = SpineSession(patient_id=patient_id)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.SetComputePerimeter(False)
        stats.Execute(mask_img)

        for lvl in spine_levels:
            if lvl not in LEVELS:
                continue
            lbl_id = LEVELS.index(lvl) + 1
            if not stats.HasLabel(lbl_id) or stats.GetNumberOfPixels(lbl_id) < 30:
                continue
            try:
                plane = propose_plane(ct_image, mask_img, lbl_id)
                ct_slice, valid_s, hint = sample_plane(ct_image, plane, mask_img, lbl_id)
                para = sample_parasagittal_slices(ct_image, plane)
                sess = BodySession(
                    fp, lvl, 'Baseline', plane, ct_slice, valid_s, hint,
                    age_group='adult', parasagittal_slices=para, source_ct=ct_image,
                )
                sess.auto_populate(approve=False, recompute_classification=False)
                spine.add_session(sess)
            except Exception:
                continue

        if spine.sessions:
            diag_res = spine.recompute_aospine_classification(
                inference_engine=inference_engine,
                prototype_bank=prototype_bank,
            )
            if diag_res.get('ranked_vertebrae'):
                return diag_res

    # Approach B: Fallback via mid-sagittal 2D extraction with 3-Pillar Fusion
    mask_arr = sitk.GetArrayFromImage(mask_img)
    ct_arr = sitk.GetArrayFromImage(ct_image)
    sp_xyz = mask_img.GetSpacing()
    spacing_2d = (sp_xyz[1], sp_xyz[2])

    tl_indices = [LEVELS.index(lvl) + 1 for lvl in spine_levels if lvl in LEVELS]
    tl_voxels = np.isin(mask_arr, tl_indices)
    mid_x = int(np.median(np.where(tl_voxels)[2])) if tl_voxels.any() else mask_arr.shape[2] // 2

    ct_2d = ct_arr[:, :, mid_x].astype(np.float32)
    mask_2d = mask_arr[:, :, mid_x]

    class SPlane:
        def __init__(self, sp, sz):
            self.spacing_xy_mm = sp
            self.size_xy = sz

    splane = SPlane(spacing_2d, (mask_2d.shape[1], mask_2d.shape[0]))
    vertebrae_data = []

    for lvl in spine_levels:
        if lvl not in LEVELS:
            continue
        lvl_idx = LEVELS.index(lvl) + 1
        v_m = (mask_2d == lvl_idx)
        if v_m.any() and v_m.sum() >= 30:
            det = auto_segment_body(ct_2d, np.ones_like(v_m, bool), v_m, spacing_2d, return_details=True)
            b_m = det['body_mask']
            arch_m = det.get('posterior_arch_mask')
            lm_v = auto_detect_landmarks(b_m, splane)
            h_v = measure_heights(lm_v, splane)
            ret_v = measure_posterior_wall_retropulsion(
                ct_2d, spacing_2d, lm_v,
                exclude_mask=arch_m,
                search_depth_mm=6.0,
                max_gap_mm=2.5,
            )
            dp_mask = measure_retropulsion_from_mask(b_m, spacing_2d, lm_v)
            dp_ct = float(ret_v.get('delta_posterior_mm', 0.0))
            dp_v = min(dp_ct, dp_mask) if dp_mask > 0 else 0.0
            vertebrae_data.append({
                'level': lvl,
                'landmarks': lm_v,
                'heights': h_v,
                'delta_posterior_mm': float(dp_v),
            })

    if not vertebrae_data:
        raise ValueError("Không trích xuất được đủ lát cắt thân đốt sống từ mặt cắt midsagittal.")

    vertebrae_data.sort(key=lambda x: LEVELS.index(x['level']) if x['level'] in LEVELS else 999)
    inter = compute_intervertebral_metrics(vertebrae_data)
    ranked = fuse_and_classify_spine(inter)
    return ranked


def load_ground_truth_map(gt_path: Path) -> dict[str, dict[str, Any]]:
    """Load ground truth mapping from data ngay 27.09.2026.csv."""
    gt_map: dict[str, dict[str, Any]] = {}
    if not gt_path.is_file():
        return gt_map

    with gt_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            def _clean(v: Any) -> str:
                return re.sub(r'\s+', ' ', str(v or '').replace('\xa0', ' ')).strip()

            stt = _clean(r.get('STT', ''))
            name = _clean(r.get('Tên', ''))
            raw_lvl = _clean(r.get('Đốt gãy', ''))
            raw_cls = _clean(r.get('phân loại', ''))
            pid = _clean(r.get('ID', ''))
            notes = _clean(r.get('kèm theo', ''))

            slug = strip_accents(name)
            clean_slug = slug.replace(' ', '').lower()
            entry = {
                'stt': stt,
                'name': name,
                'patient_id': pid,
                'true_level': normalize_level_name(raw_lvl),
                'true_levels': parse_levels(raw_lvl),
                'true_class': normalize_class_name(raw_cls),
                'true_group': to_group(raw_cls),
                'notes': notes,
                'raw_row': r,
            }
            if clean_slug:
                gt_map[clean_slug] = entry
            if pid:
                gt_map[pid] = entry

    return gt_map


def resolve_oof_fold_for_patient(
    clean_slug: str,
    splits_data: dict[str, Any],
) -> tuple[int | None, bool, bool]:
    """Resolve zero-leakage OOF fold index for a normalized patient identifier.

    Returns (fold_id, is_matched_in_splits, is_rare_challenge).
    - Validation patient in fold k -> (k, True, False)
    - Held-out rare challenge patient (A1/B3, absent from every fold train set) -> (0, True, True)
    - Patient outside splits -> (None, False, False)
    """
    norm = strip_accents(clean_slug)
    folds = splits_data.get('folds', [])
    for f in folds:
        val_keys = {strip_accents(p) for p in f.get('val_patient_ids', [])}
        if norm in val_keys:
            return int(f['fold_id']), True, False

    rare_keys = {
        strip_accents(p)
        for p in (
            splits_data.get('rare_class_challenge_set')
            or splits_data.get('rare_class_challenge_patients')
            or []
        )
    }
    if norm in rare_keys and folds:
        return int(folds[0]['fold_id']), True, True

    return None, False, False


def quadratic_weighted_cohen_kappa(cm: np.ndarray) -> float:
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


def compute_evaluation_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_order: list[str],
) -> dict[str, Any]:
    n = len(y_true)
    k = len(class_order)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        t_idx = class_to_idx.get(t)
        p_idx = class_to_idx.get(p)
        if t_idx is not None and p_idx is not None:
            cm[t_idx, p_idx] += 1

    recalls, f1s, supports = [], [], []
    per_class = {}
    for i, c in enumerate(class_order):
        tp = float(sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c))
        fp = float(sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c))
        fn = float(sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c))
        supp = int(sum(1 for t in y_true if t == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4), 'support': supp}
        if supp > 0:
            recalls.append(rec)
        f1s.append(f1)
        supports.append(supp)

    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1))
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    tot_supp = sum(supports)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / tot_supp) if tot_supp > 0 else 0.0
    qwk = quadratic_weighted_cohen_kappa(cm)

    return {
        'total': n,
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'quadratic_weighted_kappa': round(qwk, 4),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'classes': class_order,
    }


def compute_binary_triage_metrics(y_true_pos: list[bool], y_pred_pos: list[bool]) -> dict[str, Any]:
    """Compute binary triage classification metrics (sensitivity, specificity, accuracy, F1)."""
    n = len(y_true_pos)
    tp = sum(1 for t, p in zip(y_true_pos, y_pred_pos) if t and p)
    fp = sum(1 for t, p in zip(y_true_pos, y_pred_pos) if (not t) and p)
    tn = sum(1 for t, p in zip(y_true_pos, y_pred_pos) if (not t) and (not p))
    fn = sum(1 for t, p in zip(y_true_pos, y_pred_pos) if t and (not p))
    acc = (tp + tn) / max(n, 1)
    sens = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    spec = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        'total': n,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': round(float(acc), 4),
        'sensitivity': round(float(sens), 4),
        'specificity': round(float(spec), 4),
        'f1': round(float(f1), 4),
    }


def _load_oof_engine_and_bank(
    fold_id: int,
    models_dir: Path,
    banks_dir: Path,
    splits_data: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load isolated fold-specific BatchedCPUInferenceEngine and AOSpinePrototypeBank."""
    from spinal_ct.aospine_model import BatchedCPUInferenceEngine, DualStreamAOSpineNet
    from spinal_ct.prototype_bank import AOSpinePrototypeBank

    pt_path = Path(models_dir) / f'aospine_dualstream_fold{fold_id}.pt'
    if not pt_path.is_file():
        raise FileNotFoundError(f"OOF checkpoint missing for fold {fold_id}: {pt_path}")

    net = DualStreamAOSpineNet(in_channels=5, feature_dim=128)
    net.load_checkpoint(pt_path)
    net.eval()
    engine = BatchedCPUInferenceEngine(net)

    bank_path = Path(banks_dir) / f'bank_fold_{fold_id}.json'
    if bank_path.is_file():
        bank = AOSpinePrototypeBank().load(bank_path)
    elif splits_data is not None and manifest_path is not None and Path(manifest_path).is_file():
        from scripts.prepare_aospine_dataset import filter_clean_normal_vertebrae

        manifest_data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        clean_samples = filter_clean_normal_vertebrae(manifest_data.get('samples', []))
        fold_info = splits_data['folds'][fold_id]
        bank = AOSpinePrototypeBank()
        bank.build_from_fold(
            samples=clean_samples,
            train_sample_ids=fold_info['train_sample_ids'],
            train_patient_ids=fold_info['train_patient_ids'],
            roi_dir=Path(manifest_path).parent,
            fold_id=fold_id,
        )
    else:
        raise FileNotFoundError(f"OOF prototype bank missing for fold {fold_id}: {bank_path}")

    return engine, bank


def run_batch_diagnosis(
    *,
    fast: bool = True,
    limit: int | None = None,
    force: bool = False,
    oof: bool = False,
    splits_path: Path | str = ROOT / 'data/processed/aospine_splits.json',
    models_dir: Path | str = ROOT / 'models',
    banks_dir: Path | str = ROOT / 'data/processed/prototype_banks',
    manifest_path: Path | str = ROOT / 'data/processed/aospine_rois/manifest.json',
    output_path: Path | str | None = None,
    raw_root: Path | str = ROOT / 'data/raw',
    output_root: Path | str = ROOT / 'data/output',
) -> dict[str, Any]:
    """Run batch diagnosis in Production mode or strict 5-Fold Out-Of-Fold (OOF) mode."""
    import spinal_ct.morphometry as morph_mod

    gt_path = Path(raw_root) / 'data ngay 27.09.2026.csv'
    gt_map = load_ground_truth_map(gt_path)

    splits_data: dict[str, Any] | None = None
    oof_engines_by_fold: dict[int, Any] = {}
    oof_banks_by_fold: dict[int, Any] = {}

    if oof:
        sp_file = Path(splits_path)
        if not sp_file.is_file():
            raise FileNotFoundError(f"OOF splits file not found: {sp_file}")
        splits_data = json.loads(sp_file.read_text(encoding='utf-8'))
        # Strict singleton isolation: clear any lingering globals prior to OOF evaluation
        morph_mod._INFERENCE_ENGINE = None
        morph_mod._PROTOTYPE_BANK = None

    raw_categories = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']
    patients: list[tuple[str, Path]] = []
    skipped_unlabeled_oof: list[str] = []
    for cat in raw_categories:
        cat_dir = Path(raw_root) / cat
        if not cat_dir.is_dir():
            continue
        for p_dir in sorted(cat_dir.iterdir()):
            if not p_dir.is_dir():
                continue
            if oof:
                clean_key = strip_accents(p_dir.name).replace(' ', '').lower()
                if clean_key not in gt_map:
                    skipped_unlabeled_oof.append(f"{cat}/{p_dir.name}")
                    continue
            patients.append((cat, p_dir))

    if limit:
        patients = patients[:limit]

    mode_label = '5-FOLD OOF' if oof else 'PRODUCTION 3-PILLAR'
    print(f"=== BẮT ĐẦU CHẨN ĐOÁN {mode_label} TRÊN {len(patients)} BỆNH NHÂN ===")
    print(f"Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Chế độ Force Re-diagnosis: {force} | OOF: {oof}\n")
    if skipped_unlabeled_oof:
        print(f"OOF Zero-Leakage Guard: Đã loại {len(skipped_unlabeled_oof)} thư mục ngoài CSV: {skipped_unlabeled_oof}\n")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    backend = ROOT / '.venv-model/Scripts/TotalSegmentator.exe'

    for idx, (cat, p_dir) in enumerate(patients, 1):
        p_name = p_dir.name
        slug = slugify(p_name)
        clean_slug = strip_accents(p_name).replace(' ', '').lower()
        print(f"[{idx}/{len(patients)}] Xử lý ca [{cat}] '{p_name}' (slug: {slug})...")

        out_patient_dir = Path(output_root) / slug
        out_dvcs = out_patient_dir / 'dinh_vi_cot_song'
        out_dvcs.mkdir(parents=True, exist_ok=True)

        # Keep OOF artifacts separated from production diagnosis_result.json
        diag_filename = 'diagnosis_result_oof.json' if oof else 'diagnosis_result.json'
        diag_file = out_patient_dir / diag_filename
        if not force and diag_file.is_file():
            try:
                cached_res = json.loads(diag_file.read_text(encoding='utf-8'))
                if cached_res.get('has_3pillar_fusion') and (not oof or cached_res.get('oof_fold_id') is not None):
                    print(f"  -> BỎ QUA: Đã có kết quả trong {diag_file.name}.")
                    skipped.append({'category': cat, 'patient_name': p_name, 'slug': slug, 'diagnosis': cached_res})
                    results.append({'category': cat, 'patient_name': p_name, 'slug': slug, 'status': 'SKIPPED_3PILLAR_EXISTS', 'diagnosis': cached_res})
                    continue
            except Exception:
                pass

        # Identify existing segmentation mask: prioritize direct vertebrae.nii.gz over consensus
        v_masks = sorted(out_dvcs.glob('**/vertebrae.nii.gz'), key=lambda p: p.stat().st_mtime, reverse=True)
        c_masks = sorted(out_dvcs.glob('**/consensus.nii.gz'), key=lambda p: p.stat().st_mtime, reverse=True)
        best_mask = v_masks[0] if v_masks else (c_masks[0] if c_masks else None)

        selected_series = None
        ct_img = None

        try:
            # Resolve fold-isolated models when OOF is active
            fold_id: int | None = None
            is_rare_challenge = False
            fold_engine = None
            fold_bank = None
            if oof and splits_data is not None:
                fold_id, in_splits, is_rare_challenge = resolve_oof_fold_for_patient(clean_slug, splits_data)
                if not in_splits or fold_id is None:
                    raise RuntimeError(f"Zero-Leakage Guard: '{clean_slug}' không nằm trong {splits_path}.")
                if fold_id not in oof_engines_by_fold:
                    oof_engines_by_fold[fold_id], oof_banks_by_fold[fold_id] = _load_oof_engine_and_bank(
                        fold_id,
                        models_dir=Path(models_dir),
                        banks_dir=Path(banks_dir),
                        splits_data=splits_data,
                        manifest_path=Path(manifest_path),
                    )
                fold_engine = oof_engines_by_fold[fold_id]
                fold_bank = oof_banks_by_fold[fold_id]

                # Verify mathematically that test patient was NOT trained in this fold
                trained_keys = {strip_accents(p).replace(' ', '').lower() for p in fold_bank.trained_patient_ids}
                if clean_slug in trained_keys:
                    raise RuntimeError(
                        f"Patient Leakage Violation: '{clean_slug}' found in trained_patient_ids of Fold {fold_id}!"
                    )

            # 1. Select series matching mask geometry
            selected_series = select_best_series(p_dir, mask_path=best_mask)
            if not selected_series:
                raise ValueError("Không tìm thấy chuỗi CT cắt lớp trục (axial) hợp lệ.")

            series_uid = selected_series['uid']
            print(f"  -> Chọn chuỗi CT: UID={series_uid[:20]}... ({selected_series['slices']} lát, kernel={selected_series.get('kernel', '')})")

            # 2. Load CT image
            ct_img = load_ct(p_dir, series_uid)

            # 3. Localize if mask not already present
            if not best_mask:
                if not backend.is_file():
                    raise FileNotFoundError(f"Không tìm thấy backend phân đoạn: {backend}")
                run_tag = f"run_{'fast_' if fast else ''}{time.strftime('%Y%m%d_%H%M%S')}"
                run_out_dir = out_dvcs / run_tag
                print(f"  -> Đang chạy phân đoạn TotalSegmentator ({'fast' if fast else 'detailed'} mode)...")

                cmd = [
                    str(ROOT / '.venv/Scripts/python.exe'),
                    '-m', 'tools.localize_vertebrae',
                    '--backend', str(backend),
                    '--input', str(p_dir),
                    '--series-uid', series_uid,
                    '--output', str(run_out_dir),
                    '--device', 'cpu',
                ]
                if fast:
                    cmd.append('--fast')

                t0 = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                elapsed = time.time() - t0

                if res.returncode != 0:
                    err_msg = res.stderr.strip() or res.stdout.strip()
                    raise RuntimeError(f"TotalSegmentator thất bại (exit {res.returncode}): {err_msg[:300]}")

                print(f"  -> Phân đoạn thành công trong {elapsed:.1f}s.")
                best_mask = run_out_dir / 'vertebrae.nii.gz'
                if not best_mask.is_file():
                    cons = run_out_dir / 'monte_carlo/consensus.nii.gz'
                    best_mask = cons if cons.is_file() else None
                    if not best_mask:
                        raise FileNotFoundError("Không tìm thấy tệp mask sau phân đoạn.")

            # 4. Run Three-Pillar AI Fusion
            fold_note = f" [OOF Fold {fold_id}{' Rare-Challenge' if is_rare_challenge else ''}]" if oof else ""
            print(f"  -> Thực thi Hợp nhất 3 Trụ cột AI{fold_note} từ mask {best_mask.name}...")
            t_diag_start = time.time()
            diag_res = run_morphometry_and_classification(
                best_mask,
                ct_img,
                patient_id=slug,
                inference_engine=fold_engine,
                prototype_bank=fold_bank,
            )
            elapsed_diag = time.time() - t_diag_start

            if oof and (morph_mod._INFERENCE_ENGINE is not None or morph_mod._PROTOTYPE_BANK is not None):
                raise RuntimeError("Singleton Isolation Violation: _INFERENCE_ENGINE/_PROTOTYPE_BANK populated during OOF!")

            top1 = diag_res['ranked_vertebrae'][0]
            top2_lvls = [v['level'] for v in diag_res['ranked_vertebrae'][:2]]
            summary = diag_res['patient_summary']

            # Match Ground Truth strictly from CSV (no hardcoded patient labels)
            gt_entry = gt_map.get(clean_slug)
            if not gt_entry and not oof:
                for k, v in gt_map.items():
                    if k in clean_slug or clean_slug in k:
                        gt_entry = v
                        break

            loc_top1_match = False
            loc_top2_match = False
            cls_match = False
            grp_match = False

            if gt_entry:
                t_lvls = gt_entry['true_levels']
                t_cls = gt_entry['true_class']
                t_grp = gt_entry['true_group']
                loc_top1_match = (top1['level'] in t_lvls)
                loc_top2_match = any(lvl in t_lvls for lvl in top2_lvls)
                cls_match = (top1['predicted_class'] == t_cls)
                grp_match = (to_group(top1['predicted_class']) == t_grp)

            print(f"  -> KẾT QUẢ AI: {top1['level']} · {top1['predicted_class']} (Độ tin cậy: {top1['confidence']:.1%}, thời gian: {elapsed_diag:.2f}s)")
            if gt_entry:
                loc_mark = '✅' if loc_top1_match else ('⚠️ (Top-2)' if loc_top2_match else '❌')
                cls_mark = '✅' if cls_match else ('⚠️ (Nhóm)' if grp_match else '❌')
                print(f"  -> GROUND TRUTH: {gt_entry['true_level']} · {gt_entry['true_class']} | Vị trí: {loc_mark} | Phân loại: {cls_mark}")

            # Check triggered guards
            fd = top1.get('fusion_diagnostics', {})
            guards = fd.get('triggered_guards', [])
            if guards:
                print(f"  -> Chốt bảo vệ kích hoạt: {guards}")

            # Save full diagnosis result
            save_payload = {
                'patient_id': p_name,
                'slug': slug,
                'category': cat,
                'series_uid': series_uid,
                'slices': selected_series['slices'],
                'has_data': True,
                'has_3pillar_fusion': True,
                'evaluation_mode': '5fold_oof' if oof else 'production',
                'oof_fold_id': fold_id,
                'oof_rare_challenge': is_rare_challenge,
                'top1_level': top1['level'],
                'top1_class': top1['predicted_class'],
                'top1_confidence': top1['confidence'],
                'top2_levels': top2_lvls,
                'patient_summary': summary,
                'ranked_vertebrae': diag_res['ranked_vertebrae'],
                'ground_truth': gt_entry,
                'evaluation': {
                    'loc_top1_match': loc_top1_match,
                    'loc_top2_match': loc_top2_match,
                    'cls_match': cls_match,
                    'grp_match': grp_match,
                },
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            diag_file.write_text(json.dumps(save_payload, ensure_ascii=False, indent=2), encoding='utf-8')

            results.append({
                'category': cat,
                'patient_name': p_name,
                'slug': slug,
                'status': 'SUCCESS',
                'diagnosis': save_payload,
            })

        except Exception as exc:
            err_trace = traceback.format_exc()
            print(f"  -> THẤT BẠI: {exc}")
            failure_entry = {
                'category': cat,
                'patient_name': p_name,
                'slug': slug,
                'error_type': type(exc).__name__,
                'error_message': str(exc),
                'traceback': err_trace,
                'series_info': selected_series,
            }
            failures.append(failure_entry)
            results.append({
                'category': cat,
                'patient_name': p_name,
                'slug': slug,
                'status': 'FAILED',
                'error': str(exc),
            })

        print("-" * 65)

    # Compute Final Evaluation Metrics strictly on labeled CSV cases; track unlabeled cases separately
    evaluable = [r['diagnosis'] for r in results if r.get('diagnosis') and r['diagnosis'].get('ground_truth')]
    unlabeled_cases = [
        {
            'category': r['category'],
            'patient_name': r['patient_name'],
            'slug': r['slug'],
            'top1_level': r['diagnosis']['top1_level'],
            'top1_class': r['diagnosis']['top1_class'],
            'top1_confidence': r['diagnosis']['top1_confidence'],
            'top2_levels': r['diagnosis']['top2_levels'],
        }
        for r in results
        if r.get('diagnosis') and not r['diagnosis'].get('ground_truth')
    ]
    n_eval = len(evaluable)
    loc_top1_hits = sum(1 for e in evaluable if e['evaluation']['loc_top1_match'])
    loc_top2_hits = sum(1 for e in evaluable if e['evaluation']['loc_top2_match'])

    y_true_cls = [e['ground_truth']['true_class'] for e in evaluable]
    y_pred_cls = [e['top1_class'] for e in evaluable]

    y_true_grp = [e['ground_truth']['true_group'] for e in evaluable]
    y_pred_grp = [to_group(e['top1_class']) for e in evaluable]

    metrics_6cls = compute_evaluation_metrics(y_true_cls, y_pred_cls, AOSPINE_6CLASSES)
    metrics_grp = compute_evaluation_metrics(y_true_grp, y_pred_grp, AOSPINE_GROUPS)

    instability_classes = ('C', 'B3', 'B2', 'A4')
    surgical_instability_triage = {
        'instability_c_b3_b2_a4': compute_binary_triage_metrics(
            [c in instability_classes for c in y_true_cls],
            [normalize_class_name(p) in instability_classes for p in y_pred_cls],
        ),
        'non_a1_intervention': compute_binary_triage_metrics(
            [c != 'A1' for c in y_true_cls],
            [normalize_class_name(p) != 'A1' for p in y_pred_cls],
        ),
    }

    singleton_isolated = bool(
        oof and morph_mod._INFERENCE_ENGINE is None and morph_mod._PROTOTYPE_BANK is None
    )

    if output_path is not None:
        summary_file = Path(output_path)
    else:
        out_eval_dir = Path(output_root) / 'aospine_batch_run'
        default_name = 'batch_run_results_3pillar_oof.json' if oof else 'batch_run_results_3pillar.json'
        summary_file = out_eval_dir / default_name

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        'evaluation_mode': '5fold_oof' if oof else 'production',
        'singleton_isolated': singleton_isolated,
        'skipped_unlabeled_oof': skipped_unlabeled_oof,
        'total_cases': len(patients),
        'evaluated_cases': n_eval,
        'unlabeled_cases_count': len(unlabeled_cases),
        'unlabeled_cases': unlabeled_cases,
        'skipped_count': len(skipped),
        'success_count': len([r for r in results if r['status'] == 'SUCCESS']),
        'failure_count': len(failures),
        'localization': {
            'top1_hits': loc_top1_hits,
            'top1_accuracy': round(loc_top1_hits / max(n_eval, 1), 4),
            'top2_hits': loc_top2_hits,
            'top2_accuracy': round(loc_top2_hits / max(n_eval, 1), 4),
        },
        'classification_6class': metrics_6cls,
        'classification_group': metrics_grp,
        'surgical_instability_triage': surgical_instability_triage,
        'results': results,
        'failures': failures,
    }
    summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    triage_main = surgical_instability_triage['instability_c_b3_b2_a4']
    print(f"\n=================== TỔNG KẾT ĐÁNH GIÁ LÂM SÀNG ({mode_label}) ===================")
    print(f"Tổng số ca bệnh nhân: {len(patients)} (Có nhãn CSV: {n_eval}, Chưa có nhãn CSV: {len(unlabeled_cases)})")
    if oof:
        print(f"Cách ly Singleton (_INFERENCE_ENGINE/_PROTOTYPE_BANK == None): {singleton_isolated}")
    print(f"Chạy thành công: {summary_payload['success_count']}/{len(patients)}")
    print(f"Thất bại: {len(failures)}")
    print(f"Định vị tổn thương Top-1: {loc_top1_hits}/{n_eval} ({summary_payload['localization']['top1_accuracy']:.1%})")
    print(f"Định vị tổn thương Top-2: {loc_top2_hits}/{n_eval} ({summary_payload['localization']['top2_accuracy']:.1%})")
    print(f"Phân loại 6 Lớp AO Spine (A1-C): Độ chính xác = {metrics_6cls['accuracy']:.1%} | Balanced Acc = {metrics_6cls['balanced_accuracy']:.1%} | Macro-F1 = {metrics_6cls['macro_f1']:.4f} | Kappa = {metrics_6cls['quadratic_weighted_kappa']:.4f}")
    print(f"Phân loại Cấp Nhóm (A vs B vs C): Độ chính xác = {metrics_grp['accuracy']:.1%} | Macro-F1 = {metrics_grp['macro_f1']:.4f}")
    print(f"Sàng lọc Mất vững Phẫu thuật (C/B3/B2/A4): Acc = {triage_main['accuracy']:.1%} | Sens = {triage_main['sensitivity']:.1%} | Spec = {triage_main['specificity']:.1%} | F1 = {triage_main['f1']:.4f}")
    print(f"Dữ liệu tổng hợp lưu tại: {summary_file}")
    print("==================================================================\n")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true', default=True, help='Use fast 3mm model')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--force', action='store_true', default=False, help='Force re-diagnosis even if previous diagnosis exists')
    parser.add_argument('--oof', action='store_true', default=False, help='Strict 5-Fold Out-Of-Fold evaluation on CSV-labeled patients with singleton isolation')
    parser.add_argument('--splits', type=str, default=str(ROOT / 'data/processed/aospine_splits.json'), help='Path to 5-fold splits JSON')
    parser.add_argument('--models-dir', type=str, default=str(ROOT / 'models'), help='Directory containing fold CNN checkpoints')
    parser.add_argument('--banks-dir', type=str, default=str(ROOT / 'data/processed/prototype_banks'), help='Directory containing per-fold prototype bank JSONs')
    parser.add_argument('--manifest', type=str, default=str(ROOT / 'data/processed/aospine_rois/manifest.json'), help='Path to ROI manifest JSON')
    parser.add_argument('--output', type=str, default=None, help='Optional output summary JSON path')
    args = parser.parse_args()

    run_batch_diagnosis(
        fast=args.fast,
        limit=args.limit,
        force=args.force,
        oof=args.oof,
        splits_path=args.splits,
        models_dir=args.models_dir,
        banks_dir=args.banks_dir,
        manifest_path=args.manifest,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
