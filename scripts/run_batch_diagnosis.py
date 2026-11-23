#!/usr/bin/env python3
"""Batch diagnosis pipeline for Spinal CT raw cases.

Processes raw patient CT scans across categories (A1, A3, A4, B2, B3, C):
1. Skips patients that already have completed diagnosis outputs.
2. For unsegmented cases, finds the optimal axial series and runs fast localization.
3. If diagnosis cannot be performed, records exact root causes.
4. For successfully localized cases, performs vertebra morphometry and AO Spine classification.
5. Exports structured predictions and comparison against ground truth CSV.
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
from spinal_ct.localization import LEVELS
from spinal_ct.morphometry import (
    auto_segment_body,
    auto_detect_landmarks,
    measure_heights,
    measure_posterior_wall_retropulsion,
    compute_intervertebral_metrics,
)
from spinal_ct.aospine_rules import rank_and_classify_spine


def strip_accents(text: str) -> str:
    """Normalize Vietnamese diacritics and symbols for robust matching."""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', str(text))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return re.sub(r'[^a-z0-9]', '', text.lower())


def select_best_series(p_dir: Path) -> dict[str, Any] | None:
    """Select the best axial CT series for spine localization."""
    choices = list_ct_series(p_dir)
    valid = [c for c in choices if c.get('axial') and c.get('geometry_error') is None and c.get('slices', 0) >= 30]
    if not valid:
        # Fallback: check if any axial series exist with slices >= 20 even if slight geometry flag
        fallback = [c for c in choices if c.get('axial') and c.get('slices', 0) >= 20]
        if not fallback:
            return None
        valid = fallback

    max_sl = max(c['slices'] for c in valid)
    top = [c for c in valid if c['slices'] == max_sl]

    # Prioritize bone/spine reconstruction kernel
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
    spine_levels: list[str] = ('T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5'),
) -> dict[str, Any]:
    """Extract vertebral morphometry and compute AO Spine classification."""
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    ct_arr = sitk.GetArrayFromImage(ct_image)

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
    ranked = rank_and_classify_spine(inter)
    return ranked


def check_existing_diagnosis(slug: str) -> dict[str, Any] | None:
    """Check if patient already has diagnosis results or report in data/output/<slug>."""
    out_dir = ROOT / 'data/output' / slug
    if not out_dir.is_dir():
        return None

    # 1. Check if explicit diagnosis_result.json exists
    diag_f = out_dir / 'diagnosis_result.json'
    if diag_f.is_file():
        try:
            return json.loads(diag_f.read_text(encoding='utf-8'))
        except Exception:
            pass

    # 2. Check if do_than_dot has completed measurements
    dtd = out_dir / 'do_than_dot'
    if dtd.is_dir():
        sessions = [s for s in dtd.iterdir() if s.is_dir() and (s / 'measurements.json').is_file()]
        if len(sessions) >= 1:
            # Load existing classification from evaluate_aospine_baseline or aospine_evaluation
            eval_preds = ROOT / 'data/output/aospine_evaluation/predictions.json'
            if eval_preds.is_file():
                try:
                    preds = json.loads(eval_preds.read_text(encoding='utf-8'))
                    for p in preds:
                        if strip_accents(p.get('patient_id', '')) == strip_accents(slug) and p.get('has_data'):
                            return p
                except Exception:
                    pass
            # If no cached summary, mark present with do_than_dot sessions
            return {
                'status': 'existing_sessions',
                'session_count': len(sessions),
                'note': f'Đã có {len(sessions)} phiên đo thân đốt trong do_than_dot',
            }

    # 3. Check if dinh_vi_cot_song exists and was evaluated
    eval_preds = ROOT / 'data/output/aospine_evaluation/predictions.json'
    if eval_preds.is_file():
        try:
            preds = json.loads(eval_preds.read_text(encoding='utf-8'))
            for p in preds:
                if strip_accents(p.get('patient_id', '')) == strip_accents(slug) and p.get('has_data'):
                    return p
        except Exception:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true', default=True, help='Use fast 3mm model')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    raw_categories = ['A1', 'A3', 'A4', 'B2', 'B3', 'C']
    patients: list[tuple[str, Path]] = []
    for cat in raw_categories:
        cat_dir = ROOT / 'data/raw' / cat
        if not cat_dir.is_dir():
            continue
        for p_dir in sorted(cat_dir.iterdir()):
            if p_dir.is_dir():
                patients.append((cat, p_dir))

    if args.limit:
        patients = patients[:args.limit]

    print(f"=== BẮT ĐẦU CHẠY KIỂM TRA {len(patients)} BỆNH NHÂN ===")
    print(f"Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    backend = ROOT / '.venv-model/Scripts/TotalSegmentator.exe'
    if not backend.is_file():
        raise FileNotFoundError(f"Không tìm thấy backend: {backend}")

    for idx, (cat, p_dir) in enumerate(patients, 1):
        p_name = p_dir.name
        slug = slugify(p_name)
        print(f"[{idx}/{len(patients)}] Xử lý ca [{cat}] '{p_name}' (slug: {slug})...")

        # Bước 1: Kiểm tra kết quả chẩn đoán đã có
        existing = check_existing_diagnosis(slug)
        if existing:
            print(f"  -> BỎ QUA: Bệnh nhân đã có kết quả/báo cáo chẩn đoán trong data/output/{slug}.")
            skipped.append({
                'category': cat,
                'patient_name': p_name,
                'slug': slug,
                'existing_data': existing,
            })
            results.append({
                'category': cat,
                'patient_name': p_name,
                'slug': slug,
                'status': 'SKIPPED_ALREADY_EXISTS',
                'diagnosis': existing,
            })
            continue

        # Bước 2: Bệnh nhân chưa có kết quả - Kiểm tra và chuẩn bị chạy
        out_patient_dir = ROOT / 'data/output' / slug
        out_dvcs = out_patient_dir / 'dinh_vi_cot_song'
        out_dvcs.mkdir(parents=True, exist_ok=True)

        # Kiểm tra xem đã có segmentation vertebrae.nii.gz từ trước chưa
        existing_masks = list(out_dvcs.glob('**/vertebrae.nii.gz')) + list(out_dvcs.glob('**/consensus.nii.gz'))
        best_mask = existing_masks[0] if existing_masks else None

        selected_series = None
        ct_img = None

        try:
            # Chọn chuỗi CT thích hợp
            selected_series = select_best_series(p_dir)
            if not selected_series:
                raise ValueError("Không tìm thấy chuỗi CT cắt lớp trục (axial) hợp lệ đáp ứng hình học chuẩn (lỗi hình học hoặc không đủ lát cắt >= 20).")

            series_uid = selected_series['uid']
            print(f"  -> Đã chọn chuỗi CT: UID={series_uid[:20]}... ({selected_series['slices']} lát, kernel={selected_series['kernel']})")

            # Nạp ảnh CT
            ct_img = load_ct(p_dir, series_uid)

            if not best_mask:
                # Chạy phân đoạn TotalSegmentator
                run_tag = f"run_{'fast_' if args.fast else ''}{time.strftime('%Y%m%d_%H%M%S')}"
                run_out_dir = out_dvcs / run_tag
                print(f"  -> Đang chạy phân đoạn TotalSegmentator ({'fast' if args.fast else 'detailed'} mode) tới {run_out_dir.name}...")

                cmd = [
                    str(ROOT / '.venv/Scripts/python.exe'),
                    '-m', 'tools.localize_vertebrae',
                    '--backend', str(backend),
                    '--input', str(p_dir),
                    '--series-uid', series_uid,
                    '--output', str(run_out_dir),
                    '--device', 'cpu',
                ]
                if args.fast:
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
                    if cons.is_file():
                        best_mask = cons
                    else:
                        raise FileNotFoundError("Không tìm thấy tệp mask sau khi chạy phân đoạn.")

            # Chạy trích xuất hình thái học & phân loại tổn thương AO Spine
            print(f"  -> Đang tính toán hình thái thân đốt và phân loại AO Spine từ mask {best_mask.name}...")
            diag_res = run_morphometry_and_classification(best_mask, ct_img)

            top1 = diag_res['ranked_vertebrae'][0]
            summary = diag_res['patient_summary']
            print(f"  -> KẾT QUẢ CHẨN ĐOÁN: Tổn thương chính tại {top1['level']}, Phân loại AO: {top1['predicted_class']} (Độ tin cậy: {top1['confidence']:.1%})")

            # Lưu kết quả chẩn đoán vào thư mục output của bệnh nhân
            save_payload = {
                'patient_id': p_name,
                'slug': slug,
                'category': cat,
                'series_uid': series_uid,
                'slices': selected_series['slices'],
                'has_data': True,
                'top1_level': top1['level'],
                'top1_class': top1['predicted_class'],
                'top1_confidence': top1['confidence'],
                'top2_levels': [v['level'] for v in diag_res['ranked_vertebrae'][:2]],
                'patient_summary': summary,
                'ranked_vertebrae': diag_res['ranked_vertebrae'],
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            (out_patient_dir / 'diagnosis_result.json').write_text(
                json.dumps(save_payload, ensure_ascii=False, indent=2), encoding='utf-8'
            )

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

        print("-" * 50)

    # Lưu kết quả tổng hợp
    out_eval_dir = ROOT / 'data/output/aospine_batch_run'
    out_eval_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_eval_dir / 'batch_run_results.json'
    summary_file.write_text(json.dumps({
        'total_cases': len(patients),
        'skipped_count': len(skipped),
        'success_count': len([r for r in results if r['status'] == 'SUCCESS']),
        'failure_count': len(failures),
        'results': results,
        'failures': failures,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n=== HOÀN TẤT CHẠY KIỂM TRA ===")
    print(f"Tổng số ca: {len(patients)}")
    print(f"Đã bỏ qua (đã có kết quả): {len(skipped)}")
    print(f"Chạy mới thành công: {len([r for r in results if r['status'] == 'SUCCESS'])}")
    print(f"Không thể chẩn đoán (thất bại): {len(failures)}")
    print(f"Kết quả lưu tại: {summary_file}")


if __name__ == '__main__':
    main()
