#!/usr/bin/env python3
"""Audit 54 clinical CSV cases against data/raw/ DICOM folders & masks, and run
resume-safe TotalSegmentator (--fast --device cpu) on unsegmented CSV cases.

Adheres strictly to:
- Zero Label Leakage: segmentation adapter receives only sanitized CT volumes.
- Resume-safe: validates existing vertebrae.nii.gz via select_best_series +
  same_geometry(ct, mask) and never re-segments valid cases unless --force.
- Fault-tolerant: logs any per-case error into segmentation_54cases_audit.json
  and continues processing remaining patients.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import SimpleITK as sitk

from spinal_ct.ct_io import list_ct_series, load_ct, slugify
from spinal_ct.review import same_geometry
from scripts.evaluate_aospine_baseline import (
    discover_patient_cases,
    load_ground_truth,
    strip_accents,
)
from scripts.run_batch_diagnosis import select_best_series


DEFAULT_AUDIT_OUT = ROOT / 'data/output/aospine_batch_run/segmentation_54cases_audit.json'


def _find_valid_mask(
    p_dir: Path,
    slug: str,
) -> tuple[Path | None, dict[str, Any] | None, bool, str | None]:
    """Return (best_mask, selected_series, geom_ok, note) for a patient directory."""
    out_dvcs = ROOT / 'data/output' / slug / 'dinh_vi_cot_song'
    cand_masks: list[Path] = []
    if out_dvcs.is_dir():
        cand_masks.extend(out_dvcs.glob('**/vertebrae.nii.gz'))
    if (p_dir / 'dinh_vi_cot_song').is_dir():
        cand_masks.extend((p_dir / 'dinh_vi_cot_song').glob('**/vertebrae.nii.gz'))

    cand_masks.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    if not cand_masks:
        ser = select_best_series(p_dir, mask_path=None)
        return None, ser, False, None

    for mf in cand_masks:
        try:
            ser = select_best_series(p_dir, mask_path=mf)
            if not ser:
                continue
            ct_img = load_ct(p_dir, ser['uid'])
            mask_img = sitk.ReadImage(str(mf))
            if same_geometry(ct_img, mask_img):
                return mf, ser, True, None
        except Exception as exc:
            note = f'{type(exc).__name__}: {exc}'
            continue

    ser = select_best_series(p_dir, mask_path=cand_masks[0])
    return cand_masks[0], ser, False, 'Existing mask(s) failed geometry check against axial CT'


def _write_audit(audit_path: Path, payload: dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def run_audit_and_segmentation(
    *,
    data_dir: Path,
    ground_truth_file: Path,
    audit_output: Path,
    audit_only: bool = False,
    force: bool = False,
    limit: int | None = None,
    device: str = 'cpu',
) -> dict[str, Any]:
    """Audit CSV vs data/raw vs masks, then segment missing CSV cases safely."""
    gt_map = load_ground_truth(ground_truth_file)
    csv_records: list[dict[str, Any]] = []
    seen_csv_keys: set[str] = set()
    for k, rec in gt_map.items():
        if k.startswith('case_'):
            csv_records.append(rec)
            seen_csv_keys.add(strip_accents(rec.get('name', '')))

    case_dirs = discover_patient_cases(data_dir)
    backend = ROOT / '.venv-model/Scripts/TotalSegmentator.exe'
    python_exe = ROOT / '.venv/Scripts/python.exe'

    dir_entries: list[dict[str, Any]] = []
    matched_csv_slugs: set[str] = set()
    to_segment: list[tuple[Path, str, dict[str, Any]]] = []

    print(f"=== KIỂM KÊ DỮ LIỆU CT ({len(case_dirs)} thư mục vs {len(csv_records)} dòng CSV) ===", flush=True)

    for p_dir in case_dirs:
        cat = p_dir.parent.name if p_dir.parent != data_dir else 'FLAT'
        p_name = p_dir.name
        slug = slugify(p_name)
        clean_key = strip_accents(p_name)
        gt_item = gt_map.get(clean_key)
        in_csv = gt_item is not None
        if in_csv:
            matched_csv_slugs.add(clean_key)

        sers = list_ct_series(p_dir)
        ax_valid = [s for s in sers if s.get('axial') and s.get('geometry_error') is None and s.get('slices', 0) >= 20]
        best_mask, sel_ser, geom_ok, mask_note = _find_valid_mask(p_dir, slug)

        if not ax_valid:
            status = 'MISSING_DICOM'
        elif best_mask is not None and geom_ok and not force:
            status = 'ALREADY_SEGMENTED_VALID'
        elif not in_csv:
            status = 'NOT_IN_CSV_SKIPPED_SEG'
        else:
            status = 'NEEDS_SEGMENTATION'
            to_segment.append((p_dir, slug, sel_ser))

        entry = {
            'category': cat,
            'patient_name': p_name,
            'slug': slug,
            'clean_key': clean_key,
            'in_csv': in_csv,
            'csv_stt': gt_item.get('stt') if gt_item else None,
            'csv_true_level': gt_item.get('true_level') if gt_item else None,
            'csv_true_class': gt_item.get('true_class') if gt_item else None,
            'total_series': len(sers),
            'valid_axial_series': len(ax_valid),
            'selected_series_uid': sel_ser['uid'] if sel_ser else None,
            'selected_series_slices': sel_ser['slices'] if sel_ser else None,
            'selected_series_kernel': sel_ser.get('kernel') if sel_ser else None,
            'mask_path': str(best_mask.relative_to(ROOT)) if best_mask and best_mask.is_relative_to(ROOT) else (str(best_mask) if best_mask else None),
            'mask_geometry_valid': bool(geom_ok),
            'status': status,
            'note': mask_note,
        }
        dir_entries.append(entry)

        csv_tag = f"CSV#{gt_item.get('stt')} ({gt_item.get('true_level')} {gt_item.get('true_class')})" if gt_item else 'NO_CSV'
        print(f"  [{cat:<11}] {p_name:<22} | {csv_tag:<18} | ax={len(ax_valid):<2} | geom_ok={str(geom_ok):<5} | {status}", flush=True)

    missing_raw_csv = [
        {
            'stt': r.get('stt'),
            'name': r.get('name'),
            'patient_id': r.get('patient_id'),
            'true_level': r.get('true_level'),
            'true_class': r.get('true_class'),
            'clean_key': strip_accents(r.get('name', '')),
        }
        for r in csv_records
        if strip_accents(r.get('name', '')) not in matched_csv_slugs
    ]

    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'csv_total_rows': len(csv_records),
        'raw_dirs_total': len(case_dirs),
        'matched_in_csv_and_raw': len(matched_csv_slugs),
        'csv_missing_raw_dir_count': len(missing_raw_csv),
        'raw_not_in_csv_count': sum(1 for e in dir_entries if not e['in_csv']),
        'already_segmented_valid_count': sum(1 for e in dir_entries if e['status'] == 'ALREADY_SEGMENTED_VALID'),
        'needs_segmentation_count': len(to_segment),
        'newly_segmented_count': 0,
        'segmentation_failed_count': 0,
        'missing_raw_dir_csv_rows': missing_raw_csv,
        'entries': dir_entries,
        'segmentation_runs': [],
    }
    _write_audit(audit_output, summary)

    print("\n--- TÓM TẮT KIỂM KÊ ---", flush=True)
    print(f"Tổng dòng CSV: {summary['csv_total_rows']} | Thư mục data/raw: {summary['raw_dirs_total']}", flush=True)
    print(f"Khớp CSV & có DICOM: {summary['matched_in_csv_and_raw']} | Thiếu thư mục DICOM: {summary['csv_missing_raw_dir_count']} | Ngoài CSV: {summary['raw_not_in_csv_count']}", flush=True)
    print(f"Đã có mask hợp lệ (Resume-skip): {summary['already_segmented_valid_count']} | Cần phân đoạn mới: {summary['needs_segmentation_count']}", flush=True)

    if audit_only or not to_segment:
        return summary

    if not backend.is_file():
        raise FileNotFoundError(f"Không tìm thấy TotalSegmentator backend tại {backend}")

    targets = to_segment[:limit] if limit is not None else to_segment
    entry_by_slug = {e['slug']: e for e in dir_entries}

    print(f"\n=== BẮT ĐẦU PHÂN ĐOẠN BỔ SUNG ({len(targets)} ca, --fast --device {device}) ===", flush=True)
    for idx, (p_dir, slug, sel_ser) in enumerate(targets, 1):
        out_dvcs = ROOT / 'data/output' / slug / 'dinh_vi_cot_song'
        out_dvcs.mkdir(parents=True, exist_ok=True)
        run_out = out_dvcs / f"run_fast_{time.strftime('%Y%m%d_%H%M%S')}"
        while run_out.exists():
            time.sleep(1.0)
            run_out = out_dvcs / f"run_fast_{time.strftime('%Y%m%d_%H%M%S')}"

        run_rec: dict[str, Any] = {
            'patient_name': p_dir.name,
            'slug': slug,
            'series_uid': sel_ser['uid'] if sel_ser else None,
            'output_dir': str(run_out.relative_to(ROOT)),
            'status': 'RUNNING',
        }
        print(f"[{idx}/{len(targets)}] Segmenting {p_dir.name} ({slug}) -> {run_out.name}...", flush=True)
        t0 = time.monotonic()
        try:
            if not sel_ser:
                raise ValueError('No valid axial CT series selected for segmentation.')
            cmd = [
                str(python_exe),
                '-m', 'tools.localize_vertebrae',
                '--backend', str(backend),
                '--input', str(p_dir),
                '--series-uid', sel_ser['uid'],
                '--output', str(run_out),
                '--device', device,
                '--fast',
            ]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
            elapsed = round(time.monotonic() - t0, 2)
            if proc.returncode != 0:
                err_txt = (proc.stderr or proc.stdout or '').strip()
                raise RuntimeError(f"exit={proc.returncode}: {err_txt[:500]}")

            mask_file = run_out / 'vertebrae.nii.gz'
            if not mask_file.is_file():
                raise FileNotFoundError(f"Expected mask missing at {mask_file}")

            ct_img = load_ct(p_dir, sel_ser['uid'])
            mask_img = sitk.ReadImage(str(mask_file))
            if not same_geometry(ct_img, mask_img):
                raise RuntimeError('Generated vertebrae.nii.gz geometry does not match selected CT series')

            run_rec['status'] = 'SUCCESS'
            run_rec['elapsed_seconds'] = elapsed
            run_rec['mask_path'] = str(mask_file.relative_to(ROOT))
            entry_by_slug[slug]['status'] = 'NEWLY_SEGMENTED'
            entry_by_slug[slug]['mask_path'] = str(mask_file.relative_to(ROOT))
            entry_by_slug[slug]['mask_geometry_valid'] = True
            summary['newly_segmented_count'] += 1
            print(f"  -> OK in {elapsed:.1f}s ({mask_file.relative_to(ROOT)})", flush=True)
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            run_rec['status'] = 'FAILED'
            run_rec['elapsed_seconds'] = elapsed
            run_rec['error'] = f'{type(exc).__name__}: {exc}'
            entry_by_slug[slug]['status'] = 'SEGMENTATION_FAILED'
            entry_by_slug[slug]['note'] = run_rec['error']
            summary['segmentation_failed_count'] += 1
            print(f"  -> FAILED after {elapsed:.1f}s: {run_rec['error']}", flush=True)

        summary['segmentation_runs'].append(run_rec)
        summary['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
        _write_audit(audit_output, summary)

    print(
        f"\nHoàn tất phân đoạn bổ sung: Thành công {summary['newly_segmented_count']}/{len(targets)} | "
        f"Lỗi {summary['segmentation_failed_count']}",
        flush=True,
    )
    print(f"Audit JSON: {audit_output}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--ground-truth', type=Path, default=ROOT / 'data/raw/data ngay 27.09.2026.csv')
    parser.add_argument('--audit-output', type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument('--audit-only', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', choices=['cpu', 'gpu', 'mps'], default='cpu')
    args = parser.parse_args()

    run_audit_and_segmentation(
        data_dir=args.data_dir,
        ground_truth_file=args.ground_truth,
        audit_output=args.audit_output,
        audit_only=args.audit_only,
        force=args.force,
        limit=args.limit,
        device=args.device,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
