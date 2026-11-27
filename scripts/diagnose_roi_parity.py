#!/usr/bin/env python3
"""Diagnostic script comparing Production BodySession ROIs vs Training Reference ROIs.

Inspects axis conventions, channel-by-channel statistics (Ch 0..4), correlation,
16-dim Prototype Bank feature vectors, p_anomaly, and DualStream CNN raw_p_dl
on huynh_triet and duong_kim_hoa across T10-L5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import list_ct_series, load_ct, slugify
from spinal_ct.localization import LEVELS, AOSPINE_TARGET_LEVELS
from spinal_ct.body_region import propose_plane, sample_plane, sample_parasagittal_slices
from spinal_ct.morphometry import (
    BodySession,
    ct_fingerprint,
    get_aospine_prototype_bank,
    get_aospine_inference_engine,
)
from spinal_ct.prototype_bank import extract_roi_prototype_features
from scripts.run_batch_diagnosis import select_best_series, strip_accents


def inspect_patient_parity(cat: str, patient_name: str, use_source_ct: bool = False) -> dict:
    p_dir = ROOT / 'data/raw' / cat / patient_name
    slug = slugify(patient_name)
    clean_slug = strip_accents(patient_name).replace(' ', '').lower()

    out_dvcs = ROOT / 'data/output' / slug / 'dinh_vi_cot_song'
    v_masks = sorted(out_dvcs.glob('**/vertebrae.nii.gz'), key=lambda p: p.stat().st_mtime, reverse=True)
    best_mask = v_masks[0]

    series = select_best_series(p_dir, mask_path=best_mask)
    ct_img = load_ct(p_dir, series['uid'])
    mask_img = sitk.ReadImage(str(best_mask))
    fp = ct_fingerprint(ct_img)

    print(f"\n{'='*100}")
    print(f"PATIENT: {patient_name} (slug={slug}, clean={clean_slug}) | use_source_ct={use_source_ct}")
    print(f"CT size={ct_img.GetSize()}, spacing={ct_img.GetSpacing()}, direction={tuple(round(d, 3) for d in ct_img.GetDirection())}")
    print(f"Mask: {best_mask.relative_to(ROOT)}")

    bank = get_aospine_prototype_bank()
    engine = get_aospine_inference_engine()

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(False)
    stats.Execute(mask_img)

    rows_summary = []
    for lvl in AOSPINE_TARGET_LEVELS:
        lbl_id = LEVELS.index(lvl) + 1
        if not stats.HasLabel(lbl_id):
            continue

        ref_body_path = ROOT / f'data/processed/aospine_rois/{clean_slug}_{lvl}_body.npy'
        ref_arch_path = ROOT / f'data/processed/aospine_rois/{clean_slug}_{lvl}_arch.npy'
        if not ref_body_path.is_file() or not ref_arch_path.is_file():
            print(f"  [{lvl}] Reference ROI not found.")
            continue

        ref_body = np.load(ref_body_path).astype(np.float32)
        ref_arch = np.load(ref_arch_path).astype(np.float32)

        plane = propose_plane(ct_img, mask_img, lbl_id)
        ct_slice, valid_s, hint = sample_plane(ct_img, plane, mask_img, lbl_id)
        para = sample_parasagittal_slices(ct_img, plane)
        kwargs = {}
        if use_source_ct:
            kwargs['source_ct'] = ct_img
        sess = BodySession(
            fp, lvl, 'Baseline', plane, ct_slice, valid_s, hint,
            age_group='adult', parasagittal_slices=para, **kwargs,
        )
        sess.auto_populate(approve=False)
        prod_rois = sess.extract_dualstream_rois()
        if prod_rois is None:
            print(f"  [{lvl}] extract_dualstream_rois returned None!")
            continue
        prod_body, prod_arch = prod_rois

        # Compare channels 0..4 for body ROI
        ch_corrs = []
        ch_diffs = []
        for ch in range(5):
            r_ch = ref_body[ch].ravel()
            p_ch = prod_body[ch].ravel()
            if np.std(r_ch) > 1e-6 and np.std(p_ch) > 1e-6:
                corr = float(np.corrcoef(r_ch, p_ch)[0, 1])
            else:
                corr = 0.0
            ch_corrs.append(corr)
            ch_diffs.append((float(np.mean(p_ch)), float(np.mean(r_ch)), float(np.std(p_ch)), float(np.std(r_ch))))

        # Compare 16-dim feature vector
        ref_feat = extract_roi_prototype_features(ref_body, ref_arch)
        prod_feat = extract_roi_prototype_features(prod_body, prod_arch)
        feat_cos = float(np.dot(ref_feat, prod_feat))
        feat_l2 = float(np.linalg.norm(ref_feat - prod_feat))

        # Compare Prototype Bank p_anomaly
        ref_proto = bank.predict(ref_feat, level=lvl) if bank else {}
        prod_proto = bank.predict(prod_feat, level=lvl) if bank else {}

        # Compare CNN raw_p_dl
        ref_dl = engine.predict_batch([ref_body], [ref_arch])[0][0] if engine else {}
        prod_dl = engine.predict_batch([prod_body], [prod_arch])[0][0] if engine else {}

        print(f"\n--- Level {lvl} (plane size={plane.size_xy}, step={plane.spacing_xy_mm[0]:.3f}mm) ---")
        print(f"  Body ROI Ch mean (prod vs ref) & corr:")
        for ch in range(5):
            pm, rm, ps, rs = ch_diffs[ch]
            print(f"    Ch{ch}: prod={pm:.3f}±{ps:.3f} | ref={rm:.3f}±{rs:.3f} | corr={ch_corrs[ch]:+.3f}")
        print(f"  16-dim Feat: cos_sim={feat_cos:.4f}, L2_diff={feat_l2:.4f}")
        print(f"  Prototype p_anomaly: prod={prod_proto.get('p_anomaly', -1):.3f} ({prod_proto.get('predicted_class')}) "
              f"vs ref={ref_proto.get('p_anomaly', -1):.3f} ({ref_proto.get('predicted_class')})")
        print(f"  CNN raw_p_dl prod: frac={prod_dl.get('p_fracture', -1):.3f}, pw={prod_dl.get('p_pw', -1):.3f}, "
              f"ep2={prod_dl.get('p_ep2', -1):.3f}, tb={prod_dl.get('p_post_tb', -1):.3f}, trans={prod_dl.get('p_trans', -1):.3f}")
        print(f"  CNN raw_p_dl  ref: frac={ref_dl.get('p_fracture', -1):.3f}, pw={ref_dl.get('p_pw', -1):.3f}, "
              f"ep2={ref_dl.get('p_ep2', -1):.3f}, tb={ref_dl.get('p_post_tb', -1):.3f}, trans={ref_dl.get('p_trans', -1):.3f}")

        rows_summary.append({
            'level': lvl,
            'ch0_corr': ch_corrs[0],
            'ch3_corr': ch_corrs[3],
            'feat_cos': feat_cos,
            'prod_p_anom': prod_proto.get('p_anomaly', -1),
            'ref_p_anom': ref_proto.get('p_anomaly', -1),
            'prod_p_frac': prod_dl.get('p_fracture', -1),
            'ref_p_frac': ref_dl.get('p_fracture', -1),
        })

    return {'patient': patient_name, 'levels': rows_summary}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-source-ct', action='store_true', default=False)
    args = parser.parse_args()

    res_ht = inspect_patient_parity('A1', 'huynh triet', use_source_ct=args.with_source_ct)
    res_dkh = inspect_patient_parity('A3', 'duong kim hoa', use_source_ct=args.with_source_ct)

    print("\n" + "="*100)
    print("SUMMARY TABLE")
    for res in (res_ht, res_dkh):
        print(f"\nPatient: {res['patient']}")
        print(f"{'Lvl':<5} | {'Ch0 Corr':<9} | {'Ch3 Corr':<9} | {'Feat Cos':<9} | {'P_anom(Prod/Ref)':<18} | {'P_frac(Prod/Ref)':<18}")
        for r in res['levels']:
            print(f"{r['level']:<5} | {r['ch0_corr']:<+9.3f} | {r['ch3_corr']:<+9.3f} | {r['feat_cos']:<9.4f} | "
                  f"{r['prod_p_anom']:.3f} / {r['ref_p_anom']:.3f}      | {r['prod_p_frac']:.3f} / {r['ref_p_frac']:.3f}")


if __name__ == '__main__':
    main()
