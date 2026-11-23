"""Evaluate 2-Step Vertebra Prototype Bank across 5 Folds (Phase 2 - Step 2.2).

Adheres to:
- Production Contract: Absolute Zero Patient Leakage between Train and Val across all folds.
- Step 1 Metrics: AUROC (Normal vs Fractured), Top-1 and Top-2 Anomaly Localization Accuracy.
- Step 2 Metrics: Fracture-only Classification Accuracy & Confusion Matrix (A3, A4, B2, C).
- Overall Metric: 5-Fold 7-Class Accuracy vs previous 52.2% baseline.
"""
from typing import Any
from pathlib import Path
import json
import sys
from collections import Counter

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spinal_ct.prototype_bank import (
    AOSpinePrototypeBank,
    SPINE_LEVELS,
    ALL_AOSPINE_CLASSES,
)
from scripts.prepare_aospine_dataset import filter_clean_normal_vertebrae


def compute_auroc_pure(y_true: list[int], y_scores: list[float]) -> float:
    """Compute Area Under ROC curve using exact Wilcoxon-Mann-Whitney U statistic (no sklearn dependency)."""
    pos = [s for y, s in zip(y_true, y_scores) if y == 1]
    neg = [s for y, s in zip(y_true, y_scores) if y == 0]
    if not pos or not neg:
        return 0.5
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return float(u / (len(pos) * len(neg)))


def evaluate_prototype_bank_5fold(
    splits_path: Path | str = 'data/processed/aospine_splits.json',
    manifest_path: Path | str = 'data/processed/aospine_rois/manifest.json',
    roi_dir: Path | str = 'data/processed/aospine_rois',
    output_dir: Path | str = 'data/processed/prototype_banks',
    anomaly_threshold: float = 0.50,
    external_normals_dir: Path | str | None = None,
    external_weight: float = 0.5,
) -> dict[str, Any]:
    """Execute rigorous 5-fold cross-validation of 2-step Prototype Bank."""
    splits_f = Path(splits_path)
    manifest_f = Path(manifest_path)
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    if not splits_f.is_file() or not manifest_f.is_file():
        raise FileNotFoundError(f"Missing {splits_f} or {manifest_f}")

    splits = json.loads(splits_f.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_f.read_text(encoding='utf-8'))

    raw_samples = manifest.get('samples', [])
    clean_samples = filter_clean_normal_vertebrae(raw_samples)
    print(f"Loaded {len(clean_samples)} clean vertebra samples from {len(raw_samples)} manifest samples.")

    # Storage for Step 1 metrics
    all_val_y_true: list[int] = []
    all_val_p_anomaly: list[float] = []
    all_val_anomaly_dists: list[float] = []

    patient_top1_hits: list[bool] = []
    patient_top2_hits: list[bool] = []
    patient_localization_records: list[dict[str, Any]] = []

    # Storage for Step 2 metrics (fractures with training representation)
    frac_y_true: list[str] = []
    frac_y_pred: list[str] = []

    # Storage for Overall 7-class decision
    all_y_true: list[str] = []
    all_y_pred: list[str] = []

    fold_results = []

    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION OF 2-STEP VERTEBRA PROTOTYPE BANK")
    print("=" * 80)

    for f in splits['folds']:
        fid = f['fold_id']
        train_sids = set(f['train_sample_ids'])
        val_sids = set(f['val_sample_ids'])
        train_pids = set(f['train_patient_ids'])
        val_pids = set(f['val_patient_ids'])

        # Strict Zero Patient Leakage Verification
        leakage = train_pids & val_pids
        if leakage:
            raise RuntimeError(f"FATAL: Patient leakage detected in fold {fid}: {leakage}")

        # Build prototype bank strictly from train set
        bank = AOSpinePrototypeBank()
        bank.build_from_fold(
            samples=clean_samples,
            train_sample_ids=train_sids,
            train_patient_ids=train_pids,
            roi_dir=roi_dir,
            fold_id=fid,
        )

        # Ingest external multi-scanner normal vertebrae if provided
        if external_normals_dir and Path(external_normals_dir).exists():
            cnt = bank.ingest_external_normals(external_normals_dir, external_weight=external_weight)
            if fid == 0:
                print(f"[Fold {fid}] Ingested {cnt} clean external normal vertebrae (weight={external_weight}).")

        # Save bank for this fold
        bank_save_file = out_d / f"bank_fold_{fid}.json"
        bank.save(bank_save_file)

        # Evaluate on validation samples of this fold
        val_samples = [s for s in clean_samples if s['sample_id'] in val_sids]

        fold_correct = 0
        fold_total = len(val_samples)
        fold_patient_preds: dict[str, list[dict[str, Any]]] = {}

        for s in val_samples:
            sid = s['sample_id']
            pid = s['patient_id']
            gt_cls = s['aospine_class']
            lvl = s['level']
            feats = s['prototype_features']

            # 2-Step Prediction
            pred = bank.predict(np.array(feats, dtype=np.float32), level=lvl, anomaly_threshold=anomaly_threshold)
            pred_cls = pred['predicted_class']
            p_anom = pred['p_anomaly']
            anom_dist = pred['anomaly_distance']

            is_match = (pred_cls == gt_cls)
            if is_match:
                fold_correct += 1

            all_y_true.append(gt_cls)
            all_y_pred.append(pred_cls)

            # Step 1 Anomaly tracking
            is_frac = 1 if gt_cls != 'Normal' else 0
            all_val_y_true.append(is_frac)
            all_val_p_anomaly.append(p_anom)
            all_val_anomaly_dists.append(anom_dist)

            fold_patient_preds.setdefault(pid, []).append({
                'sample_id': sid,
                'level': lvl,
                'aospine_class': gt_cls,
                'is_fractured': is_frac,
                'p_anomaly': p_anom,
                'anomaly_distance': anom_dist,
            })

            # Step 2 Lesion tracking (for true fractures where class exists in train bank)
            if gt_cls != 'Normal' and gt_cls in bank.prototypes:
                # Direct lesion matching
                lesion_match = bank.match_lesion_prototypes(np.array(feats, dtype=np.float32))
                frac_y_true.append(gt_cls)
                frac_y_pred.append(lesion_match['predicted_lesion_class'])

        # Anomaly localization per patient in this fold
        for pid, v_list in fold_patient_preds.items():
            if any(v['is_fractured'] == 1 for v in v_list):
                sorted_by_anomaly = sorted(v_list, key=lambda x: x['p_anomaly'], reverse=True)
                top1_hit = (sorted_by_anomaly[0]['is_fractured'] == 1)
                top2_hit = any(x['is_fractured'] == 1 for x in sorted_by_anomaly[:2])

                patient_top1_hits.append(top1_hit)
                patient_top2_hits.append(top2_hit)

                fractured_lvls = [x['level'] for x in v_list if x['is_fractured'] == 1]
                ranked_lvls = [f"{x['level']} ({x['p_anomaly']:.2f})" for x in sorted_by_anomaly[:3]]

                patient_localization_records.append({
                    'patient_id': pid,
                    'fold_id': fid,
                    'fractured_levels': fractured_lvls,
                    'top_ranked_levels': ranked_lvls,
                    'top1_hit': top1_hit,
                    'top2_hit': top2_hit,
                })

        fold_acc = float(fold_correct / fold_total) if fold_total > 0 else 0.0
        fold_results.append({
            'fold_id': fid,
            'train_patients': len(train_pids),
            'val_patients': len(val_pids),
            'train_samples': len(train_sids),
            'val_samples': fold_total,
            'accuracy': fold_acc,
            'correct_count': fold_correct,
            'available_lesion_prototypes': [c for c, p in bank.lesion_prototypes_by_class.items() if p['has_samples']],
        })

        print(f"Fold {fid}: Accuracy = {fold_acc*100:.1f}% ({fold_correct}/{fold_total} val samples) | "
              f"Patients: {len(train_pids)} train, {len(val_pids)} val | "
              f"Classes: {fold_results[-1]['available_lesion_prototypes']}")

    # --- Step 1 Overall Metrics ---
    auroc = compute_auroc_pure(all_val_y_true, all_val_p_anomaly)
    top1_acc = float(np.mean(patient_top1_hits)) if patient_top1_hits else 0.0
    top2_acc = float(np.mean(patient_top2_hits)) if patient_top2_hits else 0.0

    # Normal vs Fracture distance statistics
    norm_dists = [d for y, d in zip(all_val_y_true, all_val_anomaly_dists) if y == 0]
    frac_dists = [d for y, d in zip(all_val_y_true, all_val_anomaly_dists) if y == 1]

    # --- Step 2 Overall Metrics ---
    frac_acc = float(np.mean([t == p for t, p in zip(frac_y_true, frac_y_pred)])) if frac_y_true else 0.0
    confusion_pairs = Counter(zip(frac_y_true, frac_y_pred))
    confusion_matrix_dict = {f"True_{t}_Pred_{p}": count for (t, p), count in sorted(confusion_pairs.items())}

    # Per-class accuracy on fractures
    frac_classes = sorted(list(set(frac_y_true)))
    per_class_frac_acc = {}
    for c in frac_classes:
        total_c = sum(1 for t in frac_y_true if t == c)
        correct_c = sum(1 for t, p in zip(frac_y_true, frac_y_pred) if t == c and p == c)
        per_class_frac_acc[c] = {
            'correct': correct_c,
            'total': total_c,
            'accuracy': float(correct_c / total_c) if total_c > 0 else 0.0,
        }

    # --- Overall 7-Class Metric ---
    overall_acc = float(np.mean([t == p for t, p in zip(all_y_true, all_y_pred)]))
    mean_fold_acc = float(np.mean([r['accuracy'] for r in fold_results]))

    print("\n" + "=" * 80)
    print("STEP 1: ANOMALY DETECTION & LOCALIZATION PERFORMANCE")
    print("=" * 80)
    print(f"AUROC (Normal vs Fractured)          : {auroc:.4f}")
    print(f"Normal Anomaly Distance (mean ± std) : {np.mean(norm_dists):.2f} ± {np.std(norm_dists):.2f} (median: {np.median(norm_dists):.2f})")
    print(f"Fracture Anomaly Distance (mean ± std): {np.mean(frac_dists):.2f} ± {np.std(frac_dists):.2f} (median: {np.median(frac_dists):.2f})")
    print(f"Top-1 Anomaly Localization Accuracy   : {top1_acc*100:.1f}% ({sum(patient_top1_hits)}/{len(patient_top1_hits)} fracture cases)")
    print(f"Top-2 Anomaly Localization Accuracy   : {top2_acc*100:.1f}% ({sum(patient_top2_hits)}/{len(patient_top2_hits)} fracture cases)")

    print("\n" + "=" * 80)
    print("STEP 2: LESION IDENTIFICATION PERFORMANCE ON FRACTURED VERTEBRAE")
    print("=" * 80)
    print(f"Fracture-only Classification Accuracy : {frac_acc*100:.1f}% ({sum(t == p for t, p in zip(frac_y_true, frac_y_pred))}/{len(frac_y_true)})")
    print(f"Confusion Matrix (True -> Pred):")
    for (t, p), count in sorted(confusion_pairs.items()):
        print(f"  True {t:2s} -> Pred {p:2s} : {count} cases")
    print(f"Per-Class Breakdown:")
    for c, stats in per_class_frac_acc.items():
        print(f"  Class {c:2s}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")

    print("\n" + "=" * 80)
    print("OVERALL 7-CLASS ACCURACY COMPARISON")
    print("=" * 80)
    print(f"Previous 1-Step Baseline Accuracy    : 52.2%")
    print(f"Upgraded 2-Step 5-Fold Overall Acc   : {overall_acc*100:.1f}% ({sum(t == p for t, p in zip(all_y_true, all_y_pred))}/{len(all_y_true)})")
    print(f"Mean Fold Accuracy                   : {mean_fold_acc*100:.1f}%")
    print(f"Net Improvement                      : +{(overall_acc - 0.522)*100:+.1f}%")
    print("=" * 80)

    summary_report = {
        'schema_version': 2,
        'zero_patient_leakage_verified': True,
        'anomaly_threshold': anomaly_threshold,
        'total_evaluated_samples': len(all_y_true),
        'step1_anomaly_detection': {
            'auroc': float(round(auroc, 4)),
            'normal_distance_mean': float(round(np.mean(norm_dists), 2)),
            'fracture_distance_mean': float(round(np.mean(frac_dists), 2)),
            'top1_localization_accuracy': float(round(top1_acc, 4)),
            'top2_localization_accuracy': float(round(top2_acc, 4)),
            'total_fracture_patients': len(patient_top1_hits),
            'patient_details': patient_localization_records,
        },
        'step2_lesion_identification': {
            'fracture_only_accuracy': float(round(frac_acc, 4)),
            'confusion_matrix': confusion_matrix_dict,
            'per_class_breakdown': per_class_frac_acc,
        },
        'overall_comparison': {
            'baseline_1step_accuracy': 0.522,
            'upgraded_2step_overall_accuracy': float(round(overall_acc, 4)),
            'mean_fold_accuracy': float(round(mean_fold_acc, 4)),
            'net_improvement': float(round(overall_acc - 0.522, 4)),
        },
        'external_normals': {
            'external_normals_dir': str(external_normals_dir) if external_normals_dir else None,
            'external_weight': float(external_weight) if external_normals_dir else None,
            'external_ingested_count': int(bank.external_ingested_count),
        },
        'fold_results': fold_results,
    }

    report_suffix = f"_ext_w{int(external_weight*100)}" if external_normals_dir else ""
    report_file = Path(f'data/processed/prototype_bank_evaluation{report_suffix}.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nEvaluation summary exported to {report_file}")
    return summary_report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate 2-Step Vertebra Prototype Bank 5-Fold.")
    parser.add_argument('--splits', type=str, default='data/processed/aospine_splits.json')
    parser.add_argument('--manifest', type=str, default='data/processed/aospine_rois/manifest.json')
    parser.add_argument('--roi-dir', type=str, default='data/processed/aospine_rois')
    parser.add_argument('--output-dir', type=str, default='data/processed/prototype_banks')
    parser.add_argument('--anomaly-threshold', type=float, default=0.50)
    parser.add_argument('--external-normals-dir', type=str, default=None,
                        help="Optional directory containing clean external normal vertebrae manifest/ROIs.")
    parser.add_argument('--external-weight', type=float, default=0.5,
                        help="Weight w in (0, 1) assigned to external normal distribution (default: 0.5).")
    args = parser.parse_args()

    evaluate_prototype_bank_5fold(
        splits_path=args.splits,
        manifest_path=args.manifest,
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        anomaly_threshold=args.anomaly_threshold,
        external_normals_dir=args.external_normals_dir,
        external_weight=args.external_weight,
    )
