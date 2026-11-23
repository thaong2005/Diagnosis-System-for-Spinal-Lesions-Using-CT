"""Build production 2-step vertebra prototype bank for real-time clinical deployment (Stage 3).

Ingests local clean vertebrae and TotalSegmentator v201 multi-scanner clean normal vertebrae
using the optimal weighted variance pooling (w = 0.3) without modifying benchmark OOF folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_aospine_dataset import filter_clean_normal_vertebrae
from spinal_ct.prototype_bank import AOSpinePrototypeBank, SPINE_LEVELS, ALL_AOSPINE_CLASSES


def build_production_prototype_bank(
    manifest_path: Path | str = 'data/processed/aospine_rois/manifest.json',
    external_normals_dir: Path | str = 'data/processed/external_normal_rois',
    output_path: Path | str = 'models/aospine_prototype_bank.json',
    external_weight: float = 0.3,
    exclude_rare: bool = False,
) -> dict:
    """Build and serialize production AOSpinePrototypeBank."""
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Missing ROI manifest: {m_path}")

    manifest = json.loads(m_path.read_text(encoding='utf-8'))
    raw_samples = manifest.get('samples', [])
    clean_samples = filter_clean_normal_vertebrae(raw_samples)

    if exclude_rare:
        clean_samples = [s for s in clean_samples if s.get('aospine_class') not in ('A1', 'B3')]
        print("Note: Excluded rare challenge classes (A1, B3) from prototype bank training.")

    sample_ids = [s['sample_id'] for s in clean_samples]
    patient_ids = [s['patient_id'] for s in clean_samples]

    bank = AOSpinePrototypeBank()
    bank.build_from_fold(
        samples=clean_samples,
        train_sample_ids=sample_ids,
        train_patient_ids=patient_ids,
        roi_dir=m_path.parent,
        fold_id=-1,  # -1 designates production full-cohort model
    )

    ext_dir = Path(external_normals_dir)
    ingested_count = 0
    if ext_dir.exists():
        ingested_count = bank.ingest_external_normals(ext_dir, external_weight=external_weight)
        print(f"Ingested {ingested_count} multi-scanner clean normal vertebrae from {ext_dir} (weight={external_weight}).")
    else:
        print(f"Warning: External normal directory not found: {ext_dir}. Running with local samples only.")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out_file)
    print(f"Production prototype bank saved to: {out_file} ({out_file.stat().st_size:,} bytes)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("PRODUCTION PROTOTYPE BANK SUMMARY")
    print("=" * 60)
    print("Normal Prototypes by Level (T10-L5):")
    for lvl in SPINE_LEVELS:
        lvl_data = bank.normal_prototypes_by_level.get(lvl, {})
        cnt = lvl_data.get('count', 0)
        mean_norm = float(np.linalg.norm(lvl_data['mean'])) if lvl_data.get('mean') is not None else 0.0
        print(f"  {lvl:4s}: {cnt:3d} vertebrae (mean norm: {mean_norm:.3f})")

    g_norm = bank.global_normal_prototype
    print(f"Global Normal Prototype: {g_norm.get('count', 0)} vertebrae total")
    print(f"External Ingested Count: {bank.external_ingested_count}")

    print("\nLesion Prototypes by Class:")
    for c in ALL_AOSPINE_CLASSES:
        p_data = bank.lesion_prototypes_by_class.get(c, {})
        has_s = p_data.get('has_samples', False)
        s_cnt = p_data.get('sample_count', 0)
        print(f"  Class {c:3s}: {s_cnt:2d} samples (Active: {has_s})")
    print("=" * 60)

    return {
        'output_path': str(out_file),
        'total_local_samples': len(clean_samples),
        'external_ingested_count': bank.external_ingested_count,
        'normal_counts_by_level': {lvl: bank.normal_prototypes_by_level.get(lvl, {}).get('count', 0) for lvl in SPINE_LEVELS},
        'lesion_counts_by_class': {c: bank.lesion_prototypes_by_class.get(c, {}).get('sample_count', 0) for c in ALL_AOSPINE_CLASSES},
    }


def main():
    import numpy as np  # Ensure numpy is imported for __main__
    parser = argparse.ArgumentParser(description="Build production 2-step vertebra prototype bank.")
    parser.add_argument('--manifest', type=str, default='data/processed/aospine_rois/manifest.json')
    parser.add_argument('--external-normals-dir', type=str, default='data/processed/external_normal_rois')
    parser.add_argument('--output', type=str, default='models/aospine_prototype_bank.json')
    parser.add_argument('--external-weight', type=float, default=0.3)
    parser.add_argument('--exclude-rare', action='store_true', help="Exclude A1/B3 from prototype bank")
    args = parser.parse_args()

    build_production_prototype_bank(
        manifest_path=args.manifest,
        external_normals_dir=args.external_normals_dir,
        output_path=args.output,
        external_weight=args.external_weight,
        exclude_rare=args.exclude_rare,
    )


if __name__ == '__main__':
    main()
