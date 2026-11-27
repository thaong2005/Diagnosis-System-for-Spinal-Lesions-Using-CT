"""Integration tests for Direction C (Stage 3): Three-Pillar AI Fusion in Production.

Tests the end-to-end integration of:
1. Production AOSpinePrototypeBank singleton and graceful fallback.
2. BodySession 3-Pillar fusion (Rules + Prototype Bank + CNN) with signature caching.
3. SpineSession multi-vertebra sequence triage and ranking.
4. Export verification (features.csv with p_anomaly, prototype_class, triggered_guards,
   and report.html with 3-pillar breakdown).
5. UI Banner & state logic (apps.demo and apps.body_review).
6. Execution latency benchmark on CPU.
"""
from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from spinal_ct.body_region import BodyPlane
from spinal_ct.morphometry import (
    BodySession,
    SpineSession,
    get_aospine_prototype_bank,
    get_aospine_inference_engine,
    save_session,
)


def make_dummy_production_session(
    level: str = 'L1',
    fracture_type: str = 'normal',
    spacing_xy: tuple[float, float] = (1.0, 1.0),
) -> BodySession:
    """Create a 2D BodySession with realistic parasagittal slices for 2.5D feature extraction."""
    shape = (80, 80)
    plane = BodyPlane(
        origin_lps_mm=(0.0, 0.0, 0.0),
        u_lps=(0.0, 1.0, 0.0),
        v_lps=(0.0, 0.0, 1.0),
        spacing_xy_mm=spacing_xy,
        size_xy=shape,
    )
    ct = np.full(shape, 100.0, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    hint = np.zeros(shape, dtype=bool)

    # Vertebral body region: col 15..45, row 20..60
    hint[20:60, 15:45] = True
    ct[20:60, 15:45] = 400.0

    # Spinal canal (CSF / soft tissue): col 46..58
    ct[20:60, 46:58] = 30.0

    # Posterior arch / spinous process: col 60..75, row 35..45
    hint[35:45, 60:75] = True
    ct[35:45, 60:75] = 350.0

    mask = np.zeros(shape, dtype=bool)
    mask[20:60, 15:45] = True

    if fracture_type == 'burst':
        # Retropulsion into canal + anterior collapse
        ct[30:50, 46:54] = 380.0
        mask[30:50, 46:54] = False
        landmarks = [
            [15.0, 36.0], [15.0, 24.0],  # Ha = 12 mm
            [30.0, 36.0], [30.0, 24.0],  # Hm = 12 mm
            [45.0, 36.0], [45.0, 24.0],  # Hp = 12 mm
        ]
    elif fracture_type == 'wedge':
        # Pure anterior wedge
        landmarks = [
            [15.0, 38.0],  # AS collapsed
            [15.0, 22.0],  # AI
            [30.0, 48.0],  # MS
            [30.0, 22.0],  # MI
            [45.0, 58.0],  # PS
            [45.0, 22.0],  # PI
        ]
    else:  # Normal vertebra
        landmarks = [
            [15.0, 58.0],  # AS
            [15.0, 22.0],  # AI
            [30.0, 58.0],  # MS
            [30.0, 22.0],  # MI
            [45.0, 58.0],  # PS
            [45.0, 22.0],  # PI
        ]

    # Dummy parasagittal slices (-4mm, 0mm, +4mm)
    parasagittal_slices = [
        np.array(ct, copy=True),
        np.array(ct, copy=True),
        np.array(ct, copy=True),
    ]

    session = BodySession(
        input_fingerprint='test_3pillar_fp',
        level=level,
        source_kind='Baseline',
        plane=plane,
        ct=ct,
        valid=valid,
        hint=hint,
        mask=mask,
        landmarks=landmarks,
        age_group='adult',
        parasagittal_slices=parasagittal_slices,
    )

    if fracture_type == 'burst':
        session.retropulsion = {
            'delta_posterior_mm': 3.8,
            'canal_encroachment_ratio': 0.25,
            'canal_line_points': [[45.0, 10.0], [45.0, 70.0]],
            'is_retropulsed': True,
            'retropulsed_bone_mask': np.zeros(shape, bool),
        }

    return session


class TestProductionThreePillarIntegration(unittest.TestCase):
    """Test Three-Pillar AI Fusion in Production Core."""

    def test_prototype_bank_singleton_and_graceful_fallback(self):
        """Test singleton loading and fallback when prototype file is absent."""
        bank = get_aospine_prototype_bank()
        self.assertIsNotNone(bank, "Production prototype bank should load from models/aospine_prototype_bank.json")
        self.assertGreater(len(bank.normal_prototypes_by_level), 0, "Bank should have normal prototypes")
        self.assertIn('L1', bank.normal_prototypes_by_level)
        self.assertIn('T12', bank.normal_prototypes_by_level)

        # Test fallback with non-existent file
        import spinal_ct.morphometry as morph
        morph._PROTOTYPE_BANK = None
        bank_fallback = morph.get_aospine_prototype_bank(bank_path=Path(tempfile.gettempdir()) / 'nonexistent_bank.json')
        self.assertIsNone(bank_fallback, "Should gracefully return None if bank file not found")
        # Reset
        morph._PROTOTYPE_BANK = None
        bank_restored = morph.get_aospine_prototype_bank()
        self.assertIsNotNone(bank_restored)

    def test_body_session_3pillar_fusion_caching(self):
        """Test BodySession recompute produces 3-pillar diagnostics and caches correctly."""
        session = make_dummy_production_session('L1', fracture_type='normal')
        res = session.recompute_aospine_classification()

        self.assertIn('predicted_class', res)
        self.assertIn('fusion_diagnostics', res)
        diag = res['fusion_diagnostics']

        # Prototype bank check
        self.assertIn('prototype_result', diag)
        proto_res = diag['prototype_result']
        if proto_res:
            self.assertIn('p_anomaly', proto_res)
            self.assertIn('predicted_class', proto_res)
            self.assertGreaterEqual(proto_res['p_anomaly'], 0.0)
            self.assertLessEqual(proto_res['p_anomaly'], 1.0)

        # Biomechanical guards check
        self.assertIn('triggered_guards', diag)
        self.assertIsInstance(diag['triggered_guards'], list)

        # Verify caching
        sig_before = session._cached_proto_sig
        self.assertIsNotNone(sig_before)
        res_before = session._cached_proto_result
        self.assertIsNotNone(res_before)

        # Second recompute should use cache
        res2 = session.recompute_aospine_classification()
        self.assertIs(session._cached_proto_result, res_before)

        # Invalidate directly should clear cached signature
        session.invalidate()
        self.assertIsNone(session._cached_proto_sig)
        self.assertIsNone(session._cached_proto_result)

        # Changing landmarks recomputes and yields a new signature
        session.set_landmarks([[15.0, 50.0]] + session.landmarks[1:])
        self.assertIsNotNone(session._cached_proto_sig)
        self.assertNotEqual(session._cached_proto_sig, sig_before)

    def test_save_session_exports_3pillar_diagnostics(self):
        """Test save_session generates features.csv and report.html with 3-pillar details."""
        session = make_dummy_production_session('T12', fracture_type='wedge')
        session.approve()
        session.recompute_aospine_classification()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / 'T12'
            save_session(session, out_path, samples=500, seed=42)

            # 1. Check features.csv
            csv_path = out_path / 'features.csv'
            self.assertTrue(csv_path.is_file(), "features.csv must exist")
            with csv_path.open('r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                self.assertIn('p_anomaly', row)
                self.assertIn('prototype_class', row)
                self.assertIn('triggered_guards', row)
                self.assertIn('aospine_class', row)

            # 2. Check report.html
            html_path = out_path / 'report.html'
            self.assertTrue(html_path.is_file(), "report.html must exist")
            html_content = html_path.read_text(encoding='utf-8')
            self.assertIn('Kiến trúc Hợp nhất 3 Trụ cột (Three-Pillar AI Fusion)', html_content)
            self.assertIn('Trụ cột 1: Luật Hình thái 3D (Geometric Rules)', html_content)
            self.assertIn('Trụ cột 2: Ngân hàng Mẫu 2 Bước', html_content)
            self.assertIn('Trụ cột 3: Mạng Nơ-ron 2.5D Dual-Stream CNN', html_content)
            self.assertIn('Chốt bảo vệ Cơ sinh học (Guards)', html_content)
            if get_aospine_inference_engine() is not None:
                self.assertIn('P_frac=', html_content)
                self.assertNotIn('Không khả dụng (Chạy Fallback)', html_content)

            # 3. Check measurements.json
            meas_path = out_path / 'measurements.json'
            self.assertTrue(meas_path.is_file(), "measurements.json must exist")
            meas_data = json.loads(meas_path.read_text(encoding='utf-8'))
            self.assertIn('aospine', meas_data)
            self.assertIn('fusion_diagnostics', meas_data['aospine'])

    def test_spine_session_multi_level_3pillar_triage(self):
        """Test multi-vertebra sequence triage and ranking in SpineSession."""
        s_t11 = make_dummy_production_session('T11', fracture_type='normal')
        s_t12 = make_dummy_production_session('T12', fracture_type='burst')
        s_l1 = make_dummy_production_session('L1', fracture_type='normal')

        spine = SpineSession(patient_id='PT_TEST_001')
        spine.add_session(s_t11)
        spine.add_session(s_t12)
        spine.add_session(s_l1)

        result = spine.recompute_aospine_classification()
        ranked = result['ranked_vertebrae']
        psum = result['patient_summary']

        self.assertEqual(len(ranked), 3)
        # T12 should be ranked #1 due to burst severity
        self.assertEqual(ranked[0]['level'], 'T12')
        self.assertEqual(ranked[0]['rank'], 1)
        self.assertEqual(psum['primary_lesion_level'], 'T12')
        self.assertTrue(psum['has_surgical_instability'])

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / 'do_than_dot'
            s_t11.approve()
            s_t12.approve()
            s_l1.approve()
            spine.save_session(out_root, samples=200, seed=42)

            summary_file = out_root / 'spine_patient_summary.json'
            self.assertTrue(summary_file.is_file())
            sum_doc = json.loads(summary_file.read_text(encoding='utf-8'))
            self.assertEqual(sum_doc['patient_summary']['primary_lesion_level'], 'T12')
            self.assertEqual(len(sum_doc['ranked_vertebrae']), 3)

    def test_demo_and_body_review_headless_ui(self):
        """Test headless banner styling and region state parsing logic."""
        from apps.demo import DemoWindow

        win = MagicMock(spec=DemoWindow)
        win.study = {'patient': {'slug': 'test_patient'}}
        win.mode = MagicMock()
        win.mode.currentText.return_value = 'Baseline'
        win.triage_banner = MagicMock()
        win.region_paths = {}

        # 1. Test empty triage banner
        DemoWindow._update_triage_banner(win)
        win.triage_banner.setText.assert_called_with(
            'Chưa có dữ liệu đo thân đốt. Chọn đốt và bấm "Đo thân đốt" để kích hoạt chẩn đoán AO Spine 3 Trụ cột.'
        )

        # 2. Test region state helper with temporary measurements
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / 'T12'
            folder.mkdir()
            reg_file = folder / 'region.json'
            meas_file = folder / 'measurements.json'

            reg_file.write_text(json.dumps({'reviewed': True}), encoding='utf-8')
            meas_file.write_text(json.dumps({
                'aospine': {
                    'predicted_class': 'A3',
                    'confidence': 0.88,
                    'secondary_subtype': None,
                }
            }), encoding='utf-8')

            win.region_paths = {('Baseline', 'T12'): str(folder)}
            state_text = DemoWindow._get_region_state_text(win, 'Baseline', 'T12')
            self.assertIn('Đã lưu · Đã xác nhận · AO Spine: A3 (88%)', state_text)

            # Test triage banner with unstable vertebra
            DemoWindow._update_triage_banner(win)
            call_text = win.triage_banner.setText.call_args[0][0]
            self.assertIn('CẢNH BÁO MẤT VỮNG', call_text)
            self.assertIn('T12', call_text)

    def test_whole_spine_recompute_latency_cpu(self):
        """Verify real-time performance: whole-spine (8 levels T10-L5) recompute < 100 ms on CPU."""
        levels = ['T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
        spine = SpineSession(patient_id='PT_PERF_TEST')
        for lvl in levels:
            spine.add_session(make_dummy_production_session(lvl, fracture_type='normal'))

        # Warm-up
        spine.recompute_aospine_classification()

        # Benchmark
        t0 = time.perf_counter()
        res = spine.recompute_aospine_classification()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(elapsed_ms, 100.0, f"Whole spine recompute took {elapsed_ms:.2f} ms, expected < 100 ms")
        self.assertEqual(len(res['ranked_vertebrae']), 8)

    def test_out_of_domain_level_skips_ai_and_triage_prefers_t10_l5(self):
        """Verify that C1-T9 vertebrae skip Prototype Bank & CNN and do not usurp Triage Top-1."""
        s_t5 = make_dummy_production_session('T5', fracture_type='burst')
        s_t12 = make_dummy_production_session('T12', fracture_type='burst')

        spine = SpineSession(patient_id='PT_OOD_TEST')
        spine.add_session(s_t5)
        spine.add_session(s_t12)

        res = spine.recompute_aospine_classification()

        # T5 is out of domain (T10-L5)
        self.assertTrue(s_t5.aospine.get('is_out_of_aospine_scope'))
        self.assertEqual(s_t5.aospine.get('lesion_severity_score'), 0.0)
        self.assertIsNone(getattr(s_t5, '_cached_proto_sig', None))
        self.assertIsNone(getattr(s_t5, '_cached_dl_sig', None))

        # T12 is in domain and should be Top-1 in triage
        self.assertFalse(s_t12.aospine.get('is_out_of_aospine_scope', False))
        self.assertEqual(spine.patient_summary['primary_lesion_level'], 'T12')
        self.assertEqual(spine.ranked_vertebrae[0]['level'], 'T12')
        self.assertEqual(spine.ranked_vertebrae[-1]['level'], 'T5')

    def test_render_body_title_compact(self):
        """Verify that draw_body produces concise titles without Matplotlib canvas edge clipping."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from spinal_ct.morphometry_render import draw_body

        fig, (ax1, ax2) = plt.subplots(1, 2)
        try:
            # 1. In-domain vertebra
            s_l1 = make_dummy_production_session('L1', fracture_type='normal')
            s_l1.recompute_aospine_classification()
            draw_body(ax1, s_l1)
            t1 = ax1.get_title()
            self.assertIn('L1', t1)
            self.assertIn('AO:', t1)
            self.assertLessEqual(len(t1), 45)

            # 2. Out-of-domain vertebra (e.g. T5)
            s_t5 = make_dummy_production_session('T5', fracture_type='normal')
            s_t5.recompute_aospine_classification()
            draw_body(ax2, s_t5)
            t2 = ax2.get_title()
            self.assertIn('T5', t2)
            self.assertIn('Ngoài T10–L5', t2)
            self.assertLessEqual(len(t2), 45)
        finally:
            plt.close(fig)

    def test_body_session_source_ct_roi_parity_and_signature(self):
        """Hotfix 3.2 Task 1: source_ct gives 80x80mm full context + true Axial Ch3 without altering _signature."""
        import SimpleITK as sitk
        from spinal_ct.body_region import propose_plane, sample_plane, sample_parasagittal_slices
        from spinal_ct.morphometry import ct_fingerprint

        # Build synthetic 3D LPS CT (Z=100, Y=100, X=60) with T12, L1, L2 bodies + canal
        vol = np.full((100, 100, 60), -200.0, dtype=np.float32)
        lbl = np.zeros((100, 100, 60), dtype=np.uint8)
        # Body block cols Y=30..55, X=20..40 for three levels T12(14), L1(15), L2(16)
        vol[65:85, 30:55, 20:40] = 420.0
        vol[38:58, 30:55, 20:40] = 420.0
        vol[12:32, 30:55, 20:40] = 420.0
        lbl[65:85, 30:55, 20:40] = 14
        lbl[38:58, 30:55, 20:40] = 15
        lbl[12:32, 30:55, 20:40] = 16
        # Posterior elements
        vol[38:58, 70:82, 27:33] = 360.0

        ct_img = sitk.GetImageFromArray(vol)
        ct_img.SetSpacing((1.0, 1.0, 1.0))
        lbl_img = sitk.GetImageFromArray(lbl)
        lbl_img.SetSpacing((1.0, 1.0, 1.0))

        # propose_plane keeps default margin_mm=5.0
        plane = propose_plane(ct_img, lbl_img, 15)
        self.assertLessEqual(max(plane.size_xy), 60)

        ct_slice, valid_s, hint = sample_plane(ct_img, plane, lbl_img, 15)
        para = sample_parasagittal_slices(ct_img, plane)
        fp = ct_fingerprint(ct_img)

        sess_2d = BodySession(fp, 'L1', 'Baseline', plane, ct_slice, valid_s, hint,
                              mask=hint.copy(), age_group='adult', parasagittal_slices=para)
        sess_3d = BodySession(fp, 'L1', 'Baseline', plane, ct_slice, valid_s, hint,
                              mask=hint.copy(), age_group='adult', parasagittal_slices=para,
                              source_ct=ct_img)

        self.assertEqual(sess_2d._signature(), sess_3d._signature())

        roi2d_body, _ = sess_2d.extract_dualstream_rois()
        roi3d_body, _ = sess_3d.extract_dualstream_rois()

        # 2D clipped plane has substantial zero (-1024 HU) border padding at top/bottom,
        # whereas 3D wide extraction captures adjacent T12 & L2 context (> -200 HU)
        self.assertGreater(float(np.mean(roi2d_body[0] == 0.0)), float(np.mean(roi3d_body[0] == 0.0)))
        # Ch 3 (Axial bone window vs 2D sagittal soft-tissue window) must differ substantially
        self.assertFalse(np.allclose(roi2d_body[3], roi3d_body[3]))

    def test_run_batch_diagnosis_oof_fold_resolution_and_singleton_isolation(self):
        """Hotfix 3.3 / Phase 4.2: OOF batch mode excludes non-CSV folders and keeps singletons None."""
        import SimpleITK as sitk
        import torch
        from unittest.mock import patch
        import spinal_ct.morphometry as morph
        from spinal_ct.aospine_model import DualStreamAOSpineNet
        from spinal_ct.prototype_bank import AOSpinePrototypeBank
        from scripts.run_batch_diagnosis import resolve_oof_fold_for_patient, run_batch_diagnosis

        splits = {
            'rare_class_challenge_set': ['huynhtriet', 'havanquyet'],
            'folds': [
                {'fold_id': 0, 'val_patient_ids': ['patalpha'], 'train_patient_ids': ['patbeta']},
                {'fold_id': 1, 'val_patient_ids': ['patbeta'], 'train_patient_ids': ['patalpha']},
            ],
        }
        self.assertEqual(resolve_oof_fold_for_patient('patalpha', splits), (0, True, False))
        self.assertEqual(resolve_oof_fold_for_patient('patbeta', splits), (1, True, False))
        self.assertEqual(resolve_oof_fold_for_patient('huynhtriet', splits), (0, True, True))
        self.assertEqual(resolve_oof_fold_for_patient('alal', splits), (None, False, False))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / 'raw'
            out_root = tmp / 'output'
            models_dir = tmp / 'models'
            banks_dir = tmp / 'banks'
            models_dir.mkdir(parents=True)
            banks_dir.mkdir(parents=True)

            # CSV with only pat_alpha (A3 L1) and pat_beta (A4 L2); A4/a lal is absent from CSV
            (raw_root / 'A3' / 'pat alpha').mkdir(parents=True)
            (raw_root / 'A4' / 'pat beta').mkdir(parents=True)
            (raw_root / 'A4' / 'a lal').mkdir(parents=True)
            (raw_root / 'data ngay 27.09.2026.csv').write_text(
                'STT,Tên,Đốt gãy,phân loại,ID,kèm theo\n'
                '1,pat alpha,L1,A3,P01,\n'
                '2,pat beta,L2,A4,P02,\n',
                encoding='utf-8',
            )
            splits_file = tmp / 'splits.json'
            splits_file.write_text(json.dumps(splits), encoding='utf-8')

            # Create minimal synthetic 3D CT/mask shared via patched DICOM loader
            from spinal_ct.localization import LEVELS
            vol = np.full((90, 80, 60), -200.0, dtype=np.float32)
            lbl = np.zeros((90, 80, 60), dtype=np.uint8)
            for z0, lvl_name in [(62, 'T12'), (38, 'L1'), (14, 'L2')]:
                lid = LEVELS.index(lvl_name) + 1
                vol[z0:z0 + 16, 15:35, 20:40] = 400.0
                lbl[z0:z0 + 16, 15:35, 20:40] = lid
                vol[z0 + 4:z0 + 12, 46:60, 27:33] = 350.0
                lbl[z0 + 4:z0 + 12, 46:60, 27:33] = lid
            ct_img = sitk.GetImageFromArray(vol)
            lbl_img = sitk.GetImageFromArray(lbl)

            for slug in ('pat_alpha', 'pat_beta', 'a_lal'):
                m_dir = out_root / slug / 'dinh_vi_cot_song' / 'run_fast_test'
                m_dir.mkdir(parents=True)
                sitk.WriteImage(lbl_img, str(m_dir / 'vertebrae.nii.gz'))

            # Save two fold checkpoints and two fold prototype banks
            for fid, trained_pid in [(0, 'patbeta'), (1, 'patalpha')]:
                net = DualStreamAOSpineNet(in_channels=5, feature_dim=128)
                torch.save(net.state_dict(), str(models_dir / f'aospine_dualstream_fold{fid}.pt'))
                bank = AOSpinePrototypeBank()
                dummy_feats = np.ones((2, 16), dtype=np.float32)
                dummy_feats[1] *= 2.0
                samples = [
                    {'sample_id': f'{trained_pid}_T12', 'patient_id': trained_pid, 'level': 'T12', 'aospine_class': 'Normal', 'prototype_features': dummy_feats[0].tolist()},
                    {'sample_id': f'{trained_pid}_L1', 'patient_id': trained_pid, 'level': 'L1', 'aospine_class': 'A3', 'prototype_features': dummy_feats[1].tolist()},
                ]
                bank.build_from_fold(samples, {s['sample_id'] for s in samples}, {trained_pid}, fold_id=fid)
                bank.save(banks_dir / f'bank_fold_{fid}.json')

            morph._INFERENCE_ENGINE = None
            morph._PROTOTYPE_BANK = None
            fake_ser = {'uid': '1.2.3.4', 'slices': 60, 'kernel': 'BONEPLUS', 'axial': True}

            with patch('scripts.run_batch_diagnosis.select_best_series', return_value=fake_ser), \
                 patch('scripts.run_batch_diagnosis.load_ct', return_value=ct_img):
                summary = run_batch_diagnosis(
                    force=True,
                    oof=True,
                    splits_path=splits_file,
                    models_dir=models_dir,
                    banks_dir=banks_dir,
                    raw_root=raw_root,
                    output_root=out_root,
                )

            self.assertEqual(summary['evaluation_mode'], '5fold_oof')
            self.assertTrue(summary['singleton_isolated'])
            self.assertEqual(summary['evaluated_cases'], 2)
            self.assertEqual(summary['unlabeled_cases_count'], 0)
            self.assertEqual(summary['skipped_unlabeled_oof'], ['A4/a lal'])
            self.assertEqual(summary['failure_count'], 0)
            fold_by_slug = {r['slug']: r['diagnosis']['oof_fold_id'] for r in summary['results']}
            self.assertEqual(fold_by_slug, {'pat_alpha': 0, 'pat_beta': 1})
            self.assertIsNone(morph._INFERENCE_ENGINE)
            self.assertIsNone(morph._PROTOTYPE_BANK)


if __name__ == '__main__':
    unittest.main()
