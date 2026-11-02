import json
from pathlib import Path
import tempfile
import unittest

from spinal_ct.result_catalog import discover_results, compatible_result, discover_regions


class ResultCatalogTests(unittest.TestCase):
    def report(self):
        return dict(task='vertebral_localization', input_fingerprint='same',
                    model=dict(backend='TotalSegmentator', version='2.18.0', resolution='3mm',
                               device='cpu', robust_crop=False))

    def test_discovery_excludes_wrong_ct_partial_and_malformed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, report in [('valid', self.report()), ('wrong', dict(self.report(), input_fingerprint='other')),
                                 ('partial', self.report()), ('malformed', [])]:
                folder = root/name
                folder.mkdir()
                (folder/'prediction.json').write_text(json.dumps(report))
                if name != 'partial':
                    (folder/'vertebrae.nii.gz').touch()
            (root/'broken').mkdir()
            (root/'broken/prediction.json').write_text('{')
            found, errors = discover_results([root], 'same')
            self.assertEqual([item['folder'].name for item in found], ['valid'])
            self.assertEqual(errors, 1)

    def test_compatibility_checks_model_resolution_crop_and_mc(self):
        report = self.report()
        self.assertTrue(compatible_result(report, fast=True))
        self.assertFalse(compatible_result(report, fast=False))
        for key, value in [('version', 'old'), ('device', 'gpu'), ('robust_crop', True)]:
            changed = dict(report, model=dict(report['model'], **{key: value}))
            self.assertFalse(compatible_result(changed, fast=True))
        mc = dict(samples=2, seed=42, noise_sigma_hu=20., method='monte_carlo_input_perturbation')
        report['monte_carlo'] = mc
        self.assertFalse(compatible_result(report, fast=True))
        self.assertTrue(compatible_result(report, fast=True, samples=2))
        for config in (dict(samples=3), dict(samples=2, seed=1), dict(samples=2, noise=10.)):
            self.assertFalse(compatible_result(report, fast=True, **config))

    def test_latest_draft_is_scoped_to_ct_level_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, fp, kind, date in [('old', 'same', 'Baseline', '2026-10-01'),
                                         ('new', 'same', 'Baseline', '2026-10-02'),
                                         ('other', 'other', 'Baseline', '2026-10-03'),
                                         ('consensus', 'same', 'Consensus', '2026-10-01')]:
                folder = root/name
                folder.mkdir()
                (folder/'body_mask.npy').touch()
                (folder/'region.json').write_text(json.dumps(dict(task='vertebral_body_region',
                    input_fingerprint=fp, level='L3', source_kind=kind, created_utc=date)))
            found = discover_regions(root, 'same')
            self.assertEqual(found[('Baseline', 'L3')].name, 'new')
            self.assertEqual(found[('Consensus', 'L3')].name, 'consensus')

    def test_discovery_deduplicates_overlapping_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root/'duong_kim_hoa'/'dinh_vi_cot_song'/'run_1'
            sub.mkdir(parents=True)
            (sub/'prediction.json').write_text(json.dumps(self.report()))
            (sub/'vertebrae.nii.gz').touch()

            # Passing both the child and parent directories
            found, errors = discover_results([sub.parent, root], 'same')
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]['folder'].name, 'run_1')
