"""Cross-platform adapter tests for progress, cancellation and publication."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from spinal_ct.localization import LEVELS
from spinal_ct.review import ct_fingerprint
from tools.localize_vertebrae import main


class DemoPipelineTests(unittest.TestCase):
    def run_fixture(self, root, cancel=False, fail=False, fast=False, binary=False, fail_sample=False):
        source = root/'ct.nii.gz'
        image = sitk.GetImageFromArray(np.zeros((8, 9, 10), dtype=np.float32))
        sitk.WriteImage(image, str(source))
        cancel_file = root/'cancel'
        calls = 0

        def backend(command, **kwargs):
            nonlocal calls
            if '--version' in command:
                return subprocess.CompletedProcess(command, 0, stdout='2.18.0\n', stderr='')
            calls += 1
            # Keep both baseline and MC out of the memory-heavy worker pool.
            self.assertIn('--nr_thr_saving', command)
            self.assertEqual(command[command.index('--nr_thr_saving')+1], '1')
            self.assertEqual('--robust_crop' in command, not fast)
            self.assertEqual('--fast' in command, fast)
            self.assertIn('--ml', command)
            self.assertEqual(Path(command[command.index('-i')+1]).suffix, '.nii')
            self.assertEqual(Path(command[command.index('-o')+1]).suffix, '.nii')
            if fail or (fail_sample and calls == 2):
                return subprocess.CompletedProcess(command, 9, stdout='', stderr='fixture failure')
            labels = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            labels.CopyInformation(image)
            labels[2:6, 2:7, 2:6] = 31  # Backend L1, not project label 20.
            sitk.WriteImage(labels, command[command.index('-o')+1])
            if cancel:
                cancel_file.touch()
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        arguments = ['localize', '--input', str(source), '--output', str(root/'result'),
                     '--backend', 'fixture', '--mc-samples', '2', '--progress-json',
                     '--cancel-file', str(cancel_file), '--model-cache', str(root/'cache')]
        arguments += ['--fast'] if fast else ['--robust-crop']
        if binary:
            arguments += ['--export-binary-masks']
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), patch('tools.localize_vertebrae.shutil.which', return_value='fixture'), \
                patch('tools.localize_vertebrae.subprocess.run', side_effect=backend), redirect_stdout(output):
            code = main()
        return code, output.getvalue(), image

    def test_progress_report_and_mc_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, output, image = self.run_fixture(root)
            self.assertEqual(code, 0, output)
            report = json.loads((root/'result/prediction.json').read_text(encoding='utf-8'))
            self.assertEqual(report['input_fingerprint'], ct_fingerprint(image))
            self.assertEqual(report['monte_carlo']['samples'], 2)
            self.assertEqual(report['model']['saving_threads'], 1)
            self.assertEqual(report['model']['resolution'], '1.5mm')
            self.assertTrue(report['model']['robust_crop'])
            self.assertGreater(report['runtime_seconds'], 0)
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(report['predictions'][0]['mask_encoding'], 'label_map')
            self.assertEqual(report['predictions'][0]['level'], 'L1')
            np.testing.assert_array_equal(
                sitk.GetArrayFromImage(sitk.ReadImage(str(root/'result/vertebrae.nii.gz')))[2:6, 2:7, 2:6], 20)
            self.assertFalse((root/'result/masks').exists())
            self.assertFalse((root/'result/monte_carlo/consensus_masks').exists())
            self.assertFalse((root/'result/ct.nii').exists())
            self.assertFalse((root/'result/backend-labels.nii').exists())
            for key in ('load_ct', 'baseline_inference', 'mc_001_inference', 'mc_002_inference', 'mc_aggregation'):
                self.assertGreaterEqual(report['timings_seconds'][key], 0)
            self.assertAlmostEqual(sum(report['timings_seconds'].values()), report['runtime_seconds'], delta=.01)
            stages = [json.loads(line[9:])['message'] for line in output.splitlines() if line.startswith('PROGRESS ')]
            self.assertTrue(any('mẫu 2/2' in stage for stage in stages))
            self.assertFalse(list(root.glob('.localization-*')))

    def test_optional_binary_export_matches_shared_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, output, _ = self.run_fixture(root, binary=True)
            self.assertEqual(code, 0, output)
            for folder, label_path in (('masks', 'vertebrae.nii.gz'),
                                       ('monte_carlo/consensus_masks', 'monte_carlo/consensus.nii.gz')):
                labels = sitk.GetArrayFromImage(sitk.ReadImage(str(root/'result'/label_path)))
                for label_id, level in enumerate(LEVELS, 1):
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(root/'result'/folder/f'vertebrae_{level}.nii.gz')))
                    np.testing.assert_array_equal(mask, labels == label_id)

    def test_failed_mc_sample_never_publishes_baseline_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, output, _ = self.run_fixture(root, fail_sample=True)
            self.assertEqual(code, 1, output)
            self.assertIn('MC sample 1', output)
            self.assertFalse((root/'result').exists())
            self.assertFalse(list(root.glob('.localization-*')))

    def test_fast_mode_is_retained_for_baseline_and_mc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, output, _ = self.run_fixture(root, fast=True)
            self.assertEqual(code, 0, output)
            report = json.loads((root/'result/prediction.json').read_text(encoding='utf-8'))
            self.assertEqual(report['model']['resolution'], '3mm')
            self.assertFalse(report['model']['robust_crop'])

    def test_cancel_and_backend_failure_never_publish_partial_result(self):
        for cancel, fail in ((True, False), (False, True)):
            with self.subTest(cancel=cancel, fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                code, output, _ = self.run_fixture(root, cancel, fail)
                self.assertEqual(code, 1)
                self.assertIn('cancelled' if cancel else 'exit 9', output)
                self.assertFalse((root/'result').exists())
                self.assertFalse(list(root.glob('.localization-*')))
