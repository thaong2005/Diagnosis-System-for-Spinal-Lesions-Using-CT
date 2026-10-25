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
    def run_fixture(self, root, cancel=False, fail=False):
        source = root/'ct.nii.gz'
        image = sitk.GetImageFromArray(np.zeros((8, 9, 10), dtype=np.float32))
        sitk.WriteImage(image, str(source))
        cancel_file = root/'cancel'

        def backend(command, **kwargs):
            if '--version' in command:
                return subprocess.CompletedProcess(command, 0, stdout='2.18.0\n', stderr='')
            # Keep both baseline and MC out of the memory-heavy worker pool.
            self.assertIn('--nr_thr_saving', command)
            self.assertEqual(command[command.index('--nr_thr_saving')+1], '1')
            if fail:
                return subprocess.CompletedProcess(command, 9, stdout='', stderr='fixture failure')
            masks = Path(command[command.index('-o')+1])
            masks.mkdir()
            for level in LEVELS:
                mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
                mask.CopyInformation(image)
                if level == 'L1':
                    mask[2:6, 2:7, 2:6] = 1
                sitk.WriteImage(mask, str(masks/f'vertebrae_{level}.nii.gz'))
            if cancel:
                cancel_file.touch()
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        arguments = ['localize', '--input', str(source), '--output', str(root/'result'),
                     '--backend', 'fixture', '--mc-samples', '2', '--progress-json',
                     '--cancel-file', str(cancel_file), '--model-cache', str(root/'cache')]
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
            self.assertGreater(report['runtime_seconds'], 0)
            stages = [json.loads(line[9:])['message'] for line in output.splitlines() if line.startswith('PROGRESS ')]
            self.assertTrue(any('mẫu 2/2' in stage for stage in stages))
            self.assertFalse(list(root.glob('.localization-*')))

    def test_cancel_and_backend_failure_never_publish_partial_result(self):
        for cancel, fail in ((True, False), (False, True)):
            with self.subTest(cancel=cancel, fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                code, output, _ = self.run_fixture(root, cancel, fail)
                self.assertEqual(code, 1)
                self.assertIn('cancelled' if cancel else 'exit 9', output)
                self.assertFalse((root/'result').exists())
                self.assertFalse(list(root.glob('.localization-*')))
