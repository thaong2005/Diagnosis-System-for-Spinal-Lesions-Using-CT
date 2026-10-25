"""Exercise the desktop workflow with synthetic CT; no model or patient input."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import SimpleITK as sitk
from PySide6.QtWidgets import QApplication

from apps.demo import DemoWindow
from spinal_ct.review import ct_fingerprint


class DemoUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_idle(self, window):
        deadline = time.monotonic()+30
        while window._busy() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertFalse(window._busy(), 'UI task did not finish')

    def test_sample_path_drives_label_series_and_clears_previous_case(self):
        choices = [dict(uid='a', slices=567, rows=512, columns=512, kernel='B70s',
                        thickness='1', axial=True, geometry_error=None),
                   dict(uid='b', slices=397, rows=512, columns=512, kernel='B20s',
                        thickness='1', axial=True, geometry_error=None)]
        sample = Path('data/raw/example/new patient')
        with patch('apps.demo.SAMPLE_DIR', sample), \
                patch.object(DemoWindow, 'verify_environment'), \
                patch('apps.demo.list_ct_series', return_value=choices) as scan, \
                patch.object(DemoWindow, 'load_path') as load:
            window = DemoWindow()
            self.assertIn('New Patient', window.sample_button.text())
            window.study = {'old': True}
            window.review = {'old': True}
            window.open_sample()
            self.wait_idle(window)
            scan.assert_called_once_with(sample)
            load.assert_called_once_with(sample, 'a')
            self.assertEqual(window.pending_path, sample)
            self.assertEqual(window.series_combo.currentData(), 'a')
            self.assertIsNone(window.study)
            self.assertIsNone(window.review)
            load.reset_mock()
            choices[1]['slices'] = 567
            window.open_sample()
            self.wait_idle(window)
            load.assert_not_called()
            self.assertEqual(window.series_combo.count(), 2)
            self.assertTrue(window.load_button.isEnabled())
            self.assertFalse(window.run_button.isEnabled())
            window.series_combo.setCurrentIndex(1)
            window.load_series()
            load.assert_called_once_with(sample, 'b')
            window.close()

    def test_load_switch_modes_select_level_and_failed_case_clears_results(self):
        with tempfile.TemporaryDirectory() as temporary, patch('apps.demo.check_backend', return_value=(True, 'Fixture')), \
                patch('apps.demo.QMessageBox.warning') as warning:
            root = Path(temporary)
            data = np.full((40, 48, 36), -700, dtype=np.float32)
            data[8:32, 12:36, 8:28] = 500
            image = sitk.GetImageFromArray(data)
            image.SetSpacing((.8, .9, 1.3))
            source = root/'ct.nii.gz'
            sitk.WriteImage(image, str(source))
            # Hash the same decoded image that the application reads.
            image = sitk.ReadImage(str(source), sitk.sitkFloat32)
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)*20
            result = root/'result'
            mc = result/'monte_carlo'
            mc.mkdir(parents=True)
            sitk.WriteImage(mask, str(result/'vertebrae.nii.gz'))
            sitk.WriteImage(mask, str(mc/'consensus.nii.gz'))
            sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkFloat32)*.5, str(mc/'disagreement.nii.gz'))
            prediction = dict(level='L1', label_id=20, volume_mm3=100,
                              centroid_lps_mm=list(image.TransformIndexToPhysicalPoint((18, 24, 20))),
                              touches_boundary=False)
            report = dict(task='vertebral_localization', status='ok', predictions=[prediction], warnings=[],
                          input_fingerprint=ct_fingerprint(image),
                          monte_carlo=dict(samples=2, per_level=[dict(level='L1', detected_samples=2)],
                                           consensus=dict(predictions=[prediction])))
            (result/'prediction.json').write_text(json.dumps(report), encoding='utf-8')
            window = DemoWindow()
            window.resize(1320, 740)
            self.assertEqual(window.width(), 1320)
            window.show()
            # Let the startup environment check fire.
            for _ in range(20):
                self.app.processEvents()
                time.sleep(.01)
            self.wait_idle(window)
            window.load_path(source, results=result)
            self.wait_idle(window)
            self.assertTrue(window.review['verified'])
            self.assertEqual(window.mode.currentText(), 'Consensus')
            self.assertEqual(window.table.rowCount(), 1)
            window.table.selectRow(0)
            self.assertEqual([s.value() for s in window.sliders], [20, 18, 24])
            window.mode.setCurrentText('Bất đồng')
            window.canvas.draw()
            self.assertEqual(len(window.axes), 3)
            self.assertTrue(window.export_button.isEnabled())
            window.load_path(root/'missing.npy')
            self.wait_idle(window)
            self.assertIsNone(window.study)
            self.assertIsNone(window.review)
            self.assertFalse(window.run_button.isEnabled())
            self.assertEqual(window.table.rowCount(), 0)
            warning.assert_called_once()
            window.close()
