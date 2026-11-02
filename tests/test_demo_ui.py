"""Exercise the desktop workflow with synthetic CT; no model or patient input."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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
            sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkUInt8)*21, str(mc/'consensus.nii.gz'))
            sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkFloat32)*.5, str(mc/'disagreement.nii.gz'))
            prediction = dict(level='L1', label_id=20, volume_mm3=100,
                              centroid_lps_mm=list(image.TransformIndexToPhysicalPoint((18, 24, 20))),
                              touches_boundary=False)
            report = dict(task='vertebral_localization', status='ok', predictions=[prediction], warnings=[],
                          input_fingerprint=ct_fingerprint(image),
                          monte_carlo=dict(samples=2, per_level=[dict(level='L1', detected_samples=2)],
                                           consensus=dict(predictions=[dict(prediction, level='L2', label_id=21)])))
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
            for name, expected in (('Baseline', 'L1'), ('Consensus', 'L2')):
                window.mode.setCurrentText(name)
                window.canvas.draw()
                self.assertEqual([t.get_text() for t in window.axes[0].texts], [])
                for ax in window.axes[1:]:
                    self.assertEqual([t.get_text() for t in ax.texts], [expected])
                # The label gutter is outside the CT: clicking it must not move slices.
                window.click_plane(SimpleNamespace(inaxes=window.axes[1], xdata=-3, ydata=20))
                self.assertEqual([s.value() for s in window.sliders], [20, 18, 24])
            window.show_names.setChecked(False)
            self.assertTrue(all(not ax.texts for ax in window.axes))
            window.show_names.setChecked(True)
            window.overlay.setChecked(False)
            self.assertTrue(all(not ax.texts for ax in window.axes))
            window.overlay.setChecked(True)
            window._set_indices(20, 0, 0)
            self.assertTrue(all(not ax.texts for ax in window.axes[1:]))
            window._set_indices(20, 18, 24)
            self.assertEqual([t.get_text() for t in window.axes[1].texts], ['L2'])
            window.mode.setCurrentText('Bất đồng')
            window.canvas.draw()
            self.assertEqual(len(window.axes), 3)
            self.assertTrue(all(not ax.texts for ax in window.axes))
            self.assertTrue(window.export_button.isEnabled())
            window.load_path(root/'missing.npy')
            self.wait_idle(window)
            self.assertIsNone(window.study)
            self.assertIsNone(window.review)
            self.assertFalse(window.run_button.isEnabled())
            self.assertEqual(window.table.rowCount(), 0)
            warning.assert_called_once()
            window.close()

    def test_discovered_result_reopens_without_inference_and_navigation_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(DemoWindow, 'verify_environment'):
            root = Path(tmp)
            image = sitk.GetImageFromArray(np.zeros((30, 20, 20), np.float32))
            source = root/'ct.nii.gz'
            sitk.WriteImage(image, str(source))
            folder = root/'data/demo_runs/saved'
            folder.mkdir(parents=True)
            labels = sitk.Image(image.GetSize(), sitk.sitkUInt8)
            sitk.WriteImage(labels, str(folder/'vertebrae.nii.gz'))
            predictions = [dict(level=level, volume_mm3=10., centroid_lps_mm=[10., 10., z])
                           for level, z in [('L1', 20.), ('L2', 15.), ('L3', 10.)]]
            report = dict(task='vertebral_localization', status='ok', warnings=[], predictions=predictions,
                          input_fingerprint=ct_fingerprint(image), model=dict(backend='TotalSegmentator',
                          version='2.18.0', resolution='3mm', device='cpu', robust_crop=False))
            (folder/'prediction.json').write_text(json.dumps(report))
            with patch('apps.demo.ROOT', root):
                window = DemoWindow()
                window.load_path(source)
                self.wait_idle(window)
                self.assertEqual(window.saved_combo.count(), 1)
                self.assertIn('Khác / thiếu cấu hình', window.saved_combo.currentText())
                window.resolution.setCurrentIndex(1)
                self.assertIn('Khớp cấu hình', window.saved_combo.currentText())
                self.assertTrue(window.reuse_button.isEnabled())
                window.reuse_result()
                self.wait_idle(window)
                self.assertTrue(window.review['verified'])
                self.assertIsNone(window.process)
                self.assertEqual(window.table.currentRow(), 0)
                self.assertFalse(window.previous_button.isEnabled())
                window.next_button.click()
                self.assertEqual(window.table.currentRow(), 1)
                self.assertEqual(window.sliders[0].value(), 15)
                window.move_level(100)
                self.assertEqual(window.table.currentRow(), 2)
                self.assertFalse(window.next_button.isEnabled())
                window.move_level(-100)
                self.assertEqual(window.table.currentRow(), 0)
                window.close()

    def test_detail_and_fast_run_options_reach_cli(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(DemoWindow, 'verify_environment'), \
                patch('apps.demo.ROOT', Path(temporary)), patch('apps.demo.QProcess') as process:
            window = DemoWindow()
            data = np.zeros((4, 5, 6), np.float32)
            window.study = dict(path=Path(temporary)/'ct.nii.gz', uid=None,
                                data=data, reference=sitk.GetImageFromArray(data))
            self.assertFalse(window.fast.isChecked())
            window.mc.setChecked(True)
            window.start_run()
            arguments = process.return_value.setArguments.call_args.args[0]
            self.assertIn('--robust-crop', arguments)
            self.assertNotIn('--fast', arguments)
            self.assertIn('--mc-samples', arguments)
            window.process = None
            window.fast.setChecked(True)
            window.start_run()
            arguments = process.return_value.setArguments.call_args.args[0]
            self.assertIn('--fast', arguments)
            self.assertNotIn('--robust-crop', arguments)
            window.process = None
            window.study = None
            window.close()


    def test_open_body_measurements_folder_without_predictions_does_not_crash(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(DemoWindow, 'verify_environment'), \
                patch('apps.demo.ROOT', Path(temporary)), \
                patch('apps.demo.QDesktopServices.openUrl') as mock_open:
            window = DemoWindow()
            self.assertEqual(window.current_predictions, [])
            self.assertIn('#123e50', window.patient_label.styleSheet())
            window.study = dict(path=Path(temporary)/'ct.nii.gz', uid=None,
                                patient=dict(display_name='Test Patient', slug='test_patient'))
            window._sync()
            self.assertTrue(window.body_folder_button.isEnabled())
            window.open_body_measurements_folder()
            mock_open.assert_called_once()
            window.close()
