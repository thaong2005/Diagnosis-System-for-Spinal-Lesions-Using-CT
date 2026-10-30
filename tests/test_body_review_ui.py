"""Body editor state invalidation and user actions on synthetic CT."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from PySide6.QtWidgets import QApplication
from apps.body_review import BodyReviewDialog
from apps.demo import DemoWindow
from spinal_ct.morphometry import save_session
from spinal_ct.review import ct_fingerprint
from test_body_morphometry import fixture


class BodyReviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self):
        source, labels = fixture()
        return BodyReviewDialog(source, labels, ct_fingerprint(source), 'L1', 'Baseline', Path('data/body_measurements'))

    def draw(self, dialog):
        dialog.apply_polygon([[12, 12], [30, 12], [30, 28], [12, 28]])

    def test_polygon_confirm_measure_edit_and_undo(self):
        with patch('apps.body_review.QMessageBox.warning') as warning:
            dialog = self.dialog()
            self.assertFalse(dialog.measure_button.isEnabled())
            self.draw(dialog)
            original = dialog.session.mask.copy()
            dialog.confirm.setChecked(True)
            self.assertTrue(dialog.measure_button.isEnabled())
            dialog.samples.setValue(1000)
            dialog.measure()
            self.assertGreater(dialog.table.rowCount(), 0)
            dialog.apply_polygon([[15, 15], [18, 15], [18, 18], [15, 18]], subtract=True)
            self.assertFalse(dialog.confirm.isChecked())
            self.assertEqual(dialog.table.rowCount(), 0)
            dialog.undo()
            np.testing.assert_array_equal(dialog.session.mask, original)
            self.assertFalse(dialog.session.reviewed)
            warning.assert_not_called()
            dialog.close()

    def test_plane_and_seed_change_clear_measurements(self):
        dialog = self.dialog()
        self.draw(dialog)
        dialog.confirm.setChecked(True)
        dialog.samples.setValue(100)
        dialog.measure()
        dialog.seed.setText('123')
        self.assertIsNone(dialog.session.result)
        self.assertEqual(dialog.table.rowCount(), 0)
        self.assertTrue(dialog.session.reviewed)
        dialog.offset.setValue(1)
        self.assertIsNone(dialog.session)
        self.assertFalse(dialog.measure_button.isEnabled())
        dialog.rebuild_plane()
        self.assertFalse(dialog.session.mask.any())
        dialog.close()

    def test_landmark_clicks_and_reopen(self):
        dialog = self.dialog()
        self.draw(dialog)
        dialog.start_landmarks()
        for x, y in ((12, 27), (12, 12), (20, 29), (20, 12), (28, 30), (28, 12)):
            dialog.click(SimpleNamespace(inaxes=dialog.body_ax, xdata=x, ydata=y, button=1))
        self.assertEqual(len(dialog.session.landmarks), 6)
        dialog.confirm.setChecked(True)
        with tempfile.TemporaryDirectory() as tmp:
            save_session(dialog.session, Path(tmp)/'region', samples=100)
            dialog.clear_region()
            self.assertFalse(dialog.session.mask.any())
            dialog.load_region(Path(tmp)/'region')
            self.assertTrue(dialog.session.reviewed)
            self.assertEqual(len(dialog.session.landmarks), 6)
            self.assertIsNone(dialog.session.result)
        dialog.close()

    def test_brush_and_age_group_invalidate_review(self):
        dialog = self.dialog()
        self.draw(dialog)
        dialog.confirm.setChecked(True)
        dialog.start_brush()
        dialog.click(SimpleNamespace(inaxes=dialog.body_ax, xdata=20, ydata=20, button=3))
        dialog.release(None)
        self.assertFalse(dialog.session.reviewed)
        dialog.confirm.setChecked(True)
        dialog.age.setCurrentIndex(1)
        self.assertFalse(dialog.session.reviewed)
        self.assertEqual(dialog.session.age_group, 'adult')
        dialog.close()

    def test_log_rerun_and_parameter_edits_preserve_body_measurements(self):
        dialog = self.dialog()
        self.draw(dialog)
        dialog.confirm.setChecked(True)
        dialog.samples.setValue(100)
        dialog.measure()
        previous = dialog.session.result['monte_carlo_area'].copy()
        rows = dialog.table.rowCount()
        dialog.find_blobs()
        self.assertEqual(dialog.session.result['monte_carlo_area'], previous)
        self.assertEqual(dialog.table.rowCount(), rows)
        self.assertIsNotNone(dialog.session.result['blob_detection'])
        dialog.log_threshold.setValue(.07)
        self.assertIsNone(dialog.session.candidates)
        self.assertIsNone(dialog.session.result['blob_detection'])
        self.assertEqual(dialog.blob_choices.count(), 0)
        self.assertFalse(dialog.show_blobs.isEnabled())
        self.assertEqual(dialog.table.rowCount(), rows)
        self.assertTrue(dialog.session.reviewed)
        self.assertIn('Thông số đã đổi', dialog.log_status.text())
        dialog.find_blobs()
        self.assertEqual(dialog.session.result['blob_detection']['config']['threshold'], .07)
        dialog.close()

    def test_saved_settings_restore_plane_log_and_large_seed(self):
        dialog = self.dialog()
        dialog.offset.setValue(2.5)
        dialog.yaw.setValue(12)
        dialog.tilt.setValue(-7)
        dialog.rebuild_plane()
        self.draw(dialog)
        dialog.log_min.setValue(6)
        dialog.log_max.setValue(15)
        dialog.log_threshold.setValue(.07)
        dialog.find_blobs()
        dialog.confirm.setChecked(True)
        large_seed = 2**63-1
        with tempfile.TemporaryDirectory() as tmp:
            save_session(dialog.session, Path(tmp)/'region', samples=700, seed=large_seed)
            dialog.offset.setValue(0)
            dialog.yaw.setValue(0)
            dialog.tilt.setValue(0)
            dialog.log_min.setValue(2)
            dialog.seed.setText('1')
            dialog.load_region(Path(tmp)/'region')
            self.assertAlmostEqual(dialog.offset.value(), 2.5)
            self.assertAlmostEqual(dialog.yaw.value(), 12)
            self.assertAlmostEqual(dialog.tilt.value(), -7)
            self.assertEqual(dialog.log_min.value(), 6)
            self.assertEqual(dialog.log_max.value(), 15)
            self.assertEqual(dialog.log_threshold.value(), .07)
            self.assertEqual(dialog.current_sampling(), dict(samples=700, seed=large_seed))
            self.assertTrue(dialog.session.reviewed)
            dialog.measure()
            self.assertEqual(dialog.session.result['monte_carlo_area']['seed'], large_seed)
        dialog.close()

    def test_experimental_panels_collapsed_and_sigma_guard(self):
        dialog = self.dialog()
        self.assertTrue(dialog.experiment_panel.isHidden())
        self.assertFalse(dialog.experiment_toggle.isChecked())
        dialog.experiment_toggle.setChecked(True)
        self.assertFalse(dialog.experiment_panel.isHidden())
        dialog.log_min.setValue(.2)
        self.assertGreaterEqual(dialog.log_min.value(), max(dialog.session.plane.spacing_xy_mm))
        self.draw(dialog)
        dialog.confirm.setChecked(True)
        dialog.seed.setText('')
        self.assertFalse(dialog.measure_button.isEnabled())
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.seed.setText('42')
        self.assertTrue(dialog.measure_button.isEnabled())
        dialog.close()
        with patch.object(DemoWindow, 'verify_environment'):
            window = DemoWindow()
            self.assertTrue(window.mc_panel.isHidden())
            window.mc.setChecked(True)
            self.assertTrue(window.mc_toggle.isChecked())
            self.assertFalse(window.mc_panel.isHidden())
            self.assertIn('đang bật', window.mc_toggle.text())
            window.close()

    def test_demo_opens_body_editor_for_selected_source_and_level(self):
        source, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(DemoWindow, 'verify_environment'), \
                patch('apps.body_review.BodyReviewDialog') as editor:
            folder = Path(tmp)
            sitk.WriteImage(labels, str(folder/'vertebrae.nii.gz'))
            window = DemoWindow()
            window.study = dict(source=source, fingerprint=ct_fingerprint(source),
                                reference=source, data=sitk.GetArrayFromImage(source))
            prediction = dict(level='L1', volume_mm3=100., centroid_lps_mm=[9, 20, 30])
            window.review = dict(folder=folder, report=dict(predictions=[prediction]),
                                 masks=dict(Baseline=sitk.GetArrayFromImage(labels)))
            window.mode.addItem('Baseline')
            window.mode.setCurrentText('Baseline')
            self.assertTrue(window.body_button.isEnabled())
            with patch.object(window, '_task', side_effect=lambda function, callback, message: callback(function())):
                window.open_body_review()
            self.assertEqual(editor.call_args.args[3:5], ('L1', 'Baseline'))
            editor.return_value.exec.assert_called_once()
            window._clear_study()
            self.assertFalse(window.body_button.isEnabled())
            window.close()


if __name__ == '__main__':
    unittest.main()
