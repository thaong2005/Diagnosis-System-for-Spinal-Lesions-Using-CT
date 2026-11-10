"""Body editor state invalidation and user actions on synthetic CT."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from apps.body_review import BodyReviewDialog
from apps.demo import DemoWindow
from spinal_ct.morphometry import save_session
from spinal_ct.review import ct_fingerprint
from test_body_morphometry import fixture


class BodyReviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        prompt = patch('apps.body_review.QMessageBox.question', return_value=QMessageBox.Discard)
        prompt.start()
        self.addCleanup(prompt.stop)

    def wait_idle(self, dialog):
        deadline = time.monotonic()+20
        while dialog.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertIsNone(dialog.worker)

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
            self.wait_idle(dialog)
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
        self.wait_idle(dialog)
        dialog.seed.setText('123')
        self.assertIsNone(dialog.session.result)
        self.assertEqual(dialog.table.rowCount(), 0)
        self.assertTrue(dialog.session.reviewed)
        dialog.offset.setValue(1)
        self.assertTrue(dialog.session.mask.any())
        self.assertTrue(dialog.pending_plane)
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
        self.wait_idle(dialog)
        previous = dialog.session.result['monte_carlo_area'].copy()
        rows = dialog.table.rowCount()
        dialog.find_blobs()
        self.wait_idle(dialog)
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
        self.wait_idle(dialog)
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
        self.wait_idle(dialog)
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
            self.wait_idle(dialog)
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

    def test_worker_keeps_event_loop_alive_and_blocks_edits(self):
        dialog = self.dialog()
        self.draw(dialog)
        dialog.confirm.setChecked(True)
        released = threading.Event()
        from spinal_ct.morphometry import BodySession
        original = BodySession.measure
        def delayed(session, **config):
            released.wait(5)
            return original(session, **config)
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        with patch.object(BodySession, 'measure', delayed):
            dialog.measure()
            mask = dialog.session.mask.copy()
            self.assertFalse(dialog.controls_panel.isEnabled())
            dialog.close()
            self.assertFalse(dialog._closing)
            for _ in range(12): QTest.qWait(50)
            self.assertGreater(len(ticks), 2)
            np.testing.assert_array_equal(mask, dialog.session.mask)
            released.set()
            self.wait_idle(dialog)
        timer.stop()
        self.assertIsNotNone(dialog.session.result)
        self.assertTrue(dialog.controls_panel.isEnabled())
        dialog.close()

    def test_cancel_plane_change_and_close_preserves_unsaved_region(self):
        dialog = self.dialog()
        self.draw(dialog)
        original = dialog.session.mask.copy()
        dialog.offset.setValue(3)
        with patch('apps.body_review.QMessageBox.question', return_value=QMessageBox.Cancel):
            dialog.rebuild_plane()
            dialog.reject()
            self.assertFalse(dialog._closing)
        np.testing.assert_array_equal(original, dialog.session.mask)
        dialog.restore_plane_controls()
        self.assertFalse(dialog.pending_plane)
        self.assertEqual(dialog.offset.value(), 0)
        self.assertTrue(dialog.add_button.isEnabled())
        dialog.close()

    def test_save_draft_before_plane_change_and_failure_preserves_region(self):
        dialog = self.dialog()
        self.draw(dialog)
        original = dialog.session.mask.copy()
        with tempfile.TemporaryDirectory() as tmp:
            dialog.output_root = Path(tmp)
            dialog.offset.setValue(2)
            with patch('apps.body_review.QMessageBox.question', return_value=QMessageBox.Save), \
                    patch('apps.body_review.QFileDialog.getExistingDirectory', return_value=tmp), \
                    patch('apps.body_review.save_session', side_effect=OSError('disk full')), \
                    patch('apps.body_review.QMessageBox.warning'):
                dialog.rebuild_plane()
                self.wait_idle(dialog)
            np.testing.assert_array_equal(dialog.session.mask, original)
            self.assertTrue(dialog.pending_plane)
            with patch('apps.body_review.QMessageBox.question', return_value=QMessageBox.Save), \
                    patch('apps.body_review.QFileDialog.getExistingDirectory', return_value=tmp):
                dialog.rebuild_plane()
                self.wait_idle(dialog)
            self.assertTrue((dialog.last_saved_path/'region.json').is_file())
            self.assertFalse(dialog.session.mask.any())
            dialog.load_region(dialog.last_saved_path)
            np.testing.assert_array_equal(dialog.session.mask, original)
            self.assertFalse(dialog.is_dirty())
            self.assertIn('Vùng nháp', dialog.workflow_state())
        dialog.close()

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

    def test_auto_populate_worker_and_undo(self):
        dialog = self.dialog()
        self.assertTrue(dialog.auto_button.isEnabled())
        dialog.auto_button.click()
        self.wait_idle(dialog)

        self.assertTrue(dialog.session.mask.any())
        self.assertEqual(len(dialog.session.landmarks), 6)
        self.assertIsNotNone(dialog.session.auto_provenance)
        self.assertFalse(dialog.session.reviewed)
        self.assertTrue(dialog.landmark_toggle.isChecked())
        self.assertIn('Đã tự động tách vùng và đặt 6 mốc', dialog.status.text())

        dialog.undo()
        self.assertFalse(dialog.session.mask.any())
        self.assertEqual(len(dialog.session.landmarks), 0)
        self.assertIsNone(dialog.session.auto_provenance)
        self.assertFalse(dialog.session.reviewed)
        dialog.close()

    def test_overlay_toggles_and_quick_clean_shortcut(self):
        """Test toggling overlay layers and H/Space quick clean comparison."""
        from PySide6.QtCore import Qt
        dialog = self.dialog()
        self.draw(dialog)

        # 1. Overlay checkboxes
        self.assertTrue(dialog.toggle_mask.isChecked())
        self.assertTrue(dialog.toggle_landmarks.isChecked())
        self.assertTrue(dialog.toggle_aospine.isChecked())

        dialog.toggle_mask.setChecked(False)
        dialog.toggle_landmarks.setChecked(False)
        dialog.toggle_aospine.setChecked(False)
        self.assertFalse(dialog.hide_overlays_temp)

        # 2. Quick clean button
        dialog.toggle_clean_quick.setChecked(True)
        self.assertTrue(dialog.hide_overlays_temp)
        dialog.toggle_clean_quick.setChecked(False)
        self.assertFalse(dialog.hide_overlays_temp)

        # 3. Key press H and Space
        QTest.keyClick(dialog, Qt.Key_H)
        self.assertTrue(dialog.toggle_clean_quick.isChecked())
        self.assertTrue(dialog.hide_overlays_temp)

        QTest.keyClick(dialog, Qt.Key_Space)
        self.assertFalse(dialog.toggle_clean_quick.isChecked())
        self.assertFalse(dialog.hide_overlays_temp)

        dialog.close()

    def test_doctor_override_and_navigator_switch(self):
        """Test doctor confirmation dropdown and spine level navigator."""
        dialog = self.dialog()
        self.draw(dialog)

        # 1. Doctor override combo
        self.assertIsNotNone(dialog.doctor_override_combo)
        self.assertIsNone(dialog.session.doctor_confirmed_class)

        dialog.doctor_override_combo.setCurrentText('A3')
        self.assertEqual(dialog.session.doctor_confirmed_class, 'A3')
        self.assertIn('BS: A3', dialog.aospine_badge.text())
        self.assertTrue(dialog.is_dirty())

        # Reset to AI recommendation
        dialog.doctor_override_combo.setCurrentText('Theo gợi ý AI')
        self.assertIsNone(dialog.session.doctor_confirmed_class)

        # 2. Spine level navigator
        self.assertIn('L1', dialog.available_levels)
        self.assertIn('L1', dialog.level_buttons)

        # Level switch
        if len(dialog.available_levels) > 1:
            target_level = [lvl for lvl in dialog.available_levels if lvl != 'L1'][0]
            # Add 6 landmarks to L1
            dialog.start_landmarks()
            for x, y in ((12, 27), (12, 12), (20, 29), (20, 12), (28, 30), (28, 12)):
                dialog.click(SimpleNamespace(inaxes=dialog.body_ax, xdata=x, ydata=y, button=1))
            self.assertEqual(len(dialog.session.landmarks), 6)

            dialog.switch_to_level(target_level)
            self.assertEqual(dialog.level, target_level)

            # Switch back to L1 and verify content is NOT lost
            dialog.switch_to_level('L1')
            self.assertEqual(dialog.level, 'L1')
            self.assertEqual(len(dialog.session.landmarks), 6)
            self.assertTrue(dialog.session.mask.any())

        dialog.close()


if __name__ == '__main__':
    unittest.main()
