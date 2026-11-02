"""Manual body-region review, experimental LoG and area measurements."""
from datetime import datetime
import copy
import json
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt, QRegularExpression, QTimer
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector

from spinal_ct.body_region import propose_plane, sample_plane, polygon_mask, plane_parameters
from spinal_ct.blob_detection import detect_blobs
from spinal_ct.morphometry import BodySession, load_session, save_session, LANDMARK_NAMES
from spinal_ct.morphometry_render import draw_body, draw_normalized, draw_convergence
from spinal_ct.area_monte_carlo import validate_sampling
from apps.ui_fonts import ui_font
from apps.tasks import Task

LANDMARK_LABELS = ('Trước – trên', 'Trước – dưới', 'Giữa – trên', 'Giữa – dưới', 'Sau – trên', 'Sau – dưới')
WARNINGS = dict(small_region='Vùng đo quá nhỏ', touches_plane_boundary='Vùng đo chạm biên mặt phẳng',
                multiple_components='Có nhiều vùng rời: cần kiểm tra',
                ellipse_orientation_unstable='Hai trục gần bằng nhau: hướng ellipse chưa ổn định',
                touches_ct_field_boundary='Vùng đo chạm giới hạn CT',
                height_landmarks_missing='Chưa đặt 6 mốc chiều cao', age_group_unspecified='Chưa xác định nhóm tuổi')


class BodyReviewDialog(QDialog):
    def __init__(self, source, labels, fingerprint, level, source_kind, output_root, parent=None):
        super().__init__(parent)
        self.setFont(QFont(ui_font(), 10))
        self.source, self.labels, self.fingerprint = source, labels, fingerprint
        self.level, self.source_kind, self.output_root = level, source_kind, Path(output_root)
        self.session = None
        self.selector = None
        self.history = []
        self.worker = None
        self.pending_plane = False
        self.saved_state = None
        self.last_saved_path = None
        self.last_saved_reviewed = False
        self.timings = []
        self._closing = False
        self._busy_started = None
        self._busy_message = ''
        self.mode = 'view'
        self.brushing = False
        self.blob_extras = dict(scales=8, max_candidates=30, window_hu=[-200., 1500.])
        self.setWindowTitle(f'Đo thân đốt · {level} · {source_kind}')
        self.resize(1380, 900)
        self.setMinimumSize(1050, 650)
        outer = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        panel = QWidget()
        self.controls_panel = panel
        controls = QVBoxLayout(panel)
        intro = QLabel('Vẽ đường bao thân đốt, gồm phần bên trong; loại cấu trúc phía sau. '
                       'Đường nét đứt xanh chỉ là mask toàn đốt để tham khảo.')
        intro.setWordWrap(True)
        controls.addWidget(intro)
        self.state_label = QLabel('Chưa kiểm tra')
        self.state_label.setWordWrap(True)
        controls.addWidget(self.state_label)
        self.tool_label = QLabel('Công cụ: Xem')
        controls.addWidget(self.tool_label)
        self.plane_note = QLabel('1. Chọn mặt phẳng đi qua thân đốt')
        self.plane_note.setWordWrap(True)
        controls.addWidget(self.plane_note)
        self.plane_toggle = QPushButton('+ Điều chỉnh mặt phẳng')
        self.plane_toggle.setCheckable(True)
        controls.addWidget(self.plane_toggle)
        self.plane_panel = QWidget()
        plane_layout = QVBoxLayout(self.plane_panel)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.offset = self._double(-80, 80, 0, ' mm')
        self.yaw = self._double(-45, 45, 0, '°')
        self.tilt = self._double(-45, 45, 0, '°')
        form.addRow('Dịch mặt phẳng', self.offset)
        form.addRow('Xoay quanh trục đứng', self.yaw)
        form.addRow('Nghiêng mặt phẳng', self.tilt)
        plane_layout.addLayout(form)
        self.apply_plane = QPushButton('Tạo / áp dụng mặt phẳng')
        self.apply_plane.clicked.connect(self.rebuild_plane)
        plane_layout.addWidget(self.apply_plane)
        self.reset_plane = QPushButton('Giữ mặt phẳng hiện tại')
        self.reset_plane.clicked.connect(self.restore_plane_controls)
        plane_layout.addWidget(self.reset_plane)
        controls.addWidget(self.plane_panel)
        self.plane_panel.hide()
        self.plane_toggle.toggled.connect(self.plane_panel.setVisible)
        controls.addWidget(QLabel('2. Vẽ / chỉnh vùng thân đốt'))
        for widget in (self.offset, self.yaw, self.tilt):
            widget.valueChanged.connect(self.plane_dirty)
        self.add_button = QPushButton('Vẽ / thêm vùng')
        self.add_button.clicked.connect(lambda: self.start_polygon(False))
        controls.addWidget(self.add_button)
        self.remove_button = QPushButton('Xóa một phần vùng')
        self.remove_button.clicked.connect(lambda: self.start_polygon(True))
        controls.addWidget(self.remove_button)
        self.brush_button = QPushButton('Cọ thêm (chuột trái) / xóa (chuột phải)')
        self.brush_button.clicked.connect(self.start_brush)
        controls.addWidget(self.brush_button)
        self.radius = self._double(.5, 10, 2, ' mm')
        controls.addWidget(self.radius)
        self.undo_button = QPushButton('Hoàn tác · Ctrl+Z')
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setShortcut('Ctrl+Z')
        controls.addWidget(self.undo_button)
        self.clear_button = QPushButton('Xóa vùng và mốc để vẽ lại')
        self.clear_button.clicked.connect(self.clear_region)
        controls.addWidget(self.clear_button)
        self.landmark_toggle = QPushButton('+ Đo chiều cao · 6 mốc tùy chọn')
        self.landmark_toggle.setCheckable(True)
        controls.addWidget(self.landmark_toggle)
        self.landmark_panel = QWidget()
        landmark_layout = QVBoxLayout(self.landmark_panel)
        landmark_layout.setContentsMargins(0, 0, 0, 0)
        self.landmark_button = QPushButton('Đặt lại 6 mốc chiều cao')
        self.landmark_button.clicked.connect(self.start_landmarks)
        landmark_layout.addWidget(self.landmark_button)
        self.clear_landmarks = QPushButton('Bỏ các mốc chiều cao')
        self.clear_landmarks.clicked.connect(self.remove_landmarks)
        landmark_layout.addWidget(self.clear_landmarks)
        self.landmark_choice = QComboBox()
        self.landmark_choice.addItems(LANDMARK_LABELS)
        landmark_layout.addWidget(self.landmark_choice)
        self.edit_landmark = QPushButton('Sửa mốc đã chọn')
        self.edit_landmark.clicked.connect(self.start_edit_landmark)
        landmark_layout.addWidget(self.edit_landmark)
        controls.addWidget(self.landmark_panel)
        self.landmark_panel.hide()
        self.landmark_toggle.toggled.connect(self.landmark_panel.setVisible)
        self.experiment_panel = QWidget()
        experiment_controls = QVBoxLayout(self.experiment_panel)
        experiment_controls.setContentsMargins(0, 0, 0, 0)
        experiment_controls.addWidget(QLabel('LoG thử nghiệm · sigma theo mm'))
        log_form = QFormLayout()
        self.log_min = self._double(.2, 40, 2, ' mm')
        self.log_max = self._double(.2, 60, 12, ' mm')
        self.log_threshold = self._double(.001, 1, .03)
        self.log_threshold.setDecimals(3)
        self.log_threshold.setSingleStep(.01)
        log_form.addRow('Sigma nhỏ nhất', self.log_min)
        log_form.addRow('Sigma lớn nhất', self.log_max)
        log_form.addRow('Ngưỡng đáp ứng', self.log_threshold)
        experiment_controls.addLayout(log_form)
        self.log_button = QPushButton('Tìm ứng viên blob trong ROI')
        self.log_button.clicked.connect(self.find_blobs)
        experiment_controls.addWidget(self.log_button)
        self.blob_choices = QComboBox()
        self.blob_choices.currentIndexChanged.connect(self.candidate_changed)
        experiment_controls.addWidget(self.blob_choices)
        self.show_blobs = QCheckBox('Hiện ứng viên LoG')
        self.show_blobs.setChecked(False)
        self.show_blobs.toggled.connect(self.redraw)
        experiment_controls.addWidget(self.show_blobs)
        self.log_status = QLabel('Chưa chạy LoG trên mặt phẳng này.')
        self.log_status.setWordWrap(True)
        experiment_controls.addWidget(self.log_status)
        for widget in (self.log_min, self.log_max, self.log_threshold):
            widget.valueChanged.connect(self.log_parameters_changed)
        controls.addWidget(QLabel('3. Xác nhận, đo và lưu'))
        self.age = QComboBox()
        for title, value in (('Chưa xác định nhóm tuổi', 'unspecified'), ('Người lớn', 'adult'), ('Trẻ em', 'pediatric')):
            self.age.addItem(title, value)
        self.age.currentIndexChanged.connect(self.age_changed)
        controls.addWidget(self.age)
        self.confirm = QCheckBox('Xác nhận vùng đo và các mốc')
        self.confirm.setToolTip('Kiểm tra bằng mắt trước khi xác nhận. Thay đổi vùng/mốc sẽ hủy xác nhận và phép đo cũ.')
        self.confirm.toggled.connect(self.approve)
        controls.addWidget(self.confirm)
        self.sampling_toggle = QPushButton('+ Cấu hình MC diện tích')
        self.sampling_toggle.setCheckable(True)
        controls.addWidget(self.sampling_toggle)
        self.sampling_panel = QWidget()
        mc_form = QFormLayout(self.sampling_panel)
        self.samples = QSpinBox()
        self.samples.setRange(100, 10_000_000)
        self.samples.setValue(100_000)
        self.samples.setSingleStep(10_000)
        self.seed = QLineEdit('42')
        self.seed.setValidator(QRegularExpressionValidator(QRegularExpression('[0-9]{1,19}'), self.seed))
        mc_form.addRow('MC diện tích · số điểm', self.samples)
        mc_form.addRow('Seed', self.seed)
        controls.addWidget(self.sampling_panel)
        self.sampling_panel.hide()
        self.sampling_toggle.toggled.connect(self.sampling_panel.setVisible)
        self.samples.valueChanged.connect(self.mc_changed)
        self.seed.textChanged.connect(self.mc_changed)
        self.measure_button = QPushButton('Đo thân đốt')
        self.measure_button.setObjectName('primary')
        self.measure_button.clicked.connect(self.measure)
        controls.addWidget(self.measure_button)
        self.show_points = QCheckBox('Hiện tối đa 1.000 điểm MC')
        self.show_points.toggled.connect(self.redraw)
        controls.addWidget(self.show_points)
        self.save_button = QPushButton('Lưu vùng nháp / kết quả')
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        self.save_as_button = QPushButton('Lưu vào thư mục khác…')
        self.save_as_button.clicked.connect(self.save_as)
        controls.addWidget(self.save_as_button)
        self.open_button = QPushButton('Mở vùng thân đốt đã lưu')
        self.open_button.clicked.connect(self.open_saved)
        controls.addWidget(self.open_button)
        self.experiment_toggle = QPushButton('+ Thử nghiệm LoG')
        self.experiment_toggle.setCheckable(True)
        self.experiment_toggle.toggled.connect(self.experiment_panel.setVisible)
        self.experiment_toggle.toggled.connect(lambda checked: self.experiment_toggle.setText(
            '- Thử nghiệm LoG' if checked else '+ Thử nghiệm LoG'))
        controls.addWidget(self.experiment_toggle)
        controls.addWidget(self.experiment_panel)
        self.experiment_panel.setVisible(False)
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)
        split.addWidget(scroll)
        right = QWidget()
        layout = QVBoxLayout(right)
        self.figure = Figure(layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 3)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Phép đo trên vùng gốc', 'Giá trị'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(45)
        layout.addWidget(self.status)
        split.addWidget(right)
        split.setSizes([360, 1020])
        self.canvas.mpl_connect('button_press_event', self.click)
        self.canvas.mpl_connect('motion_notify_event', self.motion)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.rebuild_plane()
        self.saved_state = self.state_token()
        self.sync()
        self.setStyleSheet('''
            QDialog,QWidget {background:#eef3f6;color:#233c4b;}
            QPushButton {background:white;border:1px solid #c5d5df;border-radius:5px;padding:6px;}
            QPushButton:hover {background:#e0f2f2;}
            QPushButton:disabled {color:#93a4ac;background:#e7edef;}
            QPushButton#primary {background:#136979;color:white;font-weight:600;}
            QPushButton#primary:disabled {background:#92adb2;}
            QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit {background:white;padding:4px;}
            QTableWidget {background:white;}
        ''')
        self.busy_timer = QTimer(self)
        self.busy_timer.timeout.connect(self.busy_tick)
        self.busy_timer.start(250)

    @staticmethod
    def _double(low, high, value, suffix=''):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def error(self, error):
        self.status.setText(str(error))
        QMessageBox.warning(self, 'Cần kiểm tra vùng đo', str(error))

    def stop_editing(self):
        if self.selector:
            self.selector.set_visible(False)
            self.selector.disconnect_events()
            self.selector = None
        self.mode = 'view'
        self.brushing = False
        self.tool_label.setText('Công cụ: Xem')

    def sync(self):
        active = self.session is not None and not self.pending_plane and self.worker is None
        nonempty = active and bool(self.session.mask.any())
        self.confirm.blockSignals(True)
        self.confirm.setChecked(self.session is not None and self.session.reviewed)
        self.confirm.blockSignals(False)
        self.confirm.setEnabled(nonempty)
        for widget in (self.add_button, self.remove_button, self.brush_button, self.log_button):
            widget.setEnabled(active)
        for widget in (self.clear_button, self.landmark_button, self.clear_landmarks, self.save_button):
            widget.setEnabled(nonempty)
        self.undo_button.setEnabled(active and bool(self.history))
        try:
            self.current_sampling()
            valid_sampling = True
        except ValueError:
            valid_sampling = False
        self.measure_button.setEnabled(active and self.session.reviewed and valid_sampling)
        can_save = self.worker is None and self.session is not None and bool(self.session.mask.any()) and valid_sampling
        self.save_button.setEnabled(can_save)
        self.save_as_button.setEnabled(can_save)
        has_blobs = active and self.session.candidates is not None
        self.blob_choices.setEnabled(has_blobs and self.blob_choices.count() > 0)
        self.show_blobs.setEnabled(has_blobs)
        self.edit_landmark.setEnabled(active and len(self.session.landmarks) == 6)
        self.reset_plane.setEnabled(self.pending_plane and self.worker is None)
        state = self.workflow_state()
        self.state_label.setText(state + (' · Chưa lưu thay đổi' if self.is_dirty() else ''))
        color = '#956000' if self.is_dirty() or 'cảnh báo' in state else '#136979'
        self.state_label.setStyleSheet(f'font-weight:600;color:{color};padding:6px')

    def current_sampling(self):
        return validate_sampling(self.samples.value(), int(self.seed.text()))

    def log_config(self):
        return dict(self.blob_extras, min_sigma_mm=self.log_min.value(), max_sigma_mm=self.log_max.value(),
                    threshold=self.log_threshold.value())

    def refresh_blob_choices(self):
        self.blob_choices.blockSignals(True)
        self.blob_choices.clear()
        for i, candidate in enumerate((self.session.candidates or {}).get('candidates', []), 1):
            x, y = candidate['centre_xy_mm']
            self.blob_choices.addItem(f'B{i}: ({x:.1f}, {y:.1f}) mm · sigma {candidate["sigma_mm"]:.1f}')
        self.blob_choices.blockSignals(False)

    def log_parameters_changed(self, *_):
        if self.session:
            self.session.set_candidates(None)
            self.refresh_blob_choices()
            self.log_status.setText('Thông số đã đổi. Bấm Tìm ứng viên để chạy lại LoG.')
            self.redraw()

    def plane_dirty(self, *_):
        self.stop_editing()
        self.pending_plane = True
        self.sync()
        self.status.setText('Chưa áp dụng mặt phẳng mới. Vùng hiện tại được giữ; áp dụng hoặc chọn Giữ mặt phẳng hiện tại.')

    def restore_plane_controls(self):
        for widget, value in zip((self.offset, self.yaw, self.tilt), self.applied_controls):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.pending_plane = False
        self.sync()
        self.status.setText('Đã giữ mặt phẳng và vùng đang chỉnh.')

    def rebuild_plane(self, *_):
        if self.worker is not None:
            return
        if self.session and self.session.mask.any():
            if not self.confirm_replace(self._rebuild_plane, force=True):
                return
        self._rebuild_plane()

    def _rebuild_plane(self):
        try:
            from spinal_ct.localization import LEVELS
            label_id = LEVELS.index(self.level)+1
            plane = propose_plane(self.source, self.labels, label_id, offset_mm=self.offset.value(),
                                  yaw_deg=self.yaw.value(), tilt_deg=self.tilt.value())
            ct, valid, hint = sample_plane(self.source, plane, self.labels, label_id)
            session = BodySession(self.fingerprint, self.level, self.source_kind, plane, ct, valid, hint,
                                  age_group=self.age.currentData())
            self.stop_editing()
            self.session = session
            self.pending_plane = False
            self.applied_controls = (self.offset.value(), self.yaw.value(), self.tilt.value())
            self.history.clear()
            self.blob_choices.clear()
            minimum_sigma = float(np.ceil(max(plane.spacing_xy_mm)*100)/100)
            for widget in (self.log_min, self.log_max):
                widget.blockSignals(True)
                widget.setMinimum(minimum_sigma)
                widget.blockSignals(False)
            self.log_status.setText('Chưa chạy LoG trên mặt phẳng này.')
            self.plane_note.setText('1. Mặt phẳng sagittal từ ROI · có thể điều chỉnh')
            self.status.setText('Kiểm tra mặt phẳng, rồi vẽ đường bao thân đốt. Nhấp đúp hoặc chọn lại điểm đầu để khép vùng.')
        except (ValueError, RuntimeError) as error:
            self.error(error)
        self.redraw()

    def remember(self):
        self.history.append((self.session.mask.copy(), list(self.session.landmarks)))
        self.history = self.history[-20:]

    def undo(self):
        self.stop_editing()
        if self.session and self.history:
            mask, points = self.history.pop()
            self.session.set_mask(mask)
            self.session.set_landmarks(points)
            self.redraw()
            self.status.setText('Đã hoàn tác. Kiểm tra và xác nhận lại vùng đo.')

    def clear_region(self):
        if self.session:
            self.stop_editing()
            self.remember()
            self.session.set_mask(np.zeros(self.session.mask.shape, bool))
            self.redraw()
            self.status.setText('Đã xóa vùng đo và các mốc. Có thể vẽ lại.')

    def start_polygon(self, subtract=False):
        if not self.session:
            return
        self.stop_editing()
        self.mode = 'polygon'
        self.tool_label.setText('Công cụ: '+('Xóa vùng' if subtract else 'Vẽ / thêm vùng'))
        self.status.setText('Chọn các điểm đường bao; nhấp đúp / chọn lại điểm đầu để khép vùng. Esc để vẽ lại.')
        self.selector = PolygonSelector(self.body_ax, lambda vertices: self.apply_polygon(vertices, subtract),
                                        useblit=True, props=dict(color='#ff7bcc', linewidth=1.5))

    def apply_polygon(self, vertices, subtract=False):
        try:
            region = polygon_mask(vertices, self.session.mask.shape, self.session.plane.spacing_xy_mm)
            updated = self.session.mask & ~region if subtract else self.session.mask | region
            self.remember()
            self.session.set_mask(updated)
            self.stop_editing()
            self.redraw()
            self.status.setText('Đã sửa vùng đo; hãy kiểm tra và xác nhận lại trước khi đo.')
        except ValueError as error:
            self.stop_editing()
            self.error(error)
            self.redraw()

    def start_brush(self):
        if self.session:
            self.stop_editing()
            self.mode = 'brush'
            self.tool_label.setText('Công cụ: Cọ thêm / xóa')
            self.status.setText('Cọ: giữ chuột trái để thêm, chuột phải để xóa. Bán kính theo mm.')

    def start_landmarks(self):
        if self.session:
            self.stop_editing()
            self.remember()
            self.session.set_landmarks([])
            self.mode = 'landmark'
            self.tool_label.setText('Công cụ: Đặt mốc')
            self.redraw()
            self.status.setText('Mốc 1/6: '+LANDMARK_LABELS[0])

    def remove_landmarks(self):
        if self.session:
            self.stop_editing()
            self.remember()
            self.session.set_landmarks([])
            self.redraw()

    def _on_image(self, event):
        if self.worker is not None or self.pending_plane or not self.session or event.inaxes != self.body_ax or event.xdata is None or event.ydata is None:
            return False
        sx, sy = self.session.plane.spacing_xy_mm
        h, w = self.session.mask.shape
        return -sx/2 <= event.xdata <= (w-.5)*sx and -sy/2 <= event.ydata <= (h-.5)*sy

    def click(self, event):
        if not self._on_image(event):
            return
        if self.mode == 'edit_landmark' and event.button == 1:
            self.remember()
            points = list(self.session.landmarks)
            points[self.landmark_choice.currentIndex()] = [event.xdata, event.ydata]
            self.session.set_landmarks(points)
            self.stop_editing()
            self.redraw()
            self.status.setText('Đã sửa mốc. Kiểm tra và xác nhận lại vùng đo.')
        elif self.mode == 'landmark' and event.button == 1:
            try:
                self.session.set_landmarks(self.session.landmarks + [[event.xdata, event.ydata]])
                count = len(self.session.landmarks)
                self.redraw()
                if count < len(LANDMARK_NAMES):
                    self.status.setText(f'Mốc {count+1}/6: '+LANDMARK_LABELS[count])
                else:
                    self.stop_editing()
                    self.status.setText('Đủ 6 mốc. Kiểm tra thứ tự và xác nhận vùng đo.')
            except ValueError as error:
                self.error(error)
        elif self.mode == 'brush' and event.button in (1, 3):
            self.remember()
            self.brushing = event.button
            self.paint(event)

    def paint(self, event):
        if not self._on_image(event):
            return
        sx, sy = self.session.plane.spacing_xy_mm
        rows, cols = np.indices(self.session.mask.shape)
        region = (cols*sx-event.xdata)**2 + (rows*sy-event.ydata)**2 <= self.radius.value()**2
        mask = self.session.mask | (region & self.session.valid) if self.brushing == 1 else self.session.mask & ~region
        self.session.set_mask(mask)
        self.redraw()

    def motion(self, event):
        if self.mode == 'brush' and self.brushing:
            self.paint(event)

    def release(self, event):
        self.brushing = False

    def age_changed(self, *_):
        if self.session:
            self.session.age_group = self.age.currentData()
            self.session.invalidate()
            self.redraw()
            self.status.setText('Đã đổi nhóm tuổi. Kiểm tra và xác nhận lại vùng đo.')

    def approve(self, checked):
        if not self.session:
            return
        self.stop_editing()
        try:
            if checked:
                self.session.approve()
                self.status.setText('Đã xác nhận vùng hiện tại. Có thể tính đặc trưng và MC diện tích.')
            else:
                self.session.invalidate()
        except ValueError as error:
            self.error(error)
        self.redraw()

    def mc_changed(self, *_):
        if self.session:
            self.session.result = None
            self.redraw()
            self.status.setText('Đã đổi số điểm / seed. Bấm Đo để tính lại Monte Carlo diện tích.')

    def start_edit_landmark(self):
        if self.worker is None and self.session and len(self.session.landmarks) == 6:
            self.stop_editing()
            self.mode = 'edit_landmark'
            self.tool_label.setText('Công cụ: Sửa mốc')
            self.status.setText('Nhấp vị trí mới cho mốc: '+self.landmark_choice.currentText())

    def busy_tick(self):
        if self.worker is not None:
            self.status.setText(f'{self._busy_message} · {time.perf_counter()-self._busy_started:.1f} giây')

    def run_task(self, function, callback, message):
        if self.worker is not None:
            return
        self.stop_editing()
        self._busy_message = message
        self._busy_started = time.perf_counter()
        self.worker = Task(function, self)
        self._task_callback = callback
        self.controls_panel.setEnabled(False)
        self.sync()
        self.worker.finished.connect(self.task_done)
        self.busy_tick()
        self.worker.start()

    def task_done(self):
        task, callback = self.worker, self._task_callback
        self.worker = None
        self.controls_panel.setEnabled(True)
        self.timings.append(dict(stage=self._busy_message, seconds=round(task.elapsed, 4)))
        if task.error:
            self.error(task.error)
        else:
            try:
                callback(task.value)
                self.status.setText(self.status.text()+f' · {task.elapsed:.2f} giây')
            except Exception as error:
                self.error(error)
        task.deleteLater()
        self.sync()

    def measure(self):
        if self.worker is not None or self.pending_plane or not self.session:
            return
        try:
            config = self.current_sampling()
            snapshot = copy.deepcopy(self.session)
            def compute():
                snapshot.measure(**config)
                return snapshot
            def accepted(session):
                self.session = session
                warnings = [WARNINGS.get(w, w) for w in session.result['warnings']]
                self.status.setText('Đã đo. '+ ('; '.join(warnings) if warnings else 'Có thể lưu kết quả.'))
                self.redraw()
            self.run_task(compute, accepted, 'Đang đo hình dạng và diện tích')
        except (ValueError, RuntimeError) as error:
            self.error(error)

    def find_blobs(self):
        if self.worker is not None or self.pending_plane or not self.session:
            return
        snapshot, config = self.session, self.log_config()
        def accepted(detection):
            self.session.set_candidates(detection)
            self.refresh_blob_choices()
            self.show_blobs.setChecked(True)
            self.redraw()
            self.log_status.setText(f'Đã chạy LoG: {self.blob_choices.count()} ứng viên.')
            self.status.setText('Chọn tâm tham khảo nếu cần; vòng tròn LoG biểu diễn thang đo.')
        self.run_task(lambda: detect_blobs(snapshot.ct, snapshot.plane.spacing_xy_mm, snapshot.valid, **config),
                      accepted, 'Đang tìm ứng viên LoG')

    def candidate_changed(self, *_):
        self.redraw()

    def save(self, checked=False, *, target_dir=None, after=None):
        if self.worker is not None or not self.session:
            return
        try:
            config = self.current_sampling()
            parent = Path(target_dir) if target_dir else Path(self.output_root)
            parent.mkdir(parents=True, exist_ok=True)
            output = parent/f'{self.level}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
            snapshot = copy.deepcopy(self.session)
            def compute():
                save_session(snapshot, output, **config)
                return snapshot
            def accepted(session):
                self.session = session
                self.last_saved_path = output
                self.last_saved_reviewed = session.reviewed
                self.saved_state = self.state_token()
                self.status.setText(f'Đã lưu {"kết quả đo" if session.reviewed else "vùng nháp"}: {output.name}')
                self.redraw()
                if after:
                    after()
            self.run_task(compute, accepted, 'Đang lưu vùng và xuất kết quả')
        except (ValueError, OSError, RuntimeError) as error:
            self.error(error)

    def save_as(self):
        selected = QFileDialog.getExistingDirectory(self, 'Chọn thư mục lưu kết quả', str(self.output_root))
        if selected:
            self.save(target_dir=Path(selected))

    def open_saved(self):
        if not self.confirm_replace(self._choose_saved):
            return
        self._choose_saved()

    def _choose_saved(self):
        selected = QFileDialog.getExistingDirectory(self, 'Chọn thư mục chứa region.json', str(self.output_root))
        if selected:
            try:
                self.load_region(Path(selected))
            except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
                self.error(error)

    def load_region(self, folder):
        session = load_session(folder, self.source, fingerprint=self.fingerprint)
        if session.level != self.level or session.source_kind != self.source_kind:
            raise ValueError('Vùng đã lưu thuộc đốt hoặc nguồn mask khác. Hãy chọn đúng đốt và Baseline/Consensus trong demo.')
        from spinal_ct.localization import LEVELS
        _, _, session.hint = sample_plane(self.source, session.plane, self.labels, LEVELS.index(self.level)+1)
        try:
            parameters = plane_parameters(self.source, self.labels, LEVELS.index(self.level)+1, session.plane)
        except ValueError:
            parameters = None
        self.stop_editing()
        self.session = session
        self.pending_plane = False
        self.history.clear()
        self.refresh_blob_choices()
        for widget, key in ((self.offset, 'offset_mm'), (self.yaw, 'yaw_deg'), (self.tilt, 'tilt_deg')):
            widget.blockSignals(True)
            widget.setValue(parameters[key] if parameters else 0.)
            widget.blockSignals(False)
        self.applied_controls = (self.offset.value(), self.yaw.value(), self.tilt.value())
        self.plane_note.setText('1. Mặt phẳng đã tải; các thông số được khôi phục' if parameters else
                               '1. Mặt phẳng tùy chỉnh đã tải; các ô bên dưới chỉ dùng để tạo mới')
        config = (session.candidates or {}).get('config')
        for widget in (self.log_min, self.log_max):
            widget.blockSignals(True)
            widget.setMinimum(float(np.ceil(max(session.plane.spacing_xy_mm)*100)/100))
            widget.blockSignals(False)
        displayed = config or dict(scales=8, max_candidates=30, window_hu=[-200., 1500.],
                                   min_sigma_mm=2., max_sigma_mm=12., threshold=.03)
        self.blob_extras = {key: displayed[key] for key in ('scales', 'max_candidates', 'window_hu')}
        for widget, key in ((self.log_min, 'min_sigma_mm'), (self.log_max, 'max_sigma_mm'), (self.log_threshold, 'threshold')):
            widget.blockSignals(True)
            widget.setMaximum(max(widget.maximum(), displayed[key]))
            widget.setValue(displayed[key])
            widget.blockSignals(False)
        self.log_status.setText('Đã khôi phục thông số và ứng viên LoG đã lưu.' if config else 'Vùng đã lưu chưa có kết quả LoG.')
        self.samples.blockSignals(True)
        self.samples.setValue(session.sampling_config['samples'])
        self.samples.blockSignals(False)
        self.seed.blockSignals(True)
        self.seed.setText(str(session.sampling_config['seed']))
        self.seed.blockSignals(False)
        self.age.blockSignals(True)
        self.age.setCurrentIndex(self.age.findData(session.age_group))
        self.age.blockSignals(False)
        self.last_saved_path = Path(folder)
        self.last_saved_reviewed = session.reviewed
        self.saved_state = self.state_token()
        self.status.setText('Đã mở vùng, kiểm tra khớp CT. Phép đo sẽ được tính lại khi bấm Đo. '
                            'Số điểm và seed được khôi phục theo bản lưu.')
        self.redraw()

    def redraw(self, *_):
        # A polygon selector owns its axes until completion; don't replace them.
        if self.selector:
            self.sync()
            return
        self.figure.clear()
        axes = self.figure.subplots(1, 3)
        self.body_ax, normalized_ax, convergence_ax = axes
        self.table.setRowCount(0)
        if self.session:
            draw_body(self.body_ax, self.session, show_points=self.show_points.isChecked(), show_blobs=self.show_blobs.isChecked())
            draw_normalized(normalized_ax, self.session)
            candidates = (self.session.candidates or {}).get('candidates', [])
            index = self.blob_choices.currentIndex()
            if self.show_blobs.isChecked() and 0 <= index < len(candidates):
                self.body_ax.plot(*candidates[index]['centre_xy_mm'], '+', color='#fc64f0', markersize=15, markeredgewidth=2)
            report = self.session.result if self.session.reviewed else None
            if report and 'monte_carlo_area' in report and 'convergence' in report['monte_carlo_area']:
                draw_convergence(convergence_ax, report)
            else:
                convergence_ax.text(.5, .5, 'Xác nhận và đo\nđể xem hội tụ MC', ha='center', va='center',
                                     transform=convergence_ax.transAxes, color='#94aebc', fontsize=10)
                convergence_ax.set_axis_off()
            if report:
                f, mc = report['features'], report['monte_carlo_area']
                values = [('Diện tích đếm pixel (mm²)', f['area_mm2']), ('Cao/rộng theo mặt phẳng', f['height_width_ratio']),
                          ('Tỷ lệ lấp đầy hộp bao', f['fill_ratio']), ('Solidity', f['solidity']),
                          ('Góc ellipse (độ)', f['ellipse']['angle_deg']), ('Tỷ lệ trục ellipse', f['ellipse']['axis_ratio'])]
                for name, key in (('Chiều cao trước (mm)', 'anterior_mm'), ('Chiều cao giữa (mm)', 'middle_mm'),
                                  ('Chiều cao sau (mm)', 'posterior_mm'), ('Trước/sau', 'anterior_posterior_ratio'),
                                  ('Giữa/sau', 'middle_posterior_ratio')):
                    values.append((name, (report['heights'] or {}).get(key, 'Chưa đặt mốc')))
                low, high = mc['sampling_interval95_mm2']
                values += [('MC diện tích (mm²)', mc['area_mm2']), ('Sai lệch MC so với đếm pixel (mm²)', mc['error_vs_pixel_area_mm2']),
                           ('Khoảng lấy mẫu 95% MC (mm²)', f'{low:.2f} – {high:.2f}'), ('Số điểm / seed', f'{mc["samples"]} / {mc["seed"]}')]
                if 'elbow' in mc:
                    elbow = mc['elbow']
                    values.append(('MC điểm cùi chỏ (mm²)', f"{elbow['area_mm2']:.4f} (N={elbow['samples']:,})"))
                self.table.setRowCount(len(values))
                for row, (name, value) in enumerate(values):
                    self.table.setItem(row, 0, QTableWidgetItem(name))
                    self.table.setItem(row, 1, QTableWidgetItem(f'{value:,.4f}' if isinstance(value, (float, int)) else value))
        else:
            for ax in axes:
                ax.set_axis_off()
            self.body_ax.text(.5, .5, 'Áp dụng mặt phẳng để chọn vùng mới', ha='center', va='center', transform=self.body_ax.transAxes)
        self.canvas.draw_idle()
        self.sync()

    def state_token(self):
        if not self.session:
            return None
        return (self.session._signature(), self.session.reviewed, self.samples.value(), self.seed.text(),
                json.dumps(self.log_config(), sort_keys=True))

    def is_dirty(self):
        return self.state_token() != self.saved_state

    def workflow_state(self):
        if self.last_saved_path and not self.is_dirty():
            return 'Đã lưu · '+('Đã xác nhận' if self.session.reviewed else 'Vùng nháp')
        if self.session and self.session.reviewed:
            if self.session.result and self.session.result['warnings']:
                return 'Đã xác nhận · Có cảnh báo'
            return 'Đã xác nhận'
        return 'Chưa kiểm tra'

    def confirm_replace(self, continuation, *, force=False):
        if self.worker is not None:
            self.status.setText('Đang xử lý. Hãy chờ tác vụ hoàn tất.')
            return False
        if not force and not self.is_dirty():
            return True
        choice = QMessageBox.question(self, 'Giữ vùng đang chỉnh',
            'Lưu vùng hiện tại trước khi tiếp tục? Bạn có thể lưu nháp khi chưa xác nhận.',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if choice == QMessageBox.Save:
            self.save(after=continuation)
            return False
        return choice == QMessageBox.Discard

    def _finish_close(self):
        self.stop_editing()
        self._closing = True
        super().reject()

    def reject(self):
        if self._closing:
            super().reject()
        elif self.confirm_replace(self._finish_close):
            self._finish_close()

    def closeEvent(self, event):
        event.ignore()
        self.reject()
