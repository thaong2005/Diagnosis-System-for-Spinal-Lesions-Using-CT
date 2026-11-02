"""Desktop review workflow: choose CT, run localization, inspect saved results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from uuid import uuid4

import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from spinal_ct.ct_io import list_ct_series, load_ct, select_sample_series, extract_patient_identifier, slugify
from spinal_ct.review import ct_fingerprint, display_image, load_review, report_html
from spinal_ct.overlays import draw_vertebral_overlay, vertebral_palette
from apps.ui_fonts import ui_font
from apps.tasks import Task
from spinal_ct.result_catalog import discover_results, compatible_result, discover_regions

ROOT = Path(__file__).resolve().parents[1]
# Keep patient-specific paths in the ignored local configuration.
try:
    from sample_config import SAMPLE_DIR
except ModuleNotFoundError as exc:
    if exc.name != 'sample_config':
        raise
    SAMPLE_DIR = ROOT / 'data/raw/sample_patient'


def read_study(path, uid):
    source = load_ct(path, uid)
    reference = display_image(source)
    fingerprint = ct_fingerprint(source)
    patient = extract_patient_identifier(path, series_uid=uid, fingerprint=fingerprint)
    search_roots = [
        ROOT/'data/output'/patient['slug']/'dinh_vi_cot_song',
        ROOT/'data/output',
        ROOT/'data/demo_runs',
        ROOT/'data/processed'
    ]
    saved, unreadable = discover_results(search_roots, fingerprint)
    region_roots = [
        ROOT/'data/output'/patient['slug']/'do_than_dot',
        ROOT/'data/output',
        ROOT/'data/body_measurements'
    ]
    saved_regions = discover_regions(region_roots, fingerprint)
    return dict(source=source, reference=reference, data=sitk.GetArrayFromImage(reference),
                fingerprint=fingerprint, path=path, uid=uid, saved_results=saved, unreadable_results=unreadable,
                saved_regions=saved_regions, patient=patient)


def check_backend():
    backend = ROOT/'.venv-model/Scripts/TotalSegmentator.exe'
    if not backend.is_file():
        return False, 'Thiếu backend. Xem hướng dẫn cài đặt Windows trong docs/guides.'
    try:
        result = subprocess.run([str(backend), '--version'], capture_output=True,
                                text=True, timeout=60,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or result.stdout.strip().splitlines()[-1] != '2.18.0':
            return False, 'Backend chưa sẵn sàng hoặc khác phiên bản 2.18.0.'
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return False, 'Không khởi động được backend. Kiểm tra .venv-model.'
    weights = ROOT/'.tmp/totalsegmentator'
    cached = weights.exists() and any(weights.rglob('*.pth'))
    note = 'Đã tìm thấy cache trọng số.' if cached else 'Lần chạy đầu có thể cần mạng để tải trọng số.'
    return True, 'TotalSegmentator 2.18.0 · CPU\n'+note


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.font_family = ui_font()
        QApplication.instance().setFont(QFont(self.font_family, 10))
        self.setWindowTitle('Spinal CT · Research Review')
        self.resize(1500, 960)
        self.study = None
        self.review = None
        self.thread = None
        self.process = None
        self.backend_ready = False
        self.pending_path = None
        self.run_output = None
        self.cancel_path = None
        self.stdout_buffer = ''
        self.started = None
        self.saved_results = []
        self.current_predictions = []
        self.region_states = {}
        self.region_paths = {}
        self.timings = []
        self.active_stage = 'Chọn CT hoặc mở ca mẫu để bắt đầu.'
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)
        self._sync()
        QTimer.singleShot(100, self.verify_environment)

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 16)
        header = QHBoxLayout()
        title = QLabel('SPINAL CT  /  Research Review')
        title.setObjectName('title')
        header.addWidget(title)
        header.addStretch()
        help_button = QPushButton('Hướng dẫn')
        help_button.clicked.connect(self.show_help)
        header.addWidget(help_button)
        outer.addLayout(header)
        subtitle = QLabel('Chọn ảnh CT • Kiểm tra phân đoạn • Đối chiếu kết quả trên ba mặt phẳng')
        subtitle.setObjectName('muted')
        outer.addWidget(subtitle)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(390)
        controls = QVBoxLayout(left)
        controls.setContentsMargins(0, 12, 16, 0)
        controls.addWidget(QLabel('01  DỮ LIỆU CT'))
        self.file_button = QPushButton('Chọn file CT (.nii / .nii.gz / .npy)')
        self.file_button.clicked.connect(self.choose_file)
        self.file_button.setText('Mở CT…')
        self.file_button.clicked.disconnect()
        menu = QMenu(self.file_button)
        menu.addAction('File NIfTI / NPY…', self.choose_file)
        menu.addAction('Thư mục DICOM…', self.choose_dicom)
        self.file_button.setMenu(menu)
        controls.addWidget(self.file_button)
        self.dicom_button = QPushButton('Chọn thư mục DICOM')
        self.dicom_button.clicked.connect(self.choose_dicom)
        self.dicom_button.hide()
        self.sample_button = QPushButton(f'Mở ca mẫu · {SAMPLE_DIR.name.title()}')
        self.sample_button.setToolTip(str(SAMPLE_DIR))
        self.sample_button.clicked.connect(self.open_sample)
        controls.addWidget(self.sample_button)
        self.patient_row = QHBoxLayout()
        self.patient_label = QLabel('Bệnh nhân: Chưa chọn')
        self.patient_label.setStyleSheet('font-weight: bold; color: #123e50;')
        self.patient_label.setWordWrap(True)
        self.patient_row.addWidget(self.patient_label, 1)
        self.rename_patient_button = QPushButton('Đổi tên…')
        self.rename_patient_button.setFixedWidth(70)
        self.rename_patient_button.clicked.connect(self.rename_patient)
        self.rename_patient_button.setEnabled(False)
        self.patient_row.addWidget(self.rename_patient_button)
        controls.addLayout(self.patient_row)
        self.source_label = QLabel()
        self.source_label.hide()
        self.series_combo = QComboBox()
        controls.addWidget(self.series_combo)
        self.load_button = QPushButton('Đọc series đã chọn')
        self.load_button.clicked.connect(self.load_series)
        controls.addWidget(self.load_button)
        self.geometry_label = QLabel('HU và geometry sẽ được kiểm tra trước khi chạy.')
        self.geometry_label.setWordWrap(True)
        controls.addWidget(self.geometry_label)
        controls.addSpacing(12)
        controls.addWidget(QLabel('02  XỬ LÝ'))
        self.fast = QCheckBox('Chạy nhanh · mô hình 3 mm')
        self.fast.setChecked(False)
        self.fast.setToolTip('Bỏ chọn để dùng mô hình 1,5 mm và định vị vùng cắt kỹ hơn. '
                             'Chạy chậm hơn và cần nhiều RAM hơn, nhất là khi bật Monte Carlo.')
        self.fast.hide()
        self.resolution = QComboBox()
        self.resolution.addItems(['Chi tiết · 1,5 mm', 'Nhanh · 3 mm'])
        self.resolution.currentIndexChanged.connect(lambda i: self.fast.setChecked(i == 1))
        self.fast.toggled.connect(lambda enabled: self.resolution.setCurrentIndex(int(enabled)))
        controls.addWidget(self.resolution)
        self.resolution_note = QLabel()
        self.resolution_note.setWordWrap(True)
        self.resolution_note.setObjectName('muted')
        controls.addWidget(self.resolution_note)
        self.fast.toggled.connect(self.update_resolution_note)
        self.update_resolution_note()
        self.mc_toggle = QPushButton('+ Thử nghiệm nhiễu HU')
        self.mc_toggle.setCheckable(True)
        controls.addWidget(self.mc_toggle)
        self.mc_panel = QWidget()
        mc_layout = QVBoxLayout(self.mc_panel)
        mc_layout.setContentsMargins(0, 0, 0, 0)
        self.mc = QCheckBox('Bổ sung Monte Carlo nhiễu HU')
        self.mc.toggled.connect(self._sync)
        self.mc.toggled.connect(lambda enabled: self.mc_toggle.setChecked(True) if enabled else None)
        mc_layout.addWidget(self.mc)
        form = QFormLayout()
        self.samples = QSpinBox()
        self.samples.setRange(2, 64)
        self.samples.setValue(2)
        self.noise = QDoubleSpinBox()
        self.noise.setRange(.1, 1000)
        self.noise.setValue(20)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        form.addRow('Số mẫu', self.samples)
        form.addRow('Nhiễu sigma (HU)', self.noise)
        form.addRow('Seed', self.seed)
        mc_layout.addLayout(form)
        controls.addWidget(self.mc_panel)
        self.mc_panel.setVisible(False)
        self.mc_toggle.toggled.connect(self.mc_panel.setVisible)
        self.mc_toggle.toggled.connect(self.update_mc_toggle)
        self.backend_label = QLabel('Đang kiểm tra backend…')
        self.backend_label.setWordWrap(True)
        self.backend_label.setObjectName('muted')
        self.environment_toggle = QPushButton('+ Chi tiết môi trường')
        self.environment_toggle.setCheckable(True)
        controls.addWidget(self.environment_toggle)
        controls.addWidget(self.backend_label)
        self.backend_label.hide()
        self.environment_toggle.toggled.connect(self.backend_label.setVisible)
        self.check_button = QPushButton('Kiểm tra môi trường')
        self.check_button.clicked.connect(self.verify_environment)
        controls.addWidget(self.check_button)
        self.check_button.hide()
        self.environment_toggle.toggled.connect(self.check_button.setVisible)
        self.run_button = QPushButton('Chạy định vị')
        self.run_button.setObjectName('primary')
        self.run_button.clicked.connect(self.start_run)
        controls.addWidget(self.run_button)
        self.stop_button = QPushButton('Dừng sau lượt xử lý hiện tại')
        self.stop_button.clicked.connect(self.request_stop)
        controls.addWidget(self.stop_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel('03  KẾT QUẢ'))
        self.saved_combo = QComboBox()
        self.saved_combo.setToolTip('Các kết quả có cùng dấu vân tay CT. Cấu hình khác được ghi rõ.')
        controls.addWidget(self.saved_combo)
        self.reuse_button = QPushButton('Tải lại lần chạy đã chọn')
        self.reuse_button.clicked.connect(self.reuse_result)
        controls.addWidget(self.reuse_button)
        self.open_button = QPushButton('Duyệt kết quả khác từ máy…')
        self.open_button.clicked.connect(self.choose_results)
        controls.addWidget(self.open_button)
        self.export_button = QPushButton('Xuất báo cáo HTML')
        self.export_button.clicked.connect(self.export_report)
        controls.addWidget(self.export_button)
        self.body_button = QPushButton('Đo thân đốt')
        self.body_button.clicked.connect(self.open_body_review)
        self.folder_button = QPushButton('Mở thư mục kết quả (Explorer)')
        self.folder_button.clicked.connect(self.open_result_folder)
        controls.addWidget(self.folder_button)
        self.body_folder_button = QPushButton('Mở thư mục đo thân đốt')
        self.body_folder_button.clicked.connect(self.open_body_measurements_folder)
        controls.addWidget(self.body_folder_button)
        controls.addStretch()
        note = QLabel('Mask toàn đốt sống · thể tích mm³\nĐo thân đốt trên vùng được duyệt riêng.\nKết quả cần được kiểm tra.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        controls.addWidget(note)
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setWidget(left)
        sidebar.setFrameShape(QScrollArea.NoFrame)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(410)
        split.addWidget(sidebar)

        vsplit = QSplitter(Qt.Vertical)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 8, 0, 0)
        toolbar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem('Chỉ CT')
        self.mode.currentTextChanged.connect(self.mode_changed)
        toolbar.addWidget(self.mode)
        self.overlay = QCheckBox('Hiện overlay')
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(self.redraw)
        toolbar.addWidget(self.overlay)
        self.show_names = QCheckBox('Tên đốt sống')
        self.show_names.setChecked(True)
        self.show_names.setToolTip('Tên và đường dẫn tới mask trên lát Sagittal / Coronal đang xem.')
        self.show_names.toggled.connect(self.redraw)
        toolbar.addWidget(self.show_names)
        toolbar.addWidget(QLabel('Độ đậm'))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 90)
        self.opacity.setValue(18)
        self.opacity.setMaximumWidth(100)
        self.opacity.valueChanged.connect(self.redraw)
        toolbar.addWidget(self.opacity)
        self.window_level = QSpinBox()
        self.window_level.setRange(-1000, 3000)
        self.window_level.setValue(400)
        self.window_level.valueChanged.connect(self.redraw)
        self.window_width = QSpinBox()
        self.window_width.setRange(1, 6000)
        self.window_width.setValue(1800)
        self.window_width.valueChanged.connect(self.redraw)
        toolbar.addWidget(QLabel('WL'))
        toolbar.addWidget(self.window_level)
        toolbar.addWidget(QLabel('WW'))
        toolbar.addWidget(self.window_width)
        top_layout.addLayout(toolbar)
        self.figure = Figure(facecolor='#111d29')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(240)
        top_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect('button_press_event', self.click_plane)
        self.canvas.mpl_connect('resize_event', self.redraw)
        sliders = QHBoxLayout()
        self.sliders = []
        self.slice_labels = []
        for title in ('Axial · z', 'Sagittal · x', 'Coronal · y'):
            column = QVBoxLayout()
            label = QLabel(title)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 0)
            slider.valueChanged.connect(self.redraw)
            column.addWidget(label)
            column.addWidget(slider)
            sliders.addLayout(column)
            self.sliders.append(slider)
            self.slice_labels.append(label)
        top_layout.addLayout(sliders)
        self.result_label = QLabel('Chọn CT để xem. Nhấp trên ảnh để đồng bộ vị trí ba mặt phẳng.')
        self.result_label.setWordWrap(True)
        top_layout.addWidget(self.result_label)
        navigation = QHBoxLayout()
        self.previous_button = QPushButton('← Đốt trước')
        self.next_button = QPushButton('Đốt tiếp theo →')
        self.previous_button.clicked.connect(lambda: self.move_level(-1))
        self.next_button.clicked.connect(lambda: self.move_level(1))
        self.level_label = QLabel('Chọn đốt để kiểm tra')
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.level_label, 1)
        navigation.addWidget(self.body_button)
        self.body_button.setObjectName('primary')
        navigation.addWidget(self.next_button)
        top_layout.addLayout(navigation)

        tabs = QTabWidget()
        tabs.setMinimumHeight(130)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Đốt sống', 'Thể tích (mm³)', 'MC phát hiện', 'Cảnh báo', 'Vùng đo thân đốt'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.select_level)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(26)
        tabs.addTab(self.table, 'Các đốt sống')
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        tabs.addTab(self.warnings, 'Lưu ý kết quả')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        tabs.addTab(self.log, 'Tiến trình')

        vsplit.addWidget(top_widget)
        vsplit.addWidget(tabs)
        vsplit.setCollapsible(0, False)
        vsplit.setCollapsible(1, False)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        vsplit.setSizes([480, 260])

        split.addWidget(vsplit)
        split.setSizes([325, 1135])
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(5)
        self.progress.setTextVisible(False)
        outer.addWidget(self.progress)
        self.status = QLabel(self.active_stage)
        outer.addWidget(self.status)
        for control in (self.samples, self.noise, self.seed, self.window_level, self.window_width,
                        self.mode, self.series_combo):
            control.setMinimumHeight(32)
        self.setStyleSheet('''
            QMainWindow,QWidget {background:#eef3f6;color:#233c4b;font:10pt "Segoe UI";}
            QLabel#title {font-size:20pt;font-weight:700;color:#123e50;}
            QLabel#muted {color:#56717f;font-size:9pt;}
            QPushButton {background:white;border:1px solid #c5d5df;border-radius:6px;padding:8px;}
            QPushButton:hover {background:#e0f2f2;border-color:#2f8c96;}
            QPushButton:disabled {color:#93a4ac;background:#e7edef;}
            QPushButton#primary {background:#136979;color:white;font-weight:700;border:0;}
            QPushButton#primary:disabled {background:#92adb2;}
            QComboBox,QSpinBox,QDoubleSpinBox {background:white;border:1px solid #c5d5df;padding:5px;border-radius:4px;}
            QPlainTextEdit,QTableWidget {background:white;border:1px solid #d4e0e6;}
            QHeaderView::section {background:#dfe9ee;padding:6px;border:0;}
            QProgressBar {border:0;background:#dfe9ee;} QProgressBar::chunk {background:#168a91;}
            QSlider::groove:horizontal {background:#cbdbe3;height:4px;}
            QSlider::handle:horizontal {background:#136979;width:12px;margin:-5px 0;border-radius:5px;}
            QSplitter::handle:vertical {background:#c8d8e2;height:6px;margin:1px 0;border-radius:3px;}
            QSplitter::handle:vertical:hover {background:#136979;}
            QTableWidget {alternate-background-color:#f5f8fa;}
            QTableWidget::item {padding:2px 4px;}
        '''.replace('Segoe UI', self.font_family))
        for control in (self.fast, self.mc, self.samples, self.noise, self.seed):
            signal = control.toggled if isinstance(control, QCheckBox) else control.valueChanged
            signal.connect(self.refresh_saved_results)
        self.axes = []
        self.redraw()

    def _busy(self):
        return self.thread is not None or self.process is not None

    def update_resolution_note(self, *_):
        self.resolution_note.setText(
            '3 mm · nhanh hơn; đường biên ít chi tiết.' if self.fast.isChecked() else
            '1,5 mm · ưu tiên chi tiết; chạy lâu hơn và cần nhiều RAM hơn.')

    def _sync(self, *_):
        busy = self._busy()
        for button in (self.file_button, self.dicom_button, self.sample_button, self.check_button):
            button.setEnabled(not busy)
        self.sample_button.setEnabled(not busy and SAMPLE_DIR.is_dir())
        self.load_button.setEnabled(not busy and self.series_combo.count() > 0)
        self.series_combo.setEnabled(not busy)
        self.series_combo.setVisible(self.series_combo.count() > 0)
        self.load_button.setVisible(self.series_combo.count() > 0)
        self.run_button.setEnabled(not busy and self.study is not None and self.backend_ready)
        self.open_button.setEnabled(not busy and self.study is not None)
        self.export_button.setEnabled(not busy and self.review is not None)
        self.rename_patient_button.setEnabled(not busy and self.study is not None)
        self.body_folder_button.setEnabled(not busy and self.study is not None)
        self.body_button.setEnabled(not busy and self.review is not None
                                    and self.mode.currentText() in ('Baseline', 'Consensus') and self.table.rowCount() > 0)
        self.mode.setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.folder_button.setEnabled(self.review is not None)
        self.stop_button.setVisible(self.process is not None)
        self.stop_button.setEnabled(self.process is not None and not (self.cancel_path and self.cancel_path.exists()))
        for widget in (self.fast, self.mc):
            widget.setEnabled(not busy)
        for widget in (self.samples, self.noise, self.seed):
            widget.setEnabled(not busy and self.mc.isChecked())
        self.resolution.setEnabled(not busy)
        self.saved_combo.setVisible(bool(self.saved_results))
        self.reuse_button.setVisible(bool(self.saved_results))
        self.saved_combo.setEnabled(not busy and self.saved_combo.count() > 0)
        self.reuse_button.setEnabled(not busy and self.study is not None and self.saved_combo.count() > 0)
        row = self.table.currentRow()
        self.previous_button.setEnabled(not busy and row > 0)
        self.next_button.setEnabled(not busy and self.table.rowCount() > 0 and row < self.table.rowCount()-1)
        self.mc_toggle.setEnabled(not busy)
        self.update_mc_toggle()
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0 if busy else 1)

    def update_mc_toggle(self, *_):
        state = ' · đang bật' if self.mc.isChecked() else ''
        prefix = '- ' if self.mc_toggle.isChecked() else '+ '
        self.mc_toggle.setText(prefix+'Thử nghiệm nhiễu HU'+state)

    def _stage(self, message):
        self.active_stage = message
        self.log.appendPlainText(message)
        self._tick()

    def _tick(self):
        elapsed = f' · {int(time.monotonic()-self.started)} giây' if self.started and self._busy() else ''
        self.status.setText(self.active_stage+elapsed)

    def _task(self, function, callback, message):
        if self._busy():
            return
        self.started = time.monotonic()
        self._callback = callback
        self.thread = Task(function, self)
        self.thread.finished.connect(self._task_done)
        self._stage(message)
        self._sync()
        self.thread.start()

    def _task_done(self):
        task = self.thread
        callback = self._callback
        self.timings.append(dict(stage=self.active_stage, seconds=round(task.elapsed, 4)))
        self.log.appendPlainText(f'Thời gian tác vụ: {task.elapsed:.2f} giây')
        self.thread = None
        self._sync()
        if task.error:
            self.error(task.error)
        else:
            try:
                callback(task.value)
            except Exception as error:
                self.error(str(error))
        task.deleteLater()
        self._sync()

    def error(self, message):
        self._stage('Không hoàn tất: '+message)
        QMessageBox.warning(self, 'Cần kiểm tra', message)

    def verify_environment(self):
        def accepted(result):
            self.backend_ready, message = result
            self.backend_label.setText(message)
            self._stage('Đã kiểm tra môi trường. Có thể mở CT và kết quả đã lưu.')
        self._task(check_backend, accepted, 'Kiểm tra backend…')

    def choose_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Chọn CT', str(ROOT/'data'),
                                                 'CT (*.nii *.nii.gz *.npy)')
        if filename:
            self.load_path(Path(filename))

    def choose_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục DICOM', str(ROOT/'data'))
        if not folder:
            return
        self.pending_path = Path(folder)
        self.series_combo.clear()
        self._task(lambda: list_ct_series(self.pending_path), self._series_ready, 'Tìm các CT series…')

    def _series_ready(self, choices):
        self.series_combo.clear()
        for i, series in enumerate(choices, 1):
            self.series_combo.addItem(f"Series {i}: {series['slices']} lát · {series['rows']}×{series['columns']} · "
                                      f"{series['kernel']} · {series['thickness']} mm", series['uid'])
        self._stage(f'Tìm thấy {len(choices)} CT series. Chọn series rồi bấm Đọc series đã chọn.')
        if not choices:
            self.error('Không tìm thấy CT series hỗ trợ trong thư mục đã chọn.')

    def load_series(self):
        if self.pending_path and self.series_combo.currentData():
            self.load_path(self.pending_path, self.series_combo.currentData())

    def _clear_study(self):
        # Clear old data before selection/loading, including ambiguous new samples.
        self.study = None
        self.review = None
        self.current_predictions = []
        self.saved_results = []
        self.saved_combo.clear()
        self.region_states.clear()
        self.region_paths.clear()
        self.level_label.setText('Chọn đốt để kiểm tra')
        self.mode.clear()
        self.mode.addItem('Chỉ CT')
        self.table.setRowCount(0)
        self.warnings.clear()
        self.patient_label.setText('Bệnh nhân: Chưa chọn')
        self.rename_patient_button.setEnabled(False)
        self.source_label.setText('Đang đọc ca mới…')
        self.geometry_label.setText('Đang kiểm tra HU và geometry…')
        self.result_label.setText('Chưa có kết quả cho ca mới.')
        self.redraw()
        self._sync()

    def rename_patient(self):
        if not self.study:
            return
        patient = self.study.get('patient') or {}
        current_name = patient.get('display_name', '')
        new_name, ok = QInputDialog.getText(self, 'Đổi tên bệnh nhân',
                                            'Nhập tên hoặc mã định danh bệnh nhân mới:',
                                            QLineEdit.Normal, current_name)
        if ok and new_name.strip():
            clean_name = new_name.strip()
            old_slug = patient.get('slug', '')
            new_slug = slugify(clean_name, fallback=old_slug or 'benh_nhan')
            patient['display_name'] = clean_name
            patient['slug'] = new_slug
            self.study['patient'] = patient
            old_dir = ROOT/'data/output'/old_slug
            new_dir = ROOT/'data/output'/new_slug
            if old_slug and old_dir.exists() and old_slug != new_slug:
                try:
                    old_dir.rename(new_dir)
                except Exception as exc:
                    self.log.appendPlainText(f'Không đổi được tên thư mục cũ: {exc}')
            self.patient_label.setText(f'Bệnh nhân: {clean_name} ({new_slug})')
            self._stage(f'Đã đổi tên bệnh nhân thành: {clean_name}')
            if 'fingerprint' in self.study:
                saved, _ = discover_results([
                    ROOT/'data/output'/new_slug/'dinh_vi_cot_song',
                    ROOT/'data/output', ROOT/'data/demo_runs', ROOT/'data/processed'
                ], self.study['fingerprint'])
                self.saved_results = saved
                self.refresh_saved_results()

    def open_body_measurements_folder(self):
        slug = (self.study.get('patient') or {}).get('slug') if self.study else None
        target = None
        row = max(0, self.table.currentRow())
        current_preds = getattr(self, 'current_predictions', [])
        if 0 <= row < len(current_preds):
            level = current_preds[row]['level']
            kind = self.mode.currentText()
            saved = self.region_paths.get((kind, level))
            if saved and Path(saved).exists():
                target = Path(saved)
        if not target or not target.exists():
            if slug:
                candidate = ROOT/'data/output'/slug/'do_than_dot'
                if candidate.exists():
                    target = candidate
            if not target:
                target = ROOT/'data/output' if (ROOT/'data/output').exists() else ROOT/'data/body_measurements'
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def load_path(self, path, uid=None, results=None):
        self._clear_study()
        def accepted(study):
            self.study = study
            patient = study.get('patient') or {}
            display = patient.get('display_name', 'Chưa rõ')
            slug = patient.get('slug', '')
            self.patient_label.setText(f"Bệnh nhân: {display} ({slug})")
            self.rename_patient_button.setEnabled(True)
            self.saved_results = study.get('saved_results', [])
            self.region_paths = study.get('saved_regions', {})
            self.region_states = {key: 'Có bản lưu · Chưa mở kiểm tra' for key in self.region_paths}
            self.refresh_saved_results()
            if study.get('unreadable_results'):
                self.log.appendPlainText('Một số báo cáo cũ không đọc được; có thể mở thư mục kết quả riêng.')
            self.source_label.setText(str(path))
            shape = study['source'].GetSize()
            spacing = study['source'].GetSpacing()
            self.geometry_label.setText('CT hợp lệ · '+ ' × '.join(map(str, shape))+' voxel\n'
                                        +'Spacing: '+' × '.join(f'{s:g}' for s in spacing)+' mm')
            z, y, x = study['data'].shape
            for slider, count in zip(self.sliders, (z, x, y)):
                slider.blockSignals(True)
                slider.setRange(0, count-1)
                slider.setValue(count//2)
                slider.blockSignals(False)
            self.redraw()
            self._stage('CT đã sẵn sàng. Có kết quả cũ: bấm Mở kết quả đã có.' if self.saved_results else
                        'CT đã sẵn sàng. Bấm Chạy định vị để nhận diện đốt sống.')
            if results and (results/'prediction.json').is_file():
                self.load_results(results)
        self._task(lambda: read_study(path, uid), accepted, 'Đọc CT và chuẩn bị ba mặt phẳng…')

    def open_sample(self):
        self._clear_study()
        self.pending_path = SAMPLE_DIR
        self.series_combo.clear()
        def select_sample(series):
            self._series_ready(series)
            if not series:
                return
            selected = select_sample_series(series)
            if selected is None:
                self._stage('Chưa xác định được duy nhất series axial phù hợp. '
                            'Chọn series rồi bấm Đọc series đã chọn.')
                return
            self.series_combo.setCurrentIndex(self.series_combo.findData(selected['uid']))
            self._stage(f"Tự chọn series axial nhiều lát nhất: {selected['slices']} lát · "
                        f"{selected['kernel']}. Có thể chọn series khác trong danh sách.")
            self.load_path(SAMPLE_DIR, selected['uid'])

        self._task(lambda: list_ct_series(SAMPLE_DIR), select_sample,
                   'Đang đọc danh sách series của ca mẫu…')

    def choose_results(self):
        slug = (self.study.get('patient') or {}).get('slug') if self.study else None
        default_dir = ROOT/'data/output'/slug/'dinh_vi_cot_song' if slug and (ROOT/'data/output'/slug/'dinh_vi_cot_song').exists() \
            else (ROOT/'data/output' if (ROOT/'data/output').exists() else ROOT/'data/demo_runs')
        default_dir.mkdir(parents=True, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, 'Thư mục chứa prediction.json', str(default_dir))
        if folder:
            p = Path(folder)
            if not (p/'prediction.json').is_file():
                if (p/'region.json').is_file() or (p/'measurements.json').is_file():
                    QMessageBox.warning(self, 'Thư mục không phù hợp',
                                        'Thư mục này chứa vùng đo thân đốt (do_than_dot / body_measurements), không phải kết quả định vị cột sống.\n\n'
                                        'Để xem vùng đo này:\n'
                                        '1. Chọn kết quả định vị ở ô bên trên rồi bấm "Tải lại lần chạy đã chọn".\n'
                                        '2. Chọn đốt sống (ví dụ L3) trong bảng rồi bấm "Đo thân đốt".')
                else:
                    QMessageBox.warning(self, 'Thiếu prediction.json',
                                        f'Không tìm thấy prediction.json trong thư mục:\n{folder}\n\n'
                                        'Vui lòng chọn thư mục chứa kết quả định vị (trong data/output hoặc data/demo_runs).')
                return
            self.load_results(p)

    def load_results(self, folder):
        folder = Path(folder)
        if not self.study:
            return
        if not (folder/'prediction.json').is_file():
            self._stage(f'Thư mục không chứa prediction.json: {folder.name}')
            return
        study = self.study
        self.review = None
        self.table.setRowCount(0)
        self.warnings.clear()
        self.result_label.setText('Đang kiểm tra kết quả đã chọn…')
        self.mode.clear()
        self.mode.addItem('Chỉ CT')
        def accepted(review):
            self.review = review
            self.saved_results = [item for item in self.saved_results if item['folder'] != folder]
            self.saved_results.insert(0, dict(folder=folder, report=review['report']))
            self.refresh_saved_results()
            self.mode.addItems(list(review['masks']))
            self.mode.setCurrentText('Consensus' if 'Consensus' in review['masks'] else 'Baseline')
            self.warnings.setPlainText('\n\n'.join(review['warnings']) or 'Không có cảnh báo tự động; vẫn cần review.')
            verified = 'CT đã xác nhận khớp' if review['verified'] else 'Kết quả cũ: cần xác nhận đúng ca'
            self.result_label.setText(f"{'Có cảnh báo' if review['warnings'] else 'Chưa kiểm tra'} · {verified} · {folder.name}")
            resolution = review['report'].get('model', {}).get('resolution')
            if resolution:
                self.result_label.setText(self.result_label.text()+f' · Mô hình: {resolution}')
            self._stage('Đã mở kết quả. Chọn đốt sống trong bảng để di chuyển đến vùng tương ứng.')
            self.mode_changed()
        self._task(lambda: load_review(folder, study['source'], study['reference'], study['fingerprint']),
                   accepted, 'Kiểm tra kết quả và căn chỉnh mask…')

    def refresh_saved_results(self, *_):
        selected = self.saved_combo.currentData()
        self.saved_combo.clear()
        config = dict(fast=self.fast.isChecked(), samples=self.samples.value() if self.mc.isChecked() else 0,
                      noise=self.noise.value(), seed=self.seed.value())
        ordered = sorted(self.saved_results, key=lambda item: not compatible_result(item['report'], **config))
        for item in ordered:
            report = item['report']
            model = report.get('model') or {}
            match = compatible_result(report, **config)
            text = ('Khớp cấu hình' if match else 'Khác / thiếu cấu hình')
            text += f" · {model.get('resolution', '?')} · MC {(report.get('monte_carlo') or {}).get('samples', 0)} · {item['folder'].name}"
            self.saved_combo.addItem(text, item['folder'])
            self.saved_combo.setItemData(self.saved_combo.count()-1, text, Qt.ToolTipRole)
        index = self.saved_combo.findData(selected)
        if index >= 0:
            self.saved_combo.setCurrentIndex(index)
        self._sync()

    def reuse_result(self):
        folder = self.saved_combo.currentData()
        if folder is not None and not self._busy():
            self.load_results(folder)

    def move_level(self, delta):
        if self._busy() or not self.table.rowCount():
            return
        row = self.table.currentRow()
        target = 0 if row < 0 else row + delta
        self.table.selectRow(max(0, min(self.table.rowCount()-1, target)))

    def start_run(self):
        if not self.study or self._busy():
            return
        self.review = None
        self.mode.clear()
        self.mode.addItem('Chỉ CT')
        self.table.setRowCount(0)
        self.warnings.clear()
        self.result_label.setText('Đang chạy · chưa có kết quả cho lần xử lý mới.')
        patient = self.study.get('patient') or {}
        slug = patient.get('slug') or 'benh_nhan'
        res = '3mm' if self.fast.isChecked() else '1.5mm'
        run_id = f'run_{res}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
        parent = ROOT/'data/output'/slug/'dinh_vi_cot_song'
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.error('Không tạo được thư mục kết quả: '+str(error))
            self._sync()
            return
        self.run_output = parent/run_id
        self.cancel_path = parent/(run_id+'.cancel')
        self.backend_log = parent/(run_id+'.log')
        args = ['-u', '-m', 'tools.localize_vertebrae', '--input', str(self.study['path']),
                '--output', str(self.run_output), '--backend', str(ROOT/'.venv-model/Scripts/TotalSegmentator.exe'),
                '--device', 'cpu', '--model-cache', str(ROOT/'.tmp/totalsegmentator'),
                '--backend-log', str(self.backend_log), '--progress-json', '--cancel-file', str(self.cancel_path)]
        if self.study['uid']:
            args += ['--series-uid', self.study['uid']]
        if self.fast.isChecked():
            args += ['--fast']
        else:
            args += ['--robust-crop']
        if self.mc.isChecked():
            args += ['--mc-samples', str(self.samples.value()), '--mc-noise-hu', str(self.noise.value()),
                     '--mc-seed', str(self.seed.value())]
        self.stdout_buffer = ''
        self.started = time.monotonic()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONIOENCODING', 'utf-8')
        environment.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_progress)
        self.process.finished.connect(self.run_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.setProgram(str(ROOT/'.venv/Scripts/python.exe'))
        self.process.setArguments(args)
        self._stage('Bắt đầu chạy · kết quả được lưu vào thư mục mới')
        self._sync()
        self.process.start()

    def read_progress(self):
        if not self.process:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        lines = self.stdout_buffer.split('\n')
        self.stdout_buffer = lines.pop()
        for line in lines:
            if line.startswith('PROGRESS '):
                try:
                    self._stage(json.loads(line[9:])['message'])
                except (ValueError, KeyError):
                    self.log.appendPlainText(line)
            elif line.strip():
                self.log.appendPlainText(line.strip())

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.run_finished(-1, QProcess.CrashExit)

    def run_finished(self, code, status):
        if self.process is None:
            return
        self.read_progress()
        process = self.process
        self.process = None
        process.deleteLater()
        stopped = self.cancel_path and self.cancel_path.exists()
        if stopped:
            self.cancel_path.unlink()
        self._sync()
        if code == 0 and (self.run_output/'prediction.json').is_file():
            self.load_results(self.run_output)
        elif stopped:
            self._stage('Đã dừng. Không công bố kết quả dở dang.')
        else:
            self.error('Xử lý chưa hoàn tất. Xem tab Tiến trình và log tại:\n'+str(self.backend_log))

    def request_stop(self):
        if self.process and self.cancel_path:
            try:
                self.cancel_path.touch(exist_ok=True)
            except OSError as error:
                self.error('Không gửi được yêu cầu dừng: '+str(error))
                return
            self._stage('Đã yêu cầu dừng. Đang chờ lượt suy luận hiện tại kết thúc…')
            self._sync()

    def mode_changed(self, *_):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if self.review:
            report = self.review['report']
            selected = report.get('monte_carlo', {}).get('consensus', report) if self.mode.currentText() == 'Consensus' else report
            self.current_predictions = selected.get('predictions', [])
            stats = {p['level']: p for p in report.get('monte_carlo', {}).get('per_level', [])}
            self.table.setRowCount(len(self.current_predictions))
            for row, prediction in enumerate(self.current_predictions):
                mc = stats.get(prediction['level'])
                count = f"{mc['detected_samples']}/{report['monte_carlo']['samples']}" if mc else '—'
                values = [prediction['level'], f"{prediction['volume_mm3']:,.1f}", count,
                          'Chạm biên ảnh' if prediction.get('touches_boundary') else '—',
                          self.region_states.get((self.mode.currentText(), prediction['level']), 'Chưa kiểm tra')]
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.blockSignals(False)
        if self.table.rowCount():
            self.table.selectRow(0)
        self.redraw()
        self._sync()

    def open_body_review(self):
        if not self.study or not self.review or self._busy() or self.mode.currentText() not in ('Baseline', 'Consensus'):
            return
        row = max(0, self.table.currentRow())
        if row >= len(self.current_predictions):
            return
        from apps.body_review import BodyReviewDialog
        study, review = self.study, self.review
        level, kind = self.current_predictions[row]['level'], self.mode.currentText()
        path = review['folder']/('monte_carlo/consensus.nii.gz' if kind == 'Consensus' else 'vertebrae.nii.gz')
        slug = (study.get('patient') or {}).get('slug') or 'benh_nhan'
        output_root = ROOT/'data/output'/slug/'do_than_dot'
        output_root.mkdir(parents=True, exist_ok=True)
        def accepted(labels):
            if self.study is study and self.review is review:
                dialog = BodyReviewDialog(study['source'], labels, study['fingerprint'], level, kind,
                                          output_root, self)
                saved_region = self.region_paths.get((kind, level))
                if saved_region:
                    try:
                        dialog.load_region(saved_region)
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                        dialog.error('Không mở được vùng cũ: '+str(error))
                dialog.exec()
                if isinstance(dialog.last_saved_path, Path):
                    self.region_paths[(kind, level)] = dialog.last_saved_path
                state = ('Đã lưu · '+('Đã xác nhận' if dialog.last_saved_reviewed else 'Vùng nháp')
                         if isinstance(dialog.last_saved_path, Path) else 'Chưa kiểm tra')
                if isinstance(state, str):
                    self.region_states[(kind, level)] = state
                    self.table.setItem(row, 4, QTableWidgetItem(state))
        self._task(lambda: sitk.ReadImage(str(path)), accepted, f'Mở công cụ đo thân đốt {level}…')

    def select_level(self):
        row = self.table.currentRow()
        if row < 0 or not self.review:
            return
        self.level_label.setText(f"{self.current_predictions[row]['level']} · {row+1}/{self.table.rowCount()}")
        self._sync()
        centroid = self.current_predictions[row]['centroid_lps_mm']
        x, y, z = self.study['reference'].TransformPhysicalPointToContinuousIndex(centroid)
        self._set_indices(z, x, y)

    def _set_indices(self, z, x, y):
        for slider, value in zip(self.sliders, (z, x, y)):
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
        self.redraw()

    def click_plane(self, event):
        if not self.study or event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        z, x, y = [s.value() for s in self.sliders]
        dx, dy, dz = self.study['reference'].GetSpacing()
        index = self.axes.index(event.inaxes)
        shape = self.study['data'].shape
        width, height, sx, sy = [(shape[2], shape[1], dx, dy),
                               (shape[1], shape[0], dy, dz),
                               (shape[2], shape[0], dx, dz)][index]
        if not (-sx/2 <= event.xdata <= (width-.5)*sx and
                -sy/2 <= event.ydata <= (height-.5)*sy):
            return  # Clicking the name gutter must not move the CT crosshair.
        if index == 0:
            x, y = event.xdata/dx, event.ydata/dy
        elif index == 1:
            y, z = event.xdata/dy, event.ydata/dz
        else:
            x, z = event.xdata/dx, event.ydata/dz
        self._set_indices(z, x, y)

    def redraw(self, *_):
        self.figure.clear()
        self.axes = list(self.figure.subplots(1, 3))
        self.figure.subplots_adjust(left=.04, right=.98, bottom=.12, top=.90, wspace=.23)
        if not self.study:
            for ax, title in zip(self.axes, ('AXIAL', 'SAGITTAL', 'CORONAL')):
                ax.set_facecolor('#111d29')
                ax.text(.5, .5, title+'\nChưa có CT', transform=ax.transAxes,
                        color='#89a5b7', ha='center', va='center', fontsize=13)
                ax.set_axis_off()
            self.canvas.draw_idle()
            return
        z, x, y = [slider.value() for slider in self.sliders]
        data = self.study['data']
        dx, dy, dz = self.study['reference'].GetSpacing()
        planes = [data[z], data[:, :, x], data[:, y, :]]
        sizes = [(dx, dy), (dy, dz), (dx, dz)]
        origins = ['upper', 'lower', 'lower']
        titles = ['Axial · R → L / A → P', 'Sagittal · A → P / I → S', 'Coronal · R → L / I → S']
        mask = self.review['masks'].get(self.mode.currentText()) if self.review and self.overlay.isChecked() else None
        mask_planes = [mask[z], mask[:, :, x], mask[:, y, :]] if mask is not None else [None]*3
        palette = vertebral_palette()
        for i, (ax, plane, overlay, spacing, origin, title) in enumerate(zip(self.axes, planes, mask_planes, sizes, origins, titles)):
            sx, sy = spacing
            h, w = plane.shape
            extent = (-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2) if origin == 'upper' else (-sx/2, (w-.5)*sx, -sy/2, (h-.5)*sy)
            ax.set_facecolor('#111d29')
            ax.imshow(plane, cmap='gray', vmin=self.window_level.value()-self.window_width.value()/2,
                      vmax=self.window_level.value()+self.window_width.value()/2, extent=extent, origin=origin, interpolation='bilinear')
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            if overlay is not None:
                if self.mode.currentText() == 'Bất đồng':
                    ax.imshow(np.ma.masked_less_equal(overlay, 0), cmap='magma', vmin=0, vmax=1,
                              extent=extent, origin=origin, alpha=self.opacity.value()/100, interpolation='nearest')
                else:
                    ax.imshow(np.ma.masked_equal(overlay, 0), cmap=palette, vmin=0, vmax=25,
                              extent=extent, origin=origin, alpha=self.opacity.value()/100, interpolation='nearest')
                    draw_vertebral_overlay(ax, overlay, spacing,
                        names=i in (1, 2) and self.show_names.isChecked(),
                        label_map=self.review['report'].get('label_map'))
            cross = [(x*dx, y*dy), (y*dy, z*dz), (x*dx, z*dz)][i]
            ax.vlines(cross[0], min(extent[2:]), max(extent[2:]), color='#7adee5', lw=.6, alpha=.6)
            ax.hlines(cross[1], *extent[:2], color='#7adee5', lw=.6, alpha=.6)
            ax.set_title(title, color='#c5e0e8', fontsize=9, pad=12)
            ax.tick_params(colors='#94aebc', labelsize=7)
        for label, title, index, count in zip(self.slice_labels, ('Axial z', 'Sagittal x', 'Coronal y'),
                                             (z, x, y), (data.shape[0], data.shape[2], data.shape[1])):
            label.setText(f'{title}: {index+1} / {count}')
        caption = 'Bất đồng 0–1: đen → sáng; không phải xác suất sai.' if self.mode.currentText() == 'Bất đồng' else 'Vị trí được đồng bộ · khoảng cách theo mm · mask toàn đốt sống'
        self.figure.text(.5, .02, caption, ha='center', color='#94aebc', fontsize=8)
        self.canvas.draw_idle()

    def export_report(self):
        if not self.review:
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Lưu báo cáo', str(self.review['folder']/'review_report.html'), 'HTML (*.html)')
        if filename:
            try:
                alias = 'CT-'+self.study['fingerprint'][:12]
                Path(filename).write_text(report_html(self.review['report'], self.review['warnings'], alias), encoding='utf-8')
                self._stage('Đã xuất báo cáo HTML. Có thể mở bằng trình duyệt và in thành PDF.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.error(str(error))

    def open_result_folder(self):
        if self.review:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.review['folder'])))

    def show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Hướng dẫn sử dụng · Quy trình 4 bước')
        msg.setText(
            '<h3>QUY TRÌNH SỬ DỤNG HỆ THỐNG SPINAL CT</h3>'
            '<p><b>Bước 1: Chọn dữ liệu CT bệnh nhân</b><br>'
            '• Bấm <i>Mở CT…</i> → chọn file NIfTI hoặc thư mục DICOM của bệnh nhân.<br>'
            '• Hệ thống tự động nhận diện danh tính bệnh nhân và hiển thị <i>Bệnh nhân: [Tên]</i>. Có thể bấm <i>Đổi tên…</i> để chỉnh sửa.<br>'
            '• Chọn Series lát cắt mỏng (khuyến nghị ≤ 1.5 mm) và bấm <i>Đọc series đã chọn</i>.</p>'
            '<p><b>Bước 2: Định vị cột sống tự động bằng AI</b><br>'
            '• Chọn <i>Chi tiết · 1,5 mm</i> (chính xác cao) hoặc <i>Nhanh · 3 mm</i>.<br>'
            '• Bấm <b>Chạy định vị</b>. AI tự động phân đoạn các đốt sống từ C1 đến S1.<br>'
            '• Kết quả tự động lưu vào: <code>data/output/[tên_bệnh_nhân]/dinh_vi_cot_song/</code>.</p>'
            '<p><b>Bước 3: Đo thân đốt sống & Phân tích lún xẹp (Morphometry)</b><br>'
            '• Click chọn đốt sống cần đo trong bảng (ví dụ: L1, L2, L3) hoặc bấm <i>Đốt tiếp theo →</i>.<br>'
            '• Bấm <b>Đo thân đốt</b> (màu xanh đậm).<br>'
            '• Khoanh vùng thân đốt (vẽ đa giác hoặc cọ) → Đặt 6 mốc chiều cao → Tích <i>Xác nhận vùng đo</i> → Bấm <b>Đo thân đốt</b>.<br>'
            '• Xem trực tiếp 3 biểu đồ: Lát cắt CT, Vùng chuẩn hóa 256×256, và <b>Biểu đồ hội tụ Monte Carlo</b> kèm <b>Điểm cùi chỏ (Elbow point)</b>.<br>'
            '• Bấm <b>Lưu vùng nháp / kết quả</b> (tự động lưu vào <code>data/output/[tên_bệnh_nhân]/do_than_dot/</code>).</p>'
            '<p><b>Bước 4: Tra cứu & Quản lý dữ liệu</b><br>'
            '• Bấm <i>Mở thư mục đo thân đốt</i> để mở nhanh file <code>preview.png</code> (ảnh 3 đồ thị) và <code>report.html</code> (báo cáo mở bằng trình duyệt).<br>'
            '• Bấm <i>Mở thư mục kết quả (Explorer)</i> để mở thư mục định vị 3D.</p>'
        )
        msg.exec()

    def closeEvent(self, event):
        if self._busy():
            QMessageBox.information(self, 'Đang xử lý', 'Hãy chờ tác vụ hoàn tất. Khi đang suy luận, có thể yêu cầu dừng sau lượt hiện tại.')
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DemoWindow()
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1500, available.width()-40), min(960, available.height()-60))
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
