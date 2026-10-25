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
from PySide6.QtCore import QProcess, QProcessEnvironment, QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib import get_data_path

from spinal_ct.ct_io import list_ct_series, load_ct, select_sample_series
from spinal_ct.review import ct_fingerprint, display_image, load_review, report_html

ROOT = Path(__file__).resolve().parents[1]
# Keep patient-specific paths in the ignored local configuration.
try:
    from sample_config import SAMPLE_DIR
except ModuleNotFoundError as exc:
    if exc.name != 'sample_config':
        raise
    SAMPLE_DIR = ROOT / 'data/raw/sample_patient'


def ui_font() -> str:
    # Offscreen Qt and some Windows hosts do not populate the system font DB.
    # Matplotlib ships DejaVu Sans, including the Vietnamese glyphs we need.
    candidates = [Path('C:/Windows/Fonts/segoeui.ttf'),
                  Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf']
    for path in candidates:
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                return families[0]
    return QApplication.font().family()


class Task(QThread):
    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.value = None
        self.error = None

    def run(self):
        try:
            self.value = self.function()
        except Exception as error:
            self.error = str(error)


def read_study(path, uid):
    source = load_ct(path, uid)
    reference = display_image(source)
    return dict(source=source, reference=reference, data=sitk.GetArrayFromImage(reference),
                fingerprint=ct_fingerprint(source), path=path, uid=uid)


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
        controls.addWidget(self.file_button)
        self.dicom_button = QPushButton('Chọn thư mục DICOM')
        self.dicom_button.clicked.connect(self.choose_dicom)
        controls.addWidget(self.dicom_button)
        self.sample_button = QPushButton(f'Mở ca mẫu · {SAMPLE_DIR.name.title()}')
        self.sample_button.setToolTip(str(SAMPLE_DIR))
        self.sample_button.clicked.connect(self.open_sample)
        controls.addWidget(self.sample_button)
        self.source_label = QLabel('Chưa chọn dữ liệu')
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName('muted')
        controls.addWidget(self.source_label)
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
        self.fast.setChecked(True)
        controls.addWidget(self.fast)
        self.mc = QCheckBox('Bổ sung Monte Carlo')
        self.mc.toggled.connect(self._sync)
        controls.addWidget(self.mc)
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
        controls.addLayout(form)
        self.backend_label = QLabel('Đang kiểm tra backend…')
        self.backend_label.setWordWrap(True)
        self.backend_label.setObjectName('muted')
        controls.addWidget(self.backend_label)
        self.check_button = QPushButton('Kiểm tra môi trường')
        self.check_button.clicked.connect(self.verify_environment)
        controls.addWidget(self.check_button)
        self.run_button = QPushButton('Chạy định vị')
        self.run_button.setObjectName('primary')
        self.run_button.clicked.connect(self.start_run)
        controls.addWidget(self.run_button)
        self.stop_button = QPushButton('Dừng sau lượt xử lý hiện tại')
        self.stop_button.clicked.connect(self.request_stop)
        controls.addWidget(self.stop_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel('03  KẾT QUẢ'))
        self.open_button = QPushButton('Mở thư mục kết quả đã lưu')
        self.open_button.clicked.connect(self.choose_results)
        controls.addWidget(self.open_button)
        self.export_button = QPushButton('Xuất báo cáo HTML')
        self.export_button.clicked.connect(self.export_report)
        controls.addWidget(self.export_button)
        self.folder_button = QPushButton('Mở vị trí kết quả')
        self.folder_button.clicked.connect(self.open_result_folder)
        controls.addWidget(self.folder_button)
        controls.addStretch()
        note = QLabel('Mask toàn đốt sống · thể tích mm³\nChưa tính diện tích / phân loại AO.\nKết quả cần được kiểm tra.')
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

        right = QWidget()
        area = QVBoxLayout(right)
        area.setContentsMargins(0, 12, 0, 0)
        toolbar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem('Chỉ CT')
        self.mode.currentTextChanged.connect(self.mode_changed)
        toolbar.addWidget(self.mode)
        self.overlay = QCheckBox('Hiện overlay')
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(self.redraw)
        toolbar.addWidget(self.overlay)
        toolbar.addWidget(QLabel('Độ đậm'))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 90)
        self.opacity.setValue(30)
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
        area.addLayout(toolbar)
        self.figure = Figure(facecolor='#111d29')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(310)
        area.addWidget(self.canvas, 3)
        self.canvas.mpl_connect('button_press_event', self.click_plane)
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
        area.addLayout(sliders)
        self.result_label = QLabel('Chọn CT để xem. Nhấp trên ảnh để đồng bộ vị trí ba mặt phẳng.')
        self.result_label.setWordWrap(True)
        area.addWidget(self.result_label)
        tabs = QTabWidget()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Đốt sống', 'Thể tích (mm³)', 'MC phát hiện', 'Cảnh báo'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.select_level)
        tabs.addTab(self.table, 'Các đốt sống')
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        tabs.addTab(self.warnings, 'Lưu ý kết quả')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        tabs.addTab(self.log, 'Tiến trình')
        area.addWidget(tabs, 1)
        split.addWidget(right)
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
        '''.replace('Segoe UI', self.font_family))
        self.axes = []
        self.redraw()

    def _busy(self):
        return self.thread is not None or self.process is not None

    def _sync(self, *_):
        busy = self._busy()
        for button in (self.file_button, self.dicom_button, self.sample_button, self.check_button):
            button.setEnabled(not busy)
        self.sample_button.setEnabled(not busy and SAMPLE_DIR.is_dir())
        self.load_button.setEnabled(not busy and self.series_combo.count() > 0)
        self.series_combo.setEnabled(not busy)
        self.run_button.setEnabled(not busy and self.study is not None and self.backend_ready)
        self.open_button.setEnabled(not busy and self.study is not None)
        self.export_button.setEnabled(not busy and self.review is not None)
        self.folder_button.setEnabled(self.review is not None)
        self.stop_button.setEnabled(self.process is not None and not (self.cancel_path and self.cancel_path.exists()))
        for widget in (self.fast, self.mc):
            widget.setEnabled(not busy)
        for widget in (self.samples, self.noise, self.seed):
            widget.setEnabled(not busy and self.mc.isChecked())
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0 if busy else 1)

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
        self.mode.clear()
        self.mode.addItem('Chỉ CT')
        self.table.setRowCount(0)
        self.warnings.clear()
        self.source_label.setText('Đang đọc ca mới…')
        self.geometry_label.setText('Đang kiểm tra HU và geometry…')
        self.result_label.setText('Chưa có kết quả cho ca mới.')
        self.redraw()
        self._sync()

    def load_path(self, path, uid=None, results=None):
        self._clear_study()
        def accepted(study):
            self.study = study
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
            self._stage('CT đã sẵn sàng. Chạy định vị hoặc mở kết quả đã lưu.')
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
        folder = QFileDialog.getExistingDirectory(self, 'Thư mục chứa prediction.json', str(ROOT/'data/processed'))
        if folder:
            self.load_results(Path(folder))

    def load_results(self, folder):
        if not self.study:
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
            self.mode.addItems(list(review['masks']))
            self.mode.setCurrentText('Consensus' if 'Consensus' in review['masks'] else 'Baseline')
            self.warnings.setPlainText('\n\n'.join(review['warnings']) or 'Không có cảnh báo tự động; vẫn cần review.')
            verified = 'CT đã xác nhận khớp' if review['verified'] else 'Kết quả cũ: cần xác nhận đúng ca'
            self.result_label.setText(f"{review['report']['status']} · {verified} · {folder.name}")
            self._stage('Đã mở kết quả. Chọn đốt sống trong bảng để di chuyển đến vùng tương ứng.')
            self.mode_changed()
        self._task(lambda: load_review(folder, study['source'], study['reference'], study['fingerprint']),
                   accepted, 'Kiểm tra kết quả và căn chỉnh mask…')

    def start_run(self):
        if not self.study or self._busy():
            return
        self.review = None
        self.mode.clear()
        self.mode.addItem('Chỉ CT')
        self.table.setRowCount(0)
        self.warnings.clear()
        self.result_label.setText('Đang chạy · chưa có kết quả cho lần xử lý mới.')
        run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S_')+uuid4().hex[:6]
        parent = ROOT/'data/demo_runs'
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
                          'Chạm biên ảnh' if prediction.get('touches_boundary') else '—']
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.blockSignals(False)
        self.redraw()

    def select_level(self):
        row = self.table.currentRow()
        if row < 0 or not self.review:
            return
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
        palette = ListedColormap([(0, 0, 0, 0)]+[colormaps['tab20'](i % 20) for i in range(25)])
        for i, (ax, plane, overlay, spacing, origin, title) in enumerate(zip(self.axes, planes, mask_planes, sizes, origins, titles)):
            sx, sy = spacing
            h, w = plane.shape
            extent = (-sx/2, (w-.5)*sx, (h-.5)*sy, -sy/2) if origin == 'upper' else (-sx/2, (w-.5)*sx, -sy/2, (h-.5)*sy)
            ax.set_facecolor('#111d29')
            ax.imshow(plane, cmap='gray', vmin=self.window_level.value()-self.window_width.value()/2,
                      vmax=self.window_level.value()+self.window_width.value()/2, extent=extent, origin=origin, interpolation='bilinear')
            if overlay is not None:
                if self.mode.currentText() == 'Bất đồng':
                    ax.imshow(np.ma.masked_less_equal(overlay, 0), cmap='magma', vmin=0, vmax=1,
                              extent=extent, origin=origin, alpha=self.opacity.value()/100, interpolation='nearest')
                else:
                    ax.imshow(np.ma.masked_equal(overlay, 0), cmap=palette, vmin=0, vmax=25,
                              extent=extent, origin=origin, alpha=self.opacity.value()/100, interpolation='nearest')
                    for label_id in np.unique(overlay):
                        if not label_id:
                            continue
                        ax.contour(np.arange(w)*sx, np.arange(h)*sy, overlay == label_id,
                                   levels=[.5], colors=[palette(int(label_id))], linewidths=.65)
            cross = [(x*dx, y*dy), (y*dy, z*dz), (x*dx, z*dz)][i]
            ax.axvline(cross[0], color='#7adee5', lw=.6, alpha=.6)
            ax.axhline(cross[1], color='#7adee5', lw=.6, alpha=.6)
            ax.set_title(title, color='#c5e0e8', fontsize=9, pad=12)
            ax.tick_params(colors='#94aebc', labelsize=7)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        for label, title, index, count in zip(self.slice_labels, ('Axial z', 'Sagittal x', 'Coronal y'),
                                             (z, x, y), (data.shape[0], data.shape[2], data.shape[1])):
            label.setText(f'{title}: {index+1} / {count}')
        caption = 'Bất đồng 0–1: đen → sáng; không phải xác suất sai.' if self.mode.currentText() == 'Bất đồng' else 'Vị trí được đồng bộ · khoảng cách theo mm · mask toàn đốt sống'
        self.figure.text(.5, .02, caption, ha='center', color='#94aebc', fontsize=8)
        self.figure.subplots_adjust(left=.04, right=.98, bottom=.12, top=.90, wspace=.23)
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
        QMessageBox.information(self, 'Hướng dẫn nhanh',
            '1. Chọn file CT hoặc thư mục DICOM; với DICOM, chọn series rồi bấm Đọc.\n'
            '2. Chạy baseline trước; bật Monte Carlo nếu cần đánh giá độ nhạy với nhiễu.\n'
            '3. Chọn Baseline / Consensus / Bất đồng; kéo thanh lát cắt hoặc nhấp trên ảnh.\n'
            '4. Chọn một đốt trong bảng để di chuyển đến tâm mask.\n'
            '5. Xuất báo cáo HTML hoặc mở thư mục để lấy mask NIfTI và JSON.\n\n'
            'Mở kết quả cũ: chọn đúng CT trước, sau đó chọn thư mục chứa prediction.json.\n'
            'Kết quả chạy mới lưu trong data/demo_runs/.\n'
            'Dừng sẽ chờ lượt suy luận hiện tại kết thúc.\n\n'
            'Chi tiết: docs/guides/teacher_demo.md')

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
