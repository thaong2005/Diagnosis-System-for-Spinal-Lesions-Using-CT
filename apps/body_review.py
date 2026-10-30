"""Manual body-region review, experimental LoG and area measurements."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector

from spinal_ct.body_region import propose_plane, sample_plane, polygon_mask
from spinal_ct.blob_detection import detect_blobs
from spinal_ct.morphometry import BodySession, load_session, save_session, LANDMARK_NAMES
from spinal_ct.morphometry_render import draw_body, draw_normalized
from apps.ui_fonts import ui_font

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
        self.mode = 'view'
        self.brushing = False
        self.setWindowTitle(f'Đo thân đốt · {level} · {source_kind}')
        self.resize(1380, 900)
        self.setMinimumSize(1050, 650)
        outer = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        panel = QWidget()
        controls = QVBoxLayout(panel)
        intro = QLabel('Vẽ đường bao thân đốt, gồm phần bên trong; loại cấu trúc phía sau. '
                       'Đường nét đứt xanh chỉ là mask toàn đốt để tham khảo.')
        intro.setWordWrap(True)
        controls.addWidget(intro)
        form = QFormLayout()
        self.offset = self._double(-80, 80, 0, ' mm')
        self.yaw = self._double(-45, 45, 0, '°')
        self.tilt = self._double(-45, 45, 0, '°')
        form.addRow('Dịch mặt phẳng', self.offset)
        form.addRow('Xoay quanh trục đứng', self.yaw)
        form.addRow('Nghiêng mặt phẳng', self.tilt)
        controls.addLayout(form)
        self.apply_plane = QPushButton('Tạo / áp dụng mặt phẳng')
        self.apply_plane.clicked.connect(self.rebuild_plane)
        controls.addWidget(self.apply_plane)
        for widget in (self.offset, self.yaw, self.tilt):
            widget.valueChanged.connect(self.plane_dirty)
        self.add_button = QPushButton('Vẽ / thêm đường bao thân đốt')
        self.add_button.clicked.connect(lambda: self.start_polygon(False))
        controls.addWidget(self.add_button)
        self.remove_button = QPushButton('Vẽ vùng cần loại khỏi mask')
        self.remove_button.clicked.connect(lambda: self.start_polygon(True))
        controls.addWidget(self.remove_button)
        self.brush_button = QPushButton('Cọ thêm (chuột trái) / xóa (chuột phải)')
        self.brush_button.clicked.connect(self.start_brush)
        controls.addWidget(self.brush_button)
        self.radius = self._double(.5, 10, 2, ' mm')
        controls.addWidget(self.radius)
        self.undo_button = QPushButton('Hoàn tác lần sửa vùng / mốc gần nhất')
        self.undo_button.clicked.connect(self.undo)
        controls.addWidget(self.undo_button)
        self.clear_button = QPushButton('Xóa vùng và mốc để vẽ lại')
        self.clear_button.clicked.connect(self.clear_region)
        controls.addWidget(self.clear_button)
        self.landmark_button = QPushButton('Đặt lại 6 mốc chiều cao')
        self.landmark_button.clicked.connect(self.start_landmarks)
        controls.addWidget(self.landmark_button)
        self.clear_landmarks = QPushButton('Bỏ các mốc chiều cao')
        self.clear_landmarks.clicked.connect(self.remove_landmarks)
        controls.addWidget(self.clear_landmarks)
        controls.addWidget(QLabel('LoG thử nghiệm · sigma theo mm'))
        log_form = QFormLayout()
        self.log_min = self._double(.2, 40, 2, ' mm')
        self.log_max = self._double(.2, 60, 12, ' mm')
        self.log_threshold = self._double(.001, 1, .03)
        self.log_threshold.setDecimals(3)
        self.log_threshold.setSingleStep(.01)
        log_form.addRow('Sigma nhỏ nhất', self.log_min)
        log_form.addRow('Sigma lớn nhất', self.log_max)
        log_form.addRow('Ngưỡng đáp ứng', self.log_threshold)
        controls.addLayout(log_form)
        self.log_button = QPushButton('Tìm ứng viên blob trong ROI')
        self.log_button.clicked.connect(self.find_blobs)
        controls.addWidget(self.log_button)
        self.blob_choices = QComboBox()
        self.blob_choices.currentIndexChanged.connect(self.candidate_changed)
        controls.addWidget(self.blob_choices)
        self.show_blobs = QCheckBox('Hiện ứng viên LoG')
        self.show_blobs.setChecked(True)
        self.show_blobs.toggled.connect(self.redraw)
        controls.addWidget(self.show_blobs)
        self.age = QComboBox()
        for title, value in (('Chưa xác định nhóm tuổi', 'unspecified'), ('Người lớn', 'adult'), ('Trẻ em', 'pediatric')):
            self.age.addItem(title, value)
        self.age.currentIndexChanged.connect(self.age_changed)
        controls.addWidget(self.age)
        self.confirm = QCheckBox('Xác nhận vùng đo và các mốc')
        self.confirm.setToolTip('Kiểm tra bằng mắt trước khi xác nhận. Thay đổi vùng/mốc sẽ hủy xác nhận và phép đo cũ.')
        self.confirm.toggled.connect(self.approve)
        controls.addWidget(self.confirm)
        mc_form = QFormLayout()
        self.samples = QSpinBox()
        self.samples.setRange(100, 10_000_000)
        self.samples.setValue(100_000)
        self.samples.setSingleStep(10_000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        mc_form.addRow('MC diện tích · số điểm', self.samples)
        mc_form.addRow('Seed', self.seed)
        controls.addLayout(mc_form)
        for widget in (self.samples, self.seed):
            widget.valueChanged.connect(self.mc_changed)
        self.measure_button = QPushButton('Đo hình dạng và Monte Carlo diện tích')
        self.measure_button.clicked.connect(self.measure)
        controls.addWidget(self.measure_button)
        self.show_points = QCheckBox('Hiện tối đa 1.000 điểm MC')
        self.show_points.toggled.connect(self.redraw)
        controls.addWidget(self.show_points)
        self.save_button = QPushButton('Lưu vùng / xuất kết quả vào thư mục mới')
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        self.open_button = QPushButton('Mở vùng thân đốt đã lưu')
        self.open_button.clicked.connect(self.open_saved)
        controls.addWidget(self.open_button)
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

    def sync(self):
        active = self.session is not None
        nonempty = active and bool(self.session.mask.any())
        self.confirm.blockSignals(True)
        self.confirm.setChecked(active and self.session.reviewed)
        self.confirm.blockSignals(False)
        self.confirm.setEnabled(nonempty)
        for widget in (self.add_button, self.remove_button, self.brush_button, self.log_button):
            widget.setEnabled(active)
        for widget in (self.clear_button, self.landmark_button, self.clear_landmarks, self.save_button):
            widget.setEnabled(nonempty)
        self.undo_button.setEnabled(active and bool(self.history))
        self.measure_button.setEnabled(active and self.session.reviewed)

    def plane_dirty(self, *_):
        self.stop_editing()
        self.session = None
        self.history.clear()
        self.blob_choices.clear()
        self.redraw()
        self.status.setText('Mặt phẳng đã thay đổi. Bấm áp dụng, sau đó chọn lại vùng đo.')

    def rebuild_plane(self):
        self.stop_editing()
        self.session = None
        self.history.clear()
        try:
            from spinal_ct.localization import LEVELS
            label_id = LEVELS.index(self.level)+1
            plane = propose_plane(self.source, self.labels, label_id, offset_mm=self.offset.value(),
                                  yaw_deg=self.yaw.value(), tilt_deg=self.tilt.value())
            ct, valid, hint = sample_plane(self.source, plane, self.labels, label_id)
            self.session = BodySession(self.fingerprint, self.level, self.source_kind, plane, ct, valid, hint,
                                       age_group=self.age.currentData())
            self.blob_choices.clear()
            self.status.setText('Mặt phẳng ban đầu từ ROI toàn đốt. Kiểm tra vị trí trung tâm thân đốt, rồi vẽ đường bao. '
                                'Khép đa giác bằng nhấp đúp hoặc chọn lại điểm đầu.')
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
            self.status.setText('Cọ: giữ chuột trái để thêm, chuột phải để xóa. Bán kính theo mm.')

    def start_landmarks(self):
        if self.session:
            self.stop_editing()
            self.remember()
            self.session.set_landmarks([])
            self.mode = 'landmark'
            self.redraw()
            self.status.setText('Mốc 1/6: '+LANDMARK_LABELS[0])

    def remove_landmarks(self):
        if self.session:
            self.stop_editing()
            self.remember()
            self.session.set_landmarks([])
            self.redraw()

    def _on_image(self, event):
        if not self.session or event.inaxes != self.body_ax or event.xdata is None or event.ydata is None:
            return False
        sx, sy = self.session.plane.spacing_xy_mm
        h, w = self.session.mask.shape
        return -sx/2 <= event.xdata <= (w-.5)*sx and -sy/2 <= event.ydata <= (h-.5)*sy

    def click(self, event):
        if not self._on_image(event):
            return
        if self.mode == 'landmark' and event.button == 1:
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

    def measure(self):
        try:
            self.stop_editing()
            report = self.session.measure(samples=self.samples.value(), seed=self.seed.value())
            warnings = [WARNINGS.get(w, w) for w in report['warnings']]
            self.status.setText('Đã đo. '+ ('; '.join(warnings) if warnings else 'Kết quả cần đối chiếu với vùng tham chiếu.'))
            self.redraw()
        except (ValueError, RuntimeError) as error:
            self.error(error)

    def find_blobs(self):
        if not self.session:
            return
        self.stop_editing()
        try:
            self.session.candidates = detect_blobs(self.session.ct, self.session.plane.spacing_xy_mm,
                self.session.valid, min_sigma_mm=self.log_min.value(), max_sigma_mm=self.log_max.value(),
                threshold=self.log_threshold.value())
            self.session.result = None
            self.blob_choices.blockSignals(True)
            self.blob_choices.clear()
            for i, candidate in enumerate(self.session.candidates['candidates'], 1):
                x, y = candidate['centre_xy_mm']
                self.blob_choices.addItem(f'B{i}: ({x:.1f}, {y:.1f}) mm · sigma {candidate["sigma_mm"]:.1f}')
            self.blob_choices.blockSignals(False)
            self.candidate_changed()
            self.status.setText(f'Tìm thấy {self.blob_choices.count()} ứng viên. Chọn tâm tham khảo rồi vẽ vùng thân đốt; '
                                'vòng tròn LoG biểu diễn thang đo.')
        except (ValueError, RuntimeError) as error:
            self.error(error)

    def candidate_changed(self, *_):
        self.redraw()

    def save(self):
        if not self.session:
            return
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.error(error)
            return
        selected = QFileDialog.getExistingDirectory(self, 'Chọn thư mục cha để lưu một kết quả mới', str(self.output_root))
        if not selected:
            return
        output = Path(selected)/f'body_{self.level}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
        try:
            save_session(self.session, output, samples=self.samples.value(), seed=self.seed.value())
            self.status.setText(f'Đã lưu {"kết quả đo" if self.session.reviewed else "vùng nháp"}: {output}')
            self.redraw()
        except (ValueError, OSError, RuntimeError) as error:
            self.error(error)

    def open_saved(self):
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
        self.stop_editing()
        self.session = session
        self.history.clear()
        self.blob_choices.blockSignals(True)
        self.blob_choices.clear()
        for i, candidate in enumerate((session.candidates or {}).get('candidates', []), 1):
            x, y = candidate['centre_xy_mm']
            self.blob_choices.addItem(f'B{i}: ({x:.1f}, {y:.1f}) mm · sigma {candidate["sigma_mm"]:.1f}')
        self.blob_choices.blockSignals(False)
        self.age.blockSignals(True)
        self.age.setCurrentIndex(self.age.findData(session.age_group))
        self.age.blockSignals(False)
        self.status.setText('Đã mở vùng, kiểm tra khớp CT. Phép đo sẽ được tính lại khi bấm Đo. '
                            'Thông số dịch/xoay bên trái chỉ dùng khi tạo mặt phẳng mới.')
        self.redraw()

    def redraw(self, *_):
        # A polygon selector owns its axes until completion; don't replace them.
        if self.selector:
            self.sync()
            return
        self.figure.clear()
        self.body_ax, normalized_ax = self.figure.subplots(1, 2)
        self.table.setRowCount(0)
        if self.session:
            draw_body(self.body_ax, self.session, show_points=self.show_points.isChecked(), show_blobs=self.show_blobs.isChecked())
            draw_normalized(normalized_ax, self.session)
            candidates = (self.session.candidates or {}).get('candidates', [])
            index = self.blob_choices.currentIndex()
            if self.show_blobs.isChecked() and 0 <= index < len(candidates):
                self.body_ax.plot(*candidates[index]['centre_xy_mm'], '+', color='#fc64f0', markersize=15, markeredgewidth=2)
            report = self.session.result if self.session.reviewed else None
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
                self.table.setRowCount(len(values))
                for row, (name, value) in enumerate(values):
                    self.table.setItem(row, 0, QTableWidgetItem(name))
                    self.table.setItem(row, 1, QTableWidgetItem(f'{value:,.4f}' if isinstance(value, (float, int)) else value))
        else:
            for ax in (self.body_ax, normalized_ax):
                ax.set_axis_off()
            self.body_ax.text(.5, .5, 'Áp dụng mặt phẳng để chọn vùng mới', ha='center', va='center', transform=self.body_ax.transAxes)
        self.canvas.draw_idle()
        self.sync()

    def closeEvent(self, event):
        self.stop_editing()
        super().closeEvent(event)
