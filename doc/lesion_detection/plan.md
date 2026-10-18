# Kế hoạch triển khai mô hình phát hiện vị trí tổn thương

## Tóm tắt

Mục tiêu phiên bản 0 là tự tìm đốt sống nghi tổn thương từ CT, không nhận trước
vị trí. Kiến trúc gồm tiền xử lý DICOM, định vị từng đốt, trích ROI, baseline
hình học/ảnh, xếp hạng ứng viên và đánh giá khóa nhãn. Với dữ liệu hiện tại,
không tuyên bố mô hình chẩn đoán hay phân loại sáu lớp hoàn chỉnh.

## Bối cảnh kỹ thuật

- Python 3.13, Windows, CPU; PyTorch Windows hiện hỗ trợ Python 3.13 nhưng mô
  hình 3D vẫn nên huấn luyện trên GPU.
- Dữ liệu nhãn: 54 dòng, 54 vị trí và 54 mã phân loại.
- Dữ liệu ảnh đang có trong workspace: 10 thư mục ca ở `data/raw`, trong đó mới
  một ca có volume HU và metadata đã kiểm tra.
- Audit DICOM hiện tìm thấy 10 PatientID, 53 CT series và 7.638 CT image; chỉ 4
  PatientID khớp trực tiếp với các ID trong CSV. Đây là gate cần giải quyết trước
  khi tạo split hoặc huấn luyện.
- Có 668 file dùng giá trị Specific Character Set không chuẩn `ISO IR 192`;
  pydicom có thể đọc bằng giá trị sửa tương ứng nhưng lỗi chất lượng này được
  giữ lại trong báo cáo audit.
- Nhãn không gian hiện có chỉ là một bounding box sagittal nháp, chưa được xác
  nhận và còn bao nhiều đốt.
- Dữ liệu và artifact ảnh luôn nằm dưới `data/` và không được commit.

## Nguyên tắc

1. Không dùng CSV trong hàm suy luận.
2. Không chia lát cắt ngẫu nhiên; chỉ chia theo bệnh nhân.
3. Mỗi phép biến đổi phải bảo toàn/ghi lại hình học vật lý.
4. Baseline đơn giản, đo được phải có trước mô hình sâu.
5. Không dùng các đốt không ghi nhãn làm âm tính chắc chắn cho đến khi xác nhận.
6. Mô hình phải có trạng thái từ chối khi CT không đạt điều kiện.

## Các giai đoạn triển khai

### P1 — Kiểm kê và hợp đồng dữ liệu

- **P1.1 [REQ-003, REQ-010]:** đọc CSV an toàn, chuẩn hóa ID trong bộ nhớ và
  chỉ xuất thống kê tổng hợp không chứa PII.
- **P1.2 [REQ-002, REQ-003]:** quét DICOM header, thống kê ca/series/kernel và
  đo tỷ lệ liên kết được với CSV.
- **P1.3 [REQ-004]:** tạo manifest ẩn danh và split khóa theo bệnh nhân sau khi
  quy tắc liên kết được xác nhận.
- **P1.4 [REQ-002]:** chuyển từng series hợp lệ thành volume HU + metadata bằng
  một pipeline dùng chung, thay cho đường dẫn ca hard-code.

### P2 — Ground truth không gian

- **P2.1 [REQ-005]:** sửa công cụ annotation để lưu ROI 3D trên ba mặt phẳng.
- **P2.2 [REQ-005, REQ-010]:** quy định schema annotation, trạng thái
  draft/reviewed và version của người kiểm tra.
- **P2.3 [REQ-005]:** tạo ít nhất một tập validation ROI đã được xác nhận độc
  lập để đo localization; nhãn mức đốt trong CSV không thay thế ROI.

### P3 — Định vị và gán nhãn đốt sống

- **P3.1 [REQ-005, REQ-009]:** benchmark mô hình segmentation tiền huấn luyện
  trên một số ca đại diện, ghi thời gian CPU và lỗi gán nhãn đốt.
- **P3.2 [REQ-005]:** chuyển mask từng đốt thành ROI 3D chuẩn hóa và kiểm tra
  thứ tự giải phẫu T/L.
- **P3.3 [REQ-007]:** phát hiện thiếu đốt, nhầm đốt lân cận, kim loại và ca ngoài
  trường chụp; không tiếp tục im lặng khi segmentation lỗi.

### P4 — Baseline phát hiện biến dạng

- **P4.1 [REQ-006, REQ-009]:** đo chiều cao trước/giữa/sau, thể tích, độ nghiêng
  mâm đốt và tỷ lệ so với hai đốt lân cận.
- **P4.2 [REQ-006]:** xây baseline xếp hạng bất thường không học hoặc mô hình
  nhỏ trên đặc trưng hình học.
- **P4.3 [REQ-006, REQ-011]:** tạo patch 2.5D/3D cho từng đốt; chỉ dùng nhãn yếu
  sau khi xác nhận quy tắc âm tính.
- **P4.4 [REQ-006, REQ-009]:** thử CNN 2.5D trước 3D CNN; ghi rõ thời gian CPU,
  nhu cầu GPU và seed.

### P5 — Đánh giá khóa nhãn

- **P5.1 [REQ-001, REQ-004, REQ-008]:** chạy inference chỉ với CT trên test set
  đã khóa, sau đó mới mở ground truth để tính metric.
- **P5.2 [REQ-008]:** báo top-1/top-3 level accuracy, khoảng cách theo số đốt,
  candidate recall và lỗi ROI.
- **P5.3 [REQ-008]:** bổ sung sensitivity và false positives/scan chỉ khi có ca
  không gãy đại diện.
- **P5.4 [REQ-011]:** đánh giá phân loại theo từng lớp; không công bố metric cho
  lớp có test support không đủ.

### P6 — Tích hợp trình xem

- **P6.1 [REQ-007]:** overlay ROI, mức đốt, điểm tin cậy và nguồn kết quả
  (ground truth hay prediction) trong viewer.
- **P6.2 [REQ-007]:** xuất JSON kết quả có version mô hình, geometry và trạng
  thái từ chối.
- **P6.3 [REQ-010]:** kiểm tra không có PatientName/PatientID trong log, ảnh và
  file có thể commit.

## Task breakdown

### Phase 1 — Setup và kiểm kê

- [x] T001 [Plan:P1.1] Tạo bộ đọc và chuẩn hóa nhãn trong `spinal_ct/labels.py`.
- [x] T002 [Plan:P1.1,P1.2] Tạo công cụ kiểm toán aggregate trong `tools/audit_dataset.py`.
- [x] T003 [Plan:P1.1] Thêm unit test không chứa PII trong `tests/test_labels.py`.
- [x] T004 [Plan:P1.2] Chạy audit toàn bộ DICOM và lưu `data/processed/dataset_audit.json`.
- [ ] T005 [Plan:P1.3] Xác nhận quy tắc liên kết DICOM–CSV và tạo manifest ẩn danh.
- [ ] T006 [Plan:P1.3] Tạo split bệnh nhân có seed và file kiểm tra leakage.
- [ ] T007 [Plan:P1.4] Refactor pipeline DICOM thành CLI không hard-code ca.

### Phase 2 — Annotation và localization

- [ ] T008 [Plan:P2.1,P2.2] Định nghĩa schema ROI 3D và validator.
- [ ] T009 [Plan:P2.1] Nâng cấp viewer thành annotation ba mặt phẳng đồng bộ.
- [ ] T010 [Plan:P2.3] Tạo tập ROI validation được xác nhận.
- [ ] T011 [Plan:P3.1] Benchmark công cụ segmentation đốt sống tiền huấn luyện.
- [ ] T012 [Plan:P3.2,P3.3] Chuyển mask thành ROI và kiểm tra thứ tự đốt.

### Phase 3 — Baseline phát hiện

- [ ] T013 [Plan:P4.1] Trích đặc trưng hình học theo từng đốt.
- [ ] T014 [Plan:P4.2] Xây baseline xếp hạng và lưu prediction độc lập nhãn.
- [ ] T015 [Plan:P4.3] Xây dataset patch sau khi chốt quy tắc nhãn âm.
- [ ] T016 [Plan:P4.4] Huấn luyện và benchmark baseline 2.5D.
- [ ] T017 [Plan:P4.4] Chỉ thử 3D CNN khi baseline và GPU đã sẵn sàng.

### Phase 4 — Đánh giá và tích hợp

- [ ] T018 [Plan:P5.1,P5.2] Khóa test set và chạy đánh giá level/ROI.
- [ ] T019 [Plan:P5.3] Bổ sung đánh giá ca không gãy.
- [ ] T020 [Plan:P5.4] Đánh giá đa lớp khi đủ support.
- [ ] T021 [Plan:P6.1,P6.2] Overlay prediction và xuất kết quả trong viewer.
- [ ] T022 [Plan:P6.3] Chạy privacy audit trước mỗi demo/release.

## Requirement Mapping

| REQ | Plan items | Bằng chứng dự kiến |
|---|---|---|
| REQ-001 | P5.1 | inference API không nhận label; test khóa nhãn |
| REQ-002 | P1.2, P1.4 | volume HU, metadata và geometry validation |
| REQ-003 | P1.1–P1.3 | audit aggregate và manifest ẩn danh |
| REQ-004 | P1.3, P5.1 | split manifest và leakage check |
| REQ-005 | P2.1–P3.3 | ROI/mask từng đốt và localization report |
| REQ-006 | P4.1–P4.4 | feature table, score và model checkpoint |
| REQ-007 | P3.3, P6.1–P6.2 | prediction JSON và viewer overlay |
| REQ-008 | P5.1–P5.4 | evaluation report theo bệnh nhân |
| REQ-009 | P3.1, P4.1–P4.4 | CPU benchmark và quyết định GPU |
| REQ-010 | P1.1, P2.2, P6.3 | privacy audit và ignore rules |
| REQ-011 | P4.3, P5.4 | class-support gate và per-class report |

## Rủi ro/gate trước khi huấn luyện

- Không huấn luyện detector nếu chưa liên kết được DICOM–CSV theo bệnh nhân.
- Không dùng ROI nháp hiện tại làm ground truth.
- Không coi đốt không ghi trong CSV là âm tính chắc chắn nếu chưa xác nhận.
- Không báo false-positive rate khi chưa có ca không gãy.
- Không huấn luyện sáu lớp khi train/test support của lớp hiếm không đủ.
