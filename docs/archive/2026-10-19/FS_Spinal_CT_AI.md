> Bản lưu lịch sử ngày 19/10/2026. Trạng thái và đường dẫn mã trong nội dung có thể đã lỗi thời. Xem [README hiện tại](../../../README.md).

# FS Spinal CT — Tài liệu vận hành cho AI agent

## 0. Mục đích

Đây là context bắt buộc cho AI agent làm việc trong repository. Agent phải đọc tài liệu này trước khi sửa code liên quan đến CT, annotation, dataset, model hoặc evaluation.

Tài liệu dành cho người đọc là [FS_Spinal_CT.md](FS_Spinal_CT.md). Hai tài liệu phải nhất quán; khi thay đổi trạng thái dự án, cập nhật cả hai.

## 1. Mission

Xây dựng hệ thống hỗ trợ đánh giá tổn thương xương cột sống từ CT. Phiên bản đầu tiên ưu tiên:

1. Dựng CT đúng geometry.
2. Định vị các đốt sống.
3. Xếp hạng đốt/vùng nghi tổn thương.
4. Trả về ROI, mức đốt, confidence và trạng thái từ chối.

Không tuyên bố hệ thống thay thế bác sĩ.

## 2. Production contract — bất biến

### Input duy nhất

```text
Một bộ ảnh CT/DICOM của một bệnh nhân
```

Production inference không được nhận:

- CSV nhãn.
- Vị trí đốt gãy.
- ROI hoặc mask ground truth.
- Mã phân loại.
- Tên, năm sinh, PatientID hoặc mapping ID.

### Output tối thiểu

```json
{
  "status": "ok|rejected|indeterminate",
  "predictions": [
    {
      "level": "L1",
      "roi_lps_mm": {"min": [], "max": []},
      "confidence": 0.0
    }
  ],
  "geometry": {},
  "model_version": "...",
  "rejection_reason": null
}
```

Nếu CT không đạt điều kiện, trả `rejected` hoặc `indeterminate`; không dự đoán im lặng.

### API tách biệt

```python
predict(ct_path, config)              # production: chỉ nhận CT
evaluate(prediction, reference_label) # development: mở ground truth sau inference
```

Không import module CSV/label vào đường gọi `predict`.

## 3. Development-only assets

File [data ngày 27.09.2026.csv](../../../data/raw/A4/data%20ngay%2027.09.2026.csv) là ground truth phát triển, gồm 54 dòng có mức đốt và mã phân loại. Không coi nó là input thực tế.

CSV được dùng để:

- Liên kết CT với nhãn trong bộ chuẩn bị dữ liệu.
- Tạo annotation/ROI.
- Huấn luyện hoặc hiệu chỉnh.
- Mở sau inference để tính metric.

CSV không có bounding box 3D, mask tổn thương, nhãn mức độ độc lập hoặc đủ ca không gãy để đo false positives per scan.

Phân bố mã: A1=1, A3=14, A4=19, B2=14, B3=1, C=5. Không suy diễn các mã này thành nhẹ/vừa/nặng.

## 4. Evidence hiện tại

- Python 3.13, Windows CPU.
- Pipeline DICOM/HU/geometry đã xác minh trên một ca mẫu `case_053`.
- Có `ct_hu.npy`, `metadata.json`, MPR và viewer sagittal nâng cao cho ca mẫu.
- Reader/normalizer nhãn: [spinal_ct/labels.py](../../../spinal_ct/labels.py).
- Schema/validator ROI 3D: [spinal_ct/annotations.py](../../../spinal_ct/annotations.py).
- CLI kiểm tra annotation: [tools/validate_annotation.py](../../../tools/validate_annotation.py).
- Audit DICOM: [tools/audit_dataset.py](../../../tools/audit_dataset.py).
- Unit test parser: [tests/test_labels.py](../../../tests/test_labels.py).
- Unit test annotation: [tests/test_annotations.py](../../../tests/test_annotations.py).
- Audit tổng hợp: [data/processed/dataset_audit.json](../../../data/processed/dataset_audit.json).
- Audit có 10 PatientID, 53 CT series, 7.638 CT images; chỉ 4 PatientID khớp trực tiếp với CSV.
- JSON annotation cũ vẫn là sagittal 2D nháp; `annotate_ct.py` hiện đã hỗ trợ tạo ROI 3D draft, chưa có ROI 3D reviewed.
- Viewer hiện chỉ xem/xuất, chưa tự phát hiện hoặc phân loại.

## 5. Trạng thái task

### Đã hoàn thành

- Đọc DICOM và chọn series ca mẫu.
- Kiểm tra hình học và chuyển HU.
- Chuẩn hóa LPS, sagittal MPR và viewer cơ bản/nâng cao.
- Xuất ảnh và metadata tái lập.
- Parse CSV không xuất PII.
- Audit DICOM aggregate.
- Unit test parser nhãn.

### Đang làm

- Chuẩn bị chuyển sang localization.
- Xác định mapping DICOM–CSV.
- Đã có schema/validator annotation 3D; đang chờ tích hợp viewer.

### Chưa làm

- Manifest ẩn danh đã xác nhận.
- Split theo bệnh nhân và leakage check.
- CLI DICOM không hard-code case.
- ROI 3D reviewed và tập annotation validation.
- Annotation đồng bộ ba mặt phẳng.
- Định vị từng đốt.
- Baseline xếp hạng biến dạng.
- Prediction/evaluation API.
- Đánh giá test khóa.

## 6. Execution order bắt buộc

### T005 — Mapping và manifest

Tạo mapping riêng tư và manifest ẩn danh. Trạng thái mỗi ca phải là `matched`, `ambiguous` hoặc `unmatched`. Chỉ `matched` đã xác nhận mới đủ điều kiện dùng cho phát triển.

DoD:

- Không có PII trong manifest commit được.
- Có lý do cho mọi ca chưa khớp.
- Có test phát hiện mapping một-nhiều hoặc nhiều-một.

### T006 — Split theo bệnh nhân

Tạo splitter có seed. Tất cả series của một bệnh nhân phải cùng split. Không chia lát cắt độc lập.

DoD:

- Có file split ẩn danh.
- Có leakage test.
- Test set được khóa trước khi metric.

### T007 — DICOM CLI

Refactor code hard-code `case_053` thành CLI nhận input/output. Giữ nguyên kiểm tra shape, spacing, origin, direction, HU và ảnh xiên.

DoD:

- Chạy được trên nhiều series.
- Output có geometry metadata.
- Output không có PatientName/PatientID.

### T008–T010 — Annotation 3D

Định nghĩa schema ROI 3D, validator và viewer ba mặt phẳng. Trước tiên sửa ca mẫu L1, sau đó tạo tập validation được review.

DoD:

- ROI lưu cả voxel index và LPS mm.
- Có `draft/reviewed/rejected`.
- Geometry annotation khớp volume.
- Có ít nhất một ROI `reviewed`.

### T011–T014 — Localization baseline

Benchmark segmentation đốt sống tiền huấn luyện, tạo ROI từng đốt, trích đặc trưng hình học và xếp hạng ứng viên. Chưa thử 3D CNN khi chưa có baseline, dữ liệu và GPU phù hợp.

DoD:

- Inference không đọc CSV.
- Có prediction JSON độc lập ground truth.
- Có top-1/top-k level metric trên test khóa.

## 7. Gates

Dừng và báo blocker nếu:

- Không xác nhận được mapping CT–CSV.
- Geometry volume không hợp lệ.
- ROI còn là draft nhưng bị dùng như ground truth.
- Không biết một dòng CSV đại diện một hay nhiều tổn thương.
- Dùng đốt không ghi nhãn làm âm tính chắc chắn.
- Test set có leakage theo bệnh nhân.
- Muốn báo false-positive rate nhưng chưa có ca không gãy.
- Muốn công bố sáu lớp nhưng lớp test có support không đủ.

## 8. Privacy rules

- Không in tên, năm sinh, PatientID hoặc ID CSV trong log.
- Không commit CT, CSV, mapping ID, PNG có ảnh bệnh nhân hoặc JSON chứa PII.
- Dùng `case_id` ẩn danh trong artifact.
- Audit chỉ xuất aggregate.
- Trước demo/release phải chạy privacy audit.

## 9. Evaluation protocol

```text
1. Khóa test set theo bệnh nhân.
2. Chạy predict(CT) không truyền label.
3. Lưu prediction và confidence.
4. Mở reference label sau inference.
5. Tính level accuracy, level-distance, candidate recall và ROI error.
6. Báo rõ ca rejected/indeterminate.
```

Sensitivity và false positives per scan chỉ hợp lệ khi có ca không gãy đại diện. Phân loại sáu lớp chỉ đánh giá khi đủ support và ý nghĩa mã đã được người gán nhãn xác nhận.

## 10. Agent handoff format

Mỗi agent khi hoàn thành task phải báo:

```text
Task: Txxx
Files changed: ...
Input contract preserved: yes/no
CSV used in inference: yes/no
Tests run: ...
Artifacts produced: ...
Known blockers: ...
Next recommended task: ...
```

Agent không được tự mở rộng phạm vi sang training, deployment hoặc thay đổi nhãn khi gate tương ứng chưa đạt.
