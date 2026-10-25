# Diagnosis System for Spinal Lesions Using CT

Dự án nghiên cứu xử lý CT cột sống, annotation 3D, định vị đốt sống bằng
TotalSegmentator và đánh giá độ nhạy phân đoạn bằng nhiễu Monte Carlo.

## Demo cho giáo viên

Nhấp đúp **start_demo.bat** để mở giao diện chọn CT, chạy định vị/Monte Carlo,
xem overlay ba mặt phẳng, mở lại kết quả và xuất báo cáo HTML.
[Hướng dẫn từng bước](docs/guides/teacher_demo.md).

Để đặt ca mẫu riêng, sao chép `sample_config.example.py` thành `sample_config.py`
và sửa dòng `SAMPLE_DIR`. File cấu hình cục bộ này không được commit lên GitHub.

```powershell
.\.venv\Scripts\python.exe -m apps.demo
```

## Trạng thái hiện tại

- Đã chạy baseline và Monte Carlo 2 mẫu trên Windows CPU cho `case_053`.
- Có mask, ROI, thể tích, bản đồ bất đồng và ảnh QC. Cảnh báo biên ảnh vẫn cần review.
- Chưa có tính diện tích bằng lấy mẫu điểm, phân loại AO hoặc đánh giá trên tập tham chiếu.
- Viewer, annotation và inference là các công cụ riêng.

## Cấu trúc

```text
apps/                       Giao diện tương tác
  demo.py                   Quy trình demo và kiểm tra kết quả thống nhất
  view_mpr.py               Xem axial / sagittal / coronal
  view_sagittal.py           Chỉnh cửa sổ HU, xuất ảnh sagittal
  annotate_ct.py            Tạo annotation ROI 3D nháp
spinal_ct/                  Thư viện xử lý dùng chung
  ct_io.py                  Đọc CT, HU và kiểm tra geometry
  localization.py           Mask → ROI, báo cáo và ảnh QC
  monte_carlo.py            Nhiễu HU, đồng thuận và thống kê
  annotations.py            Schema / validator annotation
  labels.py                 Nhãn tham chiếu cho phát triển
  review.py                 Kiểm tra CT–mask, lưới hiển thị và báo cáo HTML
tools/                     CLI prepare, inspect, audit, validate, localize, render
tests/                     Kiểm thử tự động
docs/
  guides/                   Hướng dẫn sử dụng
  architecture.md           Lựa chọn cấu trúc và thay đổi đường dẫn
  archive/                  Tài liệu lịch sử
data/                      Dữ liệu và kết quả cục bộ, không commit
.tmp/totalsegmentator/      Cache trọng số, không commit
```

[Xem các phương án cấu trúc và nhật ký dọn dẹp](docs/architecture.md).

## Môi trường

Python 3.13; `.venv` cho ứng dụng và `.venv-model` cho TotalSegmentator.
`requirements.txt` khai báo dependency ứng dụng; `requirements-localization.txt`
khai báo backend. Các lệnh chạy tại thư mục gốc dự án, không cần activate.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

[Cài backend Windows](docs/guides/windows_integration.md) ·
[Định vị đốt sống](docs/guides/vertebral_localization.md) ·
[Monte Carlo](docs/guides/monte_carlo.md)

## Các lệnh thường dùng

Xem CT và tạo annotation:

```powershell
.\.venv\Scripts\python.exe -m apps.view_mpr --input data/processed/case_053/ct_hu.npy
.\.venv\Scripts\python.exe -m apps.view_sagittal --case data/processed/case_053
.\.venv\Scripts\python.exe -m apps.annotate_ct --case data/processed/case_053
```

Chuẩn bị ca mới (thay đường dẫn mẫu; chọn `--series-uid` nếu có nhiều series):

```powershell
.\.venv\Scripts\python.exe -m tools.inspect_dicom --input data/raw/case_demo
.\.venv\Scripts\python.exe -m tools.prepare_ct --input data/raw/case_demo --output data/processed/case_demo
```

Lệnh chuẩn bị CT không ghi đè output có sẵn và chỉ lưu HU cùng geometry.
Annotation tham chiếu được tạo riêng bằng công cụ annotation.
Viewer và inference dùng trực tiếp output này. Công cụ annotation hiện cần thêm
`case_id` ẩn danh và `reference_label.vertebral_level` trong metadata phát triển,
như ca mẫu hiện có; bộ chuẩn bị CT không tự suy diễn các nhãn đó.

Dựng lại ảnh QC từ mask đã lưu, không chạy lại mô hình (output phải mới):

```powershell
.\.venv\Scripts\python.exe -m tools.render_localization --input data/processed/case_053/ct_hu.npy --results data/processed/case_053/localization_mc --output data/processed/case_053/localization_mc/previews_v3
```

Ảnh dùng trục mm, biên theo mask và bản đồ bất đồng. Dựng ảnh không làm trơn
mask hoặc đổi phép đo. Phân đoạn `--fast` 3 mm vẫn có thể cho biên bậc thang.

Kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các lệnh khác có hướng dẫn qua `python -m tools.<tên_lệnh> --help`.
Không commit CT, ảnh ca bệnh, môi trường Python hoặc cache trọng số.
