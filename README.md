# Diagnosis System for Spinal Lesions Using CT

Dự án nghiên cứu xây dựng công cụ hỗ trợ kiểm tra CT cột sống. Người dùng chọn
ảnh của một bệnh nhân, chạy mô hình định vị đốt sống và xem vùng xương được
phân đoạn trên ba mặt phẳng. Kết quả gồm tên đốt, vùng quan tâm (ROI), thể tích,
ảnh minh họa và báo cáo.

Hiện dự án dùng TotalSegmentator để phân đoạn và Monte Carlo để kiểm tra mức
thay đổi của kết quả khi thêm nhiễu vào CT. Mục tiêu tiếp theo là tính diện tích
thân đốt sống và đánh giá trên dữ liệu có nhãn chuyên gia. Chức năng phát hiện
tổn thương và phân loại AO chưa được triển khai.

## Demo cho giáo viên

Nhấp đúp **start_demo.bat** để mở giao diện chọn CT, chạy định vị/Monte Carlo,
xem overlay ba mặt phẳng, mở lại kết quả và xuất báo cáo HTML.
[Hướng dẫn từng bước](docs/guides/teacher_demo.md).

Để đặt ca mẫu riêng, sao chép `sample_config.example.py` thành `sample_config.py`
và sửa dòng `SAMPLE_DIR`. File cấu hình cục bộ này không được commit lên GitHub.

```powershell
.\.venv\Scripts\python.exe -m apps.demo
```

## Tiến độ dự án

Cập nhật ngày **25/10/2026**. Dự án đã có demo chạy từ CT đầu vào đến báo cáo;
chưa hoàn thành hệ thống chẩn đoán tổn thương.

| Hạng mục | Tiến độ |
|---|---|
| Đọc DICOM, NIfTI và NPY | Đã có bộ đọc chung, kiểm tra HU và hình học ảnh |
| Định vị và phân đoạn đốt sống | Đã chạy mô hình thật trên Windows CPU; xuất mask, ROI và thể tích toàn đốt |
| Giao diện demo | Đã chọn/đổi ca, xem ba mặt phẳng, mở kết quả và xuất HTML |
| Monte Carlo nhiễu HU | Đã có baseline, đồng thuận và bản đồ bất đồng; đã chạy 2 mẫu trên `case_053` |
| Annotation 3D | Đã có công cụ tạo và kiểm tra ROI nháp; cần chuyên gia duyệt |
| Diện tích thân đốt bằng Monte Carlo | Chưa triển khai lấy mẫu điểm để ước lượng diện tích |
| Đánh giá độ chính xác | Chưa có kết quả trên tập tham chiếu được chuyên gia duyệt |
| Phát hiện tổn thương, phân loại AO | Chưa triển khai |

**Kiểm chứng gần nhất:**

- Bộ kiểm thử: **35 đạt, 1 bỏ qua trên Windows** vì fixture dùng script Unix.
- Cả 10 lệnh CLI mở được trợ giúp; hai môi trường Python không có xung đột dependency.
- Baseline trên ca CT 567 lát: hoàn tất trong khoảng **7 phút 29 giây** trên máy
  kiểm tra, xuất 13 mức T6–S1. Đây là thời gian của một lượt chạy, không phải cam kết
  cho mọi máy. Kết quả `indeterminate` vì mask T6/T7 chạm biên ảnh.
- Đã xác minh kết quả khớp CT gốc, mở lại trong demo và xuất báo cáo HTML.
- Đã sửa lỗi thiếu RAM khi lưu mask song song bằng cách yêu cầu backend lưu tuần tự.
  Ca 567 lát mới kiểm tra baseline; chưa chạy Monte Carlo trên ca này.

Các kiểm tra trên xác nhận luồng phần mềm hoạt động, chưa chứng minh độ chính xác
phân đoạn. Tiến độ tiếp theo:

1. Thống nhất cách chọn lát cắt và vùng thân đốt để đo diện tích.
2. Triển khai ước lượng diện tích bằng lấy mẫu Monte Carlo và kiểm tra sai số.
3. Đối chiếu phân đoạn, tên đốt và phép đo với nhãn chuyên gia trên nhiều ca;
   sau đó phát triển và đánh giá chức năng phân loại tổn thương.

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
sample_config.example.py   Mẫu cấu hình đường dẫn ca demo
sample_config.py           Cấu hình ca trên máy, không commit
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
