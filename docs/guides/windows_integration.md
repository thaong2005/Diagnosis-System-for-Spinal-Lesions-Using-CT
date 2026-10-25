# Cài đặt và chạy trên Windows

## Phạm vi

Mã định vị và Monte Carlo đã tích hợp vào repository. Baseline và 2 mẫu nhiễu
đã chạy trên Windows CPU với `case_053` ngày 24/10/2026. Kết quả có cảnh báo
T8/T9 chạm biên; việc chạy thành công không xác nhận độ chính xác phân đoạn.

## 1. Mã đã tích hợp

Mã nằm trong `spinal_ct/`, CLI trong `tools/`, giao diện trong `apps/`,
test trong `tests/` và hướng dẫn trong `docs/guides/`. Không cần gói ZIP.

## 2. Cài đặt bằng PowerShell tại thư mục gốc dự án

Môi trường ứng dụng đã có `.venv`; sử dụng trực tiếp executable để không phụ
thuộc chính sách chạy `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Cài backend trong môi trường riêng. Máy Windows đã xác minh Python 3.13.0,
torch 2.14.1+cpu, torchvision 0.29.1+cpu và TotalSegmentator 2.18.0.
Nếu pip báo không có wheel phù hợp trên máy khác, cần kiểm tra
phiên bản Python và kiến trúc máy trước khi chọn cặp torch/torchvision khác.

```powershell
py -3.13 -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install --upgrade pip
.\.venv-model\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-model\Scripts\python.exe -m pip install -r requirements-localization.txt
.\.venv-model\Scripts\python.exe -m pip check
.\.venv-model\Scripts\TotalSegmentator.exe --version
```

Backend cần hiển thị `2.18.0`. Lần suy luận đầu tải trọng số qua HTTPS; cần mạng
đến GitHub release assets. Không sao chép môi trường Linux cloud sang Windows.
Đảm bảo `.gitignore` có `.venv*/`, `.tmp/` và `/data/`.

## 3. Kiểm thử code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test CLI fixture dùng executable script Unix nên được bỏ qua trên Windows;
các kiểm thử số học, HU, geometry và annotation vẫn chạy. Dùng lệnh suy luận
thật ở bước tiếp theo để kiểm tra tích hợp backend trên Windows.

## 4. Chạy ca đã chuẩn bị

Trong `data/processed/case_053/` cần có:

```text
ct_hu.npy       # mảng HU numeric 3D theo [z, y, x]
metadata.json   # shape_zyx, spacing_zyx_mm, origin_lps_mm và direction
```

Direction dùng `image_orientation_patient` (6 số) hoặc `direction_lps` (9 số,
row-major, hoặc ma trận 3×3). Nếu cả hai tồn tại chúng phải nhất quán.
Các trường `reference_label`, `case_id`, annotation và mã AO không được đưa
vào mô hình hoặc sao chép vào prediction. Chỉ geometry được đọc từ JSON.

Chạy baseline trước bằng một dòng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m tools.localize_vertebrae --input "data/processed/case_053/ct_hu.npy" --output "data/processed/case_053/localization" --backend ".\.venv-model\Scripts\TotalSegmentator.exe" --device cpu --fast
```

Sau đó chạy Monte Carlo với 2 mẫu để thử chi phí tính toán:

```powershell
.\.venv\Scripts\python.exe -m tools.localize_vertebrae --input "data/processed/case_053/ct_hu.npy" --output "data/processed/case_053/localization_mc" --backend ".\.venv-model\Scripts\TotalSegmentator.exe" --device cpu --fast --mc-samples 2 --mc-noise-hu 20 --mc-seed 42
```

Đây là 1 baseline + 2 lần suy luận mẫu nhiễu. Khi cần đánh giá ổn định hơn có
thể tăng lên 5–20 mẫu sau khi đo runtime/RAM; sigma 20 HU là thử nghiệm.
Output phải chưa tồn tại. Nếu chạy lại hãy chọn tên thư mục mới.
Để chạy DICOM, thay `--input` bằng thư mục một ca và thêm `--series-uid` nếu
có nhiều series. Không dùng tên thư mục phân loại A1/A3/... làm đầu vào nhãn.

## 5. Xem kết quả và quy ước tích hợp

Mở `levels_mpr.png`, `monte_carlo/consensus_mpr.png` và `prediction.json` trong
thư mục output. Mask 3D có thể xem bằng trình xem NIfTI như 3D Slicer.
Viewer Qt hiện có chưa được thêm nút chạy hoặc overlay dự đoán.

**Annotation và prediction là hai loại dữ liệu khác nhau:**

- Annotation của bạn có `target=vertebral_body`, voxel theo z/y/x, trạng thái
  `draft/reviewed/rejected`, và có thể chứa nhãn tham chiếu.
- Mask TotalSegmentator là toàn bộ đốt sống, có thể gồm các thành phần phía
  sau; prediction dùng voxel x/y/z và không được coi là ground truth.
- `apps.annotate_ct.physical_bbox_from_voxel` hiện tính từ index min/max_exclusive;
  ROI prediction tính biên voxel với offset −0.5. Không so sánh trực tiếp hai
  physical box trước khi thống nhất quy ước này và mục tiêu giải phẫu.
- Chưa chuyển prediction thành annotation `reviewed`, chưa tính Dice với ROI
  thân đốt và chưa tích hợp phân loại AO. Cần chuẩn tham chiếu tương ứng.

## Khi chạy ca mới

Không cần dữ liệu người bệnh để chạy test tổng hợp. Để chạy ca thật cần
`ct_hu.npy` + metadata đúng geometry hoặc DICOM; khi có lỗi hãy gửi thông báo
lỗi đã bỏ định danh. Để tối ưu runtime cần phiên bản Python, RAM, CPU/GPU/VRAM.
Geometry của `case_053` đã được kiểm tra; ca mới vẫn cần được kiểm tra riêng.
Monte Carlo đo độ nhạy với nhiễu, không xác định trực tiếp
mức độ tổn thương xương hoặc phân loại AO.
