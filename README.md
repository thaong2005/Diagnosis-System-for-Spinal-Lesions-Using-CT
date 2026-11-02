# Diagnosis System for Spinal Lesions Using CT

Công cụ nghiên cứu để xem CT cột sống, phân đoạn/định vị đốt sống và đo hình
dạng thân đốt trên vùng được người dùng kiểm tra. Chưa triển khai chẩn đoán
tổn thương hoặc phân loại AO; cần đối chiếu phép đo với nhãn chuyên gia.

## Bắt đầu

1. Cài môi trường theo [hướng dẫn Windows](docs/guides/windows_integration.md).
2. Mở `start_demo.bat` hoặc chạy `.\.venv\Scripts\python.exe -m apps.demo`.
3. Mở CT; chọn kết quả cũ được tìm thấy hoặc chạy định vị mới.
4. Chọn Baseline/Consensus và một đốt, bấm **Đo thân đốt**.
5. Kiểm tra mặt phẳng, vẽ/chỉnh mask thân đốt, đặt mốc nếu cần, xác nhận và đo.
6. Lưu vào thư mục kết quả mới; có thể mở lại vùng và xuất JSON/CSV/ảnh/HTML.

[Hướng dẫn sử dụng đầy đủ](docs/guides/User_guild.md) ·
[Đo thân đốt và MC diện tích](docs/guides/body_morphometry.md).

LoG nằm trong nhóm **Thử nghiệm LoG** ở cuối cửa sổ đo. Monte Carlo nhiễu HU
nằm trong nhóm **Thử nghiệm nhiễu HU** ở demo chính; hai nhóm được thu gọn
mặc định. MC diện tích là phép lấy mẫu điểm trên mask, có cấu hình riêng.

Ca mẫu được cấu hình trong `sample_config.py`, tạo từ
`sample_config.example.py`. Cấu hình cá nhân và dữ liệu không đưa vào Git.

## Phạm vi hiện tại

| Chức năng | Trạng thái |
|---|---|
| Đọc DICOM, NIfTI, NPY | Kiểm tra HU/geometry; xem ba mặt phẳng |
| TotalSegmentator | Mask toàn đốt, tên đốt, ROI, thể tích; chạy được trên Windows CPU |
| Vùng thân đốt 2D | Chọn mặt phẳng, vẽ/chỉnh/duyệt thủ công, đặt sáu mốc chiều cao |
| Blob analysis và chuẩn hóa | Moment ellipse, diện tích/tỷ lệ/solidity/chiều cao; ảnh 128 × 128 giữ tỷ lệ |
| MC diện tích | Lấy mẫu điểm, seed, sai số và hội tụ; đối chiếu đếm pixel |
| LoG | Ứng viên tâm/thang đo; cần kiểm chứng, chưa tự động tạo mask thân đốt |
| MC nhiễu HU | Baseline, consensus, disagreement và thống kê độ nhạy phân đoạn |
| Đánh giá chuyên gia và phân loại AO | Chưa hoàn thành |

và đánh giá LoG trên nhiều ca.
[Đợt cải tiến UI và đo hiệu năng 31/10](docs/reports/2026-10-31-workflow.md).
Kiểm thử phần mềm dùng dữ liệu tổng hợp; không thay thế đánh giá trên người.

## Cấu trúc

```text
apps/                 Demo chính, cửa sổ đo và công cụ giao diện phát triển
spinal_ct/            Đọc ảnh, định vị, phép đo, Monte Carlo và schema
tools/                CLI chuẩn bị dữ liệu, suy luận và đánh giá
tests/                Kiểm thử tự động
docs/guides/          Hướng dẫn đang áp dụng
docs/reports/         Báo cáo kỹ thuật đã chọn lọc
docs/archive/         Tài liệu lịch sử, có thể chứa yêu cầu đã thay đổi
data/                 CT và kết quả cục bộ, không đưa vào Git
```

Môi trường `.venv` dành cho ứng dụng, `.venv-model` cho TotalSegmentator;
trọng số ở `.tmp/totalsegmentator`. Workspace theo ngày là bản nháp/đầu ra
cục bộ, được Git bỏ qua. [Quy tắc tổ chức](docs/architecture.md).

## Lệnh thường dùng

Chạy các lệnh từ thư mục gốc, không cần activate môi trường:

```powershell
.\.venv\Scripts\python.exe -m tools.inspect_dicom --input data/raw/case_demo
.\.venv\Scripts\python.exe -m tools.prepare_ct --input data/raw/case_demo --output data/processed/case_demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Chuẩn bị CT giữ HU và geometry; annotation tham chiếu được tạo riêng.
Các CLI có `--help`; thư mục output phải mới.

[Định vị đốt sống](docs/guides/vertebral_localization.md) ·
[MC nhiễu HU](docs/guides/monte_carlo.md) ·
[Công cụ phát triển](docs/guides/development_tools.md) ·
[Kiểm chứng tối ưu 26/10/2026](docs/reports/2026-10-26-performance.md).
