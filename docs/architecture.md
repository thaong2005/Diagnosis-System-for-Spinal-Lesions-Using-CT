# Tổ chức dự án

Cập nhật: 30/10/2026.

## Luồng và mã đang dùng

Giữ ba phần `apps/`, `spinal_ct/`, `tools/`, dùng `python -m` từ gốc dự án.
`start_demo.bat` là điểm mở ứng dụng chính. Chưa cần đổi toàn bộ đường dẫn
sang `src/` hoặc chia repository thành nhiều dịch vụ.

| Thành phần | Trách nhiệm |
|---|---|
| `apps/demo.py` | Chọn CT, chạy backend, xem kết quả và mở cửa sổ đo |
| `apps/body_review.py` | Chọn mặt phẳng, chỉnh mask/mốc, xác nhận, đo và lưu |
| `spinal_ct/ct_io.py` | HU và geometry dùng chung |
| `spinal_ct/localization.py` | Ánh xạ nhãn mô hình, ROI/thể tích và preview |
| `spinal_ct/review.py`, `overlays.py` | Kiểm tra CT–kết quả, lưới hiển thị và overlay |
| `body_region.py`, `blob_analysis.py` | Mặt phẳng LPS, vùng thân đốt và đặc trưng theo mm |
| `blob_detection.py` | LoG thử nghiệm; kết quả phụ trợ cho người kiểm tra |
| `area_monte_carlo.py` | Lấy mẫu điểm trên mask để ước lượng diện tích |
| `monte_carlo.py` | Thêm nhiễu HU và tổng hợp phân đoạn; mục đích khác MC diện tích |
| `morphometry.py`, `morphometry_render.py` | Trạng thái vùng, schema, lưu/mở và báo cáo |
| `annotations.py`, `labels.py` | Annotation/nhãn phát triển; không làm đầu vào suy luận |

Đọc CT ở QThread và suy luận bằng QProcess để demo tiếp tục phản hồi. Mask và
phép đo gốc không đổi khi resample riêng lưới hiển thị. Kết quả định vị mới lưu
label map/schema 2; vùng thân đốt là schema riêng, cần người dùng xác nhận.

Thay mask/mốc/nhóm tuổi hủy xác nhận và bảng đo. Đổi tham số MC chỉ hủy bảng
đo để tính lại. LoG có trạng thái riêng: đổi thông số bỏ ứng viên cũ, chạy lại
không xóa phép đo trên mask. Mở vùng khôi phục mặt phẳng, nhóm tuổi, cấu hình
LoG và N/seed; phép đo được tính lại khi người dùng yêu cầu.

## Dữ liệu và tệp sinh ra

- `data/raw/`, `data/processed/`: dữ liệu và kết quả chuẩn bị cục bộ.
- `data/demo_runs/`: mỗi lượt suy luận một thư mục mới, chỉ công bố khi hoàn tất.
- `data/body_measurements/`: mask thân đốt, mốc, báo cáo và ảnh người dùng lưu.
- `.venv`, `.venv-model`, `.tmp`: môi trường và cache cục bộ, được Git bỏ qua.
- Thư mục `YYYY-MM-DD/`: workspace của từng phiên, gồm `work` và `outputs`,
  được Git bỏ qua. Giữ workspace đang dùng để các chat/đường dẫn tiếp tục hoạt động.
- `Expected_output/`: ảnh tham chiếu cục bộ; giữ lại đến khi đối chiếu nguồn.

Không commit CT, ảnh ca bệnh, cấu hình cá nhân hoặc kết quả chứa dữ liệu bệnh
nhân. Chỉ đưa báo cáo văn bản đã chọn lọc vào `docs/reports/`.

## Tài liệu và công cụ lịch sử

`README.md` chứa cách bắt đầu và phạm vi hiện tại; `note.txt` giữ yêu cầu.
Hướng dẫn vận hành ở `docs/guides/`, bằng chứng kỹ thuật ở `docs/reports/`.
Tài liệu cũ được giữ ở `docs/archive/FS/` và `docs/archive/lesion_detection/`,
có [mục lục archive](archive/README.md); không dùng các checkpoint cũ để suy ra tiến độ hiện tại.

Các viewer độc lập và annotation 3D vẫn có chức năng riêng; xem
[công cụ phát triển](guides/development_tools.md). Chưa gộp/xóa khi chưa thay
thế đủ chức năng xuất ảnh và chuẩn bị nhãn.

## Quy trình dọn và Git

Lưu thay đổi tính năng thành một commit riêng, rồi commit phần tinh gọn.
Kiểm tra danh sách staged, chạy test và xem lại message trước khi commit.
Tệp trung gian lịch sử chỉ được dọn sau khi đóng gói và đối chiếu hash;
bản sao và manifest ở workspace cục bộ. Không xóa dữ liệu hoặc trọng số
chỉ vì có nhiều kết quả cùng ca.
