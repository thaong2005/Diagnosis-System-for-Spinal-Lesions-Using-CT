# Demo dành cho giáo viên hướng dẫn

## Mở ứng dụng

Tại thư mục gốc, nhấp đúp **start_demo.bat**. Có thể chạy tương đương:

```powershell
.\.venv\Scripts\python.exe -m apps.demo
```

Máy cần môi trường `.venv` theo [hướng dẫn Windows](windows_integration.md).
Backend `.venv-model` chỉ cần khi chạy suy luận mới. Ứng dụng vẫn xem CT và
kết quả đã lưu khi backend chưa sẵn sàng. Không chép môi trường Python từ
máy khác; tạo môi trường trên máy đích theo hướng dẫn.

## Kiểm tra nhanh bằng ca hiện có

1. Bấm **Mở ca mẫu**. Tên trên nút được lấy từ tên thư mục `SAMPLE_DIR`;
   nút chỉ bật khi thư mục tồn tại trên máy.
2. Ứng dụng tự đọc series axial có nhiều lát nhất trong các series vượt qua
   kiểm tra hình học. Nếu có nhiều series cùng số lát hoặc không có ứng viên,
   chọn trong danh sách rồi bấm **Đọc series đã chọn**. Bấm **Chạy định vị**
   để tạo kết quả mới; ứng dụng không tự mở kết quả của bệnh nhân trước.
3. Chọn **Baseline**, **Consensus** hoặc **Bất đồng**.
4. Kéo thanh lát cắt; nhấp trên ảnh để đồng bộ vị trí ba mặt phẳng.
5. Chọn một đốt trong bảng để di chuyển đến tâm mask. Bật/tắt overlay,
   chỉnh độ đậm hoặc WL/WW để kiểm tra ranh giới.
6. Đọc tab **Lưu ý kết quả**; `indeterminate` là trạng thái cần kiểm tra,
   không đồng nghĩa lỗi cài đặt.

Ca mẫu là dữ liệu cục bộ, không được đóng gói cùng mã nguồn. Khi chia sẻ
cho giáo viên, dùng ca đã khử định danh và được phép chia sẻ hoặc CT do giáo
viên cung cấp. Không có dữ liệu bệnh nhân nào được upload bởi ứng dụng demo.

Lần đầu, sao chép `sample_config.example.py` thành `sample_config.py` ở thư mục
gốc dự án. Để đổi ca mẫu, chỉ sửa **một dòng** trong `sample_config.py`, lưu
file rồi đóng và mở lại ứng dụng:

```python
SAMPLE_DIR = ROOT / 'data/raw/sample_patient'
```

Không cần sửa tên nút, kernel hay số lát. Thư mục có thể chứa nhiều thư mục con;
ứng dụng tự tìm các CT series bên trong. Với đường dẫn ngoài dự án, dùng
`SAMPLE_DIR = Path('D:/CT/ten_thu_muc')`.
`sample_config.py` được Git bỏ qua để đường dẫn chứa tên bệnh nhân không bị
đẩy lên GitHub. Nếu chưa tạo file này, demo dùng `data/raw/sample_patient`.

Quy tắc chọn tự động chỉ dựa trên hình học và số lát, không xác nhận ảnh có
đủ vùng cột sống cần xem hay series nào tốt nhất. Axial ở đây cho phép nghiêng
tối đa 15 độ; các ảnh khác vẫn có thể được chọn thủ công nếu bộ đọc hỗ trợ.
Kiểm tra ba mặt phẳng và series đang chọn trước khi chạy. Bạn có thể chọn
series khác trong danh sách bất cứ lúc nào khi ứng dụng không bận.

Để xem lại `case_053`,
chọn file `data/processed/case_053/ct_hu.npy` rồi mở thư mục kết quả cũ.

## Chạy với bệnh nhân khác

- **DICOM:** chọn thư mục; chọn CT series theo số lát, kích thước, kernel và
  độ dày lát. Bấm **Đọc series đã chọn**. Không cần đặt ca vào một lớp A1/A3/…
- **NIfTI:** chọn `.nii` hoặc `.nii.gz` đã có HU và geometry đúng.
- **NPY:** chọn `ct_hu.npy`; cần `metadata.json` bên cạnh.

Sau khi CT hợp lệ, bấm **Chạy định vị**. Mặc định dùng CPU và mô hình nhanh
3 mm. Bỏ chọn Chạy nhanh để dùng 1.5 mm; thời gian và bộ nhớ sẽ tăng.
Chạy baseline trước, sau đó thử **Monte Carlo**, mặc định 2 mẫu, sigma 20 HU,
seed 42. Hai mẫu nghĩa là 1 baseline + 2 lượt suy luận có nhiễu.
Ứng dụng yêu cầu backend lưu mask tuần tự (`--nr_thr_saving 1`) để giảm
bộ nhớ đỉnh trên Windows, đặc biệt với CT nhiều lát; áp dụng cả baseline và MC.

Thanh tiến trình hiển thị giai đoạn và thời gian đã chạy, không ước lượng
phần trăm suy luận. Lần đầu có thể cần tải trọng số. Nút **Dừng sau lượt xử lý
hiện tại** chờ backend trả về rồi hủy lần chạy; không lưu kết quả dở dang.
Trong lúc đang xử lý, cửa sổ cần được giữ mở.

Mỗi lần chạy tạo thư mục mới trong `data/demo_runs/run_.../` và log tương ứng
bên cạnh; không ghi đè CT hay kết quả cũ. Mở **vị trí kết quả** để lấy
`prediction.json`, mask NIfTI và ảnh PNG.

## Mở lại và xuất báo cáo

Chọn CT gốc trước, sau đó **Mở thư mục kết quả đã lưu**, chọn thư mục chứa
`prediction.json`. Kết quả mới có mã kiểm tra HU + geometry để từ chối
ghép nhầm CT. Kết quả cũ chỉ kiểm tra được geometry; ứng dụng nhắc xác nhận ca.

**Xuất báo cáo HTML** tạo báo cáo đọc bằng trình duyệt, có thể in thành PDF.
Báo cáo gồm trạng thái, phiên bản mô hình, chế độ xử lý, thời gian (nếu lần chạy
có ghi), thể tích, tham số MC và cảnh báo. Báo cáo không chứa đường dẫn CT
hay tên bệnh nhân. Mask NIfTI và JSON gốc vẫn là đầu ra để kiểm tra chi tiết.

## Cách đọc kết quả và giới hạn

- Thể tích là **toàn đốt sống**, gồm cả thành phần phía sau, không chỉ thân đốt.
- Consensus là đồng thuận giữa các mẫu; bất đồng nằm trong 0–1.
- Mức bất đồng thấp không chứng minh phân đoạn đúng.
- Chưa tính diện tích bằng lấy mẫu Monte Carlo, chưa phân loại AO.
- Kiểm tra đúng/sai cần annotation/mask tham chiếu đã được chuyên gia duyệt.
- DICOM nhiều frame, khoảng cách lát không đều hoặc gantry shear chưa được hỗ trợ.
- Ảnh xiên được resample riêng cho hiển thị; phép đo dùng mask gốc.

Nếu chạy lỗi, giữ CT gốc, xem tab Tiến trình và tệp log đã chỉ ra. Lỗi tải
trọng số cần kiểm tra mạng; lỗi geometry cần kiểm tra đúng series/metadata.
