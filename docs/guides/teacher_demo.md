# Hướng dẫn demo cho giáo viên

Máy mới cần [cài môi trường Windows](windows_integration.md) trước.
Dữ liệu CT được lưu trên máy, không có sẵn trong repo GitHub.

## 1. Mở và chạy thử

1. Nhấp đúp **start_demo.bat** ở thư mục gốc dự án.
2. Bấm **Mở ca mẫu**. Nếu hiện danh sách nhiều bộ ảnh (series), chọn một bộ
   rồi bấm **Đọc series đã chọn**.
3. Kiểm tra ảnh đã mở đúng ca. Giữ **Chạy nhanh**, bấm **Chạy định vị** và
   chờ thông báo hoàn tất trong tab **Tiến trình**.
4. Kéo thanh lát cắt hoặc chọn một đốt trong bảng để xem kết quả trên ba mặt phẳng.

Nếu muốn kiểm tra Monte Carlo, bật **Bổ sung Monte Carlo**, giữ tham số mặc định
rồi chạy lại. Chế độ này chạy thêm 2 lượt có nhiễu nên mất thời gian hơn.
Để dừng, bấm **Dừng sau lượt xử lý hiện tại** và chờ lượt đang chạy kết thúc.

## 2. Đổi bệnh nhân

**Chọn trực tiếp:** bấm **Chọn thư mục DICOM**, chọn thư mục bệnh nhân,
chọn series rồi bấm **Đọc series đã chọn**.
Với file `.nii`, `.nii.gz` hoặc `ct_hu.npy`, dùng **Chọn file CT**;
file NPY cần có `metadata.json` bên cạnh.

**Đổi ca mẫu mặc định:** lần đầu sao chép `sample_config.example.py` thành
`sample_config.py`. Sau đó chỉ sửa đường dẫn trong file này:

```python
SAMPLE_DIR = ROOT / 'data/raw/sample_patient'
```

Thay đường dẫn bằng thư mục cần xem, lưu file rồi mở lại ứng dụng.
Tên nút tự cập nhật; không cần sửa số lát hay kernel. File cấu hình này không
được đưa lên GitHub. Nút **Mở ca mẫu** chỉ bật khi thư mục tồn tại.

## 3. Xem lại và lấy kết quả

- **Xem lại:** mở CT gốc, bấm **Mở thư mục kết quả đã lưu**, chọn thư mục
  chứa `prediction.json`. Không cần chạy lại mô hình.
- **Lấy file:** bấm **Mở vị trí kết quả**. Các lượt chạy từ giao diện được lưu
  riêng trong `data/demo_runs/`, gồm báo cáo JSON, mask 3D và ảnh PNG.
- **Gửi báo cáo:** bấm **Xuất báo cáo HTML**, mở file bằng trình duyệt để xem
  hoặc in thành PDF.

## Lưu ý khi xem

`indeterminate` nghĩa là cần xem tab **Lưu ý kết quả**, không đồng nghĩa chương
trình bị lỗi. Bảng hiện **thể tích toàn đốt sống**, chưa phải diện tích thân đốt.
Monte Carlo cho biết kết quả thay đổi thế nào khi thêm nhiễu; chưa dùng để
kết luận tổn thương hay phân loại AO. Khi gặp lỗi, xem tab **Tiến trình** và log.

Xem [chức năng đã có và tiến độ dự án](../../README.md#tiến-độ-dự-án).
