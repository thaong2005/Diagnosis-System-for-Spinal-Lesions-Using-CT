# Đặc tả phát hiện vị trí tổn thương cột sống — phiên bản 0

## Phạm vi

Phiên bản đầu tiên nhận một bộ CT DICOM và tự xác định đốt sống nghi tổn
thương. Vị trí và mã phân loại trong CSV chỉ được dùng làm ground truth khi
phát triển; chúng không được đưa vào đường suy luận.

## Yêu cầu

- **REQ-001 — CT-only inference:** suy luận chỉ nhận CT và cấu hình kỹ thuật,
  không nhận vị trí đốt gãy hoặc lớp tổn thương.
- **REQ-002 — Hình học đúng:** chọn đúng CT series, chuyển HU, bảo toàn
  spacing/origin/direction và chuẩn hóa hình học trước phân tích.
- **REQ-003 — Chỉ mục ẩn danh:** liên kết CT với CSV trong bộ công cụ chuẩn bị
  dữ liệu nhưng không ghi tên, năm sinh hoặc ID bệnh nhân vào artifact có thể
  commit.
- **REQ-004 — Chia theo bệnh nhân:** train/validation/test phải tách theo bệnh
  nhân; một bệnh nhân không xuất hiện ở nhiều tập.
- **REQ-005 — Định vị đốt sống:** tạo được danh sách đốt sống nhìn thấy cùng
  ROI hoặc mask 3D của từng đốt.
- **REQ-006 — Phát hiện tổn thương:** chấm điểm từng đốt từ ảnh/đặc trưng hình
  dạng và xếp hạng vùng nghi tổn thương mà không dùng nhãn vị trí ở đầu vào.
- **REQ-007 — Kết quả có độ tin cậy:** trả về mức đốt dự đoán, vùng không gian,
  điểm tin cậy và trạng thái không kết luận khi dữ liệu không phù hợp.
- **REQ-008 — Đánh giá đúng:** báo top-1/top-k level accuracy, sai lệch theo số
  mức đốt và độ chính xác ROI; chỉ báo sensitivity/false positives per scan khi
  có ca không gãy.
- **REQ-009 — CPU baseline:** có baseline chạy được trên máy Windows CPU hiện
  tại; huấn luyện 3D nặng được tách thành giai đoạn cần GPU.
- **REQ-010 — Riêng tư:** dữ liệu CT, CSV, ánh xạ ID và đầu ra chứa ảnh bệnh
  nhân không được commit.
- **REQ-011 — Phân loại có điều kiện:** chỉ huấn luyện sáu lớp A1/A3/A4/B2/B3/C
  sau khi số mẫu và ý nghĩa lâm sàng của nhãn đủ để đánh giá; không suy diễn
  các lớp này thành mức nhẹ/vừa/nặng.

## Ngoài phạm vi phiên bản 0

- Chẩn đoán thay bác sĩ.
- Phân loại sáu lớp đáng tin cậy với phân bố dữ liệu hiện tại.
- Phát hiện trên CT bất kỳ khi chưa có ca không gãy để đo false positive.
- Huấn luyện mô hình 3D lớn trên máy CPU hiện tại.

## Tiêu chí hoàn thành phiên bản 0

1. Tập đánh giá khóa trước khi huấn luyện và không rò rỉ bệnh nhân.
2. Pipeline nhận CT của một ca chưa thấy và xuất danh sách đốt nghi tổn thương.
3. Nhãn vị trí chỉ được mở sau suy luận để tính metric.
4. Mọi ca lỗi hình học hoặc ngoài phạm vi được báo rõ thay vì dự đoán im lặng.

