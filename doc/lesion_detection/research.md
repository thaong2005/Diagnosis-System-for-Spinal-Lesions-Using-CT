# Quyết định và điểm chưa biết

## Quyết định 1 — Tách định vị giải phẫu khỏi phát hiện tổn thương

**Chọn:** pipeline hai tầng: (1) định vị/segment từng đốt sống; (2) chấm điểm
tổn thương trên ROI từng đốt.

**Lý do:** CSV chỉ có nhãn mức đốt, không có mask tổn thương. Cách hai tầng cho
phép dùng nhãn yếu ở mức đốt và kiểm tra lỗi của từng thành phần.

**Phương án đã cân nhắc:** detector end-to-end trên toàn volume. Phương án này
chưa phù hợp vì chỉ có ít ca, chưa có bounding box 3D và máy hiện tại không có
GPU.

## Quyết định 2 — Dùng mô hình giải phẫu tiền huấn luyện làm baseline

**Chọn:** đánh giá một công cụ phân đoạn đốt sống tiền huấn luyện, ưu tiên công
cụ trả nhãn từng đốt. TotalSegmentator là ứng viên nghiên cứu, không mặc định là
ground truth và phải được kiểm tra trên dữ liệu dự án.

**Lý do:** tự huấn luyện segmentation từ 54 nhãn mức đốt là không khả thi.

**Phương án đã cân nhắc:** gán mask toàn bộ dữ liệu bằng tay; chính xác hơn nhưng
tốn thời gian và vẫn cần quy trình kiểm tra liên quan chuyên gia.

## Quyết định 3 — Chưa huấn luyện sáu lớp

**Chọn:** phiên bản đầu tiên tập trung xếp hạng vị trí tổn thương. Phân loại đa
lớp chỉ là nhánh thử nghiệm sau khi kiểm kê đủ CT và bổ sung dữ liệu.

**Lý do:** 54 nhãn có phân bố A1=1, A3=14, A4=19, B2=14, B3=1, C=5. Kết quả
sáu lớp hiện sẽ không ổn định và không thể đánh giá hai lớp hiếm đáng tin cậy.

## Điểm cần xác nhận

1. Mỗi dòng CSV tương ứng chính xác với thư mục/PatientID DICOM nào.
2. Một ca có thể có nhiều đốt tổn thương hay chỉ một nhãn chính.
3. Các đốt không được ghi trong ca gãy có thực sự là âm tính hay chỉ chưa được
   báo cáo.
4. Quy ước D11/D12 có được chuẩn hóa thành T11/T12 trong báo cáo hay không.
5. Có thể bổ sung ca không gãy và đánh giá ROI bởi chuyên gia hay không.

