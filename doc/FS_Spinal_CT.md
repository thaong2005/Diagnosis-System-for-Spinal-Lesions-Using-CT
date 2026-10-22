# FS — Mô tả, tiến độ và lộ trình dự án

- **Tên dự án:** Development of a Deep Learning-Based Computer-Aided Diagnosis System for Spinal Lesions Using CT
- **Ngày cập nhật:** 19/10/2026
- **Môi trường:** Python 3.13, VS Code, Windows, CPU; máy phát triển hiện tại chưa có GPU.
- **Phạm vi bản thử nghiệm:** Nhận biết vị trí và đánh giá tổn thương xương cột sống ngực–thắt lưng từ ảnh CT.

## 1. Định nghĩa dự án

Dự án xây dựng phần mềm hỗ trợ đánh giá tổn thương cột sống từ **ảnh CT của một bệnh nhân**. Hệ thống phải tự kiểm tra bộ ảnh, dựng thể tích 3D, xác định các đốt sống, tìm vùng nghi tổn thương, đánh giá biến dạng và đưa ra kết quả có thể kiểm tra được.

Deep learning chỉ được dùng khi dữ liệu, nhãn và tài nguyên tính toán phù hợp. Bản thử nghiệm không thay thế chẩn đoán của bác sĩ; kết quả phải hiển thị độ tin cậy và trạng thái chưa kết luận khi dữ liệu không đủ.

### Ranh giới đầu vào bắt buộc

Khi chạy thực tế, hệ thống chỉ nhận:

```text
Một bộ ảnh CT cột sống của một bệnh nhân
```

Ưu tiên DICOM vì DICOM chứa thứ tự lát cắt, spacing, origin, direction và giá trị cường độ cần để dựng hình học đúng. Không được yêu cầu người dùng cung cấp trước vị trí đốt gãy, ROI, mã phân loại hoặc file nhãn.

File [data ngày 27.09.2026.csv](../data/raw/A4/data%20ngay%2027.09.2026.csv) chỉ là **ground truth dùng trong phát triển**. File này dùng để liên kết ca, tạo annotation, huấn luyện và đánh giá. Nó không được đưa vào đường suy luận production.

## 2. Chức năng mục tiêu

| Bước | Hệ thống thực hiện | Kết quả |
|---|---|---|
| 1. Tiền xử lý | Chọn CT series, sắp xếp lát cắt, chuyển HU và kiểm tra hình học | Volume CT hợp lệ |
| 2. Định vị giải phẫu | Xác định các đốt sống và ROI/mask của từng đốt | Danh sách đốt sống có tọa độ |
| 3. Phát hiện tổn thương | Chấm điểm biến dạng hoặc đặc trưng bất thường trên từng đốt | Các ứng viên nghi tổn thương |
| 4. Đánh giá | Đo chiều cao, tỷ lệ giảm chiều cao, góc biến dạng hoặc đặc trưng phù hợp | Phép đo hỗ trợ |
| 5. Phân loại | Dự đoán nhóm tổn thương khi đủ dữ liệu và nhãn | Mã/lớp dự đoán |
| 6. Hiển thị | Overlay ROI, mức đốt, điểm tin cậy và trạng thái từ chối | Kết quả có thể kiểm tra |

Một thay đổi hình dạng giữa hai lát cắt chỉ là dấu hiệu nghi ngờ. Hệ thống phải phân biệt tổn thương với cấu trúc bình thường, nhiễu, kim loại và lỗi ảnh.

## 3. Tách pipeline phát triển và pipeline thực tế

### Pipeline phát triển

```text
CT + ánh xạ ca + nhãn tham chiếu
    ↓
Kiểm tra dữ liệu và tạo annotation
    ↓
Huấn luyện / hiệu chỉnh / đánh giá
```

### Pipeline thực tế

```text
CT
    ↓
Tiền xử lý và dựng volume
    ↓
Định vị đốt sống
    ↓
Xếp hạng vùng nghi tổn thương
    ↓
Đánh giá và phân loại
    ↓
Vị trí + ROI + kết quả + độ tin cậy
```

CSV và ground truth chỉ được mở **sau khi suy luận** để tính metric. Không được dùng nhãn có sẵn để chọn vùng đầu vào cho mô hình trong bước đánh giá.

## 4. Dữ liệu hiện có

- CSV có 54 ca, 54/54 ca có mức đốt và mã phân loại.
- Các mã hiện có: A1, A3, A4, B2, B3 và C.
- Phân bố nhãn không cân bằng: A1 và B3 mỗi lớp chỉ có một ca; C có năm ca.
- CSV có 5 dòng chứa nhiều mức đốt.
- Audit DICOM hiện ghi nhận 10 PatientID, 53 CT series và 7.638 ảnh CT; chỉ 4 PatientID khớp trực tiếp với giá trị trong CSV.
- Có 668 ảnh dùng giá trị Specific Character Set không chuẩn và cần được giữ lại như cảnh báo chất lượng.
- Chưa có ca không gãy đủ đại diện để đánh giá false positive trên toàn scan.

Các con số audit là thống kê tổng hợp. Không đưa tên, năm sinh, PatientID hoặc ID lâm sàng vào log, artifact commit hoặc báo cáo demo.

## 5. Đã hoàn thành

| Hạng mục | Tình trạng |
|---|---|
| Môi trường Python và viewer trên Windows CPU | Đã chạy |
| Đọc DICOM, chọn BONEPLUS, sắp xếp lát cắt và chuyển HU | Đã hoàn thành trên ca mẫu |
| Kiểm tra shape, spacing, origin, direction và chuẩn hóa LPS | Đã triển khai trên ca mẫu |
| Hiển thị axial, sagittal, coronal | Đã hoạt động |
| Lưu `ct_hu.npy` và `metadata.json` | Đã hoàn thành trên ca mẫu |
| Sagittal MPR đúng tỷ lệ vật lý | Đã hoàn thành trên ca mẫu |
| Viewer sagittal nâng cao, WL/WW và preset | Đã hoàn thành |
| Xuất ảnh sagittal và metadata tái lập | Đã hoàn thành |
| Đọc/chuẩn hóa nhãn CSV và audit DICOM | Đã có trong mã nguồn |
| Unit test parser nhãn và bảo vệ PII | Đã có |
| Schema ROI 3D và validator geometry | Đã triển khai |
| Viewer annotation axial/sagittal/coronal | Đã triển khai; chưa có ROI reviewed |
| ROI không gian chính xác và suy luận tự động | Chưa hoàn thành |

## 6. Trạng thái hiện tại

Dự án đang ở điểm chuyển từ **tiền xử lý và hiển thị** sang **tạo ground truth không gian và lập trình nhận biết vị trí tổn thương**.

Hiện mới xác minh pipeline ảnh trên `case_053`. Ca này có nhãn tham chiếu L1/A1, nhưng ROI sagittal hiện tại vẫn là bản nháp và còn bao nhiều đốt. Viewer hiện tại chỉ xem và xuất ảnh; chưa tự phát hiện, định vị, đánh giá mức độ hoặc phân loại.

| Hạng mục | Trạng thái hiện tại |
|---|---|
| Liên kết CT–CSV theo ca | Chưa xác nhận đầy đủ; đây là gate dữ liệu đầu tiên |
| Manifest ẩn danh và split theo bệnh nhân | Chưa tạo |
| Schema ROI 3D có geometry và trạng thái review | Đã có validator; chưa có ROI reviewed |
| Annotation đồng bộ axial/sagittal/coronal | Đã có trong `annotate_ct.py`; chưa xác nhận ROI |
| Định vị từng đốt sống | Chưa triển khai |
| Baseline xếp hạng biến dạng | Chưa triển khai |
| Suy luận chỉ nhận CT | Chưa đóng gói thành API |
| Phân loại sáu lớp | Chưa đủ dữ liệu để đánh giá đáng tin cậy |

## 7. Việc cần làm tiếp theo

### Giai đoạn A — Chốt hợp đồng dữ liệu

1. Xác nhận quy tắc liên kết giữa thư mục/series DICOM và từng dòng CSV.
2. Tạo manifest ẩn danh với `case_id`, đường dẫn series, nhãn tham chiếu và trạng thái liên kết.
3. Tách ca `matched`, `ambiguous` và `unmatched`; chỉ dùng ca đã xác nhận cho phát triển.
4. Tạo split theo bệnh nhân có seed và kiểm tra leakage.
5. Refactor loader DICOM thành CLI không hard-code `case_053`.

**Điều kiện hoàn thành:** mỗi ca DICOM có trạng thái liên kết rõ ràng; pipeline không ghi thông tin định danh vào output.

### Giai đoạn B — Tạo ROI 3D

1. Định nghĩa schema annotation gồm `case_id`, mức đốt, bounding box/mask 3D, geometry LPS, trạng thái `draft/reviewed/rejected` và phiên bản annotation.
2. Viết validator kiểm tra giới hạn ROI, spacing, shape và geometry.
3. Sửa ROI L1 của `case_053` để chỉ bao một thân đốt.
4. Kiểm tra viewer ba mặt phẳng đồng bộ trên `case_053`, có crosshair và ROI chung.
5. Lưu một annotation 3D ở trạng thái `draft`, sau đó tạo tập ROI validation đã được chuyên gia kiểm tra.

**Điều kiện hoàn thành:** ít nhất một ca có ROI 3D chính xác; sau đó mở rộng sang các ca đại diện khác.

### Giai đoạn C — Định vị đốt sống và baseline

1. Benchmark một công cụ phân đoạn đốt sống tiền huấn luyện trên vài ca đại diện.
2. Chuyển mask từng đốt thành ROI chuẩn hóa và kiểm tra thứ tự T/D–L.
3. Trích đặc trưng hình học: chiều cao trước/giữa/sau, tỷ lệ giảm chiều cao, góc chêm và so sánh với đốt lân cận.
4. Xây baseline xếp hạng vùng nghi tổn thương trước khi thử CNN.
5. Chỉ thử CNN 2.5D sau khi baseline và dữ liệu đã ổn định; 3D CNN cần đánh giá riêng về GPU.

### Giai đoạn D — Đánh giá và tích hợp

1. Đóng băng test set theo bệnh nhân.
2. Chạy inference chỉ với CT.
3. Sau inference mới mở ground truth để tính top-1/top-k level accuracy, khoảng cách mức đốt, candidate recall và sai lệch ROI.
4. Chỉ báo sensitivity/false positives per scan khi có đủ ca không gãy.
5. Tích hợp overlay prediction, confidence, model version và trạng thái từ chối vào viewer.

## 8. Tiêu chí không được vi phạm

- Không đọc CSV trong API suy luận production.
- Không đưa vị trí đốt, ROI hoặc mã phân loại vào đầu vào mô hình khi đánh giá.
- Không coi đốt không được ghi trong CSV là âm tính chắc chắn khi chưa xác nhận.
- Không dùng ROI nháp làm ground truth chính thức.
- Không chia lát cắt ngẫu nhiên; phải chia theo bệnh nhân.
- Không công bố metric sáu lớp khi lớp test không đủ số mẫu.
- Không báo false-positive rate khi chưa có ca không gãy phù hợp.
- Không commit CT, CSV, ánh xạ ID hoặc artifact chứa thông tin bệnh nhân.
- Ca lỗi hình học, thiếu series hoặc ngoài phạm vi phải trả về trạng thái từ chối rõ ràng.

## 9. Kết quả kỳ vọng của bản thử nghiệm đầu tiên

Bản thử nghiệm đầu tiên nên chứng minh được chuỗi sau:

```text
CT mới
  → volume có geometry đúng
  → danh sách đốt sống/ROI
  → vùng nghi tổn thương được xếp hạng
  → vị trí và confidence
  → báo cáo chưa kết luận nếu không đủ điều kiện
```

Phân loại sáu lớp chỉ là nhánh mở rộng sau khi ý nghĩa lâm sàng của các mã được xác nhận và dữ liệu đủ để đánh giá.
