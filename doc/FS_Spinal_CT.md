# FS — Mô tả và kế hoạch dự án

- **Tên dự án:** Development of a Deep Learning-Based Computer-Aided Diagnosis System for Spinal Lesions Using CT
- **Ngày cập nhật:** 18/10/2026
- **Công cụ:** Python, VS Code, Windows. Máy phát triển hiện tại không có GPU.

## 1. Dự án làm gì?

Dự án xây dựng phần mềm hỗ trợ đánh giá tổn thương cột sống từ ảnh CT của bệnh nhân.

Phần mềm nối các lát cắt CT thành hình ảnh thể tích 3D, tìm các vùng thay đổi bất thường, đánh giá sự biến dạng và đưa ra vị trí, mức độ, phân loại tổn thương. Deep learning được sử dụng ở các bước phân tích phù hợp với dữ liệu và nhãn có sẵn.

Bản thử nghiệm đầu tiên tập trung vào gãy đốt sống vùng ngực–thắt lưng. Kết quả cần được chuyên gia kiểm tra, không thay thế chẩn đoán của bác sĩ.

## 2. Input — Dữ liệu đầu vào

**Đầu vào khi hệ thống chạy thực tế chỉ là một bộ ảnh CT cột sống của một bệnh nhân.** Bộ ảnh có thể nằm trong một thư mục hoặc được đóng gói thành một file ZIP. Hệ thống không được nhận trước vị trí đốt gãy, vùng tổn thương hoặc mã phân loại.

Ưu tiên ảnh **DICOM**, vì định dạng này chứa vị trí lát cắt, hướng ảnh, khoảng cách và giá trị cường độ cần thiết để dựng CT đúng. Nếu đầu vào là JPG/PNG, phải kiểm tra thông tin hình học đi kèm; không mặc định rằng các ảnh này đủ để dựng 3D hoặc đo biến dạng chính xác.

Trong giai đoạn phát triển, file CSV hiện có cung cấp **nhãn tham chiếu** gồm vị trí đốt gãy và mã phân loại cho 54 ca. Các nhãn này dùng để tạo dữ liệu huấn luyện, kiểm tra và đo độ chính xác; không được truyền vào mô hình khi đánh giá trên ca mới hoặc khi hệ thống chạy thực tế. CSV chưa có nhãn mức độ độc lập ngoài mã phân loại.

## 3. Các nhiệm vụ chính

| Bước | Phần mềm thực hiện | Kết quả của bước |
|---|---|---|
| 1. Nối ảnh | Chọn đúng bộ lát cắt, sắp xếp theo vị trí và dựng thể tích 3D | CT có hình học đúng, xem được ba mặt phẳng |
| 2. Tìm biến đổi bất thường | Tự quét toàn bộ CT, xác định cấu trúc đốt sống và tìm thay đổi hình dạng đột ngột mà không biết trước vị trí tổn thương | Vị trí đốt sống và vùng nghi ngờ cần đánh giá |
| 3. Đánh giá biến dạng | Đo đặc điểm hình dạng phù hợp, như chiều cao thân đốt, tỷ lệ giảm chiều cao hoặc góc biến dạng | Các phép đo và mức độ theo tiêu chí đã thống nhất |
| 4. Phân loại | Phân tích vùng nghi ngờ bằng mô hình và tiêu chí chuyên gia | Nhóm tổn thương dự đoán |
| 5. Hiển thị kết quả | Đánh dấu vùng trên CT và tổng hợp kết quả | Báo cáo vị trí, mức độ, phân loại |

Biến đổi đột ngột chỉ là dấu hiệu nghi ngờ. Phần mềm cần phân biệt tổn thương với cấu trúc giải phẫu bình thường, nhiễu và lỗi ảnh; không kết luận gãy chỉ từ một thay đổi giữa hai lát cắt.

## 4. Output — Kết quả đầu ra

| Đầu ra | Nội dung |
|---|---|
| **Vị trí tổn thương** | Tên đốt sống hoặc đoạn cột sống và vùng được đánh dấu trên ảnh CT |
| **Mức độ tổn thương** | Mức đánh giá theo thang đã được chuyên gia xác nhận, kèm phép đo hỗ trợ nếu phù hợp |
| **Phân loại tổn thương** | Mã hoặc tên nhóm tổn thương theo hệ phân loại được chọn |

Giao diện hiển thị ảnh CT và các kết quả trên, có thể xuất báo cáo. Trường hợp chưa đủ dữ liệu để kết luận phải được ghi rõ là chưa xác định.

**Điểm cần thống nhất:** CSV hiện có vị trí và các mã A1, A3, A4, B2, B3, C, nhưng chưa có nhãn mức độ độc lập. Cần xác nhận hệ phân loại và thang mức độ với người gán nhãn. Không tự chuyển các mã này thành nhẹ, vừa, nặng.

## 5. Luồng hoạt động

```text
Bộ ảnh CT của một bệnh nhân
    ↓
Kiểm tra và nối lát cắt thành thể tích 3D
    ↓
Tự quét toàn bộ cột sống và tìm vùng biến dạng bất thường
    ↓
Đánh giá biến dạng và phân loại
    ↓
Vị trí tổn thương + Mức độ tổn thương + Phân loại
```

Hệ thống đích tự phân tích CT mới mà **không nhận trước vị trí tổn thương**. Vị trí đốt gãy và mã phân loại trong CSV chỉ là ground truth của giai đoạn phát triển. Công cụ đánh dấu thủ công hiện tại phục vụ tạo nhãn không gian và kiểm chứng; không phải bước người dùng phải làm trong hệ thống tự động hoàn chỉnh.

## 6. Chúng ta đã hoàn thành gì?

| Hạng mục | Tình trạng |
|---|---|
| Môi trường Python và cửa sổ xem ảnh trên Windows CPU | Đã chạy thành công |
| Đọc DICOM và chọn bộ ảnh BONEPLUS của ca mẫu | Đã hoàn thành |
| Sắp xếp 333 lát cắt, kiểm tra hình học, chuyển cường độ sang HU | Đã hoàn thành |
| Hiển thị axial, sagittal, coronal | Đã hoạt động |
| Lưu ct_hu.npy và metadata.json; kiểm tra đọc lại | Đã hoàn thành |
| Bỏ qua dữ liệu bệnh nhân và môi trường ảo trong Git | Đã kiểm tra |
| Chọn khung trên ảnh sagittal và lưu JSON | Công cụ đã hoạt động |
| Dựng lại hình học CT từ shape, spacing, origin và hướng DICOM bằng SimpleITK | Đã triển khai trên ca mẫu; chuẩn hóa về LPS và từ chối ảnh chụp xiên chưa được resample |
| Sagittal MPR đúng tỷ lệ vật lý | Đã triển khai trên ca mẫu; resample hai trục về cùng spacing để tránh méo hình khi hiển thị |
| Trình xem sagittal nâng cao | Đã triển khai: đổi lát cắt, WL/WW, preset Bone/Soft tissue và chọn nội suy hiển thị |
| Xuất ảnh sagittal và thông số tái lập | Đã chạy thành công; xuất PNG kèm JSON chứa lát cắt, WL/WW, spacing, origin và direction LPS |
| Nhãn mức đốt tổn thương của các ca | Đã có trong CSV: 54/54 ca có vị trí đốt gãy và mã phân loại |
| ROI không gian chính xác quanh tổn thương của ca mẫu | Chưa hoàn thành; vị trí L1 đã biết nhưng khung mới nhất còn bao nhiều đốt |
| Vùng quan tâm 3D và kiểm tra đồng bộ trên ba mặt phẳng | Chưa triển khai trong trình xem nâng cao |
| Tự phát hiện, đánh giá mức độ và phân loại | Chưa triển khai |

Hiện mới xác minh pipeline trên **một ca mẫu (`case_053`)**, chưa phải toàn bộ dữ liệu. `view_ct_enhanced.py` là công cụ **chỉ xem và xuất ảnh**, chưa phát hiện hay dự đoán tổn thương. L1/A1 là ground truth đã được cung cấp trước để phát triển và đối chiếu; hệ thống cuối cùng phải tự tìm ra vị trí và phân loại này chỉ từ CT.

## 7. Kế hoạch tiến độ đề xuất

Kế hoạch gồm **8 tuần kể từ khi bắt đầu giai đoạn tiếp theo**, dành cho bản thử nghiệm nghiên cứu. Thời gian phụ thuộc dữ liệu, việc xác nhận nhãn và tài nguyên tính toán.

| Thời gian | Việc cần làm | Kết quả cần đạt | Tiến độ đến 18/10/2026 |
|---|---|---|---|
| Tuần 1 | Dùng vị trí L1 đã có để sửa ROI ca mẫu; thống nhất ý nghĩa hệ phân loại và cách đánh giá mức độ | Một ca mẫu có ROI ground truth đúng và quy tắc gán nhãn rõ ràng | **Chưa hoàn thành:** đã có nhãn mức đốt L1 và công cụ lưu khung, nhưng ROI còn bao nhiều đốt; quy tắc mức độ chưa được xác nhận |
| Tuần 2 | Bổ sung chọn vùng 3D; liên kết CT với CSV bằng mã ca | Vùng đốt sống kiểm tra được trên ba mặt phẳng và danh mục ca | **Đang thực hiện:** đã có sagittal MPR đúng hình học và xuất ảnh kèm metadata; chưa có ROI 3D, kiểm tra liên kết ba mặt phẳng hoặc liên kết CSV |
| Tuần 3 | Xử lý các ca còn lại, kiểm tra ảnh và tạo nhãn | Tập dữ liệu có danh sách lỗi, thiếu nhãn và ca đủ điều kiện | **Chưa bắt đầu:** mới xác minh `case_053` |
| Tuần 4 | Chia dữ liệu theo bệnh nhân; xây cách phân tích hình dạng và biến đổi ban đầu | Pipeline tạo mẫu và phương pháp đối chiếu ban đầu | **Chưa bắt đầu** |
| Tuần 5 | Thử mô hình deep learning nhỏ; đo thời gian CPU, cân nhắc GPU nếu cần | Mô hình thử nghiệm cho nhiệm vụ đã chốt | **Chưa bắt đầu** |
| Tuần 6 | Đánh giá vị trí, phân loại và phép đo biến dạng; mức độ chỉ đánh giá khi có nhãn | Báo cáo kết quả và lỗi trên bệnh nhân chưa dùng huấn luyện | **Chưa bắt đầu** |
| Tuần 7 | Tích hợp giao diện, vùng đánh dấu và báo cáo; kiểm tra thành phần định vị tự động nếu đã có | Demo thể hiện rõ bước tự động và bước còn thủ công | **Khởi tạo một phần giao diện:** đã có trình xem sagittal và export; chưa có vùng dự đoán, kết quả chẩn đoán hoặc báo cáo tổng hợp |
| Tuần 8 | Kiểm thử lại, hoàn thiện tài liệu và báo cáo dự án | Bản demo tái lập được và hướng dẫn sử dụng | **Chưa bắt đầu** |

Tiến độ hiện tại tập trung ở **nền tảng tiền xử lý và hiển thị của bước 1**. Nhãn mức đốt và phân loại đã có, nhưng chưa thể coi Tuần 1 hoặc Tuần 2 là hoàn thành vì ROI 3D, quy tắc mức độ và liên kết CT–CSV vẫn còn thiếu. Các chức năng tự quét, định vị, đánh giá và phân loại của bước 2–4 chưa được triển khai.

**Lưu ý về tính khả thi:** CSV có 54 ca; A1 và B3 mỗi lớp chỉ có một ca. Chưa đủ để đánh giá đáng tin cậy mô hình đầy đủ sáu lớp. Cần bổ sung dữ liệu, hoặc giới hạn rõ phạm vi bản thử nghiệm. Muốn phát hiện gãy trên CT bất kỳ cần thêm ca không gãy. Các bước tự động hóa đầy đủ có thể cần giai đoạn bổ sung ngoài 8 tuần.

## 8. Kiểm tra dự án đã làm đúng hay chưa

- **Nối ảnh:** chọn đúng series, đúng thứ tự, không trùng vị trí; hình học và dữ liệu lưu khớp nhau.
- **Vị trí:** khi chạy đánh giá, chỉ đưa CT vào hệ thống; sau khi có dự đoán mới so sánh vị trí đốt và ROI dự đoán với ground truth.
- **Phân loại:** đánh giá theo từng lớp, ghi rõ số bệnh nhân và loại chưa đủ dữ liệu đánh giá.
- **Mức độ:** đối chiếu thang đo hoặc phép đo đã thống nhất với chuyên gia.
- **Dữ liệu:** chia theo bệnh nhân; ảnh của cùng một bệnh nhân không nằm trong nhiều tập train/validation/test.
- **Demo:** phân biệt rõ nhãn tham chiếu và dự đoán; báo trường hợp không hỗ trợ.

Không đưa thông tin định danh và CT vào GitHub. Giữ dữ liệu đã ẩn danh trong thư mục data, được .gitignore bỏ qua.

## 9. Việc cần làm ngay

1. **Dùng nhãn vị trí L1 đã có để chọn lại ROI chỉ bao một thân đốt sống.** Không cần tìm lại tên đốt bằng tay; cần chuyển nhãn mức đốt đã biết thành ground truth không gian chính xác. Lưu JSON thành công chỉ chứng minh chức năng lưu hoạt động; chưa chứng minh ROI được chọn là đúng.
2. Thống nhất ý nghĩa các mã A1, A3, A4, B2, B3, C và tiêu chí đánh giá mức độ với người gán nhãn.
3. Mở rộng công cụ từ lát sagittal đơn sang ROI 3D có thể kiểm tra đồng bộ trên axial, sagittal và coronal.
4. Liên kết từng CT với dòng nhãn CSV bằng mã ca để tạo ground truth, sau đó xử lý và kiểm tra chất lượng các ca còn lại. Tách pipeline dữ liệu nhãn khỏi pipeline suy luận để bảo đảm ca mới chỉ nhận CT làm đầu vào.
