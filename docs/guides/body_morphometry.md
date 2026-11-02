# Đo hình dạng và diện tích thân đốt

Công cụ dùng CT và kết quả định vị đã lưu. Các phép đo thực hiện trên một mặt phẳng 2D, với vùng thân đốt do người dùng vẽ và xác nhận. LoG đề xuất tâm/thang đo để tham khảo; cần kiểm chứng hiệu quả trên dữ liệu dự án.

## Sử dụng trong demo

1. Chạy `start_demo.bat`, mở CT rồi mở/chạy kết quả định vị.
2. Chọn **Baseline** hoặc **Consensus**, chọn một đốt trong bảng rồi bấm **Đo thân đốt**. Nếu chưa chọn hàng, công cụ dùng hàng đầu tiên theo thứ tự giải phẫu.
3. Kiểm tra mặt phẳng sagittal ban đầu từ tâm mask toàn đốt. Mở **Điều chỉnh mặt phẳng**, thay dịch chuyển, xoay hoặc nghiêng rồi bấm **Tạo / áp dụng mặt phẳng**. Thay các ô thông số giữ nguyên vùng cũ và tạm khóa chỉnh vùng. Chọn **Giữ mặt phẳng hiện tại** để quay lại; áp dụng mặt phẳng mới có lựa chọn lưu, bỏ thay đổi hoặc hủy. Nếu lưu thất bại thì vùng cũ vẫn được giữ.
4. Vẽ đường bao thân đốt, gồm phần bên trong đường bao, loại cấu trúc phía sau. Khép đa giác bằng nhấp đúp hoặc chọn lại điểm đầu. Có thể thêm/bớt bằng đa giác, dùng cọ trái/phải và hoàn tác. Đường nét đứt xanh là toàn đốt để tham khảo.
5. Có thể mở **Thử nghiệm LoG** ở cuối cột điều khiển, bấm **Tìm ứng viên blob trong ROI** và chọn B1/B2... làm tâm tham khảo khi vẽ. Vòng tròn chỉ biểu diễn thang đo LoG. Điều chỉnh sigma theo mm và ngưỡng đáp ứng nếu cần; ghi nhận cả trường hợp không tìm được ứng viên.
6. Mở **Đo chiều cao · 6 mốc tùy chọn**. Để đo chiều cao, bấm **Đặt lại 6 mốc chiều cao**, nhấp lần lượt: trước–trên, trước–dưới, giữa–trên, giữa–dưới, sau–trên, sau–dưới. Chiều cao là khoảng cách Euclid mỗi cặp điểm trong mm. Có thể chọn **Sửa mốc đã chọn** rồi nhấp vị trí mới, không cần đặt lại toàn bộ. Có thể bỏ cả sáu mốc; khi đó các chỉ số chiều cao để trống.
7. Chọn nhóm tuổi nếu biết, kiểm tra mặt phẳng/mask/mốc rồi tích xác nhận. Bấm **Đo thân đốt**.
8. Bấm **Lưu** (1-click) để tự động lưu bản đo mới vào thư mục bệnh nhân kèm mã riêng; hoặc bấm **Lưu khác…** để chọn thư mục tùy ý.

## Trạng thái và thao tác nền

- Dòng trạng thái phân biệt chưa kiểm tra, đã xác nhận, có cảnh báo và đã lưu; thay đổi chưa lưu được ghi riêng.
- Đóng cửa sổ, nhấn Esc hoặc mở vùng khác có lựa chọn lưu / bỏ thay đổi / hủy khi dữ liệu đã sửa. Bản nháp được lưu bằng cùng schema vùng, chưa xác nhận vẫn giữ trạng thái nháp.
- Đo, tìm LoG và lưu/xuất chạy trong worker. Trong lúc chạy, các điều khiển chỉnh sửa tạm khóa; cửa sổ tiếp tục xử lý sự kiện và hiển thị thời gian đã dùng. Chờ tác vụ hoàn tất trước khi đóng.
- **Cấu hình MC diện tích** chứa số điểm và seed, thu gọn mặc định. **Ctrl+Z** hoàn tác vùng/mốc; công cụ đang bật hiện ở đầu bảng điều khiển.
- Khi mở công cụ đo từ demo, bản lưu mới nhất trong `data/output/<slug>/do_than_dot` (và `data/body_measurements`) đúng CT, đốt và nguồn Baseline/Consensus được nạp lại sau kiểm tra toàn vẹn. Có thể mở bản khác bằng nút mở vùng.

## Định nghĩa phép đo

- Mặt phẳng có gốc tại tâm pixel (0,0), hai trục đơn vị trong LPS và spacing mm. Trục ngang chủ yếu hướng trước → sau, trục dọc hướng dưới → trên. CT nội suy tuyến tính; nhãn nội suy nearest-neighbour. Các pixel nằm ngoài trường chụp không được đưa vào mask thân đốt.
- Mask đo là tập hợp các ô pixel đầy đủ. Diện tích tham chiếu bằng số pixel × spacing ngang × spacing dọc; đây là tham chiếu trên mask số. Đường bao thủ công cần được chuyên gia đối chiếu để đánh giá giải phẫu.
- Ellipse có cùng moment bậc hai với vùng, tính cả diện tích ô pixel. Ellipse không bảo đảm bao kín mask; hướng được cảnh báo khi hai trục gần bằng nhau.
- Hộp bao vuông được căn theo các trục moment và bao toàn bộ ô pixel; không phải thuật toán tìm hộp có diện tích nhỏ nhất. Tỷ lệ lấp đầy bằng diện tích mask / diện tích hộp vuông. Solidity bằng diện tích mask / diện tích bao lồi của các ô pixel. Các thành phần rời và lỗ trong mask được giữ nguyên.
- Cao/rộng là tỷ lệ kích thước dọc/ngang của vùng trong mặt phẳng giải phẫu đã xác nhận, trước khi căn ellipse. Tỷ lệ trục ellipse được lưu riêng.
- Ảnh 256 × 256 được căn theo tâm và trục moment, co giãn đồng đều, thêm nền -1024 HU ngoài mask thân đốt. JSON lưu ma trận chuyển từ pixel chuẩn hóa về mm trên mặt phẳng. Các đặc trưng được đo trước khi thu nhỏ.
- MC diện tích lấy điểm độc lập, đều trong hộp bao vuông; điểm thuộc mask khi rơi vào một ô pixel được đánh dấu. Báo K/N, diện tích, seed, sai lệch so với đếm pixel, sai số chuẩn và khoảng Wilson 95% cho sai số lấy mẫu. Có các mốc hội tụ dày đặc từ 100 đến 10 triệu điểm; thuật toán tự động phát hiện điểm cùi chỏ (elbow point) nơi Monte Carlo bắt đầu hội tụ ổn định. Tăng N không bảo đảm mọi lượt đều gần giá trị tham chiếu hơn.

Các cảnh báo gồm mask nhỏ, chạm biên mặt phẳng/trường CT, nhiều thành phần, hướng ellipse chưa ổn định, thiếu mốc hoặc thiếu nhóm tuổi. Sửa mask/mốc/nhóm tuổi sẽ hủy xác nhận và phép đo cũ. Đổi seed hoặc số điểm xóa bảng đo để tính lại. Đổi thông số LoG bỏ các ứng viên cũ và báo cần chạy lại; chạy LoG giữ bảng đo trên mask. Sigma nhỏ nhất được giới hạn theo spacing mặt phẳng.

Mở lại vùng phải khớp fingerprint CT và chữ ký vùng. Số điểm, seed và cấu hình LoG được khôi phục; các ô dịch/xoay khớp mặt phẳng tạo bởi công cụ. Mặt phẳng tùy chỉnh không biểu diễn được bằng các ô này có nhãn riêng, vẫn giữ geometry đã lưu. Các bản schema 1 cũ lấy N/seed từ báo cáo cùng chữ ký vùng. Seed nhập số nguyên không âm đến 2^63−1. Bấm Đo để tính lại bảng từ dữ liệu hiện tại.

## Kết quả lưu

- `region.json`, `body_mask.npy`: vùng và mặt phẳng có thể mở lại, nguồn Baseline/Consensus, nhóm tuổi và trạng thái duyệt. Không lưu đường dẫn nguồn hay định danh bệnh nhân.
- Sau khi xác nhận: `measurements.json`, `features.csv`, `normalized_ct.npy`, `normalized_mask.npy`, `preview.png`, `report.html`.
- Nếu có ứng viên LoG: thêm `blob_candidates.png`; có thể ẩn/hiện ứng viên trong cửa sổ chỉnh vùng.
- Nút Lưu 1-click tự động lưu vào `data/output/<slug>/do_than_dot/<level>_<timestamp>_<uuid>/` tạo phiên bản mới để lưu lại lịch sử đo (Audit Trail); nút Lưu khác… cho phép chọn thư mục tùy ý. Giữ các kết quả chứa ảnh bệnh nhân trong `data/` và không commit.

Tính lại từ vùng đã xác nhận, không chạy lại TotalSegmentator:

```powershell
.\.venv\Scripts\python.exe -m tools.analyze_vertebral_body --input data/processed/case_demo/ct_hu.npy --region data/output/duong_kim_hoa/do_than_dot/L1_example --output data/output/duong_kim_hoa/do_than_dot/L1_recomputed --samples 100000 --seed 42
```

## Kiểm chứng LoG

LoG dùng `LaplacianRecursiveGaussian` của SimpleITK với chuẩn hóa theo thang đo và sigma trong mm. Cường độ được giới hạn cố định [-200, 1500] HU rồi đưa về [0,1]; thuật toán tìm blob sáng và cực đại qua không gian/thang đo, không áp bộ lọc độ tròn/độ lồi. Vùng ứng viên là ROI hộp bao đã chiếu lên mặt phẳng và giới hạn bởi trường CT. [API SimpleITK](https://simpleitk.org/doxygen/v2_5/html/classitk_1_1simple_1_1LaplacianRecursiveGaussianImageFilter.html).

Để đánh giá, chuẩn bị các vùng thân đốt được duyệt trên cùng CT, chọn trước ngưỡng khoảng cách tâm và khóa tham số bằng dữ liệu validation độc lập. Ví dụ dưới đây dùng ngưỡng thử nghiệm 5 mm, chưa phải tiêu chuẩn của dự án:

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_body_blobs --input data/processed/case_demo/ct_hu.npy --regions data/output/duong_kim_hoa/do_than_dot/L1_example --output data/output/duong_kim_hoa/do_than_dot/blob_evaluation.json --tolerance-mm 5 --min-sigma-mm 2 --max-sigma-mm 12 --threshold 0.03
```

Báo cáo ghi tỷ lệ ROI có ứng viên trong ngưỡng so với tâm mask được duyệt, sai số tâm gần nhất, số ROI không có ứng viên và số ứng viên dư. Các vùng trong một bệnh nhân có tương quan; các chỉ số này chưa phải sensitivity lâm sàng. Không tự động biến ứng viên LoG thành mask thân đốt.

## Phạm vi kiểm chứng

Các test dùng hình tổng hợp để kiểm tra geometry, công thức đo, bảo toàn tỷ lệ, tái lập MC, LoG, lưu/mở lại và luồng giao diện. Chưa có kết quả đối chiếu chuyên gia trên dữ liệu bệnh nhân cho bước tách thân đốt hoặc các phép đo này. Chưa triển khai phân loại tổn thương/AO.
