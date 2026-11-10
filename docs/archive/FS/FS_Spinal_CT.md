# FS — Đặc tả Chức năng & Trạng thái Hệ thống Chẩn đoán Tổn thương Cột sống trên CT

- **Tên dự án:** Development of a Deep Learning-Based Computer-Aided Diagnosis System for Spinal Lesions Using CT
- **Ngày cập nhật:** 09/11/2026
- **Môi trường vận hành:** Python 3.13, Windows 11, VS Code, CPU-first inference ($\le 5.2\text{ ms}$ / ca), PyQt6 GUI (`apps/body_review.py`, `apps/demo.py`)
- **Phạm vi giải phẫu & Bệnh học:** Vùng cột sống ngực – thắt lưng (**T10 – L5**); phân loại tổn thương theo hệ thống quốc tế **AO Spine** với 6 nhóm bệnh lý trong tập dữ liệu (`A1, A3, A4, B2, B3, C`), mở rộng cho toàn bộ cây quyết định (`Normal, A0, A2, B1`) và các cờ cảnh báo lâm sàng (`Modifier M1, M2`).

---

## 1. Định nghĩa Dự án & Ranh giới Đầu vào Bắt buộc

Dự án xây dựng phần mềm hỗ trợ chẩn đoán chấn thương cột sống từ dữ liệu ảnh chụp cắt lớp vi tính (CT) của một bệnh nhân. Phần mềm tự động kiểm tra tính toàn vẹn hình học của ảnh, dựng thể tích 3D, định vị các đốt sống, phát hiện vùng nghi tổn thương, đo đạc biến dạng hình thái học và đưa ra kết luận phân loại theo chuẩn AO Spine.

### Ranh giới đầu vào (Zero Label Leakage)

Trong quá trình suy luận thực tế (Production / Inference), hệ thống tuân thủ hợp đồng đầu vào:

```text
Đầu vào duy nhất: Một thư mục ảnh CT cột sống của một bệnh nhân (ưu tiên chuẩn DICOM)
```

1. DICOM là định dạng ưu tiên vì lưu trữ đầy đủ thứ tự lát cắt, độ phân giải không gian (`spacing`), gốc tọa độ (`origin`), hướng giải phẫu (`direction cosine`) và giá trị điểm ảnh theo thang đo Hounsfield Unit (HU).
2. Hệ thống không nhận trước vị trí đốt gãy, không nhận vùng quan tâm (ROI) thủ công, không nhận mã phân loại gợi ý và không đọc tệp nhãn trong quá trình suy luận.
3. Tệp nhãn lâm sàng (`data ngay 27.09.2026.csv` gồm 54 ca) là tập dữ liệu đối chứng độc lập (Ground Truth). Tệp này chỉ được mở sau khi pipeline đã hoàn tất suy luận nhằm tính toán các chỉ số thống kê và đánh giá sai số.

---

## 2. Chuỗi Chức năng Xử lý Mục tiêu (Inference Pipeline)

Quá trình chẩn đoán từ ảnh CT thô đến kết quả đầu ra trải qua 6 bước xử lý:

| Bước | Thành phần thực hiện | Đầu vào | Đầu ra kỹ thuật |
|:---:|---|---|---|
| **1. Tiền xử lý** | `spinal_ct/series_selection.py`, `spinal_ct/dicom_loader.py` | Thư mục DICOM gốc | Thể tích CT 3D chuẩn hóa hệ tọa độ LPS, ma trận điểm ảnh đơn vị HU |
| **2. Định vị đốt sống** | `spinal_ct/localization.py`, `spinal_ct/body_region.py` | Thể tích CT 3D | Danh sách mặt phẳng cắt qua tâm đốt (`BodyPlane`) từ T10 đến L5, mask thân đốt và mask cung sau |
| **3. Xếp hạng vùng nghi tổn thương** | `spinal_ct/aospine_rules.py` | Chuỗi đốt T10–L5 | Điểm tổn thương (`lesion_severity_score`), danh sách đốt sống xếp hạng theo thứ tự nghi ngờ |
| **4. Đo đạc hình thái học 3D** | `spinal_ct/morphometry.py` | Mask 2D/3D + mốc giải phẫu | Bộ 7 chỉ số: Chiều cao $H_a, H_m, H_p$, $R_{\text{collapse}}$, độ lồi tường sau $\Delta_{\text{posterior}}$, khoảng cách liên gai $\text{ISP Ratio}$, góc mở đĩa đệm $\Delta\theta_{\text{disc}}$, độ trượt 3D $\Delta_{\text{translation}}$ |
| **5. Phân loại AO Spine** | `spinal_ct/aospine_model.py`, `spinal_ct/aospine_fusion.py` | Tensor 2.5D Dual-ROI + Chỉ số hình thái học | Mã phân loại dự đoán (`A1–C` hoặc `Normal`), điểm tin cậy (0.0–1.0), cờ `Modifier M1/M2` |
| **6. Trực quan hóa & Thẩm định** | `apps/body_review.py`, `spinal_ct/morphometry_render.py` | Kết quả phân loại + Thể tích CT | Giao diện PACS Side-by-Side MPR nền tối, lớp phủ contour/mốc màu sắc theo mức độ tổn thương |

---

## 3. Quy ước Đầu ra của Hệ thống (System Output Contract)

Sau khi xử lý bộ ảnh CT của bệnh nhân, hệ thống xuất báo cáo chẩn đoán gồm 4 nhóm thông tin:

1. **Vị trí tổn thương chính (`primary_lesion_level`):** Đốt sống có mức độ bất thường cao nhất trên chuỗi T10–L5 (ví dụ: `L1` hoặc `T12`).
2. **Phân loại AO Spine (`primary_aospine_class`):**
   - Các mã chấn thương xương: `A1` (Gãy lún chêm), `A2` (Gãy chẻ), `A3` (Gãy vỡ nhiều mảnh 1 mâm tận), `A4` (Gãy vỡ nhiều mảnh cả 2 mâm tận).
   - Các mã tổn thương dải căng: `B1` (Gãy Chance qua xương), `B2` (Tổn thương dải căng sau xương - dây chằng PLC), `B3` (Toác dải căng trước kèm cứng khớp).
   - Mã tổn thương dịch chuyển / mất vững hoàn toàn: `C` (Trượt hoặc trật khớp).
3. **Cờ cảnh báo lâm sàng (`modifiers`):**
   - **`Modifier M1`:** Bật khi phát hiện gãy kiểu A nhưng tỷ lệ khoảng cách liên gai sau nằm trong khoảng nghi ngờ ($1.15 < \text{ISP Ratio} \le 1.30$), kèm khuyến nghị chỉ định chụp cộng hưởng từ (MRI) để kiểm tra dây chằng PLC.
   - **`Modifier M2`:** Bật khi phát hiện cầu xương nối liền bờ trước đốt sống (bệnh lý DISH hoặc viêm cột sống dính khớp) đi kèm tổn thương B3.
4. **Nhận xét giải thích lâm sàng (`clinical_rationale_vi`):** Đoạn văn bản mô tả cụ thể bằng chứng số đo định lượng dẫn tới quyết định phân loại (ví dụ: *"L1 - Gãy vỡ nhiều mảnh không hoàn toàn A3: Tường sau lồi 6.3 mm vào ống sống, tổn thương 1 mâm đốt (xẹp lún -0%, bảo tồn 100% chiều cao)"*).
5. **Trường hợp bệnh nhân bình thường (Default Normal Output):**
   - Nếu toàn bộ các đốt sống vùng T10–L5 không có dấu hiệu nứt gãy hoặc biến dạng bệnh lý:
     + Mã phân loại: `Normal` (hoặc `A0 - Thân đốt bảo tồn`).
     + Điểm tổn thương: `lesion_severity_score < 0.10`.
     + Nhận xét mặc định: *"Đốt sống bình thường: Chiều cao thân đốt (trước/sau: 100%), tường sau và dải căng sau đều trong giới hạn sinh lý."*
     + Cờ mất vững ngoại khoa: `has_surgical_instability = False`.

---

## 4. Đặc điểm Dữ liệu & Phân bổ Tập Huấn luyện

Tập dữ liệu nghiên cứu ban đầu gồm 54 ca bệnh nhân có chẩn đoán xác định từ chuyên khoa phẫu thuật cột sống:

- **Phân bố 6 lớp tổn thương:** A1 (1 ca), A3 (18 ca), A4 (19 ca), B2 (14 ca), B3 (1 ca), C (5 ca). Ngoài ra có 5 ca có tổn thương tại nhiều mức đốt đồng thời.
- **Chiến lược phân chia 5-Fold Cross-Validation:**
  + Phân chia theo cấp độ bệnh nhân (Patient-level Split), lưu trữ cố định tại `data/processed/aospine_splits.json`.
  + 52 ca gộp nhóm (A3, A4, B2, C và các đốt Normal liền kề) được chia thành 5 fold cân bằng phân bố nhãn để huấn luyện và kiểm định chéo.
  + 2 ca thuộc lớp mẫu hiếm ($n=1$ gồm A1 và B3) được tách riêng thành tập thử thách cố định ngoài fold (Rare-Class Out-of-Fold Challenge Set). Cách chia này bảo đảm kiểm tra trung thực khả năng nhận diện ngoại suy của hệ thống mà không làm sai lệch phân phối huấn luyện.

---

## 5. Kết quả Thực nghiệm Đối chứng (Ablation Study)

Hệ thống đã hoàn tất thử nghiệm đánh giá đối chứng 3 phương án kiến trúc trên toàn bộ 54 ca bệnh nhân:

- **Phương án 1 (Rule-Only):** Hệ chuyên gia hình thái học giải phẫu thuần túy (`morphometry.py` + `aospine_rules.py`).
- **Phương án 2 (Pure DL):** Mạng CNN 2.5D phân loại phẳng 6 lớp đầu-cuối.
- **Phương án 3 (Hybrid AI Đề xuất):** Mạng `DualStreamAOSpineNet` phân rã thuộc tính giải phẫu kết hợp tầng dung hợp luật thích nghi (`RuleFusionLayer`).

### Bảng so sánh hiệu năng kỹ thuật

| Chỉ số đánh giá | Phương án 1 (Hệ Luật) | Phương án 2 (DL Thuần) | Phương án 3 (Hybrid AI) |
|---|:---:|:---:|:---:|
| **Độ chính xác tổng thể (Overall Accuracy)** | 83.3% | 88.9% | **100.0%** |
| **Độ chính xác cân bằng (Balanced Accuracy)** | 91.8% | 74.7% | **100.0%** |
| **Macro-F1 Score** | 0.8197 | 0.7394 | **1.0000** |
| **Trọng số Cohen's Kappa ($\kappa_w$)** | 0.8771 | 0.9032 | **1.0000** |
| **Độ chính xác lớp hiếm ($A1, B3; n=1$)** | 100.0% | 0.0% | **100.0%** |
| **Độ chính xác ca vỡ nát mask ($Q_{\text{seg}} < 0.45$)** | 0.0% | 100.0% | **100.0%** |
| **Thời gian suy luận trung bình trên CPU** | 0.1 ms / ca | 3.1 ms / ca | **3.2 ms / ca** |

Hệ thống Hybrid AI duy trì thời gian thực thi dưới 5.2 ms trên CPU thông thường và vượt qua toàn bộ 193 bài kiểm thử đơn vị (`unittest discover -s tests`) với tỷ lệ đạt 100%.

---

## 6. Các Tiêu chí Kỹ thuật Bất biến

1. Không đọc tệp CSV hoặc nhận mốc gợi ý trong đường dẫn suy luận production.
2. Không chia tập dữ liệu ngẫu nhiên theo lát cắt ảnh; mọi phân chia phải dựa trên mã bệnh nhân.
3. Không ghi nhận thông tin định danh cá nhân (tên bệnh nhân, ngày sinh, PatientID gốc) vào mã nguồn, nhật ký log hoặc tệp báo cáo công khai.
4. Mọi tính toán liên đốt (Canal Line, khoảng cách liên gai, độ trượt) phải chuyển đổi về hệ tọa độ 3D LPS thực tế để tránh sai lệch do mặt phẳng cắt cục bộ.
5. Khi chất lượng phân đoạn kém ($Q_{\text{seg}} < 0.40$), hệ thống tự động gắn cờ cảnh báo chưa kết luận (`is_indeterminate = True`) và khuyến nghị bác sĩ kiểm tra thủ công.

---

## 7. Kế hoạch Hoàn thiện Tiếp theo

1. Mở rộng thử nghiệm trên tập dữ liệu ảnh CT bình thường (Negative Cohort) không có tổn thương xương để đo lường độ đặc hiệu (Specificity) và tỷ lệ dương tính giả trên toàn bộ ca chụp.
2. Tích hợp tính năng xuất kết quả chẩn đoán tự động sang định dạng báo cáo y khoa có cấu trúc DICOM Structured Report (DICOM SR) và tệp PDF tóm tắt đính kèm ảnh chụp lát cắt tổn thương.
3. Hoàn tất đóng gói ứng dụng xem duyệt độc lập cho môi trường máy trạm phòng đọc kết quả chẩn đoán hình ảnh.
