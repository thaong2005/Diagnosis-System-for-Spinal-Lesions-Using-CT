# Lộ trình Thực thi Chi tiết (Step-by-Step Roadmap) & Giao thức Đánh giá Khoa học

> **Mục đích tài liệu:** Hướng dẫn từng bước thực hiện mã nguồn (Implementation Guide), cấu trúc các file cần chỉnh sửa/tạo mới trong repo `Diagnosis-System-for-Spinal-Lesions-Using-CT`, và giao thức thực nghiệm chuẩn để báo cáo trong Tiền luận văn (Pre-thesis) và Luận văn chính thức (Thesis).

---

## 1. Tổng quan Lộ trình 3 Giai đoạn

```mermaid
gantt
    title Lộ trình Phát triển Tính năng Phân loại Tổn thương AO Spine
    dateFormat  YYYY-MM-DD
    axisFormat  Tuần %W
    
    section Giai đoạn 1: Hệ Chuyên gia Hình học (Pre-thesis)
    Bước 1: Nâng cấp spinal_ct/morphometry.py (Cột sau, Canal Line, 3D Shift) :a1, 2026-11-05, 7d
    Bước 2: Xây dựng spinal_ct/aospine_rules.py (Fuzzy Tree & Ranking T10-L5) :a2, after a1, 7d
    Bước 3: Tích hợp GUI apps/body_review.py & Chạy Baseline 54 ca            :a3, after a2, 7d

    section Giai đoạn 2: Huấn luyện Dual-Stream 2.5D CNN (Thesis)
    Bước 4: Chia 5-Fold Patient Split & Trích xuất Tensor 2.5D (.npz)         :b1, after a3, 7d
    Bước 5: Huấn luyện Concept-based CNN trên GPU & Xuất ONNX                 :b2, after b1, 14d

    section Giai đoạn 3: Hợp nhất Rule Fusion & Báo cáo
    Bước 6: Tích hợp ONNX Runtime CPU Batching & RuleFusionLayer              :c1, after b2, 7d
    Bước 7: Chạy Ablation Study (PA1 vs PA2 vs PA3) & Viết báo cáo            :c2, after c1, 7d
```

---

## 2. Chi tiết Kỹ thuật Từng Bước Thực hiện

### GIAI ĐOẠN 1: Hoàn thiện Hệ Chuyên gia Hình thái học (Pre-thesis — Chạy 100% trên CPU)

Mục tiêu: Chưa cần GPU hay mạng CNN, nâng cấp trực tiếp thuật toán hình học hiện có để phân loại được đầy đủ 6 lớp (`A1, A3, A4, B2, B3, C`) và tích hợp ngay lên giao diện PyQt6.

#### Bước 1: Mở rộng `spinal_ct/morphometry.py`
Cập nhật các hàm hiện có và bổ sung 4 hàm trích xuất chỉ số giải phẫu mới:

1. **Giữ lại Mask Cung sau (`posterior_arch_mask`):**
   - Trong `auto_segment_body()`, tại bước tìm đường cắt cổ cuống cung `cut_col`:
     - Phần `col <= cut_col`: Lưu làm `body_mask` để tìm 6 mốc $H_a, H_m, H_p$ như cũ.
     - Phần `col > cut_col`: **Không xóa bỏ**, lưu thành `posterior_arch_mask` trả về trong dictionary kết quả.
   - Tìm tọa độ **Đỉnh mỏm gai (Spinous Process Tip)** $P_{\text{spinous}}(i)$ bằng cách lấy cực đại phía sau-dưới (max `col`, min `row` trong hệ LPS) của `posterior_arch_mask`.

2. **Hàm `compute_posterior_retropulsion()` (Bắt chuẩn A3, A4 — Chống sót mảnh xương rời):**
   - **Đầu vào:** Ảnh CT Sagittal gốc (đơn vị HU), tọa độ góc sau-dưới của đốt liền trên $P_{\text{PI}}(i+1)$ và góc sau-trên của đốt liền dưới $P_{\text{PS}}(i-1)$.
   - **Thuật toán:**
     - Dựng đường thẳng **Giới hạn trước ống sống (Anterior Spinal Canal Line)** đi qua $P_{\text{PI}}(i+1)$ và $P_{\text{PS}}(i-1)$.
     - Quét vùng cửa sổ ống sống (từ đường giới hạn lùi về phía sau $12\text{ mm}$) trên **3 lát cắt** (Mid-sagittal và 2 lát Parasagittal $\pm 4\text{ mm}$).
     - Tìm điểm có mật độ xương đặc/vỏ xương ($\text{HU} \ge 250$) nhô vượt qua đường giới hạn về phía sau xa nhất. Khoảng cách vuông góc từ điểm đó tới đường giới hạn chính là $\Delta_{\text{posterior}}\text{ (mm)}$.

3. **Hàm `compute_interspinous_And_disc_metrics()` (Bắt chuẩn B2, B3, M1, M2):**
   - **Tỷ lệ giãn khoảng liên gai ($\text{ISP Ratio}$ — Cho B2 & M1):**
     $$D_{\text{spinous}}(i) = \max\left(\|P_{\text{spinous}}(i+1) - P_{\text{spinous}}(i)\|, \|P_{\text{spinous}}(i) - P_{\text{spinous}}(i-1)\|\right)$$
     $$\text{ISP Ratio}(i) = \frac{D_{\text{spinous}}(i)}{\text{median}_{j \in \text{T10..L5}} D_{\text{spinous}}(j)}$$
   - **Góc mở khe đĩa đệm phía trước ($\Delta\theta_{\text{disc}}$ — Cho B3):**
     - Tính góc giữa đường thẳng nối tấm tận dưới đốt $i+1$ và tấm tận trên đốt $i$ (và tương tự cho cặp $i, i-1$).
     - Tính tỷ số khoảng cách bờ trước chia cho bờ sau của khe đĩa đệm: $R_{\text{disc}} = d_{\text{disc\_ant}} / d_{\text{disc\_post}}$.

4. **Hàm `compute_3d_translation_and_collapse()` (Bắt chuẩn C và A4):**
   - **Độ di lệch trục 3D ($\Delta_{\text{translation}}$ — Cho C):**
     - Lấy tâm hình học 3D (3D Centroid trong hệ tọa độ thực $\text{mm}$) của `body_mask` các đốt từ T10 đến L5.
     - Nội suy đường cong mượt (Spline / Linear nối đốt $i-1$ và $i+1$) và tính khoảng cách lệch ngang + lệch trước-sau của tâm đốt $i$ so với trục nội suy: $\Delta_{\text{translation}}(i) = \sqrt{\Delta_{\text{AP}}^2 + \Delta_{\text{Lat}}^2}$.
   - **Tỷ lệ sập lún tương đối ($R_{\text{collapse}}$ và Chỉ số Vỡ 2 tấm tận — Cho A4 vs A3):**
     - So sánh chiều cao đốt $i$ với trung bình cộng đốt $i-1$ và $i+1$:
       $$R_{\text{collapse}}(i) = \frac{H_m(i)}{0.5 \times [H_m(i-1) + H_m(i+1)]}$$
     - Đo độ gồ ghề / mất liên tục (Endplate Irregularity / Step-off) trên cả bờ trên (Superior Endplate) và bờ dưới (Inferior Endplate) để phân biệt hỏng 1 tấm tận (**A3**) hay hỏng cả 2 tấm tận (**A4**).

---

#### Bước 2: Xây dựng Module `spinal_ct/aospine_rules.py` (File Mới)
Tạo module độc lập thực hiện suy luận mờ (Fuzzy Logic) từ các số đo của Bước 1:

* **Định nghĩa các ngưỡng Sigmoid chuẩn ($\tau_k, T_k$):**
  - `Translation (C)`: $\tau_{\text{trans}} = 3.5\text{ mm}, T = 0.8$
  - `Anterior Tension Band (B3)`: $\tau_{\text{disc\_angle}} = 15.0^\circ$ và $R_{\text{disc}} > 2.2$
  - `Posterior Tension Band (B2)`: $\tau_{\text{ISP}} = 1.30, T = 0.08$ (Cảnh báo `M1` nếu $1.15 < \text{ISP Ratio} \le 1.30$)
  - `Posterior Wall Retropulsion (A3/A4)`: $\tau_{\text{PW}} = 2.0\text{ mm}, T = 0.5$
  - `Both Endplates Damaged (A4 vs A3)`: Kết hợp độ sập lún toàn bộ ($R_{\text{collapse}} < 0.75$) và độ gãy gập của cả 2 bờ tấm tận trên/dưới.
* **Hàm `rank_and_classify_spine(patient_morphometry_dict)`:**
  - Duyệt qua 8 đốt **T10 – L5**, tính điểm nghi ngờ chấn thương tổng hợp cho từng đốt, sắp xếp giảm dần (Top-1, Top-2) và trả về cấu trúc JSON chuẩn gồm: `vertebra_level`, `predicted_aospine_class`, `confidence`, `modifier_flags`, `anatomical_scores`, `explanation_text`.

---

#### Bước 3: Tích hợp lên Giao diện `apps/body_review.py` & Đánh giá Baseline
1. **Hiển thị trực quan trên GUI:**
   - Vẽ thêm **Đường giới hạn trước ống sống (màu xanh lá)** và **Vùng mảnh xương lồi vượt ngưỡng (màu đỏ cam)** trên lát cắt Sagittal để bác sĩ nhìn thấy ngay bằng chứng của `A3/A4`.
   - Vẽ đoạn thẳng nối 2 đỉnh mỏm gai hiển thị chỉ số `ISP Ratio` cho `B2`.
   - Hiển thị Bảng Gợi ý AO Spine ở góc phải giao diện kèm giải thích thông số.
2. **Cập nhật Tương tác Thời gian thực (Human-in-the-loop Hook):**
   - Khi bác sĩ dùng chuột kéo chỉnh bất kỳ mốc nào trong 6 mốc thân đốt trên `apps/body_review.py`, tự động gọi lại `aospine_rules.py` (chạy mất $< 5\text{ ms}$) để cập nhật lại ngay nhãn AO Spine gợi ý.

---

### GIAI ĐOẠN 2: Xây dựng Dữ liệu & Huấn luyện Nhánh Deep Learning (Thesis)

#### Bước 4: Chuẩn bị Dữ liệu & Chia Tập Đánh giá (`scripts/prepare_aospine_dataset.py`)
1. **Chia 5-Fold Stratified Group Split theo `PatientID`:**
   - Lấy 52 ca thuộc 4 lớp chính ($\text{A4: } 19, \text{A3: } 14, \text{B2: } 14, \text{C: } 5$) chia đều vào 5 Fold sao cho không có bệnh nhân nào xuất hiện ở cả Train và Val.
   - Đưa 2 ca đơn lẻ ($\text{A1: } 1, \text{B3: } 1$) vào tập **`Rare-Class Challenge Set`** cố định (dùng để kiểm thử khả năng suy luận tổ hợp thuộc tính của mô hình ở mọi Fold).
2. **Lọc Mẫu Âm tính Sạch (Clean Normal Sampling):**
   - Trong mỗi ca bệnh, chỉ lấy các đốt sống không nằm trong CSV, **cách đốt gãy ít nhất 2 tầng** và có tỷ lệ $H_a/H_p \ge 0.85, R_{\text{collapse}} \ge 0.88$ làm mẫu `Normal`. Các đốt nằm sát ổ gãy được đánh dấu `ignore_index` để không làm nhiễu quá trình huấn luyện.
3. **Xuất Tensor 2.5D Dual-ROI (`.npz`):**
   - Với mỗi đốt sống, cắt và chuẩn hóa kích thước 2 tensor `body_roi` ($5 \times 128 \times 128$) và `posterior_roi` ($5 \times 128 \times 128$) cùng vector nhãn 4 thuộc tính giải phẫu $\mathbf{y}_{\text{concept}} \in \{0, 1\}^4$.

#### Bước 5: Huấn luyện Mạng Dual-Stream 2.5D CNN (`training/train_concept_cnn.py`)
1. **Kiến trúc mạng:**
   - Sử dụng 2 nhánh **ConvNeXt-Femto** hoặc **MobileNetV3-Small** (từ `timm`, thay tầng `conv_stem` đầu tiên nhận `in_channels=5`).
   - Nối (Concatenate) vector đặc trưng của 2 nhánh + ghép thêm vector hình học chuẩn hóa $\mathbf{v}_{\text{morph}}$ (Late Feature Fusion) trước khi đi qua tầng Fully Connected dự đoán 4 xác suất thuộc tính: $[z_{\text{Trans}}, z_{\text{TB}}, z_{\text{PW}}, z_{\text{EP2}}]$.
2. **Hàm mất mát (Loss Function):**
   - Dùng **Asymmetric Focal Binary Cross-Entropy Loss** có đánh trọng số `pos_weight` cho từng thuộc tính để cân bằng số lượng mẫu dương/âm.
3. **Xuất mô hình sang ONNX:**
   - Xuất trọng số tốt nhất của từng Fold ra file `models/aospine_concept_25d.onnx` (`opset_version=17`, bật tối ưu hóa đồ thị).

---

### GIAI ĐOẠN 3: Hợp nhất Rule Fusion Layer & Báo cáo Thực nghiệm Luận văn

#### Bước 6: Triển khai `spinal_ct/fusion_classifier.py`
1. Load model ONNX bằng `onnxruntime.InferenceSession` với `CPUExecutionProvider`.
2. Đóng gói cả 8 đốt sống (**T10 – L5**) của bệnh nhân thành 1 batch duy nhất `(Batch=8, C=5, H=128, W=128)` để suy luận trong 1 lần gọi ($\sim 0.15\text{ giây}$).
3. Kết hợp đầu ra $P_{\text{DL}}$ từ ONNX và $P_{\text{Rule}}$ từ `aospine_rules.py` theo công thức trọng số động $\alpha_k(Q_{\text{seg}})$ để xuất kết quả cuối cùng.

#### Bước 7: Giao thức Thực nghiệm & Bảng Đánh giá (Evaluation Protocol cho Luận văn)
Viết script `scripts/evaluate_benchmark.py` chạy tự động trên toàn bộ 54 ca bệnh (tuân thủ nghiêm ngặt Production Contract) và xuất 3 bảng kết quả phục vụ bảo vệ luận văn:

1. **Bảng 1 — Đánh giá Khả năng Định vị Đốt Tổn thương (Vertebra Detection Ranking trên T10–L5):**
   - **Top-1 Localization Accuracy (%):** Tỷ lệ bệnh nhân mà đốt sống chấn thương thực tế đứng vị trí số 1 trong danh sách nghi ngờ của hệ thống.
   - **Top-2 Localization Accuracy (%):** Tỷ lệ bệnh nhân mà đốt gãy nằm trong 2 vị trí đầu tiên.
   - **Sensitivity @ 1 False Positive / Scan.**

2. **Bảng 2 — Đánh giá Nhận diện 4 Thuộc tính Giải phẫu (Anatomical Concept Performance):**
   - Báo cáo **AUC-ROC, Sensitivity, Specificity, F1-score** cho từng tiêu chí: (1) Vỡ tường sau, (2) Vỡ 2 tấm tận, (3) Tổn thương dải căng sau, (4) Trật trục cột sống.

3. **Bảng 3 — Nghiên cứu Đối chứng (Ablation Study) Phân loại AO Spine:**
   So sánh trực tiếp 3 phương án trên cùng tập dữ liệu:
   - **Baseline 1 (Phương án 1):** Hệ chuyên gia Hình thái học thuần túy (`morphometry.py` + `aospine_rules.py`).
   - **Baseline 2 (Phương án 2):** Mạng CNN 2.5D phân loại phẳng 6 lớp (Pure End-to-End DL).
   - **Proposed Method (Phương án 3 Cải tiến):** Dual-Stream Concept CNN + Adaptive Rule Fusion Layer.
   - **Các chỉ số so sánh:**
     - **Group-level Macro-F1** (Type A vs Type B vs Type C).
     - **6-Class Balanced Accuracy & Macro-F1**.
     - **Quadratic Weighted Cohen's Kappa ($\kappa_w$)** (Đánh giá mức độ đồng thuận lâm sàng có tính đến trọng số mức độ nặng).
     - **Độ chính xác trên tập Rare-Class Challenge Set ($A1, B3$)**.
     - **Thời gian suy luận trung bình trên CPU (ms / ca bệnh)**.

---

## 3. Hướng dẫn Sử dụng Tài liệu này cho các Cuộc trò chuyện Sau

Khi bắt đầu một phiên trò chuyện mới để viết code cho từng module, bạn chỉ cần đính kèm 2 file `.md` này kèm theo câu lệnh ngắn gọn tương ứng với từng bước:
- **Phiên 1:** *"Dựa vào 2 file đặc tả lộ trình AO Spine, hãy viết code nâng cấp file `spinal_ct/morphometry.py` (Bước 1) để giữ lại mask cung sau, đo độ lồi tường sau theo đường giới hạn ống sống và tính tỷ lệ khoảng cách liên gai."*
- **Phiên 2:** *"Hãy viết đầy đủ file `spinal_ct/aospine_rules.py` (Bước 2) hiện thực hóa cây quyết định mờ AO Spine và hàm xếp hạng đốt sống tổn thương T10–L5."*
- **Phiên 3:** *"Hãy tích hợp kết quả từ `aospine_rules.py` lên giao diện PyQt6 trong `apps/body_review.py` (Bước 3)."*
- **Phiên 4:** *"Hãy viết script `scripts/prepare_aospine_dataset.py` và `training/train_concept_cnn.py` (Bước 4 & 5) để huấn luyện mạng Dual-Stream 2.5D CNN."*