# Monte Carlo đo độ ổn định định vị đốt sống

## Mục đích và phạm vi

Bổ sung Monte Carlo input perturbation vào pipeline TotalSegmentator. Với CT
`X`, mỗi lần lấy mẫu tạo `X_i = X + ε_i`, với nhiễu voxel độc lập
`ε_i ~ Normal(0, sigma_HU²)`, rồi chạy lại phân đoạn từng đốt.
Geometry và hệ tọa độ LPS giữ nguyên nên không cần biến đổi ngược mask.

Đây là độ nhạy với nhiễu HU do người dùng giả định, không phải MC Dropout,
Bayesian posterior, xác suất tổn thương hoặc mô phỏng vật lý thu nhận CT.
Nhiễu Gaussian độc lập chưa mô tả tương quan không gian, kernel tái tạo,
metal artifact hoặc khác biệt máy chụp. Mô hình vẫn có thể gán sai tên đốt một
cách nhất quán ở mọi lần chạy. `confidence` tiếp tục là `null`.

## Cấu trúc code

| Tệp | Vai trò |
|---|---|
| `tools/localize_vertebrae.py` | CLI, kiểm tra tham số, gọi backend cho ảnh gốc và N mẫu, công bố kết quả khi tất cả thành công |
| `spinal_ct/localization.py` | Đọc DICOM/NIfTI, kiểm tra geometry, chuyển mask thành ROI và render preview |
| `spinal_ct/monte_carlo.py` | `perturb_hu`: tạo mẫu nhiễu; `aggregate_samples`: tính đồng thuận và thống kê |
| `tests/test_monte_carlo.py` | Kiểm thử số học, seed, hình học, nhãn cạnh tranh và ca thiếu phát hiện |
| `tests/test_localization.py` | Kiểm thử CLI cả chế độ thường và Monte Carlo; kiểm tra hủy đầu ra khi backend lỗi |

Luồng thực thi:

```text
DICOM/NIfTI → CT HU + geometry
                  ├─ phân đoạn ảnh gốc → masks/, prediction.json, levels_mpr.png
                  └─ N mẫu nhiễu → N lần phân đoạn → mc_samples/
                                                    ↓
                                  mask đồng thuận + bản đồ bất đồng
                                  + thống kê từng đốt trong prediction.json
```

## Chạy trong cloud hiện tại

Từ thư mục gốc dự án, sau khi đặt dữ liệu đã ẩn danh vào `data/raw/case_demo`:

```bash
.venv/bin/python -m tools.localize_vertebrae \
  --input data/raw/case_demo \
  --output data/processed/case_demo/localization_mc \
  --backend /workspace/.venvs/spine-model/bin/TotalSegmentator \
  --device cpu --fast \
  --mc-samples 5 --mc-noise-hu 20 --mc-seed 42
```

Nếu có nhiều series, thêm `--series-uid` để chọn CT phù hợp. Có thể dùng
`--input data/processed/case_demo/ct.nii.gz` nếu đó là volume HU với geometry
đúng. Thư mục output phải mới; không ghi đè kết quả đã có.
Cài backend trên máy khác theo [hướng dẫn định vị](vertebral_localization.md).
Trên Windows đổi đường dẫn executable theo môi trường Python của máy đó.

| Tham số | Ý nghĩa |
|---|---|
| `--mc-samples 0` | Mặc định, tắt Monte Carlo |
| `--mc-samples N` | 2–64 lần phân đoạn mẫu nhiễu, ngoài 1 lần ảnh gốc |
| `--mc-noise-hu 20` | Sigma Gaussian theo HU; 20 là giá trị thử nghiệm, chưa được xác nhận cho dữ liệu dự án |
| `--mc-seed 42` | Seed không âm, tái lập nhiễu đầu vào; không bảo đảm backend/GPU hoàn toàn deterministic |
| `--backend-log /tmp/run_mc.log` | Tệp log mới, ghi log baseline và từng mẫu để chẩn đoán lỗi |

Ví dụ N=5 cần **6 lần suy luận**, nên thời gian thường tăng gần tương ứng.
Chạy tuần tự để giới hạn RAM. Các label map mẫu được lưu để kiểm tra; tổng hợp
sử dụng memory map tạm, cần dung lượng đĩa tỷ lệ với N × số voxel, cộng các
mask và đầu ra. Không đặt N lớn trước khi đo thời gian/RAM/dung lượng trên ca
đại diện. `--fast` giảm độ phân giải mô hình; cần đánh giá ảnh hưởng đến độ ổn
định trước khi chọn cho benchmark.

## Đọc kết quả

Ảnh QC dùng trục khoảng cách mm, màu theo mức đốt và nhãn ngoài đường biên.
`consensus_mpr.png` hiển thị thêm bản đồ bất đồng trên cùng lát sagittal.
Đường biên lấy trực tiếp từ mask với contour 0.5; không làm trơn mask để tránh
thay đổi hình học và phép đo. Bậc thang từ phân đoạn `--fast` 3 mm vẫn có thể thấy.
Chạy `python -m tools.render_localization --help` để dựng ảnh từ kết quả có sẵn.

Đầu ra baseline `masks/`, `vertebrae.nii.gz`, `levels_mpr.png` và
`predictions` trong JSON vẫn mô tả phân đoạn ảnh gốc. Monte Carlo bổ sung:

- `mc_samples/000.nii.gz`, ...: label map từng mẫu, dùng `label_map` chung.
- `monte_carlo/consensus.nii.gz`: label map đồng thuận.
- `monte_carlo/consensus_masks/vertebrae_<level>.nii.gz`: mask từng đốt đồng thuận.
- `monte_carlo/consensus_mpr.png`: ảnh sagittal của kết quả đồng thuận.
- `monte_carlo/disagreement.nii.gz`: giá trị voxel `1 − max_k(count_k/N)`,
  tính cả background. 0 nghĩa mọi mẫu nhất trí, giá trị cao hơn nghĩa bất đồng
  lớn hơn; không diễn giải là xác suất sai.
- `prediction.json → monte_carlo`: cấu hình, cảnh báo từng lần chạy,
  thống kê cho đủ 25 mức đốt và bản tóm tắt ROI đồng thuận.

Mask đồng thuận yêu cầu **hơn 50%** mẫu chọn cùng một nhãn. Voxel không có đa
số được đặt thành background và đếm trong `no_majority_voxels`; tránh tự chọn
một tên đốt khi hòa phiếu. Ảnh gốc không tham gia phép bỏ phiếu.

Các trường trong `monte_carlo.per_level`:

| Trường | Cách hiểu |
|---|---|
| `detected_samples` | Số mẫu có mask không rỗng cho đốt đó |
| `detection_frequency` | Tỷ lệ xuất hiện trên N mẫu; không phải xác suất đúng nhãn |
| `volume_mean_mm3`, `volume_std_mm3` | Trung bình và độ lệch chuẩn mẫu, tính cả thể tích 0 khi bỏ sót |
| `centroid_mean_lps_mm` | Tâm trung bình theo x/y/z LPS, chỉ trên những mẫu phát hiện được |
| `centroid_std_lps_mm` | Độ lệch chuẩn mẫu theo từng trục, `null` nếu có dưới 2 lần phát hiện |

Độ lệch chuẩn dùng `ddof=1`; không phải khoảng tin cậy lâm sàng. Khi không
phát hiện lần nào, tâm là `null`, thể tích trung bình bằng 0. Các cảnh báo
geometry/thứ tự/biên ảnh được giữ cho từng mẫu và consensus. Ca có cảnh báo
hoặc voxel không có đa số được đánh dấu `indeterminate`; luôn cần review.

Không tự động thay baseline bằng consensus: đồng thuận có thể loại bỏ cấu
trúc nhỏ hoặc giữ lỗi có hệ thống. Cần đối chiếu cả hai trên nhãn chuyên gia.
Mọi ảnh/mask vẫn là dữ liệu người bệnh; giữ dưới `data/` và không commit.

## Kiểm thử và đánh giá

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Các test fixture kiểm tra pipeline, không chứng minh chất lượng mô hình trên
người. Đánh giá thực tế cần CT và mask/tên đốt đã được chuyên gia duyệt:

1. Chọn sigma và số mẫu trên validation; tăng N để xem các thống kê có ổn định.
2. So sánh baseline/consensus bằng Dice, IoU, sai số tâm và độ chính xác mức đốt.
3. Kiểm tra vùng có disagreement cao có tương ứng lỗi thật hay không; báo cả
   trường hợp mô hình sai nhưng ổn định. Đo quan hệ giữa độ bất định và lỗi.
4. Khóa tham số trước khi chạy test theo bệnh nhân. Báo thời gian, bộ nhớ,
   tỷ lệ thiếu phát hiện và `indeterminate` bên cạnh chỉ số chất lượng.

Chưa có dữ liệu CT người trong workspace để đánh giá độ chính xác hoặc chọn
sigma phù hợp. Không dùng các chỉ số ổn định này để quyết định phân loại AO.

Kiểm chứng ngày 22/10/2026: 15/15 unit/integration tests đạt. Backend thật đã
chạy baseline + 2 mẫu nhiễu trên phantom tổng hợp, không tìm thấy đốt và trả
`indeterminate`; disagreement bằng 0 vì mọi mẫu đều rỗng. Điều này cũng minh
họa vì sao bất đồng thấp không chứng minh kết quả đúng. Nhánh crop rỗng đã
được thực thi; chưa kiểm chứng phân đoạn đầy đủ giải phẫu người trong MC.
