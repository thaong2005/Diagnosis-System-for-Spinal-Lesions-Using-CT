# Xác định từng đốt sống từ CT

CLI `python -m tools.localize_vertebrae` dùng TotalSegmentator 2.18.0, task
`total`, để tạo mask C1–C7, T1–T12, L1–L5 và S1. Đây là định vị giải phẫu;
chưa phát hiện gãy, đo giảm chiều cao hoặc phân loại AO Spine. Backend không
trả xác suất đã hiệu chỉnh, nên `confidence` trong JSON là `null`.

## Cài đặt

Môi trường ứng dụng: Python 3.13 và `requirements.txt` như hiện tại.
Cài backend vào môi trường riêng để không đổi các phiên bản của ứng dụng:

```bash
python3.13 -m venv .venv-model
.venv-model/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install -r requirements-localization.txt
.venv-model/bin/python -m pip check
.venv-model/bin/TotalSegmentator --version
```

Windows dùng `.venv-model\Scripts\python.exe` và
`.venv-model\Scripts\TotalSegmentator.exe`. Với GPU, cài cặp torch/torchvision
phù hợp CUDA theo hướng dẫn chính thức PyTorch rồi dùng `--device gpu`.
Không trộn torchvision CUDA với torch CPU.

Trong môi trường cloud của dự án, backend đã được cài riêng tại
`/workspace/.venvs/spine-model/bin/TotalSegmentator`.
Lần chạy đầu cần truy cập GitHub release assets để tải trọng số. Trọng số giữ
trong `.tmp/totalsegmentator` (đã được Git ignore); có thể đổi bằng
`--model-cache`. Cấu hình tạm của backend tắt gửi thống kê sử dụng.

## Chạy

Từ thư mục gốc kho mã, với môi trường ứng dụng đã kích hoạt:

```bash
python -m tools.localize_vertebrae \
  --input data/raw/anonymous_case \
  --output data/processed/anonymous_case/localization \
  --backend /workspace/.venvs/spine-model/bin/TotalSegmentator \
  --device cpu --fast
```

Đầu vào là thư mục DICOM (quét đệ quy) hoặc NIfTI 3D có cường độ HU và geometry
đúng. Không nhận PNG/JPG, ảnh chụp màn hình hoặc CSV nhãn. Với nhiều series CT,
chọn rõ `--series-uid <UID>` từ công cụ xem DICOM cục bộ. Không đưa UID vào báo
cáo chia sẻ. NIfTI phải được người dùng xác nhận là CT/HU; định dạng tệp tự nó
không chứng minh được điều này.

DICOM hỗ trợ classic single-frame CT, có orientation, position, spacing và
slope/intercept. Từ chối single-slice, enhanced multi-frame, thiếu metadata,
khoảng cách không đều, lát trùng và gantry shear. Ảnh oblique có stack đều được
bảo toàn geometry. SimpleITK áp dụng slope/intercept để chuyển sang HU.

`--fast` dùng model 3 mm để kiểm tra nhanh; bỏ cờ này để dùng model 1.5 mm.
Chọn chế độ dựa trên benchmark thực tế trước khi đánh giá chất lượng. Thời gian
và RAM phụ thuộc kích thước volume; CPU có thể mất nhiều phút.
Demo mặc định bỏ `--fast` để dùng mô hình 1,5 mm và bật `--robust-crop`:
backend dùng bộ định vị vùng cắt 3 mm thay cho 6 mm khi chạy ROI subset.
CLI truyền tùy chọn này thành `--robust_crop` của TotalSegmentator 2.18.0;
baseline và tất cả mẫu Monte Carlo dùng cùng cấu hình. Báo cáo lưu
`model.resolution` và `model.robust_crop` để đối chiếu các lượt chạy.
Đây là lựa chọn phân đoạn chi tiết hơn, chưa phải bằng chứng tăng độ chính xác
trên dữ liệu dự án; vẫn cần so sánh với nhãn chuyên gia.
Thư mục output phải chưa tồn tại. Đầu ra chỉ được công bố sau khi toàn bộ xử lý
thành công. Lỗi trả exit code 1; lỗi tham số trả 2. Có thể thêm
`--backend-log <tệp-mới>` để giữ chẩn đoán backend; tệp này chỉ nhận log từ CT đã
loại metadata và đường dẫn làm việc tạm.

## Output

- `masks/vertebrae_<level>.nii.gz`: mask nhị phân từng mức đốt; mask rỗng có
  nghĩa backend không phát hiện mức đó, không chứng minh mức đó không tồn tại.
- `vertebrae.nii.gz`: label map hợp nhất; bảng `label_map` trong JSON định nghĩa ID.
- `prediction.json`: phiên bản backend, cấu hình chạy, geometry LPS, mức đốt,
  tâm, thể tích và ROI.
- `levels_mpr.png`: một mặt phẳng sagittal có nhiều voxel đốt sống nhất, có
  đường viền và tên đốt. Có thể không cắt qua mọi đốt; cần kiểm tra mask 3D.

ROI voxel dùng thứ tự **x,y,z**, cận trên loại trừ. ROI LPS tính bằng tám góc
biên voxel, đơn vị mm, là hộp bao song song trục LPS. Mask và label map giữ lưới
ảnh gốc. Ảnh preview được resample về trục LPS khi cần.

`status=ok` chỉ có nghĩa đầu ra vượt qua kiểm tra kỹ thuật. Luôn có
`review_required=true`. `indeterminate` khi không có mask, mask chạm biên,
thiếu mức giữa hai đốt hoặc thứ tự tâm đốt theo trục trên–dưới bất thường.
Những kiểm tra này là cảnh báo sơ bộ, không bảo đảm nhãn giải phẫu đúng, nhất
là trường chụp hẹp, biến thể chuyển tiếp, vẹo nặng, gãy biến dạng và kim loại.
Các file ảnh/mask vẫn là dữ liệu bệnh nhân dù đã loại metadata; giữ dưới
`data/` và không commit.

## Kiểm thử và giới hạn bằng chứng

```bash
python -m unittest discover -s tests -v
```

Bộ kiểm thử gồm DICOM tổng hợp, HU, chọn series, hình học ROI có hướng lật,
mask thiếu/chồng lấn/sai geometry, cảnh báo chất lượng, render preview và CLI
với backend fixture. Kiểm thử fixture không đo độ chính xác mô hình.

Benchmark trên CT thực tế còn cần CT và nhãn đốt/ROI được chuyên gia duyệt.
Đánh giá theo bệnh nhân: độ chính xác gán mức, khoảng cách tâm (mm), Dice/IoU,
ca thất bại và thời gian/RAM; giữ test độc lập. Không đánh dấu hoàn thành
benchmark T011 chỉ dựa trên cài backend hoặc chạy ảnh tổng hợp.

Đã chạy phân đoạn trên CT thực tế bằng backend 2.18.0 trên Windows CPU.
Lượt kiểm tra ngày 25/10/2026 với CT 567 lát xuất 13 mức T6–S1, có cảnh báo
T6/T7 chạm biên ảnh. Kết quả đã mở lại trong demo và xác nhận khớp CT nguồn.
Chưa đo độ chính xác gán đốt bằng nhãn chuyên gia. Xem
[tiến độ và kiểm thử mới nhất](../../README.md#tiến-độ-dự-án).

## Monte Carlo tùy chọn

CLI hỗ trợ `--mc-samples N --mc-noise-hu 20 --mc-seed 42` để đo độ nhạy của
mask, tâm và thể tích với nhiễu HU. Xem [hướng dẫn code và kết quả Monte Carlo](monte_carlo.md).

## Đầu vào HU array của dự án

Có thể truyền `--input data/processed/case_053/ct_hu.npy`. Bộ đọc lấy
`metadata.json` cùng thư mục, kiểm tra shape/spacing/origin/direction và chỉ
đưa ảnh HU vào backend; không sử dụng reference_label hoặc annotation.
Xem [hướng dẫn tích hợp Windows](windows_integration.md).
