# Kiểm chứng tối ưu định vị CT — 26/10/2026

Branch: `codex/vertebra-labels-boundaries`.

CT: 512 × 512 × 567 voxel. TotalSegmentator 2.18.0, CPU, mô hình 3 mm,
baseline + 2 mẫu Monte Carlo, sigma 20 HU, seed 42. Máy i9-9980HK, RAM 32 GB.

## Thời gian quan sát

| Phép đo | Trước tối ưu | Sau tối ưu |
|---|---:|---:|
| Toàn pipeline, cùng CT/cấu hình | 1477.25 s | 248.34 s |
| Tổng hợp lại đúng hai mẫu MC đã lưu | 48.005 s | 8.376 s |

Lượt toàn pipeline mới nhanh hơn khoảng 5.95 lần so với
lượt cũ. Hai lượt không kiểm soát cùng tải hệ thống, nên đây là thời gian quan sát,
không phải cam kết tốc độ hoặc phép đo riêng tác động của mã nguồn.
Replay tổng hợp chạy tuần tự trên cùng dữ liệu; không xóa cache hệ điều hành.
Bản mới không xuất 25 mask nhị phân tùy chọn khi tổng hợp.

## Kết quả đối chiếu

- Baseline, hai mẫu MC, consensus và disagreement: 0 voxel khác nhau,
  mỗi volume gồm 148.635.648 voxel.
- Tên đốt, hộp bao voxel, cảnh báo và trạng thái giữ nguyên.
- ROI vật lý, thể tích, tâm và thống kê MC khớp trong sai số số học
  (`rtol=1e-10`, `atol=1e-8`). Phép đo luôn dùng geometry CT gốc.
- Kết quả mới mở được bằng bộ đọc của demo, xác nhận fingerprint CT khớp.
- Bộ kiểm thử: 48 đạt, 1 bỏ qua trên Windows vì fixture dùng script Unix.

## Thay đổi đã áp dụng

1. Backend xuất một label map mỗi lượt; xác thực và đổi mã nhãn về quy ước dự án.
2. CT và label map tạm không nén; kết quả lưu lâu dài vẫn là `.nii.gz`.
3. Đếm/thống kê label một lượt, bỏ phiếu theo khối; công thức MC giữ nguyên.
4. Ghi thời gian từng giai đoạn vào `timings_seconds` và tab Tiến trình.

Kết quả mới dùng báo cáo schema 2: `mask_encoding=label_map`, `mask` trỏ tới
file chung và `label_id` chỉ đốt cần lấy. Kết quả cũ vẫn xem được.
CLI có `--export-binary-masks` khi cần thêm mask nhị phân riêng.

## Thời gian từng bước của lượt mới

| Giai đoạn | Giây |
|---|---:|
| `load_ct` | 7.12 |
| `fingerprint` | 2.56 |
| `backend_version` | 4.46 |
| `baseline_prepare` | 0.95 |
| `baseline_inference` | 67.69 |
| `baseline_postprocess` | 3.45 |
| `baseline_preview` | 3.16 |
| `mc_001_prepare` | 5.65 |
| `mc_001_inference` | 63.37 |
| `mc_001_postprocess` | 4.29 |
| `mc_002_prepare` | 5.72 |
| `mc_002_inference` | 64.85 |
| `mc_002_postprocess` | 3.65 |
| `mc_aggregation` | 5.28 |
| `consensus_preview` | 6.05 |
| `cleanup` | 0.07 |
| `other_overhead` | 0.02 |

Kết quả chạy mới nằm trong `data/demo_runs/perf_ml_20261026_719d7f`.
Để dùng bản tối ưu: mở lại `start_demo.bat`, bật Chạy nhanh 3 mm, chọn 2 mẫu MC.
Không cần cài thêm thư viện. Kết quả được lưu trong thư mục mới; dữ liệu cũ giữ nguyên.

## Mốc kiểm tra trước đó — 25/10/2026

- Bộ kiểm thử khi đó: 35 đạt, 1 bỏ qua trên Windows do fixture Unix.
- Kiểm tra trợ giúp của 10 CLI và dependency của hai môi trường Python.
- Baseline trên CT 567 lát hoàn tất khoảng 7 phút 29 giây, xuất 13 mức T6–S1;
  có cảnh báo T6/T7 chạm biên. Đây là một lượt quan sát trước tối ưu.
- Đã mở lại kết quả, xác nhận khớp CT và xuất HTML. Khi đó mới kiểm tra
  baseline trên ca này; các lượt MC trong báo cáo chính được thực hiện sau.
- Đã giới hạn backend ghi mask tuần tự để tránh thiếu RAM trên Windows.
