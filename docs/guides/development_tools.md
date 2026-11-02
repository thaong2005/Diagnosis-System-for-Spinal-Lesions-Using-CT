# Công cụ phát triển

Luồng vận hành chính dùng `start_demo.bat`, mở `apps.demo` và cửa sổ đo thân
đốt. Các công cụ dưới đây phục vụ thao tác riêng hoặc chuẩn bị dữ liệu.

| Lệnh `python -m ...` | Mục đích |
|---|---|
| `apps.view_mpr` | Viewer CT ba mặt phẳng độc lập |
| `apps.view_sagittal` | Xem sagittal, điều chỉnh HU/nội suy và xuất ảnh |
| `apps.annotate_ct` | Annotation ROI 3D nháp; khác mask thân đốt 2D của demo |
| `tools.inspect_dicom` | Kiểm tra series, HU và geometry |
| `tools.audit_dataset` | Tổng hợp dataset, đối chiếu nhãn phát triển |
| `tools.prepare_ct` | Chuẩn bị HU array và metadata hình học |
| `tools.validate_annotation` | Kiểm tra schema annotation 3D |
| `tools.localize_vertebrae` | Suy luận TotalSegmentator và MC nhiễu HU tùy chọn |
| `tools.render_localization` | Dựng lại preview từ kết quả phân đoạn đã lưu |
| `tools.analyze_vertebral_body` | Tính lại đặc trưng/diện tích từ vùng đã xác nhận |
| `tools.evaluate_body_blobs` | Đối chiếu ứng viên LoG với tâm vùng tham chiếu |

Thay `python` bằng `.\.venv\Scripts\python.exe` khi chạy PowerShell ở gốc
dự án. Mỗi lệnh có `--help`. Các viewer không tự ghi đè CT.

Xem [Windows](windows_integration.md), [định vị](vertebral_localization.md),
[đo thân đốt](body_morphometry.md) và [MC nhiễu HU](monte_carlo.md).
