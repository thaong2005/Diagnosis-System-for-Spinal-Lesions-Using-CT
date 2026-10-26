# Cấu trúc và quyết định tổ chức dự án

Cập nhật: 25/10/2026.

## Các phương án

| Phương án | Ưu điểm | Chi phí / thời điểm phù hợp |
|---|---|---|
| Giữ script tại gốc | Ít thay đổi lệnh chạy | Khó phân biệt giao diện, chuyển dữ liệu và mô hình; dễ lặp bộ đọc CT |
| `apps/` + `tools/` + `spinal_ct/` | Phân trách nhiệm rõ, chạy bằng `python -m`, ít cấu hình | Phù hợp quy mô nghiên cứu hiện tại; đây là cấu trúc đang áp dụng |
| `src/spinal_ct/` + đóng gói `pyproject.toml` | Cài package và kiểm tra import độc lập | Hợp lý khi phân phối, CI hoặc nhiều máy; cần quy trình cài editable và entry point |

Chưa tách thành nhiều dịch vụ hay repository: hiện chỉ có một pipeline cục bộ
và một backend. Việc tách đó sẽ tăng cấu hình mà chưa có lợi ích rõ.

## Ranh giới chức năng

- `ct_io.py`: đọc CT và kiểm tra HU/geometry cho viewer, chuẩn bị CT và inference.
- `localization.py`: chuyển mask thành ROI/báo cáo và dựng preview.
- `monte_carlo.py`: nhiễu HU và tổng hợp mẫu; chưa phải thuật toán ước lượng diện tích.
- `overlays.py`: dùng chung cách vẽ contour và tên đốt cho demo/ảnh QC; chỉ gắn
  tên khi mask có trên lát đang xem, giữ nguyên dữ liệu phân đoạn và phép đo.
- `apps/`: nhận đường dẫn từ CLI, chạy giao diện; không tự ghi volume khi mở viewer.
- `tools/`: điều phối tác vụ; inference không nhận nhãn tham chiếu.
- Annotation có quy ước ROI riêng, không tự gộp với mask toàn đốt sống.

## Demo dành cho người kiểm tra

`start_demo.bat` mở `apps/demo.py`. Cửa sổ PySide6 chạy đọc dữ liệu ở QThread
và gọi CLI suy luận bằng QProcess, để giao diện tiếp tục phản hồi. CLI phát
sự kiện từng giai đoạn; ứng dụng hiển thị thời gian thực thay vì đoán phần trăm.
Tham khảo API chính thức: [QProcess](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html).

`spinal_ct/review.py` kiểm tra geometry và dấu vân tay HU trước khi ghép CT với
kết quả mới, resample riêng lưới hiển thị nếu ảnh xiên và xuất báo cáo HTML.
Mask gốc và phép đo không thay đổi. Kết quả cũ thiếu dấu vân tay chỉ có thể
kiểm tra geometry và cần người dùng xác nhận ca. Các lần chạy demo nằm trong
`data/demo_runs/`; dừng sau lượt suy luận hiện tại không công bố output dở dang.

## Thay đổi và phần đã loại bỏ

| Đường dẫn cũ | Thay thế / lý do |
|---|---|
| `view_ct.py` | Bỏ bộ đọc DICOM và chuyển HU trùng; tách thành `tools.prepare_ct` và `apps.view_mpr` |
| `view_ct_enhanced.py` | `apps/view_sagittal.py`; dùng bộ đọc chung, bỏ hard-code ca |
| `annotate_ct.py` | `apps/annotate_ct.py`; nhận `--case`, `--output` |
| `inspect_dicom.py` | `tools/inspect_dicom.py`; nhận `--input`, dùng chung đọc header với audit |
| `doc/` | Hướng dẫn trong `docs/guides/`; snapshot cũ trong `docs/archive/2026-10-19/` |
| `doc/note.txt` | Hai dòng ý tưởng đã được bao quát trong README và tài liệu nghiên cứu |
| `doc/Report/` | Thư mục rỗng |
| Gói ZIP, `spinal_ct_mc_integration/`, hướng dẫn ở gốc | Đã xóa ở bước trước sau khi xác nhận trùng từng tệp |

Không giữ wrapper cho script gốc: dùng các lệnh `python -m` trong README.

## Những phần trông giống nhau nhưng vẫn cần

- `view_mpr` xem ba mặt phẳng; `view_sagittal` có điều chỉnh HU, nội suy và xuất ảnh.
- `inspect_dicom` kiểm tra geometry/pixel từng series; `audit_dataset` tổng hợp
  dataset và đối chiếu nhãn. Hai công cụ phục vụ hai tác vụ khác nhau.
- `.venv` và `.venv-model` tách dependency ứng dụng khỏi backend.
- Baseline, mẫu MC, consensus và disagreement là các đầu ra khác nhau.
- Tài liệu FS, audit và checkpoint cũ có bằng chứng nghiên cứu riêng nên được
  lưu trong `archive/`, kèm nhãn lịch sử; không xóa chỉ dựa vào ngày sửa.
- `Expected_output/` chứa ảnh tham chiếu cục bộ; chưa có bằng chứng trùng nên giữ.

Dữ liệu, cache trọng số, môi trường Python và cấu hình cá nhân được giữ lại.
`__pycache__` có thể tái tạo và đã được `.gitignore` loại trừ.
