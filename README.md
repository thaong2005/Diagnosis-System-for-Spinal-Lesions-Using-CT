# Diagnosis System for Spinal Lesions Using CT

Deep Learning-Based Computer-Aided Diagnosis System for Spinal Lesions Using CT.

## Thiết lập môi trường

Project dùng Python 3.13 và một virtual environment duy nhất tại `.venv`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` là nguồn khai báo dependency duy nhất. Không commit thư mục
`.venv`, `.venv-*` hoặc dữ liệu bệnh nhân trong `data`.

## Chạy các công cụ hiện có

```powershell
python inspect_dicom.py
python view_ct.py
python annotate_ct.py
python view_ct_enhanced.py
```

Các script đọc dữ liệu từ `data/`. Hãy đặt dữ liệu đúng đường dẫn mà từng script
khai báo trước khi chạy. Trình xem nâng cao dựng hình học CT bằng `SimpleITK` và
hiển thị giao diện Qt thông qua `PySide6`.
