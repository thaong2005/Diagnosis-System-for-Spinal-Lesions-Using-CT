from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.widgets import Slider

ROOT = Path("data/raw/A1/huynh triet")
KERNEL = "BONEPLUS"

if not ROOT.is_dir():
    raise SystemExit(f"Không tìm thấy thư mục: {ROOT}")

# Tìm các series CT axial có kernel BONEPLUS.
groups = {}

for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in ("", ".dcm"):
        continue

    try:
        ds = pydicom.dcmread(
            path, stop_before_pixels=True, force=True
        )

        if (
            getattr(ds, "Modality", "") != "CT"
            or "LOCALIZER" in getattr(ds, "ImageType", [])
            or str(getattr(ds, "ConvolutionKernel", "")) != KERNEL
        ):
            continue

        required = (
            "SeriesInstanceUID", "ImagePositionPatient",
            "ImageOrientationPatient", "PixelSpacing",
            "Rows", "Columns",
        )
        if not all(hasattr(ds, name) for name in required):
            continue

        uid = str(ds.SeriesInstanceUID)
        groups.setdefault(uid, []).append((path, ds))

    except Exception:
        continue

if len(groups) != 1:
    raise SystemExit(
        f"Tìm thấy {len(groups)} series {KERNEL}. "
        "Cần đúng một series để chọn tự động."
    )

items = next(iter(groups.values()))
first = items[0][1]

# Chỉ hỗ trợ hướng ảnh đã kiểm tra ở ca mẫu.
expected_orientation = np.array([1, 0, 0, 0, 1, 0], dtype=float)
orientation = np.asarray(first.ImageOrientationPatient, dtype=float)

if not np.allclose(orientation, expected_orientation, atol=1e-5):
    raise SystemExit("Hướng ảnh khác ca mẫu; cần tái định hướng trước.")

spacing = np.asarray(first.PixelSpacing, dtype=float)
normal = np.cross(orientation[:3], orientation[3:])

items.sort(
    key=lambda item: np.dot(
        np.asarray(item[1].ImagePositionPatient, dtype=float),
        normal,
    )
)

positions = np.array([
    np.asarray(ds.ImagePositionPatient, dtype=float)
    for _, ds in items
])
slice_positions = positions @ normal
gaps = np.diff(slice_positions)

if len(gaps) == 0 or np.any(gaps <= 0):
    raise SystemExit("Không đủ lát cắt hoặc có vị trí trùng.")

dz = float(np.median(gaps))

if not np.allclose(gaps, dz, atol=0.01):
    raise SystemExit("Khoảng cách lát cắt không đều; cần resample.")

if not np.allclose(positions[:, :2], positions[0, :2], atol=0.01):
    raise SystemExit("Có dịch chuyển trong mặt phẳng; cần xử lý hình học.")

for _, ds in items:
    if (
        (int(ds.Rows), int(ds.Columns))
        != (int(first.Rows), int(first.Columns))
        or not np.allclose(
            np.asarray(ds.PixelSpacing, dtype=float),
            spacing,
            atol=1e-5,
        )
        or not np.allclose(
            np.asarray(ds.ImageOrientationPatient, dtype=float),
            orientation,
            atol=1e-5,
        )
    ):
        raise SystemExit("Hình học không đồng nhất trong series.")

# Cấp phát một lần để hạn chế bộ nhớ.
volume = np.empty(
    (len(items), int(first.Rows), int(first.Columns)),
    dtype=np.float32,
)

for index, (path, _) in enumerate(items):
    ds = pydicom.dcmread(path, force=True)

    if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
        raise SystemExit(f"Lát {index} thiếu thông số chuyển sang HU.")

    pixels = ds.pixel_array
    if pixels.shape != volume.shape[1:]:
        raise SystemExit(f"Lát {index} có kích thước pixel không phù hợp.")

    volume[index] = pixels
    volume[index] *= float(ds.RescaleSlope)
    volume[index] += float(ds.RescaleIntercept)

dy, dx = map(float, spacing)
nz, ny, nx = volume.shape

import json

output_dir = Path("data/processed/case_053")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "ct_hu.npy", volume)

metadata = {
    "case_id": "case_053",
    "kernel": KERNEL,
    "shape_zyx": list(volume.shape),
    "spacing_zyx_mm": [dz, dy, dx],
    "origin_lps_mm": positions[0].tolist(),
    "image_orientation_patient": orientation.tolist(),
    "array_axes": ["slice", "row", "column"],
    "reference_label": {
        "vertebral_level": "L1",
        "fracture_class": "A1",
        "source": "provided_clinical_label",
    },
}

with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

print("Đã lưu thể tích và metadata tại:", output_dir)

print("Kernel:", KERNEL)
print("Thể tích [z, y, x]:", volume.shape)
print("Voxel [z, y, x] (mm):", (dz, dy, dx))
print("Bộ nhớ thể tích (MiB):", round(volume.nbytes / 1024**2, 1))

# Cửa sổ xương ban đầu; có thể điều chỉnh sau.
window_center = 400
window_width = 1800
vmin = window_center - window_width / 2
vmax = window_center + window_width / 2

fig, axes = plt.subplots(1, 3, figsize=(14, 7))
fig.subplots_adjust(bottom=0.27, wspace=0.3)

common = dict(cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

axial = axes[0].imshow(
    volume[nz // 2],
    origin="upper",
    aspect=dy / dx,
    **common,
)
sagittal = axes[1].imshow(
    volume[:, :, nx // 2],
    origin="lower",
    aspect=dz / dy,
    **common,
)
coronal = axes[2].imshow(
    volume[:, ny // 2, :],
    origin="lower",
    aspect=dz / dx,
    **common,
)

axes[0].set_title("Axial")
axes[0].set_xlabel("R → L")
axes[0].set_ylabel("P ← A")

axes[1].set_title("Sagittal")
axes[1].set_xlabel("A → P")
axes[1].set_ylabel("I → S")

axes[2].set_title("Coronal")
axes[2].set_xlabel("R → L")
axes[2].set_ylabel("I → S")

slider_z = Slider(
    fig.add_axes([0.20, 0.16, 0.60, 0.025]),
    "Axial z", 0, nz - 1, valinit=nz // 2, valstep=1,
)
slider_x = Slider(
    fig.add_axes([0.20, 0.11, 0.60, 0.025]),
    "Sagittal x", 0, nx - 1, valinit=nx // 2, valstep=1,
)
slider_y = Slider(
    fig.add_axes([0.20, 0.06, 0.60, 0.025]),
    "Coronal y", 0, ny - 1, valinit=ny // 2, valstep=1,
)

def update(_):
    axial.set_data(volume[int(slider_z.val)])
    sagittal.set_data(volume[:, :, int(slider_x.val)])
    coronal.set_data(volume[:, int(slider_y.val), :])
    fig.canvas.draw_idle()

for slider in (slider_z, slider_x, slider_y):
    slider.on_changed(update)

fig.suptitle("CT BONEPLUS — trình xem, chưa phát hiện tổn thương")
plt.show()