import json
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RectangleSelector, Slider

PROJECT_DIR = Path(__file__).resolve().parent
CASE_DIR = PROJECT_DIR / "data" / "processed" / "case_053"
OUTPUT = CASE_DIR / "annotation_sagittal_draft.json"

volume = np.load(
    CASE_DIR / "ct_hu.npy", mmap_mode="r", allow_pickle=False
)
metadata = json.loads(
    (CASE_DIR / "metadata.json").read_text(encoding="utf-8")
)

if volume.ndim != 3:
    raise SystemExit("Thể tích phải có 3 chiều [z, y, x].")

if list(volume.shape) != metadata["shape_zyx"]:
    raise SystemExit("Kích thước thể tích không khớp metadata.")

spacing = np.asarray(metadata["spacing_zyx_mm"], dtype=float)
if (
    spacing.shape != (3,)
    or not np.all(np.isfinite(spacing))
    or np.any(spacing <= 0)
):
    raise SystemExit("Kích thước voxel không hợp lệ.")

nz, ny, nx = volume.shape
dz, dy, dx = metadata["spacing_zyx_mm"]

state = {"x": nx // 2, "box": None}

fig, ax = plt.subplots(figsize=(8, 9))
fig.subplots_adjust(bottom=0.20)

image = ax.imshow(
    volume[:, :, state["x"]],
    origin="lower",
    aspect=dz / dy,
    cmap="gray",
    vmin=-500,
    vmax=1300,
    interpolation="nearest",
)

ax.set_xlabel("y: A → P")
ax.set_ylabel("z: I → S")
ax.set_title("Chọn mặt cắt, rồi kéo khung quanh đốt sống tham chiếu L1")

status = fig.text(0.12, 0.03, "Chưa có vùng được chọn.", fontsize=10)

def select_region(start, end):
    if any(value is None for value in (
        start.xdata, start.ydata, end.xdata, end.ydata
    )):
        return

    # Tọa độ tâm voxel; khoảng chỉ số lưu theo [min, max_exclusive).
    y0 = max(0, int(np.floor(min(start.xdata, end.xdata))))
    y1 = min(ny, int(np.ceil(max(start.xdata, end.xdata))) + 1)
    z0 = max(0, int(np.floor(min(start.ydata, end.ydata))))
    z1 = min(nz, int(np.ceil(max(start.ydata, end.ydata))) + 1)

    if y1 <= y0 or z1 <= z0:
        return

    state["box"] = {
        "y_min": y0,
        "y_max_exclusive": y1,
        "z_min": z0,
        "z_max_exclusive": z1,
    }

    print("Đã chọn vùng:", state["box"], flush=True)
    status.set_text("Bấm Lưu nhãn nháp.")
    fig.canvas.draw_idle()

selector = RectangleSelector(
    ax,
    select_region,
    button=[1],
    minspanx=3,
    minspany=3,
    spancoords="data",
    interactive=True,
)

slider = Slider(
    fig.add_axes([0.18, 0.12, 0.65, 0.025]),
    "Sagittal x",
    0,
    nx - 1,
    valinit=state["x"],
    valstep=1,
)

def change_slice(value):
    state["x"] = int(value)
    state["box"] = None
    selector.clear()
    image.set_data(volume[:, :, state["x"]])
    status.set_text("Đổi mặt cắt: cần chọn lại vùng.")
    fig.canvas.draw_idle()

slider.on_changed(change_slice)

button = Button(
    fig.add_axes([0.32, 0.065, 0.36, 0.035]),
    "Lưu nhãn nháp",
)

def save_annotation(_):
    if state["box"] is None:
        print("Chưa có vùng chọn. Hãy kéo khung trên ảnh trước.", flush=True)
        status.set_text("Hãy kéo khung chọn trước khi lưu.")
        fig.canvas.draw_idle()
        return

    annotation = {
        "case_id": metadata["case_id"],
        "volume_file": "ct_hu.npy",
        "volume_shape_zyx": list(volume.shape),
        "plane": "sagittal",
        "slice_x_index": state["x"],
        "bbox_yz_voxel": state["box"],
        "index_convention": "zero_based_min_inclusive_max_exclusive",
        "target": "vertebral_body",
        "reference_label": metadata["reference_label"],
        "review_status": "draft_unverified",
    }

    # Lưu nguyên tử để tránh file JSON bị ghi dở.
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)

    status.set_text("Đã lưu nhãn nháp; cần xác nhận vị trí L1.")
    print("Đã lưu:", OUTPUT.resolve(), flush=True)
    fig.canvas.draw_idle()

button.on_clicked(save_annotation)
plt.show()