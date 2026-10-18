import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button, RadioButtons, Slider


PROJECT_DIR = Path(__file__).resolve().parent
CASE_DIR = PROJECT_DIR / "data" / "processed" / "case_053"
EXPORT_DIR = CASE_DIR / "exports"


def configure_console_encoding():
    """Tránh lỗi khi terminal Windows không mặc định dùng UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_ct_image(volume, metadata):
    """Dựng ảnh SimpleITK 3D cùng hình học vật lý DICOM (hệ tọa độ LPS)."""
    if volume.ndim != 3:
        raise ValueError("Thể tích phải có ba chiều [z, y, x].")

    if list(volume.shape) != metadata.get("shape_zyx"):
        raise ValueError("Kích thước thể tích không khớp metadata.")

    spacing_zyx = np.asarray(metadata.get("spacing_zyx_mm"), dtype=float)
    origin_lps = np.asarray(metadata.get("origin_lps_mm"), dtype=float)
    orientation = np.asarray(
        metadata.get("image_orientation_patient"), dtype=float
    )

    if (
        spacing_zyx.shape != (3,)
        or not np.all(np.isfinite(spacing_zyx))
        or np.any(spacing_zyx <= 0)
    ):
        raise ValueError("Kích thước voxel không hợp lệ.")

    if origin_lps.shape != (3,) or not np.all(np.isfinite(origin_lps)):
        raise ValueError("Tọa độ gốc LPS không hợp lệ.")

    if orientation.shape != (6,) or not np.all(np.isfinite(orientation)):
        raise ValueError("ImageOrientationPatient không hợp lệ.")

    axis_x = orientation[:3]
    axis_y = orientation[3:]
    if (
        not np.isclose(np.linalg.norm(axis_x), 1.0, atol=1e-5)
        or not np.isclose(np.linalg.norm(axis_y), 1.0, atol=1e-5)
        or not np.isclose(np.dot(axis_x, axis_y), 0.0, atol=1e-5)
    ):
        raise ValueError("Các vector hướng DICOM không trực chuẩn.")

    axis_z = np.cross(axis_x, axis_y)
    direction = np.column_stack((axis_x, axis_y, axis_z))

    image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    image.SetSpacing(tuple(float(value) for value in spacing_zyx[::-1]))
    image.SetOrigin(tuple(float(value) for value in origin_lps))
    image.SetDirection(tuple(float(value) for value in direction.ravel()))

    # Đưa các ca có trục bị hoán vị/lật về LPS mà không đổi vị trí vật lý
    # của voxel.
    image = sitk.DICOMOrient(image, "LPS")
    if not np.allclose(
        np.asarray(image.GetDirection()).reshape(3, 3),
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError(
            "CT có mặt phẳng chụp xiên; cần resample sang lưới LPS "
            "trước khi xem."
        )

    return image


def get_sagittal_array(image, x_index, target_spacing_mm=None):
    """Trích và tùy chọn resample lát sagittal sang pixel đẳng hướng."""
    nx, ny, nz = image.GetSize()
    if not 0 <= x_index < nx:
        raise IndexError(f"Chỉ số sagittal ngoài phạm vi: {x_index}")

    extractor = sitk.ExtractImageFilter()
    extractor.SetIndex([int(x_index), 0, 0])
    extractor.SetSize([0, ny, nz])
    sagittal = extractor.Execute(image)

    if target_spacing_mm is not None:
        target_spacing_mm = float(target_spacing_mm)
        if not np.isfinite(target_spacing_mm) or target_spacing_mm <= 0:
            raise ValueError("Spacing đích của sagittal không hợp lệ.")

        old_size = sagittal.GetSize()
        old_spacing = sagittal.GetSpacing()
        new_spacing = (target_spacing_mm, target_spacing_mm)
        new_size = [
            max(
                1,
                round(
                    (old_size[axis] - 1)
                    * old_spacing[axis]
                    / new_spacing[axis]
                )
                + 1,
            )
            for axis in range(2)
        ]

        sagittal = sitk.Resample(
            sagittal,
            new_size,
            sitk.Transform(2, sitk.sitkIdentity),
            sitk.sitkLinear,
            sagittal.GetOrigin(),
            new_spacing,
            sagittal.GetDirection(),
            -1024.0,
            sagittal.GetPixelID(),
        )

    return sitk.GetArrayFromImage(sagittal)


def main():
    configure_console_encoding()

    volume = np.load(
        CASE_DIR / "ct_hu.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = json.loads(
        (CASE_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    try:
        ct_image = build_ct_image(volume, metadata)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(
            f"Không thể dựng ảnh CT bằng SimpleITK: {error}"
        ) from error
    finally:
        del volume

    nx, ny, nz = ct_image.GetSize()
    dx, dy, dz = ct_image.GetSpacing()
    spacing = np.asarray([dz, dy, dx], dtype=float)
    sagittal_spacing_mm = min(dy, dz)
    kernel = metadata.get("kernel", "Unknown")

    initial_sagittal = get_sagittal_array(
        ct_image,
        nx // 2,
        target_spacing_mm=sagittal_spacing_mm,
    )
    slice_cache = {
        "index": nx // 2,
        "array": initial_sagittal,
    }

    def sagittal_for_index(x_index):
        if x_index != slice_cache["index"]:
            slice_cache["array"] = get_sagittal_array(
                ct_image,
                x_index,
                target_spacing_mm=sagittal_spacing_mm,
            )
            slice_cache["index"] = x_index
        return slice_cache["array"]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(
        left=0.10, right=0.80, bottom=0.23, top=0.91
    )
    ax.set_facecolor("black")

    image = ax.imshow(
        initial_sagittal,
        origin="lower",
        aspect=1.0,
        cmap="gray",
        vmin=-500,
        vmax=1300,
        interpolation="nearest",
    )
    ax.set_xlabel("y voxel: Anterior → Posterior")
    ax.set_ylabel("z voxel: Inferior → Superior")

    fig.suptitle(
        "Sagittal MPR — viewer only, no lesion prediction",
        fontsize=13,
    )
    status = fig.text(0.10, 0.025, "", fontsize=9)

    slice_slider = Slider(
        fig.add_axes([0.18, 0.16, 0.57, 0.025]),
        "Slice x", 0, nx - 1,
        valinit=nx // 2, valstep=1,
    )
    level_slider = Slider(
        fig.add_axes([0.18, 0.115, 0.57, 0.025]),
        "Level HU", -1000, 2000,
        valinit=400, valstep=1,
    )
    width_slider = Slider(
        fig.add_axes([0.18, 0.07, 0.57, 0.025]),
        "Width HU", 1, 4000,
        valinit=1800, valstep=1,
    )

    radio_ax = fig.add_axes([0.82, 0.61, 0.16, 0.13])
    radio_ax.set_title("Interpolation", fontsize=10)
    interpolation = RadioButtons(
        radio_ax, ("nearest", "bilinear"), active=0
    )

    bone_button = Button(
        fig.add_axes([0.83, 0.51, 0.14, 0.045]), "Bone"
    )
    soft_button = Button(
        fig.add_axes([0.83, 0.45, 0.14, 0.045]), "Soft tissue"
    )
    save_button = Button(
        fig.add_axes([0.83, 0.34, 0.14, 0.055]), "Export PNG"
    )

    def refresh(_=None):
        x = int(slice_slider.val)
        level = float(level_slider.val)
        width = float(width_slider.val)
        point_lps = ct_image.TransformIndexToPhysicalPoint((x, 0, 0))

        image.set_data(sagittal_for_index(x))
        image.set_clim(level - width / 2, level + width / 2)
        image.set_interpolation(interpolation.value_selected)

        ax.set_title(
            f"{kernel} | x={x} ({point_lps[0]:.1f} mm LPS) | "
            f"WL={level:.0f}, WW={width:.0f}"
        )
        fig.canvas.draw_idle()

    def set_window(level, width):
        level_slider.set_val(level)
        width_slider.set_val(width)

    def scroll_slice(event):
        if event.inaxes is not ax:
            return

        # Không thay mặt cắt khi đang dùng công cụ Pan/Zoom.
        toolbar = getattr(fig.canvas.manager, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return

        step = 1 if event.button == "up" else -1
        value = int(np.clip(slice_slider.val + step, 0, nx - 1))
        slice_slider.set_val(value)

    def export_png(_):
        x = int(slice_slider.val)
        level = float(level_slider.val)
        width = float(width_slider.val)
        method = interpolation.value_selected
        sagittal_array = sagittal_for_index(x)

        # Giữ tỷ lệ vật lý, không dùng tỷ lệ số voxel đơn thuần.
        height_px = 1800
        width_px = max(
            1, round(height_px * (ny * dy) / (nz * dz))
        )
        dpi = 150

        output = Figure(
            figsize=(width_px / dpi, height_px / dpi),
            dpi=dpi,
            facecolor="black",
        )
        FigureCanvasAgg(output)
        output_ax = output.add_axes([0, 0, 1, 1])
        output_ax.imshow(
            sagittal_array,
            origin="lower",
            aspect=1.0,
            cmap="gray",
            vmin=level - width / 2,
            vmax=level + width / 2,
            interpolation=method,
        )
        output_ax.set_axis_off()

        try:
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = EXPORT_DIR / f"sagittal_x{x:03d}_{stamp}.png"
            output.savefig(path, dpi=dpi, facecolor="black")

            settings = {
                "source_volume": "../ct_hu.npy",
                "plane": "sagittal",
                "slice_x_index": x,
                "window_level_hu": level,
                "window_width_hu": width,
                "interpolation": method,
                "spacing_zyx_mm": spacing.tolist(),
                "resampled_sagittal_spacing_zy_mm": [
                    sagittal_spacing_mm,
                    sagittal_spacing_mm,
                ],
                "resampled_sagittal_shape_zy": list(
                    sagittal_array.shape
                ),
                "resample_interpolator": "sitkLinear",
                "slice_origin_lps_mm": list(
                    ct_image.TransformIndexToPhysicalPoint((x, 0, 0))
                ),
                "direction_lps": list(ct_image.GetDirection()),
                "purpose": "display_only",
            }
            path.with_suffix(".json").write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8",
            )

            status.set_text(f"Saved: {path.name}")
            print("Đã xuất:", path.resolve(), flush=True)
        except OSError as error:
            status.set_text("Export failed — see terminal.")
            print("Không thể lưu:", error, flush=True)
        finally:
            output.clear()
            fig.canvas.draw_idle()

    for slider in (slice_slider, level_slider, width_slider):
        slider.on_changed(refresh)

    interpolation.on_clicked(refresh)
    bone_button.on_clicked(lambda _: set_window(400, 1800))
    soft_button.on_clicked(lambda _: set_window(40, 400))
    save_button.on_clicked(export_png)
    fig.canvas.mpl_connect("scroll_event", scroll_slice)

    print("Thể tích SimpleITK [x,y,z]:", ct_image.GetSize())
    print("Voxel [z,y,x] mm:", spacing.tolist())
    print("Sagittal resampled [z,y]:", initial_sagittal.shape)
    print(
        "Sagittal spacing [z,y] mm:",
        [sagittal_spacing_mm, sagittal_spacing_mm],
    )
    print("Origin LPS mm:", list(ct_image.GetOrigin()))
    print("Direction LPS:", list(ct_image.GetDirection()))
    print("CT gốc được mở ở chế độ chỉ đọc.")
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
