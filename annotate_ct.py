"""Annotate a vertebral-body ROI on synchronized CT orthogonal views.

This is a development-only tool.  It consumes a prepared HU volume and its
geometry metadata, then writes a draft 3D annotation.  The annotation is never
an input to production inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector, Slider

from spinal_ct.labels import parse_vertebral_levels


PROJECT_DIR = Path(__file__).resolve().parent
CASE_DIR = PROJECT_DIR / "data" / "processed" / "case_053"
OUTPUT = CASE_DIR / "annotation_3d_draft.json"


def configure_console_encoding() -> None:
    """Keep Vietnamese status messages readable in a Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_case(case_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate the prepared HU volume and metadata."""
    volume = np.load(case_dir / "ct_hu.npy", mmap_mode="r", allow_pickle=False)
    metadata = json.loads(
        (case_dir / "metadata.json").read_text(encoding="utf-8")
    )

    if volume.ndim != 3:
        raise ValueError("Thể tích phải có 3 chiều [z, y, x].")
    if list(volume.shape) != metadata.get("shape_zyx"):
        raise ValueError("Kích thước thể tích không khớp metadata.")

    spacing = np.asarray(metadata.get("spacing_zyx_mm"), dtype=float)
    if (
        spacing.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise ValueError("Kích thước voxel không hợp lệ.")

    origin = np.asarray(metadata.get("origin_lps_mm"), dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("Tọa độ gốc LPS không hợp lệ.")

    direction = direction_from_metadata(metadata)
    if not np.all(np.isfinite(direction)):
        raise ValueError("Direction LPS không hợp lệ.")

    return volume, metadata


def direction_from_metadata(metadata: dict[str, object]) -> np.ndarray:
    """Return a 3x3 LPS direction matrix from stored DICOM orientation."""
    if "direction_lps" in metadata:
        direction = np.asarray(metadata["direction_lps"], dtype=float)
        if direction.shape == (9,):
            direction = direction.reshape(3, 3)
        if direction.shape != (3, 3):
            raise ValueError("direction_lps phải có 9 phần tử.")
        return direction

    orientation = np.asarray(
        metadata.get("image_orientation_patient"), dtype=float
    )
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
    return np.column_stack((axis_x, axis_y, axis_z))


def physical_bbox_from_voxel(
    bbox_zyx: dict[str, int],
    spacing_zyx: np.ndarray,
    origin_lps: np.ndarray,
    direction_lps: np.ndarray,
) -> dict[str, float]:
    """Convert a half-open voxel box to an axis-aligned LPS bounding box."""
    z0 = bbox_zyx["z_min"]
    z1 = bbox_zyx["z_max_exclusive"]
    y0 = bbox_zyx["y_min"]
    y1 = bbox_zyx["y_max_exclusive"]
    x0 = bbox_zyx["x_min"]
    x1 = bbox_zyx["x_max_exclusive"]
    dz, dy, dx = spacing_zyx

    points = []
    for z in (z0, z1):
        for y in (y0, y1):
            for x in (x0, x1):
                xyz_mm = np.asarray([x * dx, y * dy, z * dz], dtype=float)
                points.append(origin_lps + direction_lps @ xyz_mm)
    points_array = np.asarray(points)
    minimum = points_array.min(axis=0)
    maximum = points_array.max(axis=0)
    return {
        "x_min": float(minimum[0]),
        "x_max": float(maximum[0]),
        "y_min": float(minimum[1]),
        "y_max": float(maximum[1]),
        "z_min": float(minimum[2]),
        "z_max": float(maximum[2]),
    }


def make_annotation(
    volume: np.ndarray,
    metadata: dict[str, object],
    bbox_zyx: dict[str, int],
    direction_lps: np.ndarray,
) -> dict[str, object]:
    """Build a schema-v1 draft annotation without patient identifiers."""
    reference = metadata.get("reference_label", {})
    reference_level = reference.get("vertebral_level") if isinstance(reference, dict) else None
    if not isinstance(reference_level, str):
        raise ValueError("metadata thiếu reference_label.vertebral_level.")

    levels = list(parse_vertebral_levels(reference_level))
    spacing = np.asarray(metadata["spacing_zyx_mm"], dtype=float)
    origin = np.asarray(metadata["origin_lps_mm"], dtype=float)

    return {
        "schema_version": 1,
        "case_id": metadata["case_id"],
        "volume_file": "ct_hu.npy",
        "vertebral_levels": levels,
        "target": "vertebral_body",
        "bbox_zyx_voxel": bbox_zyx,
        "bbox_lps_mm": physical_bbox_from_voxel(
            bbox_zyx,
            spacing,
            origin,
            direction_lps,
        ),
        "geometry": {
            "shape_zyx": list(volume.shape),
            "spacing_zyx_mm": [float(value) for value in spacing],
            "origin_lps_mm": [float(value) for value in origin],
            "direction_lps": [
                float(value) for value in direction_lps.reshape(-1)
            ],
        },
        "index_convention": "zero_based_min_inclusive_max_exclusive",
        "reference_label": reference,
        "review_status": "draft",
    }


def _clamped_bounds(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Convert a rectangle selection to half-open integer bounds."""
    if any(value is None for value in (first_x, first_y, second_x, second_y)):
        return None
    x0 = max(0, int(np.floor(min(first_x, second_x))))
    x1 = min(width, int(np.ceil(max(first_x, second_x))) + 1)
    y0 = max(0, int(np.floor(min(first_y, second_y))))
    y1 = min(height, int(np.ceil(max(first_y, second_y))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, x1, y0, y1


def run_annotation(case_dir: Path = CASE_DIR, output: Path = OUTPUT) -> None:
    """Open the synchronized three-plane annotation window."""
    volume, metadata = load_case(case_dir)
    direction_lps = direction_from_metadata(metadata)
    nz, ny, nx = volume.shape
    dz, dy, dx = (float(value) for value in metadata["spacing_zyx_mm"])

    state: dict[str, object] = {
        "z": nz // 2,
        "y": ny // 2,
        "x": nx // 2,
        "bounds": None,
        "selected": {"axial": False, "sagittal": False, "coronal": False},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(bottom=0.25, wspace=0.18)
    axial_ax, sagittal_ax, coronal_ax = axes

    images = {
        "axial": axial_ax.imshow(
            volume[int(state["z"]), :, :],
            origin="lower",
            aspect=dy / dx,
            cmap="gray",
            vmin=-500,
            vmax=1300,
            interpolation="nearest",
        ),
        "sagittal": sagittal_ax.imshow(
            volume[:, :, int(state["x"])],
            origin="lower",
            aspect=dz / dy,
            cmap="gray",
            vmin=-500,
            vmax=1300,
            interpolation="nearest",
        ),
        "coronal": coronal_ax.imshow(
            volume[:, int(state["y"]), :],
            origin="lower",
            aspect=dz / dx,
            cmap="gray",
            vmin=-500,
            vmax=1300,
            interpolation="nearest",
        ),
    }

    axial_ax.set_title("Axial — z")
    axial_ax.set_xlabel("x: R → L")
    axial_ax.set_ylabel("y: A → P")
    sagittal_ax.set_title("Sagittal — x")
    sagittal_ax.set_xlabel("y: A → P")
    sagittal_ax.set_ylabel("z: I → S")
    coronal_ax.set_title("Coronal — y")
    coronal_ax.set_xlabel("x: R → L")
    coronal_ax.set_ylabel("z: I → S")

    overlays = {
        "axial": Rectangle((0, 0), 1, 1, fill=False, color="lime", linewidth=2),
        "sagittal": Rectangle((0, 0), 1, 1, fill=False, color="lime", linewidth=2),
        "coronal": Rectangle((0, 0), 1, 1, fill=False, color="lime", linewidth=2),
    }
    for name, patch in overlays.items():
        axes[{"axial": 0, "sagittal": 1, "coronal": 2}[name]].add_patch(patch)
        patch.set_visible(False)

    selectors: dict[str, RectangleSelector] = {}

    def update_overlays() -> None:
        bounds = state["bounds"]
        if not isinstance(bounds, dict):
            for patch in overlays.values():
                patch.set_visible(False)
            return

        z0, z1 = bounds["z_min"], bounds["z_max_exclusive"]
        y0, y1 = bounds["y_min"], bounds["y_max_exclusive"]
        x0, x1 = bounds["x_min"], bounds["x_max_exclusive"]
        current_z, current_y, current_x = state["z"], state["y"], state["x"]

        axial_patch = overlays["axial"]
        axial_patch.set_bounds(x0, y0, x1 - x0, y1 - y0)
        axial_patch.set_visible(z0 <= current_z < z1)

        sagittal_patch = overlays["sagittal"]
        sagittal_patch.set_bounds(y0, z0, y1 - y0, z1 - z0)
        sagittal_patch.set_visible(x0 <= current_x < x1)

        coronal_patch = overlays["coronal"]
        coronal_patch.set_bounds(x0, z0, x1 - x0, z1 - z0)
        coronal_patch.set_visible(y0 <= current_y < y1)

    def update_status(message: str | None = None) -> None:
        selected = state["selected"]
        selected_text = ", ".join(
            name for name, value in selected.items() if value
        ) or "chưa có"
        suffix = f" {message}" if message else ""
        status.set_text(f"Đã chọn: {selected_text}.{suffix}")
        fig.canvas.draw_idle()

    def update_images() -> None:
        images["axial"].set_data(volume[int(state["z"]), :, :])
        images["sagittal"].set_data(volume[:, :, int(state["x"])])
        images["coronal"].set_data(volume[:, int(state["y"]), :])
        update_overlays()
        fig.canvas.draw_idle()

    def select_axial(start, end) -> None:
        result = _clamped_bounds(start.xdata, start.ydata, end.xdata, end.ydata, nx, ny)
        if result is None:
            return
        x0, x1, y0, y1 = result
        bounds = state["bounds"] or {}
        bounds.update({"x_min": x0, "x_max_exclusive": x1, "y_min": y0, "y_max_exclusive": y1})
        state["bounds"] = bounds
        state["selected"]["axial"] = True
        update_overlays()
        update_status("Axial đã ghi nhận.")

    def select_sagittal(start, end) -> None:
        result = _clamped_bounds(start.xdata, start.ydata, end.xdata, end.ydata, ny, nz)
        if result is None:
            return
        y0, y1, z0, z1 = result
        bounds = state["bounds"] or {}
        bounds.update({"y_min": y0, "y_max_exclusive": y1, "z_min": z0, "z_max_exclusive": z1})
        state["bounds"] = bounds
        state["selected"]["sagittal"] = True
        update_overlays()
        update_status("Sagittal đã ghi nhận.")

    def select_coronal(start, end) -> None:
        result = _clamped_bounds(start.xdata, start.ydata, end.xdata, end.ydata, nx, nz)
        if result is None:
            return
        x0, x1, z0, z1 = result
        bounds = state["bounds"] or {}
        bounds.update({"x_min": x0, "x_max_exclusive": x1, "z_min": z0, "z_max_exclusive": z1})
        state["bounds"] = bounds
        state["selected"]["coronal"] = True
        update_overlays()
        update_status("Coronal đã ghi nhận.")

    selectors["axial"] = RectangleSelector(
        axial_ax, select_axial, button=[1], minspanx=3, minspany=3,
        spancoords="data", interactive=True,
    )
    selectors["sagittal"] = RectangleSelector(
        sagittal_ax, select_sagittal, button=[1], minspanx=3, minspany=3,
        spancoords="data", interactive=True,
    )
    selectors["coronal"] = RectangleSelector(
        coronal_ax, select_coronal, button=[1], minspanx=3, minspany=3,
        spancoords="data", interactive=True,
    )

    def change_z(value) -> None:
        state["z"] = int(value)
        update_images()

    def change_y(value) -> None:
        state["y"] = int(value)
        update_images()

    def change_x(value) -> None:
        state["x"] = int(value)
        update_images()

    sliders = {
        "z": Slider(
            fig.add_axes([0.12, 0.16, 0.76, 0.025]),
            "Axial z",
            0,
            nz - 1,
            valinit=state["z"],
            valstep=1,
        ),
        "y": Slider(
            fig.add_axes([0.12, 0.12, 0.76, 0.025]),
            "Coronal y",
            0,
            ny - 1,
            valinit=state["y"],
            valstep=1,
        ),
        "x": Slider(
            fig.add_axes([0.12, 0.08, 0.76, 0.025]),
            "Sagittal x",
            0,
            nx - 1,
            valinit=state["x"],
            valstep=1,
        ),
    }
    sliders["z"].on_changed(change_z)
    sliders["y"].on_changed(change_y)
    sliders["x"].on_changed(change_x)

    status = fig.text(0.08, 0.035, "Đã chọn: chưa có.", fontsize=10)
    save_button = Button(fig.add_axes([0.38, 0.005, 0.24, 0.035]), "Lưu annotation 3D")

    def save_annotation(_event) -> None:
        selected = state["selected"]
        if not all(selected.values()):
            update_status("Cần chọn ROI trên cả ba mặt phẳng.")
            return
        bounds = state["bounds"]
        if not isinstance(bounds, dict):
            update_status("Chưa có ROI.")
            return

        annotation = make_annotation(volume, metadata, bounds, direction_lps)
        temporary = output.with_suffix(".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        update_status(f"Đã lưu nháp: {output.name}.")
        print("Đã lưu:", output.resolve(), flush=True)

    save_button.on_clicked(save_annotation)
    update_overlays()
    plt.show()


def main() -> None:
    configure_console_encoding()
    try:
        run_annotation()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Không thể mở annotation: {error}") from error


if __name__ == "__main__":
    main()
