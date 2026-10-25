"""Inspect DICOM series geometry and pixel decoding for one input directory."""
import argparse
from tools.audit_dataset import read_header
import numpy as np
from pathlib import Path
from collections import defaultdict
import pydicom


def inspect_series(root: Path) -> None:
    if not root.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục: {root}")


    series = defaultdict(list)
    unreadable = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ("", ".dcm"):
            continue

        try:
            ds = read_header(path)
        except Exception:
            unreadable += 1
            continue

        # force=True có thể đọc cả file không phải DICOM:
        # chỉ giữ file có các trường nhận dạng ảnh cần thiết.
        required = (
            "SOPClassUID", "SOPInstanceUID",
            "SeriesInstanceUID", "Rows", "Columns"
        )
        if not all(hasattr(ds, field) for field in required):
            unreadable += 1
            continue
        ds.filename = str(path)
        series[str(ds.SeriesInstanceUID)].append(ds)

    print(f"Số nhóm series ảnh: {len(series)}")
    print(f"Số file chưa xác nhận là DICOM ảnh: {unreadable}")

    for number, images in enumerate(series.values(), start=1):
        ds = images[0]
        geometry_count = sum(
            all(hasattr(img, field) for field in (
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PixelSpacing",
            ))
            for img in images
        )
        frames = sum(int(getattr(img, "NumberOfFrames", 1)) for img in images)

        print(f"\nSeries {number}")
        print(f"  Modality: {getattr(ds, 'Modality', 'Không rõ')}")
        print(f"  Số file: {len(images)}")
        print(f"  Tổng số frame: {frames}")
        print(f"  Kích thước ảnh đầu: {ds.Rows} × {ds.Columns}")
        print(f"  File có thông tin hình học: {geometry_count}/{len(images)}")
        print(f"  PixelSpacing: {getattr(ds, 'PixelSpacing', 'Thiếu')}")
        print(f"  SliceThickness: {getattr(ds, 'SliceThickness', 'Thiếu')}")
        print(
            "  TransferSyntax:",
            getattr(ds.file_meta, "TransferSyntaxUID", "Thiếu")
        )

    for number, images in enumerate(series.values(), start=1):
        ds = images[0]

        print(f"\n--- Kiểm tra chi tiết series {number} ---")
        print("ImageType:", list(getattr(ds, "ImageType", [])))
        print("ConvolutionKernel:", getattr(ds, "ConvolutionKernel", "Thiếu"))
        print("Orientation:", getattr(ds, "ImageOrientationPatient", "Thiếu"))

        try:
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            normal = np.cross(orientation[:3], orientation[3:])

            positions = np.array([
                np.dot(
                    np.asarray(img.ImagePositionPatient, dtype=float),
                    normal,
                )
                for img in images
            ])

            order = np.argsort(positions)
            gaps = np.diff(positions[order])

            if len(gaps):
                print(
                    "Khoảng cách lát cắt min / median / max (mm):",
                    np.round(
                        [gaps.min(), np.median(gaps), gaps.max()], 4
                    ).tolist(),
                )
                print(
                    "Số khoảng cách gần 0:",
                    int(np.count_nonzero(np.abs(gaps) < 0.001)),
                )

            middle = images[int(order[len(order) // 2])]
            full = pydicom.dcmread(middle.filename, force=True)
            pixels = full.pixel_array

            print("Đọc pixel thành công:", pixels.shape, pixels.dtype)
            print(
                "Có thông số chuyển sang HU:",
                hasattr(full, "RescaleSlope")
                and hasattr(full, "RescaleIntercept"),
            )

        except Exception as exc:
            print("Kiểm tra chưa thành công:", type(exc).__name__, str(exc))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    inspect_series(args.input)


if __name__ == '__main__':
    main()
