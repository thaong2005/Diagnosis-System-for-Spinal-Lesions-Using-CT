"""Measure local CT review stages without modifying images or saved regions."""
import argparse
import json
import os
from pathlib import Path
import time

import SimpleITK as sitk
from spinal_ct.ct_io import load_ct
from spinal_ct.review import ct_fingerprint, display_image, load_review
from spinal_ct.result_catalog import discover_results
from spinal_ct.morphometry import load_session
from spinal_ct.blob_detection import detect_blobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--series-uid')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--region', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ui', action='store_true', help='Also time complete MPR draws on an offscreen Qt canvas')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file.')
    result = dict(seconds={}, scope='One local run; warm caches and system load are not controlled.')

    def timed(name, function):
        started = time.perf_counter()
        value = function()
        result['seconds'][name] = round(time.perf_counter()-started, 6)
        print(f'{name}: {result["seconds"][name]:.3f} s', flush=True)
        return value

    source = timed('load_ct', lambda: load_ct(args.input, args.series_uid))
    fingerprint = timed('fingerprint', lambda: ct_fingerprint(source))
    reference = timed('display_grid', lambda: display_image(source))
    result['size_xyz'] = list(source.GetSize())
    timed('discover_results', lambda: discover_results([args.results.parent], fingerprint))
    review = timed('load_review', lambda: load_review(args.results, source, reference, fingerprint))
    if args.region:
        session = timed('load_region', lambda: load_session(args.region, source, fingerprint=fingerprint))
        if session.reviewed:
            for samples in (100_000, 10_000_000):
                timed(f'measure_{samples}', lambda n=samples: session.measure(samples=n, seed=42))
            result['pixel_area_mm2'] = session.result['features']['area_mm2']
        timed('log', lambda: detect_blobs(session.ct, session.plane.spacing_xy_mm, session.valid))
    if args.ui:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from unittest.mock import patch
        from apps.demo import DemoWindow
        app = QApplication.instance() or QApplication([])
        with patch.object(DemoWindow, 'verify_environment'):
            window = DemoWindow()
            window.study = dict(source=source, reference=reference, fingerprint=fingerprint,
                                data=sitk.GetArrayFromImage(reference))
            window.review = review
            window.mode.addItem('Baseline')
            window.mode.setCurrentText('Baseline')
            def navigate():
                for row in range(window.table.rowCount()):
                    window.table.selectRow(row)
                    window.canvas.draw()
            timed('navigate_all_vertebrae', navigate)
            result['vertebrae'] = window.table.rowCount()
            window.close()
            window.deleteLater()
            app.processEvents()
    # Only timing, dimensions and numeric measurements are exported; no source paths/identifiers.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
