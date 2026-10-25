"""Convert CT into a HU array and geometry metadata, without clinical labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct


def prepare_ct(source: Path, output: Path, series_uid: str | None = None) -> None:
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    image = load_ct(source, series_uid)
    metadata = dict(shape_zyx=list(reversed(image.GetSize())),
                    spacing_zyx_mm=list(reversed(image.GetSpacing())),
                    origin_lps_mm=list(image.GetOrigin()),
                    direction_lps=list(image.GetDirection()),
                    array_axes=['slice', 'row', 'column'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as temporary:
        stage = Path(temporary)
        np.save(stage/'ct_hu.npy', sitk.GetArrayFromImage(image), allow_pickle=False)
        (stage/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        if output.exists():
            raise ValueError('Output appeared during processing; refusing to overwrite.')
        stage.rename(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--series-uid')
    args = parser.parse_args()
    try:
        prepare_ct(args.input, args.output, args.series_uid)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'CT preparation failed: {error}\n')
    print('Prepared CT saved. No reference labels copied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
