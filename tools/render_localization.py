"""Rebuild QC previews from saved masks without rerunning the model."""
from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from spinal_ct.ct_io import load_ct
from spinal_ct.localization import render_preview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original CT input')
    parser.add_argument('--results', type=Path, required=True, help='Completed localization directory')
    parser.add_argument('--output', type=Path, required=True, help='New preview directory')
    parser.add_argument('--series-uid')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new preview directory.')
    image = load_ct(args.input, args.series_uid)
    labels = sitk.ReadImage(str(args.results/'vertebrae.nii.gz'))
    args.output.mkdir(parents=True)
    render_preview(image, labels, args.output/'levels_mpr.png')
    mc = args.results/'monte_carlo'
    if (mc/'consensus.nii.gz').is_file():
        render_preview(image, sitk.ReadImage(str(mc/'consensus.nii.gz')),
                       args.output/'consensus_mpr.png',
                       disagreement=sitk.ReadImage(str(mc/'disagreement.nii.gz')))
    print(f'Previews saved to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
