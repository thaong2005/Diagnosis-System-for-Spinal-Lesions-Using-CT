"""Recompute body shape/area from a saved manual region and its original CT."""
import argparse
from pathlib import Path
from spinal_ct.ct_io import load_ct
from spinal_ct.morphometry import load_session, save_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original DICOM, HU NIfTI or prepared ct_hu.npy')
    parser.add_argument('--series-uid')
    parser.add_argument('--region', type=Path, required=True, help='Folder containing region.json and body_mask.npy')
    parser.add_argument('--output', type=Path, required=True, help='New folder; never overwrite existing results')
    parser.add_argument('--samples', type=int, default=100_000, help='100-10,000,000 uniform area sample points')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 100 <= args.samples <= 10_000_000 or not 0 <= args.seed <= 2**63-1:
        parser.error('Invalid area MC samples or seed.')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory.')
    try:
        session = load_session(args.region, load_ct(args.input, args.series_uid))
        if not session.reviewed:
            raise ValueError('Region is a draft; confirm it in the body review tool first.')
        save_session(session, args.output, samples=args.samples, seed=args.seed)
        print(f'Body measurements saved: {args.output}')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Body measurement failed: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
