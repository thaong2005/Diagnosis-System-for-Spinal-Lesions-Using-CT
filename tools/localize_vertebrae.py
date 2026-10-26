"""Localize vertebrae from CT using TotalSegmentator; no clinical labels as input."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import SimpleITK as sitk
from spinal_ct.monte_carlo import perturb_hu, aggregate_samples
from spinal_ct.ct_io import load_ct
from spinal_ct.localization import (LEVELS, load_backend_labels, summarize_labels,
                                    export_binary_masks, render_preview)
from spinal_ct.review import ct_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='DICOM directory, HU NIfTI, or ct_hu.npy with adjacent metadata.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrites existing data')
    parser.add_argument('--mc-samples', type=int, default=0, help='0 disables MC; otherwise 2–64 perturbed inferences in addition to baseline')
    parser.add_argument('--mc-noise-hu', type=float, default=20.0, help='Experimental iid Gaussian HU noise sigma; requires validation')
    parser.add_argument('--mc-seed', type=int, default=42, help='Nonnegative seed for input perturbations only')
    parser.add_argument('--backend-log', type=Path, help='Optional new file for diagnostics from sanitized CT inference')
    parser.add_argument('--series-uid', help='Explicit CT series selection for multi-series DICOM')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'mps'], default='cpu')
    parser.add_argument('--model-cache', type=Path, default=Path('.tmp/totalsegmentator'), help='Persistent model weights directory')
    parser.add_argument('--fast', action='store_true', help='Use lower-resolution 3 mm model')
    parser.add_argument('--robust-crop', action='store_true',
                        help='Use the backend 3 mm crop locator for the 1.5 mm ROI model (slower)')
    parser.add_argument('--export-binary-masks', action='store_true',
                        help='Also export 25 binary masks for baseline/consensus (slower; shared label maps are always saved)')
    parser.add_argument('--backend', default='TotalSegmentator', help='Path to TotalSegmentator executable')
    parser.add_argument('--progress-json', action='store_true', help='Emit stage events for the desktop UI')
    parser.add_argument('--cancel-file', type=Path, help='Stop after the current backend pass if this file exists')
    args = parser.parse_args()
    started = time.monotonic()
    timings = {}

    def stage(message, **fields):
        if args.progress_json:
            print('PROGRESS ' + json.dumps(dict(message=message, **fields), ensure_ascii=True), flush=True)

    @contextmanager
    def timed(key, message):
        stage(message, stage=key)
        begin = time.monotonic()
        try:
            yield
        finally:
            seconds = round(time.monotonic()-begin, 6)
            timings[key] = seconds
            stage(f'{message} · {seconds:.1f} giây', stage=key, duration_seconds=seconds)

    def check_cancelled():
        if args.cancel_file and args.cancel_file.exists():
            raise ValueError('Run cancelled after current stage; no results published.')
    if args.mc_samples != 0 and not 2 <= args.mc_samples <= 64:
        parser.error('--mc-samples must be 0 or between 2 and 64.')
    if args.mc_seed < 0 or not np.isfinite(args.mc_noise_hu) or args.mc_noise_hu <= 0:
        parser.error('MC seed must be nonnegative and noise sigma positive and finite.')
    if args.backend_log and args.backend_log.exists():
        parser.error('Backend log already exists; choose a new file.')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory.')
    backend = shutil.which(args.backend)
    if backend is None:
        parser.error('TotalSegmentator is unavailable; install requirements-localization.txt or pass --backend.')
    try:
        check_cancelled()
        with timed('load_ct', 'Đọc CT và kiểm tra hình học'):
            image = load_ct(args.input, args.series_uid)
        with timed('fingerprint', 'Tính dấu vân tay CT'):
            fingerprint = ct_fingerprint(image)
        with timed('backend_version', 'Kiểm tra phiên bản backend'):
            version = subprocess.run([backend, '--version'], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        if version != '2.18.0':
            raise ValueError('This adapter requires TotalSegmentator 2.18.0.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.localization-', dir=args.output.parent) as tmp:
            root = Path(tmp)
            # Temporary float HU volumes, particularly noisy ones, are expensive to
            # gzip. The backend receives exactly the same values in uncompressed NIfTI.
            ct_path = root / 'ct.nii'
            raw_labels = root/'backend-labels.nii'
            with timed('baseline_prepare', 'Ghi CT tạm không nén'):
                sitk.WriteImage(image, str(ct_path))
            command = [backend, '-i', str(ct_path), '-o', str(raw_labels), '--ml',
                       # Parallel saving duplicates full volumes in Windows workers.
                       # Serial saving avoids excessive RAM use on large CT stacks.
                       '--nr_thr_saving', '1',
                       '--task', 'total', '--device', args.device, '--roi_subset',
                       *[f'vertebrae_{level}' for level in LEVELS]]
            if args.fast:
                command.append('--fast')
            if args.robust_crop:
                command.append('--robust_crop')
            env = os.environ.copy()
            config_dir = root/'backend-config'
            config_dir.mkdir()
            (config_dir/'config.json').write_text(json.dumps(dict(
                totalseg_id='local-spinal-ct', send_usage_stats=False, prediction_counter=0)))
            args.model_cache.mkdir(parents=True, exist_ok=True)
            env['TOTALSEG_HOME_DIR'] = str(config_dir.resolve())
            env['TOTALSEG_WEIGHTS_PATH'] = str(args.model_cache.resolve())
            # Only sanitized CT and temporary paths reach the backend. Suppress logs
            # in normal output; a failing backend does not publish partial results.
            check_cancelled()
            with timed('baseline_inference', 'Định vị baseline · lần đầu có thể cần tải trọng số'):
                process = subprocess.run(command, env=env, capture_output=True, text=True)
            if args.backend_log:
                with args.backend_log.open('x', encoding='utf-8') as log:
                    log.write(process.stdout + '\n' + process.stderr)
            if process.returncode:
                raise RuntimeError(f'TotalSegmentator failed (exit {process.returncode}); check model installation, weights/network access and memory.')
            check_cancelled()
            with timed('baseline_postprocess', 'Tạo ROI và lưu label map baseline'):
                labels = load_backend_labels(image, raw_labels)
                raw_labels.unlink()
                report = summarize_labels(image, labels)
                sitk.WriteImage(labels, str(root/'vertebrae.nii.gz'))
                if args.export_binary_masks:
                    export_binary_masks(labels, root/'masks')
            report['model'] = dict(backend='TotalSegmentator', version=version, task='total',
                                   resolution='3mm' if args.fast else '1.5mm', device=args.device,
                                   robust_crop=args.robust_crop,
                                   saving_threads=1, output_mode='multilabel')
            report['storage'] = dict(prediction_masks='label_map', binary_masks=args.export_binary_masks,
                                     temporary_ct_compressed=False)
            with timed('baseline_preview', 'Dựng ảnh baseline'):
                render_preview(image, labels, root/'levels_mpr.png')
            del labels
            if args.mc_samples:
                rng = np.random.default_rng(args.mc_seed)
                samples_dir = root/'mc_samples'
                samples_dir.mkdir()
                paths, run_warnings = [], []
                for index in range(args.mc_samples):
                    check_cancelled()
                    stage(f'Monte Carlo: mẫu {index+1}/{args.mc_samples}')
                    print(f'Monte Carlo sample {index+1}/{args.mc_samples}', flush=True)
                    prefix = f'mc_{index+1:03d}'
                    with timed(prefix+'_prepare', f'Tạo CT nhiễu mẫu {index+1}'):
                        sitk.WriteImage(perturb_hu(image, rng, args.mc_noise_hu), str(ct_path))
                    with timed(prefix+'_inference', f'Phân đoạn mẫu {index+1}'):
                        process = subprocess.run(command, env=env, capture_output=True, text=True)
                    if args.backend_log:
                        with args.backend_log.open('a', encoding='utf-8') as log:
                            log.write(f'\nMC sample {index+1}\n' + process.stdout + '\n' + process.stderr)
                    if process.returncode:
                        raise RuntimeError(f'TotalSegmentator failed (exit {process.returncode}) in MC sample {index+1}.')
                    check_cancelled()
                    with timed(prefix+'_postprocess', f'Kiểm tra và lưu label map mẫu {index+1}'):
                        sample_labels = load_backend_labels(image, raw_labels)
                        raw_labels.unlink()
                        sample_report = summarize_labels(image, sample_labels)
                        run_warnings.append(sample_report['warnings'])
                        path = samples_dir/f'{index:03d}.nii.gz'
                        sitk.WriteImage(sample_labels, str(path))
                        paths.append(path)
                        del sample_labels
                with timed('mc_aggregation', 'Tổng hợp đồng thuận và bản đồ bất đồng'):
                    mc = aggregate_samples(image, paths, root/'monte_carlo',
                                           binary_masks=args.export_binary_masks, check_cancelled=check_cancelled)
                consensus_report = mc['consensus']
                with timed('consensus_preview', 'Dựng ảnh đồng thuận'):
                    render_preview(image, sitk.ReadImage(str(root/'monte_carlo'/'consensus.nii.gz')),
                                   root/'monte_carlo'/'consensus_mpr.png',
                                   disagreement=sitk.ReadImage(str(root/'monte_carlo'/'disagreement.nii.gz')))
                mc.update(method='monte_carlo_input_perturbation', seed=args.mc_seed,
                          noise_sigma_hu=args.mc_noise_hu, run_warnings=run_warnings,
                          consensus=consensus_report,
                          consensus_mask='monte_carlo/consensus.nii.gz',
                          disagreement_map='monte_carlo/disagreement.nii.gz')
                report['monte_carlo'] = mc
                # Baseline remains intact; instability is explicitly surfaced.
                if any(run_warnings) or consensus_report['status'] != 'ok' or mc['no_majority_voxels']:
                    report['status'] = 'indeterminate'
                    report['warnings'].append('Monte Carlo quality warnings; review consensus and sample results.')
            check_cancelled()
            stage('Lưu kết quả')
            # CT is temporary; persist only masks, preview and non-identifying report.
            with timed('cleanup', 'Dọn CT tạm'):
                ct_path.unlink()
                shutil.rmtree(config_dir)
            elapsed = time.monotonic()-started
            timings['other_overhead'] = round(max(0., elapsed-sum(timings.values())), 6)
            report.update(input_fingerprint=fingerprint, timings_seconds=timings,
                          runtime_seconds=round(elapsed, 2),
                          created_at_utc=datetime.now(timezone.utc).isoformat())
            (root/'prediction.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
            if args.output.exists():
                raise ValueError('Output appeared during processing; refusing to overwrite.')
            root.rename(args.output)
        print(f"Localization complete: {len(report['predictions'])} vertebrae; status={report['status']}; review required.")
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError, IndexError) as exc:
        # Do not echo source paths, DICOM identifiers or backend diagnostics.
        if isinstance(exc, (subprocess.CalledProcessError, IndexError)):
            message = 'Could not determine the backend version.'
        elif isinstance(exc, OSError):
            message = 'Filesystem/backend error; check input access and available storage.'
        elif isinstance(exc, RuntimeError) and not str(exc).startswith('TotalSegmentator failed'):
            message = 'Image processing failed; check CT encoding and geometry.'
        else:
            message = str(exc)
        print(f'Localization failed: {message}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
