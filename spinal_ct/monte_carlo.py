"""Monte Carlo input-perturbation stability, not Bayesian model confidence."""
from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from spinal_ct.localization import (LEVELS, validate_label_image, label_statistics,
                                    summarize_labels, export_binary_masks)

VOTE_CHUNK_VOXELS = 262_144


def perturb_hu(image: sitk.Image, rng: np.random.Generator, sigma_hu: float) -> sitk.Image:
    if not np.isfinite(sigma_hu) or sigma_hu <= 0:
        raise ValueError('Noise standard deviation must be positive and finite.')
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    # Slice-wise draws avoid a second float64 volume allocation.
    for plane in data:
        plane += rng.normal(0, sigma_hu, plane.shape).astype(np.float32)
    result = sitk.GetImageFromArray(data)
    result.CopyInformation(image)
    return result


def aggregate_samples(image: sitk.Image, paths: list[Path], output: Path,
                      *, binary_masks: bool = False, check_cancelled=None) -> dict:
    """Strict majority consensus, including background, on the unchanged CT grid.

    Label arrays are loaded one at a time and stored as uint8 memory maps to
    bound RAM. Missing detections contribute zero volume; centroids are
    conditional on detection. Dispersion is not a clinical confidence interval.
    """
    if not 2 <= len(paths) <= 64:
        raise ValueError('Monte Carlo requires 2–64 samples.')
    output.mkdir()
    with tempfile.TemporaryDirectory(dir=output, prefix='.votes-') as tmp, ExitStack() as resources:
        shape = tuple(reversed(image.GetSize()))
        samples = []
        volumes = np.zeros((len(paths), len(LEVELS)), dtype=np.float64)
        centroids = [[] for _ in LEVELS]
        for i, path in enumerate(paths):
            if check_cancelled:
                check_cancelled()
            sample = sitk.ReadImage(str(path))
            validate_label_image(image, sample)
            # Count and find centres once per sample, instead of 25 scans + argwhere.
            measures = label_statistics(sample, image)
            voxel_volume = float(np.prod(image.GetSpacing()))
            for label_id in measures.GetLabels():
                volumes[i, label_id-1] = measures.GetNumberOfPixels(label_id)*voxel_volume
                centroids[label_id-1].append(measures.GetCentroid(label_id))
            mapped = np.memmap(Path(tmp)/f'{i}.bin', dtype=np.uint8, mode='w+', shape=shape)
            resources.callback(mapped._mmap.close)
            mapped[:] = sitk.GetArrayViewFromImage(sample)
            samples.append(mapped.reshape(-1))
            del sample
        n = len(samples)
        consensus = np.empty(shape, dtype=np.uint8)
        disagreement = np.empty(shape, dtype=np.float32)
        flat_consensus, flat_disagreement = consensus.reshape(-1), disagreement.reshape(-1)
        ambiguous = 0
        # Keep votes cache-sized, independent of full CT size and number of samples.
        for start in range(0, consensus.size, VOTE_CHUNK_VOXELS):
            if check_cancelled:
                check_cancelled()
            end = min(start+VOTE_CHUNK_VOXELS, consensus.size)
            if n == 2:
                # Strict majority of two is exactly agreement, including background.
                a, b = samples[0][start:end], samples[1][start:end]
                agreed = a == b
                flat_consensus[start:end] = np.where(agreed, a, 0)
                flat_disagreement[start:end] = np.where(agreed, 0., .5)
                ambiguous += int(np.count_nonzero(~agreed))
            else:
                votes = np.zeros((end-start, len(LEVELS)+1), dtype=np.uint8)
                positions = np.arange(end-start)
                for sample in samples:
                    # Unique positions mean fancy indexed += has no repeated indices.
                    votes[positions, sample[start:end]] += 1
                winner = votes.argmax(axis=1)
                maximum = votes[positions, winner]
                majority = maximum > n/2
                flat_consensus[start:end] = np.where(majority, winner, 0)
                flat_disagreement[start:end] = 1-maximum.astype(np.float32)/n
                ambiguous += int(np.count_nonzero(~majority))
        stats = []
        for index, level in enumerate(LEVELS):
            centres = centroids[index]
            stats.append(dict(level=level, detected_samples=len(centres),
                detection_frequency=len(centres)/n,
                volume_mean_mm3=float(np.mean(volumes[:, index])),
                volume_std_mm3=float(np.std(volumes[:, index], ddof=1)),
                centroid_mean_lps_mm=np.mean(centres, axis=0).tolist() if centres else None,
                centroid_std_lps_mm=np.std(centres, axis=0, ddof=1).tolist() if len(centres)>1 else None))
        for name, array in [('consensus.nii.gz', consensus), ('disagreement.nii.gz', disagreement)]:
            result = sitk.GetImageFromArray(array)
            result.CopyInformation(image)
            sitk.WriteImage(result, str(output/name))
            if name == 'consensus.nii.gz':
                consensus_report = summarize_labels(image, result, 'monte_carlo/consensus.nii.gz')
                if binary_masks:
                    export_binary_masks(result, output/'consensus_masks')
        # ExitStack closes all mappings before directory cleanup, including on failure.
    return dict(samples=n, per_level=stats, no_majority_voxels=ambiguous,
                consensus=consensus_report,
                consensus_rule='strict majority > N/2, otherwise background',
                disagreement_definition='1 - highest label frequency, including background',
                interpretation='Sensitivity to synthetic iid Gaussian HU noise; not calibrated confidence or a physical CT noise simulation.')
