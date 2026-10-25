"""Monte Carlo input-perturbation stability, not Bayesian model confidence."""
from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from spinal_ct.localization import LEVELS


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


def aggregate_samples(image: sitk.Image, paths: list[Path], output: Path) -> dict:
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
        for i, path in enumerate(paths):
            sample = sitk.ReadImage(str(path))
            if (sample.GetSize() != image.GetSize()
                    or sample.GetNumberOfComponentsPerPixel() != 1
                    or not np.allclose(sample.GetSpacing(), image.GetSpacing(), atol=1e-5)
                    or not np.allclose(sample.GetOrigin(), image.GetOrigin(), atol=1e-4)
                    or not np.allclose(sample.GetDirection(), image.GetDirection(), atol=1e-5)):
                raise ValueError('Monte Carlo sample geometry mismatch.')
            values = sitk.GetArrayFromImage(sample)
            if not np.isin(values, np.arange(len(LEVELS)+1)).all():
                raise ValueError('Invalid vertebral label in Monte Carlo sample.')
            mapped = np.memmap(Path(tmp)/f'{i}.bin', dtype=np.uint8, mode='w+', shape=shape)
            resources.callback(mapped._mmap.close)
            mapped[:] = values
            samples.append(mapped)
        n = len(samples)
        consensus = np.zeros(shape, dtype=np.uint8)
        max_votes = np.zeros(shape, dtype=np.uint8)
        stats = []
        mask_dir = output/'consensus_masks'
        mask_dir.mkdir()
        for label_id in range(len(LEVELS)+1):
            counts = np.zeros(shape, dtype=np.uint8)
            volumes, centroids = [], []
            for sample in samples:
                region = sample == label_id
                counts += region
                if label_id:
                    volume = int(region.sum())
                    volumes.append(volume * float(np.prod(image.GetSpacing())))
                    if volume:
                        xyz = np.argwhere(region)[:, ::-1].mean(axis=0)
                        centroids.append(image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, xyz))))
            np.maximum(max_votes, counts, out=max_votes)
            if not label_id:
                continue
            level = LEVELS[label_id-1]
            selected = counts > n/2
            consensus[selected] = label_id
            mask = sitk.GetImageFromArray(selected.astype(np.uint8))
            mask.CopyInformation(image)
            sitk.WriteImage(mask, str(mask_dir/f'vertebrae_{level}.nii.gz'))
            stats.append(dict(level=level, detected_samples=len(centroids),
                detection_frequency=len(centroids)/n,
                volume_mean_mm3=float(np.mean(volumes)),
                volume_std_mm3=float(np.std(volumes, ddof=1)),
                centroid_mean_lps_mm=np.mean(centroids, axis=0).tolist() if centroids else None,
                centroid_std_lps_mm=np.std(centroids, axis=0, ddof=1).tolist() if len(centroids)>1 else None))
        # 1 - modal frequency measures disagreement even if labels alternate
        # between two vertebrae. A no-majority voxel is left as background.
        disagreement = (1 - max_votes.astype(np.float32)/n)
        for name, array in [('consensus.nii.gz', consensus), ('disagreement.nii.gz', disagreement)]:
            result = sitk.GetImageFromArray(array)
            result.CopyInformation(image)
            sitk.WriteImage(result, str(output/name))
        ambiguous = int(np.count_nonzero(max_votes <= n/2))
        # ExitStack closes all mappings before directory cleanup, including on failure.
    return dict(samples=n, per_level=stats, no_majority_voxels=ambiguous,
                consensus_rule='strict majority > N/2, otherwise background',
                disagreement_definition='1 - highest label frequency, including background',
                interpretation='Sensitivity to synthetic iid Gaussian HU noise; not calibrated confidence or a physical CT noise simulation.')
