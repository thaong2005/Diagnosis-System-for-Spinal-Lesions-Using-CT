"""Read-only axial, sagittal and coronal viewer for DICOM or prepared CT."""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--series-uid')
    args = parser.parse_args()
    image = sitk.DICOMOrient(load_ct(args.input, args.series_uid), 'LPS')
    if not np.allclose(np.asarray(image.GetDirection()).reshape(3, 3), np.eye(3), atol=1e-5):
        raise SystemExit('Oblique CT requires resampling before orthogonal viewing.')
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    volume = sitk.GetArrayFromImage(image)
    nz, ny, nx = volume.shape
    dx, dy, dz = image.GetSpacing()
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.subplots_adjust(bottom=.27, wspace=.3)
    common = dict(cmap='gray', vmin=-500, vmax=1300, interpolation='nearest')
    axial = axes[0].imshow(volume[nz//2], origin='upper', aspect=dy/dx, **common)
    sagittal = axes[1].imshow(volume[:, :, nx//2], origin='lower', aspect=dz/dy, **common)
    coronal = axes[2].imshow(volume[:, ny//2, :], origin='lower', aspect=dz/dx, **common)
    for ax, title, xlabel, ylabel in zip(axes, ['Axial', 'Sagittal', 'Coronal'],
                                        ['R → L', 'A → P', 'R → L'],
                                        ['P ← A', 'I → S', 'I → S']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    sliders = [Slider(fig.add_axes([.20, y, .60, .025]), name, 0, size-1,
                      valinit=size//2, valstep=1)
               for y, name, size in ((.16, 'Axial z', nz), (.11, 'Sagittal x', nx),
                                     (.06, 'Coronal y', ny))]

    def update(_):
        axial.set_data(volume[int(sliders[0].val)])
        sagittal.set_data(volume[:, :, int(sliders[1].val)])
        coronal.set_data(volume[:, int(sliders[2].val), :])
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    fig.suptitle('CT MPR — viewer only')
    plt.show()


if __name__ == '__main__':
    main()
