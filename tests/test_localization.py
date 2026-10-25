import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct, list_ct_series, select_sample_series
from spinal_ct.localization import LEVELS, summarize_masks, render_preview


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = sitk.GetImageFromArray(np.zeros((12, 10, 8), np.int16))
        self.image.SetSpacing((0.7, 1.2, 2.0))
        self.image.SetOrigin((10, 20, 30))
        self.masks = self.root/'masks'
        self.masks.mkdir()
        for level in LEVELS:
            self.write_mask(level)

    def write_mask(self, level, slices=None):
        array = np.zeros((12, 10, 8), np.uint8)
        if slices:
            array[slices] = 1
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(self.image)
        sitk.WriteImage(mask, str(self.masks/f'vertebrae_{level}.nii.gz'))

    def test_physical_roi_uses_voxel_edges_and_direction(self):
        self.image.SetDirection((-1.,0.,0., 0.,-1.,0., 0.,0.,1.))
        for level in LEVELS:
            self.write_mask(level)
        self.write_mask('L1', (slice(3,6), slice(2,5), slice(1,4)))
        report, labels = summarize_masks(self.image, self.masks)
        item = report['predictions'][0]
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(item['bbox_voxel_xyz'], {'min':[1,2,3], 'max_exclusive':[4,5,6]})
        np.testing.assert_allclose(item['roi_lps_mm']['min'], [7.55,14.6,35])
        np.testing.assert_allclose(item['roi_lps_mm']['max'], [9.65,18.2,41])
        self.assertAlmostEqual(item['volume_mm3'], 27*0.7*1.2*2)
        self.assertEqual(labels.GetDirection(), self.image.GetDirection())
        render_preview(self.image, labels, self.root/'preview.png')
        self.assertGreater((self.root/'preview.png').stat().st_size, 1000)

    def test_preview_disagreement_preserves_masks_and_checks_geometry(self):
        self.write_mask('L1', (slice(0, 6), slice(0, 5), slice(1, 4)))
        _, labels = summarize_masks(self.image, self.masks)
        before = sitk.GetArrayFromImage(labels)
        disagreement = sitk.Cast(labels > 0, sitk.sitkFloat32) * 0.5
        render_preview(self.image, labels, self.root/'mc.png', disagreement=disagreement)
        np.testing.assert_array_equal(before, sitk.GetArrayFromImage(labels))
        self.assertGreater((self.root/'mc.png').stat().st_size, 1000)
        disagreement.SetOrigin((0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            render_preview(self.image, labels, self.root/'bad.png', disagreement=disagreement)

    def test_preview_empty_masks(self):
        _, labels = summarize_masks(self.image, self.masks)
        render_preview(self.image, labels, self.root/'empty.png')
        self.assertGreater((self.root/'empty.png').stat().st_size, 1000)

    def test_overlap_and_missing_outputs_rejected(self):
        box = (slice(3,6), slice(2,5), slice(1,4))
        self.write_mask('L1', box)
        self.write_mask('L2', box)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            summarize_masks(self.image, self.masks)
        self.write_mask('L2')
        (self.masks/'vertebrae_C1.nii.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            summarize_masks(self.image, self.masks)

    def test_empty_boundary_and_reversed_order_flagged(self):
        report, _ = summarize_masks(self.image, self.masks)
        self.assertEqual(report['status'], 'indeterminate')
        self.write_mask('L1', (slice(0,2), slice(2,5), slice(1,4)))
        self.write_mask('L2', (slice(5,8), slice(2,5), slice(1,4)))
        report, _ = summarize_masks(self.image, self.masks)
        self.assertTrue(report['predictions'][0]['touches_boundary'])
        self.assertTrue(any('order' in w for w in report['warnings']))

    def test_geometry_mismatch_rejected(self):
        path = self.masks/'vertebrae_L1.nii.gz'
        mask = sitk.ReadImage(str(path))
        mask.SetOrigin((0,0,0))
        sitk.WriteImage(mask, str(path))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            summarize_masks(self.image, self.masks)

    def write_dicom(self, uid, zs):
        for i, z in enumerate(zs):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            path = self.root/f'{uid}-{i}.dcm'
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = uid
            ds.StudyInstanceUID = uid
            ds.Modality = 'CT'
            ds.PatientName = 'Synthetic^Only'
            ds.Rows, ds.Columns = 4, 5
            ds.ImageOrientationPatient = [1,0,0,0,1,0]
            ds.ImagePositionPatient = [0,0,z]
            ds.PixelSpacing = [1,1]
            ds.RescaleSlope, ds.RescaleIntercept = 2, -1024
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = np.full((4,5), 100, np.uint16).tobytes()
            ds.save_as(path, enforce_file_format=True)

    def test_dicom_hu_and_series_selection(self):
        uid = generate_uid()
        self.write_dicom(uid, [4,0,2])
        image = load_ct(self.root)
        self.assertEqual(image.GetSize(), (5,4,3))
        self.assertEqual(image.GetSpacing(), (1,1,2))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), -824)
        self.assertFalse(image.GetMetaDataKeys())
        self.write_dicom(generate_uid(), [0,2])
        choices = list_ct_series(self.root)
        self.assertEqual(sorted(choice['slices'] for choice in choices), [2, 3])
        self.assertEqual(select_sample_series(choices)['uid'], uid)
        self.assertNotIn('Synthetic', str(choices))
        with self.assertRaisesRegex(ValueError, 'select one'):
            load_ct(self.root)
        self.assertEqual(load_ct(self.root, uid).GetSize(), (5,4,3))

    def test_irregular_dicom_rejected(self):
        self.write_dicom(generate_uid(), [0,2,5])
        self.assertIsNone(select_sample_series(list_ct_series(self.root)))
        with self.assertRaisesRegex(ValueError, 'irregular'):
            load_ct(self.root)

    def test_sample_series_ties_and_non_axial_require_manual_selection(self):
        self.assertIsNone(select_sample_series([]))
        first, second = generate_uid(), generate_uid()
        self.write_dicom(first, [0, 2, 4])
        self.write_dicom(second, [0, 2, 4])
        self.assertIsNone(select_sample_series(list_ct_series(self.root)))
        # A sagittal stack with more slices must not displace an axial candidate.
        self.write_dicom(second, [0, 2, 4, 6])
        for path in self.root.glob(f'{second}-*.dcm'):
            ds = pydicom.dcmread(path)
            ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
            ds.ImagePositionPatient = [float(ds.ImagePositionPatient[2]), 0, 0]
            ds.save_as(path, enforce_file_format=True)
        choices = list_ct_series(self.root)
        self.assertFalse(next(c for c in choices if c['uid'] == second)['axial'])
        self.assertEqual(select_sample_series(choices)['uid'], first)

    @unittest.skipIf(sys.platform == "win32", "Fixture backend uses a Unix shebang; validate real backend CLI separately on Windows")
    def test_cli_pipeline_with_fixture_backend(self):
        # This verifies orchestration and serialization, NOT model accuracy.
        source = self.root/'ct.nii.gz'
        sitk.WriteImage(self.image, str(source))
        backend = self.root/'fixture_backend'
        backend.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
import SimpleITK as s
if '--version' in sys.argv:
    print('2.18.0'); sys.exit(0)
image=s.ReadImage(sys.argv[sys.argv.index('-i')+1])
out=Path(sys.argv[sys.argv.index('-o')+1]); out.mkdir()
for level in {LEVELS!r}:
    mask=s.Image(image.GetSize(), s.sitkUInt8); mask.CopyInformation(image)
    if level == 'L1':
        mask[2:5, 2:6, 3:7] = 1
    s.WriteImage(mask,str(out/f'vertebrae_{{level}}.nii.gz'))
''')
        backend.chmod(0o755)
        output = self.root/'result'
        command = [sys.executable, '-m','tools.localize_vertebrae', '--input',str(source),
                   '--output',str(output),'--backend',str(backend), '--model-cache',str(self.root/'cache')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = json.loads((output/'prediction.json').read_text())
        self.assertEqual(report['predictions'][0]['level'], 'L1')
        self.assertTrue((output/'levels_mpr.png').is_file())
        self.assertFalse((output/'ct.nii.gz').exists())
        self.assertFalse((output/'backend-config').exists())
        again = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(json.loads((output/'prediction.json').read_text()), report)
        prepared = self.root/'ct_hu.npy'
        np.save(prepared, sitk.GetArrayFromImage(self.image))
        metadata = dict(shape_zyx=list(reversed(self.image.GetSize())),
                        spacing_zyx_mm=list(reversed(self.image.GetSpacing())),
                        origin_lps_mm=list(self.image.GetOrigin()),
                        direction_lps=list(self.image.GetDirection()),
                        reference_label={'vertebral_level':'L5'})
        (self.root/'metadata.json').write_text(json.dumps(metadata))
        prepared_output = self.root/'prepared-result'
        prepared_command = [str(prepared) if arg == str(source) else
                            str(prepared_output) if arg == str(output) else arg for arg in command]
        prepared_result = subprocess.run(prepared_command, capture_output=True, text=True)
        self.assertEqual(prepared_result.returncode, 0, prepared_result.stdout+prepared_result.stderr)
        prediction = json.loads((prepared_output/'prediction.json').read_text())
        self.assertEqual(prediction['predictions'][0]['level'], 'L1')
        self.assertNotIn('reference_label', prediction)
        mc_output = self.root/'mc-result'
        mc_command = [str(mc_output) if arg == str(output) else arg for arg in command]
        mc_command += ['--mc-samples', '3', '--mc-seed', '123']
        mc_result = subprocess.run(mc_command, capture_output=True, text=True)
        self.assertEqual(mc_result.returncode, 0, mc_result.stdout+mc_result.stderr)
        mc_report = json.loads((mc_output/'prediction.json').read_text())['monte_carlo']
        self.assertEqual(mc_report['samples'], 3)
        row = next(p for p in mc_report['per_level'] if p['level'] == 'L1')
        self.assertEqual(row['detection_frequency'], 1)
        self.assertEqual(row['volume_std_mm3'], 0)
        self.assertTrue((mc_output/'monte_carlo/consensus_mpr.png').is_file())
        self.assertTrue((mc_output/mc_report['consensus']['predictions'][0]['mask']).is_file())
        self.assertEqual(len(list((mc_output/'mc_samples').glob('*.nii.gz'))), 3)
        self.assertFalse((mc_output/'ct.nii.gz').exists())
        # A failed stochastic sample must invalidate the entire run.
        backend.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n    print('2.18.0'); sys.exit(0)\nif 'sample-masks' in sys.argv[sys.argv.index('-o')+1]:\n    sys.exit(9)\n" +
                           "import shutil\nshutil.copytree(" + repr(str(output/'masks')) + ", sys.argv[sys.argv.index('-o')+1])\n")
        failed_mc = self.root/'failed-mc'
        cmd = [str(failed_mc) if arg == str(mc_output) else arg for arg in mc_command]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('MC sample 1', result.stdout)
        self.assertFalse(failed_mc.exists())
        self.assertFalse(list(self.root.glob('.localization-*')))
        # A failed model must not leave a published result or a temporary CT.
        backend.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n    print('2.18.0'); sys.exit(0)\nsys.exit(7)\n")
        failed_output = self.root/'failed-result'
        failed_command = [str(failed_output) if arg == str(output) else arg for arg in command]
        failed = subprocess.run(failed_command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn('exit 7', failed.stdout)
        self.assertFalse(failed_output.exists())
        self.assertFalse(list(self.root.glob('.localization-*')))


if __name__ == '__main__':
    unittest.main()
