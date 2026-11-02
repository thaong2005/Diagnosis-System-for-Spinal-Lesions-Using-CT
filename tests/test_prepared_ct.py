import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct, slugify, extract_patient_identifier


class PreparedCTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'ct_hu.npy'
        self.data = np.arange(120, dtype=np.int16).reshape(4,5,6)-1000
        np.save(self.path, self.data)
        self.metadata = dict(shape_zyx=[4,5,6], spacing_zyx_mm=[2,1,0.5],
                             origin_lps_mm=[10,20,30],
                             image_orientation_patient=[0,1,0,-1,0,0])

    def write(self):
        (self.root/'metadata.json').write_text(json.dumps(self.metadata),encoding='utf-8')

    def test_geometry_hu_and_labels_are_independent(self):
        self.write()
        image = load_ct(self.path)
        self.assertEqual(image.GetSpacing(), (0.5,1,2))
        self.assertEqual(image.TransformIndexToPhysicalPoint((2,3,1)), (7,21,32))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), self.data)
        self.metadata['reference_label'] = {'vertebral_level':'L5', 'fracture_class':'A4'}
        self.metadata['annotation'] = {'bbox': 'not consumed'}
        self.write()
        other = load_ct(self.path)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image),sitk.GetArrayFromImage(other))
        self.assertEqual(image.GetDirection(),other.GetDirection())
        self.assertFalse(other.GetMetaDataKeys())

    def test_direction_only_and_conflict(self):
        direction = [0,-1,0,1,0,0,0,0,1]
        self.metadata['direction_lps'] = direction
        self.write()
        self.assertEqual(load_ct(self.path).GetDirection(),tuple(direction))
        self.metadata.pop('image_orientation_patient')
        self.write()
        self.assertEqual(load_ct(self.path).GetDirection(),tuple(direction))
        self.metadata['image_orientation_patient'] = [1,0,0,0,1,0]
        self.write()
        with self.assertRaisesRegex(ValueError,'disagree'):
            load_ct(self.path)

    def test_invalid_geometry_and_shape(self):
        for key, value in [('shape_zyx',[1,2,3]), ('spacing_zyx_mm',[0,1,1]),
                           ('origin_lps_mm',[0,0,float('nan')]),
                           ('image_orientation_patient',[2,0,0,0,1,0])]:
            original = self.metadata[key]
            self.metadata[key] = value
            self.write()
            with self.assertRaises(ValueError):
                load_ct(self.path)
            self.metadata[key] = original
        self.write()
        with self.assertRaisesRegex(ValueError,'series-uid'):
            load_ct(self.path,'unused')

    def test_slugify_handles_accents_and_special_chars(self):
        self.assertEqual(slugify('Đường Kim Hoa'), 'duong_kim_hoa')
        self.assertEqual(slugify('NGUYỄN VĂN A^BS'), 'nguyen_van_a_bs')
        self.assertEqual(slugify('---test!!patient...'), 'test_patient')
        self.assertEqual(slugify('', fallback='fallback_id'), 'fallback_id')

    def test_patient_identifier_directory_fallbacks(self):
        # Walking up skipping technical names, syngo hashes, batch codes
        nested = Path('/data/raw/A3/duong kim hoa/csc/MBBN3ULK/en-GB/image.dcm')
        info = extract_patient_identifier(nested, fingerprint='abcdef123456')
        self.assertEqual(info['slug'], 'duong_kim_hoa')
        self.assertEqual(info['display_name'], 'Duong Kim Hoa')

        # Fallback to fingerprint
        only_tech = Path('/data/raw/temp/dicom/exam')
        fallback = extract_patient_identifier(only_tech, fingerprint='123456789abc')
        self.assertEqual(fallback['slug'], 'benh_nhan_12345678')


if __name__ == '__main__':
    unittest.main()
