import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from spinal_ct.ct_io import load_ct
from tools.prepare_ct import prepare_ct


class PrepareCTTests(unittest.TestCase):
    def test_roundtrip_hu_geometry_and_no_reference_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            source.mkdir()
            values = np.arange(120, dtype=np.float32).reshape(4, 5, 6)-1000
            np.save(source/'ct_hu.npy', values)
            metadata = dict(shape_zyx=[4, 5, 6], spacing_zyx_mm=[2, 1, .5],
                            origin_lps_mm=[10, 20, 30],
                            direction_lps=[0, -1, 0, 1, 0, 0, 0, 0, 1],
                            reference_label={'fracture_class': 'A1'}, case_id='case_053')
            (source/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            output = root/'prepared'
            prepare_ct(source/'ct_hu.npy', output)
            image = load_ct(output/'ct_hu.npy')
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image), values)
            self.assertEqual(image.GetSpacing(), (.5, 1, 2))
            self.assertEqual(image.GetOrigin(), (10, 20, 30))
            self.assertEqual(image.GetDirection(), tuple(metadata['direction_lps']))
            saved = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
            self.assertNotIn('reference_label', saved)
            self.assertNotIn('case_id', saved)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare_ct(source/'ct_hu.npy', output)
            np.testing.assert_array_equal(np.load(output/'ct_hu.npy'), values)
