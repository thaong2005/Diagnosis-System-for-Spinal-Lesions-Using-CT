import unittest

from spinal_ct.annotations import Annotation3D
from spinal_ct.annotations import AnnotationValidationError
from spinal_ct.annotations import validate_annotation


def valid_annotation():
    return {
        "schema_version": 1,
        "case_id": "case_053",
        "vertebral_levels": ["L1"],
        "target": "vertebral_body",
        "bbox_zyx_voxel": {
            "z_min": 10,
            "z_max_exclusive": 50,
            "y_min": 20,
            "y_max_exclusive": 80,
            "x_min": 30,
            "x_max_exclusive": 90,
        },
        "bbox_lps_mm": {
            "x_min": -45.0,
            "x_max": 15.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "z_min": 10.0,
            "z_max": 70.0,
        },
        "geometry": {
            "shape_zyx": [100, 120, 140],
            "spacing_zyx_mm": [1.0, 1.0, 1.0],
            "origin_lps_mm": [-100.0, -100.0, -100.0],
            "direction_lps": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "review_status": "draft",
    }


class Annotation3DTests(unittest.TestCase):
    def test_valid_annotation_round_trips(self):
        annotation = Annotation3D.from_dict(valid_annotation())
        self.assertEqual(annotation.to_dict()["case_id"], "case_053")

    def test_rejects_old_sagittal_only_annotation(self):
        payload = valid_annotation()
        payload.pop("bbox_zyx_voxel")
        payload.pop("bbox_lps_mm")
        errors = validate_annotation(payload)
        self.assertTrue(any("bbox_zyx_voxel" in error for error in errors))
        self.assertTrue(any("bbox_lps_mm" in error for error in errors))

    def test_rejects_voxel_bbox_outside_volume(self):
        payload = valid_annotation()
        payload["bbox_zyx_voxel"]["z_max_exclusive"] = 101
        with self.assertRaises(AnnotationValidationError):
            Annotation3D.from_dict(payload)

    def test_rejected_annotation_requires_reason(self):
        payload = valid_annotation()
        payload["review_status"] = "rejected"
        with self.assertRaises(AnnotationValidationError):
            Annotation3D.from_dict(payload)

    def test_rejects_patient_identifier_as_case_id(self):
        payload = valid_annotation()
        payload["case_id"] = "Patient Name 053"
        with self.assertRaises(AnnotationValidationError):
            Annotation3D.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
