import io
import unittest

from spinal_ct.labels import parse_identifiers
from spinal_ct.labels import parse_reference_labels, parse_vertebral_levels
from spinal_ct.labels import summarize_reference_labels


class LabelParsingTests(unittest.TestCase):
    def test_parse_identifiers_splits_multiple_values(self):
        self.assertEqual(parse_identifiers("123 $$$ 456\u00a0"), ("123", "456"))

    def test_parse_levels_normalizes_vietnamese_dorsal_prefix(self):
        self.assertEqual(parse_vertebral_levels("D11-D12"), ("T11", "T12"))
        self.assertEqual(parse_vertebral_levels("Gãy L1"), ("L1",))
        self.assertEqual(parse_vertebral_levels("L3L4"), ("L3", "L4"))

    def test_load_and_summarize_without_names(self):
        content = (
            "STT,Tên,Năm sinh,Đốt gãy,ID,phân loại,kèm theo\n"
            "1,Example Person,1970,L1,123,A3,\n"
            "2,Another Person,1980,D12-L1,456$$$789,C,note\n"
        )
        labels = parse_reference_labels(io.StringIO(content))

        summary = summarize_reference_labels(labels)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["class_counts"], {"A3": 1, "C": 1})
        self.assertEqual(summary["multi_level_rows"], 1)
        self.assertNotIn("Example Person", repr(labels))
        self.assertNotIn("123", repr(labels))


if __name__ == "__main__":
    unittest.main()
