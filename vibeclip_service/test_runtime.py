import unittest

from vibeclip_service.runtime import _replace_temporal_span, decode_temporal_prediction


class DecodeTemporalPredictionTest(unittest.TestCase):
    def test_decodes_structured_time_tokens(self) -> None:
        start, end = decode_temporal_prediction(
            "Answer. <time_start> <t250> <time_end> <t499>.",
            duration_sec=100.0,
            num_bins=1000,
        )
        self.assertAlmostEqual(start, 25.025, places=3)
        self.assertAlmostEqual(end, 50.050, places=3)

    def test_rejects_unparseable_output(self) -> None:
        with self.assertRaises(ValueError):
            decode_temporal_prediction("No structured interval", duration_sec=100.0)

    def test_normalizes_reversed_and_out_of_range_bins(self) -> None:
        start, end = decode_temporal_prediction(
            "<time_start><t1200><time_end><t0>",
            duration_sec=20.0,
        )
        self.assertEqual((start, end), (0.0, 20.0))

    def test_rejects_invalid_temporal_scale(self) -> None:
        with self.assertRaises(ValueError):
            decode_temporal_prediction("<time_start><t0><time_end><t1>", float("nan"))
        with self.assertRaises(ValueError):
            decode_temporal_prediction("<time_start><t0><time_end><t1>", 10.0, num_bins=1)

    def test_explanation_normalizes_reversed_bins(self) -> None:
        text = _replace_temporal_span(
            "Located at <time_start><t900><time_end><t100>.",
            duration_sec=10.0,
            num_bins=1000,
        )
        self.assertEqual(text, "Located at between 1.0 seconds and 9.0 seconds.")


if __name__ == "__main__":
    unittest.main()
