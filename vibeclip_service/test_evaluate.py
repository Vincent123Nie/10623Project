import unittest

from vibeclip_service.evaluate import evaluate_records, interval_iou


class EvaluateTest(unittest.TestCase):
    def test_interval_iou(self) -> None:
        self.assertAlmostEqual(interval_iou((10, 20), (15, 25)), 1 / 3)

    def test_metrics_include_unparseable_predictions(self) -> None:
        metrics = evaluate_records(
            [
                {"predictedStart": 10, "predictedEnd": 20, "temporal_segments_sec": [[10, 20]]},
                {"predictedStart": None, "predictedEnd": None, "temporal_segments_sec": [[0, 5]]},
            ]
        )
        self.assertEqual(metrics["parseability"], 0.5)
        self.assertEqual(metrics["mIoU"], 0.5)

    def test_malformed_numeric_values_do_not_crash_or_count_as_parseable(self) -> None:
        metrics = evaluate_records(
            [
                {"predictedStart": True, "predictedEnd": 5, "targets": [[0, 5]]},
                {"predictedStart": 0, "predictedEnd": float("inf"), "targets": [[0, 5]]},
                {"predictedStart": 0, "predictedEnd": 5, "targets": [["bad", 5], [0, 5]]},
            ]
        )
        self.assertEqual(metrics["parseability"], 1 / 3)
        self.assertEqual(metrics["mIoU"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
