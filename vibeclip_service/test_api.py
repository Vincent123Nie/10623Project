import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vibeclip_service.runtime import GroundingPrediction

try:
    from fastapi import HTTPException

    from vibeclip_service.api import MomentSearchRequest, get_frame_budget, moment_search
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "pydantic"}:
        raise
    API_DEPENDENCIES_AVAILABLE = False
else:
    API_DEPENDENCIES_AVAILABLE = True


@unittest.skipUnless(API_DEPENDENCIES_AVAILABLE, "FastAPI service dependencies are not installed")
class ApiContractTest(unittest.TestCase):
    def test_response_matches_frontend_contract(self) -> None:
        prediction = GroundingPrediction(
            predicted_start=12.5,
            predicted_end=19.0,
            reason="The requested action occurs in this interval.",
            raw_model_output="<time_start> <t125> <time_end> <t190>",
            sampled_frames=[{"timestamp": 12.5, "relevanceScore": 1.0}],
        )
        grounder = Mock()
        grounder.predict.return_value = prediction

        with patch("vibeclip_service.api.resolve_video", return_value=Path("video.mp4")), patch(
            "vibeclip_service.api.get_grounder", return_value=grounder
        ), patch.dict(os.environ, {"VIBECLIP_FRAME_BUDGET": "16"}):
            response = moment_search(MomentSearchRequest(video_id="demo", query="Find the action"))

        payload = response.dict()
        self.assertEqual(payload["predictedStart"], 12.5)
        self.assertEqual(payload["predictedEnd"], 19.0)
        self.assertEqual(payload["reason"], prediction.reason)
        self.assertEqual(payload["sampledFrames"][0]["timestamp"], 12.5)
        self.assertEqual(payload["sampledFrames"][0]["relevanceScore"], 1.0)

    def test_invalid_frame_budget_is_reported_as_service_configuration_error(self) -> None:
        with patch.dict(os.environ, {"VIBECLIP_FRAME_BUDGET": "invalid"}):
            with self.assertRaises(RuntimeError):
                get_frame_budget()

        with patch("vibeclip_service.api.resolve_video", return_value=Path("video.mp4")), patch(
            "vibeclip_service.api.get_grounder"
        ), patch.dict(os.environ, {"VIBECLIP_FRAME_BUDGET": "4"}):
            with self.assertRaises(HTTPException) as raised:
                moment_search(MomentSearchRequest(video_id="demo", query="Find the action"))
        self.assertEqual(raised.exception.status_code, 503)

    def test_unparseable_model_output_is_reported_as_upstream_failure(self) -> None:
        grounder = Mock()
        grounder.predict.side_effect = ValueError("No temporal span")
        with patch("vibeclip_service.api.resolve_video", return_value=Path("video.mp4")), patch(
            "vibeclip_service.api.get_grounder", return_value=grounder
        ), patch.dict(os.environ, {"VIBECLIP_FRAME_BUDGET": "16"}):
            with self.assertRaises(HTTPException) as raised:
                moment_search(MomentSearchRequest(video_id="demo", query="Find the action"))
        self.assertEqual(raised.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
