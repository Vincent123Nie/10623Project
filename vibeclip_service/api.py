from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vibeclip_service.runtime import LlavaVideoGrounder


class MomentSearchRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=160)
    query: str = Field(min_length=1, max_length=2_000)
    transcript: list[dict] | None = None


class SampledFrame(BaseModel):
    timestamp: float
    relevanceScore: float


class MomentSearchResponse(BaseModel):
    query: str
    videoId: str
    predictedStart: float
    predictedEnd: float
    reason: str
    sampledFrames: list[SampledFrame]
    rawModelOutput: str


app = FastAPI(title="VibeClip Temporal Grounding API", version="1.0.0")


@lru_cache(maxsize=1)
def get_grounder() -> LlavaVideoGrounder:
    return LlavaVideoGrounder.from_environment()


def get_frame_budget() -> int:
    raw_value = os.environ.get("VIBECLIP_FRAME_BUDGET", "16")
    try:
        frame_budget = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VIBECLIP_FRAME_BUDGET must be an integer.") from exc
    if frame_budget < 8:
        raise RuntimeError("VIBECLIP_FRAME_BUDGET must be at least 8.")
    return frame_budget


def resolve_video(video_id: str) -> Path:
    video_root = Path(os.environ.get("VIBECLIP_VIDEO_ROOT", "./videos")).resolve()
    candidate = (video_root / f"{video_id}.mp4").resolve()
    if video_root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video is not registered with the model service.")
    return candidate


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "configured" if os.environ.get("LLAVA_MODEL_PATH") else "not-configured"}


@app.post("/moment-search", response_model=MomentSearchResponse)
def moment_search(request: MomentSearchRequest) -> MomentSearchResponse:
    try:
        prediction = get_grounder().predict(
            resolve_video(request.video_id),
            request.query,
            frame_budget=get_frame_budget(),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return MomentSearchResponse(
        query=request.query,
        videoId=request.video_id,
        predictedStart=prediction.predicted_start,
        predictedEnd=prediction.predicted_end,
        reason=prediction.reason,
        sampledFrames=prediction.sampled_frames,
        rawModelOutput=prediction.raw_model_output,
    )
