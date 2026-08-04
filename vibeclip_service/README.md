# VibeClip Temporal Grounding Service

This adapter exposes the repository's LLaVA temporal-token work as the API consumed by the VibeClip AI web application.

## Runtime flow

1. Load the base model or LoRA adapter once at process startup.
2. Run a coarse query-conditioned grounding pass over uniformly distributed frames.
3. Preserve global coverage and add a denser window around the coarse prediction.
4. Run a refined grounding pass with a 16-frame default budget.
5. Parse `<time_start> <t###> <time_end> <t###>` into seconds and return the evidence interval.

The service does not invent a semantic confidence score. It returns the interval, raw model output, explanation, and sampled-frame provenance. The web UI displays confidence only when a future calibrated model supplies it.

## Configuration

```bash
set LLAVA_MODEL_PATH=D:\models\vibeclip-lora
set LLAVA_MODEL_BASE=D:\models\LLaVA-NeXT-Video-7B-DPO
set VIBECLIP_VIDEO_ROOT=D:\vibeclip-videos
set VIBECLIP_FRAME_BUDGET=16
uvicorn vibeclip_service.api:app --host 0.0.0.0 --port 8000
```

Register videos as `<VIBECLIP_VIDEO_ROOT>/<video_id>.mp4`. The web app's `TEMPORAL_GROUNDING_API_URL` should point to this service.

## Endpoints

- `GET /health`
- `POST /moment-search`

The service requires a CUDA-capable environment and the main repository's inference dependencies in addition to `vibeclip_service/requirements.txt`.

## Evaluation

`python -m vibeclip_service.evaluate predictions.jsonl` reports parseability, mIoU, R@0.3, and R@0.5. Predictions that do not contain a valid interval remain in the denominator and receive IoU 0, so the report cannot hide parsing failures.
