"""
FastAPI Safety Dashboard backend server.
Provides MJPEG video feeds, REST status APIs, telemetry control, fault injection, and event audits.
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
import time
from typing import Optional
import cv2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.config import SystemConfig
from src.core.schemas import AlertState, MachineTelemetry, SensorHealthStatus
from src.pipeline import SafetyPipeline

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Global pipeline instance
pipeline: Optional[SafetyPipeline] = None


class TelemetryUpdate(BaseModel):
    speed_mps: float
    gear: str = "FORWARD"
    direction_deg: float = 0.0


class FaultInjectionRequest(BaseModel):
    sensor: str  # "rgb", "thermal", "telemetry"
    action: str  # "disconnect", "stale", "drop", "recover"
    stale_duration_sec: float = 1.0


def get_pipeline() -> SafetyPipeline:
    global pipeline
    if pipeline is None:
        cfg = SystemConfig.load_from_yaml("config/default_config.yaml")
        pipeline = SafetyPipeline(cfg)
        pipeline.start()
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield
    global pipeline
    if pipeline is not None:
        pipeline.stop()


app = FastAPI(title="Pedestrian Safety System Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Pedestrian Safety System API Active</h1>")


@app.get("/health")
async def health_check():
    pipe = get_pipeline()
    if pipe.last_health_report is None:
        pipe.process_frame()
    report = pipe.last_health_report
    return {
        "status": "UP",
        "overall_health": report.overall_health,
        "alert_state": report.alert_state.value,
        "fps": report.fps,
        "sensors": {k: v.value for k, v in report.sensors.items()},
    }


@app.get("/api/status")
async def get_system_status():
    pipe = get_pipeline()
    if pipe.last_assessment is None:
        pipe.process_frame()

    return {
        "assessment": pipe.last_assessment.model_dump() if pipe.last_assessment else None,
        "health": pipe.last_health_report.model_dump() if pipe.last_health_report else None,
        "tracks": [t.model_dump() for t in pipe.last_tracks],
        "telemetry": pipe.machine_adapter.get_telemetry().model_dump(),
        "simulator_outputs": {
            "visual_strobe": pipe.machine_adapter.visual_strobe_active,
            "buzzer_hz": pipe.machine_adapter.audio_buzzer_hz,
            "cabin_display": pipe.machine_adapter.cabin_display_text,
            "slowdown_requested": pipe.machine_adapter.advisory_slowdown_requested,
        },
    }


@app.get("/api/events")
async def get_events(limit: int = 50):
    pipe = get_pipeline()
    return pipe.event_logger.get_recent_events(limit=limit)


@app.post("/api/telemetry")
async def update_telemetry(payload: TelemetryUpdate):
    pipe = get_pipeline()
    pipe.machine_adapter.set_machine_state(
        speed_mps=payload.speed_mps,
        gear=payload.gear,
        direction_deg=payload.direction_deg,
    )
    return {"status": "updated", "telemetry": pipe.machine_adapter.get_telemetry().model_dump()}


@app.post("/api/step")
async def step_pipeline():
    pipe = get_pipeline()
    assessment, health = pipe.process_frame()
    return {
        "alert_state": assessment.alert_state.value,
        "reason": assessment.reason,
        "active_tracks": len(pipe.last_tracks),
        "fps": health.fps,
    }


@app.post("/api/fault_injection")
async def inject_fault(payload: FaultInjectionRequest):
    pipe = get_pipeline()
    target_sensor = None
    if payload.sensor.lower() == "rgb":
        target_sensor = pipe.rgb_sensor
    elif payload.sensor.lower() == "thermal":
        target_sensor = pipe.thermal_sensor
    elif payload.sensor.lower() == "telemetry":
        if payload.action == "disconnect":
            pipe.machine_adapter.set_connection_fault(True)
        else:
            pipe.machine_adapter.set_connection_fault(False)
        return {"status": "fault_applied", "target": "telemetry", "action": payload.action}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sensor target: {payload.sensor}")

    if hasattr(target_sensor, "inject_drop"):
        if payload.action == "drop":
            target_sensor.inject_drop(True)
        elif payload.action == "stale":
            target_sensor.inject_stale(True, payload.stale_duration_sec)
        elif payload.action == "disconnect":
            target_sensor.disconnect()
        elif payload.action == "recover":
            target_sensor.inject_drop(False)
            target_sensor.inject_stale(False)
            target_sensor.connect()
    else:
        if payload.action == "disconnect":
            target_sensor.disconnect()
        elif payload.action == "recover":
            target_sensor.connect()

    return {"status": "fault_applied", "target": payload.sensor, "action": payload.action}


def generate_mjpeg_stream(stream_type: str):
    pipe = get_pipeline()
    while True:
        if stream_type == "rgb":
            frame = pipe.render_annotated_rgb_frame()
        else:
            frame = pipe.render_annotated_thermal_frame()

        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.04)  # ~25 fps stream


@app.get("/stream/rgb")
async def stream_rgb():
    return StreamingResponse(
        generate_mjpeg_stream("rgb"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/thermal")
async def stream_thermal():
    return StreamingResponse(
        generate_mjpeg_stream("thermal"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Mount static assets directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
