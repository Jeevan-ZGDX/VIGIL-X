"""
FastAPI Safety Dashboard backend server.
Runs continuous real-time pipeline execution, live camera streams,
standard metric telemetry, fault injection, and scenario replay switching.
"""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
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


class SourceSelectionRequest(BaseModel):
    source_type: str  # "live" or scenario name (e.g. "scenario_1_normal")


def get_pipeline() -> SafetyPipeline:
    global pipeline
    if pipeline is None:
        cfg = SystemConfig.load_from_yaml("config/default_config.yaml")

        # Probe for physical camera availability
        cap_test = cv2.VideoCapture(0)
        has_camera = cap_test.isOpened()
        if has_camera:
            cap_test.release()
            cfg.rgb_sensor.mode = "live"
            cfg.rgb_sensor.device_index = 0
            cfg.thermal_sensor.mode = "live"
            cfg.thermal_sensor.device_index = 1
            if Path("models/yolov5n.onnx").exists():
                cfg.rgb_detector.backend = "onnx"
                cfg.rgb_detector.model_path = "models/yolov5n.onnx"
            logger.info("Auto-detected live webcam on /dev/video0; configuring live real-time pipeline with YOLOv5n ONNX")
        else:
            cfg.rgb_sensor.mode = "replay"
            cfg.rgb_sensor.replay_path = "data/synthetic/scenario_1_normal"
            cfg.thermal_sensor.mode = "replay"
            cfg.thermal_sensor.replay_path = "data/synthetic/scenario_1_normal"
            logger.info("No physical webcam detected; initializing with scenario 1 replay")

        pipeline = SafetyPipeline(cfg)
        pipeline.start_realtime_loop(target_fps=25.0)

    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield
    global pipeline
    if pipeline is not None:
        pipeline.stop_realtime_loop()
        pipeline.stop()


app = FastAPI(
    title="Pedestrian Safety System Dashboard",
    description="Multimodal RGB + Thermal Infrared Edge Safety System",
    version="0.2.0",
    lifespan=lifespan,
)

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
    return HTMLResponse("<h1>Pedestrian Safety System Active</h1>")


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
        "latency_p95_ms": report.latency_p95_ms,
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
        "latency_ms": health.latency_total_ms,
    }


@app.post("/api/set_source")
async def set_source(payload: SourceSelectionRequest):
    """Switch pipeline between live cameras and pre-recorded evaluation scenarios."""
    global pipeline
    pipe = get_pipeline()
    pipe.stop_realtime_loop()
    pipe.stop()

    cfg = SystemConfig.load_from_yaml("config/default_config.yaml")
    if payload.source_type == "live":
        cfg.rgb_sensor.mode = "live"
        cfg.rgb_sensor.device_index = 0
        cfg.thermal_sensor.mode = "live"
        cfg.thermal_sensor.device_index = 1
    else:
        scen_path = f"data/synthetic/{payload.source_type}"
        cfg.rgb_sensor.mode = "replay"
        cfg.rgb_sensor.replay_path = scen_path
        cfg.thermal_sensor.mode = "replay"
        cfg.thermal_sensor.replay_path = scen_path

    pipeline = SafetyPipeline(cfg)
    pipeline.start_realtime_loop(target_fps=25.0)
    return {"status": "source_switched", "source": payload.source_type}


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

        if frame is not None and frame.size > 0:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30 FPS streaming pace


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
