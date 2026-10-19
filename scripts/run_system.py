"""
Main system CLI entrypoint.
Launches the Pedestrian Safety System pipeline, dashboard web server, or benchmark suites.
"""

from __future__ import annotations
import argparse
import logging
import sys
import uvicorn

from src.core.config import SystemConfig
from src.pipeline import SafetyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SystemEntry")


def main():
    parser = argparse.ArgumentParser(description="Multipurpose Pedestrian Safety System (RGB + Thermal Fusion)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--mode", type=str, choices=["dashboard", "headless", "benchmark", "generate-data"], default="dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard bind port")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario folder to replay (if applicable)")

    args = parser.parse_args()

    if args.mode == "generate-data":
        from scripts.generate_scenario_data import generate_all_scenarios
        generate_all_scenarios()
        sys.exit(0)

    if args.mode == "benchmark":
        from scripts.run_benchmark import run_full_benchmark
        run_full_benchmark()
        sys.exit(0)

    if args.mode == "dashboard":
        logger.info(f"Starting Safety Dashboard on http://{args.host}:{args.port}")
        uvicorn.run("src.dashboard.app:app", host=args.host, port=args.port, reload=False, access_log=False)
        sys.exit(0)

    if args.mode == "headless":
        logger.info(f"Loading system configuration from {args.config}")
        cfg = SystemConfig.load_from_yaml(args.config)
        if args.scenario:
            cfg.rgb_sensor.mode = "replay"
            cfg.rgb_sensor.replay_path = args.scenario
            cfg.thermal_sensor.mode = "replay"
            cfg.thermal_sensor.replay_path = args.scenario

        pipeline = SafetyPipeline(cfg)
        if not pipeline.start():
            logger.error("Failed to start one or more sensors")
            sys.exit(1)

        logger.info("Pipeline started in headless loop. Press Ctrl+C to terminate.")
        try:
            while True:
                assessment, health = pipeline.process_frame()
                logger.info(
                    f"Frame {pipeline.frame_index:04d} | State: {assessment.alert_state.value:<8s} | "
                    f"Tracks: {len(pipeline.last_tracks):2d} | FPS: {health.fps:4.1f} | Latency: {health.latency_total_ms:5.1f}ms | "
                    f"Reason: {assessment.reason}"
                )
                import time
                time.sleep(0.033)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            pipeline.stop()


if __name__ == "__main__":
    main()
