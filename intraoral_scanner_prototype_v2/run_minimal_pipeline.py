"""Minimal in-process synthetic pipeline runner.
Runs the ScanningService in simulation mode for N frames without multiprocessing/UI.
Outputs a summary JSON and optional mesh export placeholder.
"""
import time, json, os
from pathlib import Path
from config.system_config import get_config
from services.scanning_service import ScanningService

DEFAULT_FRAMES = 60


def run(frames=DEFAULT_FRAMES):
    cfg = get_config()
    # Force simulation and disable heavy subsystems
    cfg.flags.simulation_mode = True
    cfg.flags.enable_meshroom = False
    cfg.flags.enable_ai = False

    service = ScanningService(service_port=0)  # port unused in in-process run
    if not service.start_service():
        print("Failed to start service")
        return 1

    # Start scan via internal method
    start_resp = service._start_scan({})
    if start_resp.get('status') != 'success':
        print("Failed to start synthetic scan:", start_resp)
        return 1

    print(f"Running synthetic scan for {frames} frames...")
    target = frames
    start_time = time.time()
    last_report = 0
    timeout_sec = max(5, frames * 0.5)
    while service.frame_count < target:
        time.sleep(0.02)
        if time.time() - start_time > timeout_sec:
            print(f"Timeout reached after {timeout_sec}s with {service.frame_count} frames")
            break
        if time.time() - last_report > 1:
            print(f" .. progress: {service.frame_count}/{target} frames")
            last_report = time.time()
    # Stop scan
    stop_resp = service._stop_scan()

    duration = time.time() - start_time
    summary = {
        'requested_frames': frames,
        'processed_frames': service.frame_count,
        'duration_sec': duration,
        'avg_fps': service.frame_count / duration if duration > 0 else 0.0,
        'scan_stats': stop_resp.get('statistics', {})
    }

    Path('artifacts').mkdir(exist_ok=True)
    with open('artifacts/synthetic_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Synthetic run complete. Summary written to artifacts/synthetic_run_summary.json")

    service.stop_service()
    return 0

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=DEFAULT_FRAMES)
    args = p.parse_args()
    raise SystemExit(run(args.frames))
