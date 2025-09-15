import json, sys
from pathlib import Path
import os

# Ensure the prototype root is on sys.path so `run_minimal_pipeline` can be imported
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Basic smoke test for minimal pipeline runner

def test_synthetic_run_import():
    try:
        import run_minimal_pipeline  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover - explicit failure detail
        raise AssertionError(f"Could not import run_minimal_pipeline from {ROOT}. sys.path={sys.path}") from e


def test_synthetic_run_exec():
    import run_minimal_pipeline
    code = run_minimal_pipeline.run(frames=10)
    assert code == 0
    summary_path = Path('artifacts/synthetic_run_summary.json')
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data['processed_frames'] >= 10
    assert data['avg_fps'] > 0
