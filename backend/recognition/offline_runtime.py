"""Local model preflight: prevent InsightFace's implicit model download."""
from pathlib import Path


def require_local_detector():
    directory = Path.home() / ".insightface/models/buffalo_l"
    if not (directory / "det_10g.onnx").is_file():
        raise FileNotFoundError("Offline detector requires local ~/.insightface/models/buffalo_l/det_10g.onnx")
    return directory
