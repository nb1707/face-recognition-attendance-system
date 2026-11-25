import math
import cv2
import numpy as np

from enrollment.errors import enrollment_error


def measure_quality(image, face):
    height, width = image.shape[:2]
    bbox = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
    if bbox.size != 4 or not np.isfinite(bbox).all():
        raise enrollment_error("INVALID_LANDMARKS")
    x1, y1, x2, y2 = bbox
    face_width, face_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    confidence = float(getattr(face, "det_score", getattr(face, "detector_confidence", 0.0)))
    return {
        "face_width": float(face_width),
        "face_height": float(face_height),
        "face_area_ratio": float(face_width * face_height / max(1, width * height)),
        "detector_confidence": confidence,
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(np.mean(gray)),
        "horizontal_offset": float(abs(center_x - width / 2.0) / max(1, width)),
        "vertical_offset": float(abs(center_y - height / 2.0) / max(1, height)),
    }


def evaluate_quality(measurements, policy):
    numeric = ("face_width", "face_height", "face_area_ratio", "detector_confidence",
               "sharpness", "brightness", "horizontal_offset", "vertical_offset")
    if any(not math.isfinite(float(measurements[name])) for name in numeric):
        raise enrollment_error("INVALID_IMAGE", details={"quality": measurements})
    if measurements["sharpness"] < policy.min_sharpness:
        raise enrollment_error("BLURRED_FACE", details={"quality": measurements})
    if not policy.min_brightness <= measurements["brightness"] <= policy.max_brightness:
        raise enrollment_error("POOR_LIGHTING", details={"quality": measurements})
    if (not 0 <= measurements["detector_confidence"] <= 1 or
            measurements["detector_confidence"] < policy.min_detector_confidence):
        raise enrollment_error("LOW_DETECTOR_CONFIDENCE", details={"quality": measurements})
    if measurements["face_area_ratio"] < policy.min_face_area_ratio:
        raise enrollment_error("FACE_TOO_SMALL", details={"quality": measurements})
    if (measurements["horizontal_offset"] > policy.max_horizontal_offset or
            measurements["vertical_offset"] > policy.max_vertical_offset):
        raise enrollment_error("FACE_NOT_CENTERED", details={"quality": measurements})
    components = (
        min(1.0, measurements["sharpness"] / 150.0),
        max(0.0, 1.0 - abs(measurements["brightness"] - 130.0) / 130.0),
        min(1.0, measurements["face_area_ratio"] / 0.20),
        max(0.0, min(1.0, measurements["detector_confidence"])),
    )
    score = float(sum(components) / len(components))
    if not math.isfinite(score):
        raise enrollment_error("INVALID_IMAGE")
    return {"accepted": True, "score": score, "measurements": measurements}
