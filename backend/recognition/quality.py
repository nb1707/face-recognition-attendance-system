"""Crop-based measurements separated from provisional development gate policy."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    # Provisional development settings, NOT calibrated classroom thresholds.
    crop_size: int = 112
    min_size: float = 32
    min_area_ratio: float = 0
    min_detector: float = .5
    min_sharpness: float = 20
    min_brightness: float = 35
    max_brightness: float = 225
    max_nose_offset: float = .65
    max_roll_degrees: float = 40
    sharpness_reference: float = 200
    size_reference: float = 112
    brightness_target: float = 128
    component_weights: tuple[float, ...] = (1, 1, 1, 1, 1)

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != "component_weights"]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite quality configuration")
        if self.crop_size < 2 or self.min_size <= 0 or self.min_sharpness < 0:
            raise ValueError("Invalid quality size/sharpness")
        if not 0 <= self.min_detector <= 1 or not 0 <= self.min_area_ratio <= 1:
            raise ValueError("Invalid quality confidence/area")
        if not 0 <= self.min_brightness < self.brightness_target < self.max_brightness <= 255:
            raise ValueError("Invalid brightness configuration")
        if min(self.max_nose_offset, self.max_roll_degrees, self.sharpness_reference, self.size_reference) <= 0:
            raise ValueError("Invalid quality reference")
        if len(self.component_weights) != 5 or not all(math.isfinite(w) and w >= 0 for w in self.component_weights) or sum(self.component_weights) <= 0:
            raise ValueError("Invalid quality component weights")


@dataclass(frozen=True)
class QualityMeasurements:
    valid: bool
    reason: str | None
    width: float = 0
    height: float = 0
    area_ratio: float = 0
    detector_confidence: float = 0
    sharpness: float = 0
    brightness: float = 0
    nose_offset: float = 0
    roll_degrees: float = 0


@dataclass(frozen=True)
class QualityDecision:
    passed: bool
    quality_score: float
    reasons: tuple[str, ...]
    components: dict[str, float]


def measure_quality(frame, bbox, landmarks, detector_confidence, crop_size=112):
    """Measure grayscale of the clipped bbox resized to a fixed square crop.

    Nose offset is projected along the eye line / inter-eye distance, not yaw.
    No full-frame blur or brightness enters the measurements.
    """
    try:
        box, points = np.asarray(bbox, dtype=float), np.asarray(landmarks, dtype=float)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("invalid_frame")
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("invalid_bbox")
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("invalid_landmarks")
        if not math.isfinite(detector_confidence) or not 0 <= detector_confidence <= 1:
            raise ValueError("invalid_detector_confidence")
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        crop = frame[y1:y2, x1:x2]
        if x2 <= x1 or y2 <= y1 or not crop.size:
            raise ValueError("invalid_bbox")
        if np.any(points < [x1, y1]) or np.any(points > [x2, y2]):
            raise ValueError("invalid_landmarks")
        eyes = points[1] - points[0]
        distance = float(np.linalg.norm(eyes))
        if distance < 1 or eyes[0] <= 0 or points[3:5, 1].mean() <= points[:2, 1].mean():
            raise ValueError("invalid_landmarks")
        offset = abs(float((points[2] - points[:2].mean(axis=0)) @ eyes)) / distance**2
        roll = abs(math.degrees(math.atan2(eyes[1], eyes[0])))
        gray = cv2.cvtColor(cv2.resize(crop, (crop_size, crop_size)), cv2.COLOR_BGR2GRAY)
        return QualityMeasurements(True, None, x2-x1, y2-y1, (x2-x1)*(y2-y1)/(w*h),
                                   float(detector_confidence), float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                                   float(gray.mean()), offset, roll)
    except (ValueError, TypeError, AttributeError, cv2.error) as error:
        return QualityMeasurements(False, str(error))


def decide_quality(m, config=None):
    """Score = weighted arithmetic mean of five visible clipped [0,1] signals.

    Signals: detector, sharpness/reference, size/reference,
    1-max(normalized nose offset, normalized roll), triangular lighting fitness.
    This is a deterministic ranking score, never a probability.
    """
    c = config or QualityConfig()
    if not m.valid:
        return QualityDecision(False, 0, (m.reason or "invalid_observation",), {})
    reasons = []
    checks = [(min(m.width, m.height) < c.min_size or m.area_ratio < c.min_area_ratio, "face_too_small"),
              (m.detector_confidence < c.min_detector, "low_detector_confidence"),
              (m.sharpness < c.min_sharpness, "blurred_face"),
              (not c.min_brightness <= m.brightness <= c.max_brightness, "poor_lighting"),
              (m.nose_offset > c.max_nose_offset or m.roll_degrees > c.max_roll_degrees, "extreme_pose")]
    reasons.extend(reason for failed, reason in checks if failed)
    light_scale = c.brightness_target-c.min_brightness if m.brightness <= c.brightness_target else c.max_brightness-c.brightness_target
    components = dict(detector=m.detector_confidence,
                      sharpness=m.sharpness/c.sharpness_reference,
                      size=min(m.width, m.height)/c.size_reference,
                      pose=1-max(m.nose_offset/c.max_nose_offset, m.roll_degrees/c.max_roll_degrees),
                      lighting=1-abs(m.brightness-c.brightness_target)/light_scale)
    components = {k: float(np.clip(v, 0, 1)) for k, v in components.items()}
    score = sum(w*v for w, v in zip(c.component_weights, components.values())) / sum(c.component_weights)
    return QualityDecision(not reasons, score, tuple(reasons), components)
