from dataclasses import dataclass
from types import MappingProxyType


POSE_TARGETS = MappingProxyType({
    "front": 5,
    "left": 3,
    "right": 3,
    "up": 2,
    "down": 2,
})

# Development policy: small pose-complete production set, not a calibrated optimum.
REPRESENTATIVE_TARGETS = MappingProxyType({
    "front": 2,
    "left": 1,
    "right": 1,
    "up": 1,
    "down": 1,
})

VALID_ROLES = frozenset({"student", "teacher", "staff"})
SESSION_STATES = frozenset({"CAPTURING", "REVIEW", "FINALIZING", "COMPLETE", "ERROR"})


@dataclass(frozen=True)
class EnrollmentPolicy:
    min_sharpness: float = 50.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_face_area_ratio: float = 0.05
    max_horizontal_offset: float = 0.18
    max_vertical_offset: float = 0.20
    # PROVISIONAL: mirrors the detector's conservative development operating region.
    min_detector_confidence: float = 0.50
    min_eye_distance_pixels: float = 8.0
    front_max_yaw: float = 0.15
    # Side poses begin outside the frontal band, leaving a deliberate dead zone.
    side_min_yaw: float = 0.20
    side_max_yaw: float = 0.45
    front_min_vertical_ratio: float = 0.42
    front_max_vertical_ratio: float = 0.60
    up_min_vertical_ratio: float = 0.18
    up_max_vertical_ratio: float = 0.40
    down_min_vertical_ratio: float = 0.62
    down_max_vertical_ratio: float = 0.85
    vertical_pose_max_yaw: float = 0.15
    # Mean absolute aligned-pixel difference / 255. Exact equality is handled first.
    near_duplicate_max_difference: float = 0.012

    def __post_init__(self):
        if not (0 <= self.front_max_yaw < self.side_min_yaw < self.side_max_yaw):
            raise ValueError("Pose yaw bands must satisfy front_max < side_min < side_max")
        if not (0 <= self.vertical_pose_max_yaw <= self.front_max_yaw):
            raise ValueError("Vertical poses must remain inside the frontal yaw band")
        if not (
            self.up_min_vertical_ratio
            < self.up_max_vertical_ratio
            < self.front_min_vertical_ratio
            < self.front_max_vertical_ratio
            < self.down_min_vertical_ratio
            < self.down_max_vertical_ratio
        ):
            raise ValueError("Up, front, and down vertical bands must be ordered and disjoint")


DEFAULT_POLICY = EnrollmentPolicy()
