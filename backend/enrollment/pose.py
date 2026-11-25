import math
import numpy as np

from enrollment.errors import enrollment_error


def measure_landmarks(landmarks, min_eye_distance_pixels=8.0):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise enrollment_error("INVALID_LANDMARKS")
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if (not math.isfinite(eye_distance) or eye_distance < min_eye_distance_pixels or
            abs(float(eye_vector[0])) < min_eye_distance_pixels):
        raise enrollment_error("INVALID_LANDMARKS")
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    eye_mouth_distance = float(mouth_center[1] - eye_center[1])
    mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
    if eye_mouth_distance <= min_eye_distance_pixels or mouth_width < min_eye_distance_pixels / 2:
        raise enrollment_error("INVALID_LANDMARKS")
    return {
        "yaw_ratio": float((nose[0] - eye_center[0]) / abs(eye_vector[0])),
        # Landmark-geometry heuristic, not pitch degrees.
        "vertical_ratio": float((nose[1] - eye_center[1]) / eye_mouth_distance),
        "eye_distance": eye_distance,
        "eye_mouth_distance": eye_mouth_distance,
        "mouth_width": mouth_width,
    }


def evaluate_pose(pose, measurements, policy):
    yaw = measurements["yaw_ratio"]
    vertical = measurements["vertical_ratio"]
    accepted = False
    guidance = "Adjust your face to match the requested pose."
    if pose == "front":
        accepted = (abs(yaw) <= policy.front_max_yaw and
                    policy.front_min_vertical_ratio <= vertical <= policy.front_max_vertical_ratio)
        guidance = "Look straight at the camera."
    elif pose == "left":
        accepted = (policy.side_min_yaw <= yaw <= policy.side_max_yaw and
                    policy.front_min_vertical_ratio <= vertical <= policy.front_max_vertical_ratio)
        guidance = "Turn slightly more to your left; avoid an extreme angle."
    elif pose == "right":
        accepted = (-policy.side_max_yaw <= yaw <= -policy.side_min_yaw and
                    policy.front_min_vertical_ratio <= vertical <= policy.front_max_vertical_ratio)
        guidance = "Turn slightly more to your right; avoid an extreme angle."
    elif pose == "up":
        accepted = (abs(yaw) <= policy.vertical_pose_max_yaw and
                    policy.up_min_vertical_ratio <= vertical <= policy.up_max_vertical_ratio)
        guidance = "Raise your face slightly while looking toward the camera."
    elif pose == "down":
        accepted = (abs(yaw) <= policy.vertical_pose_max_yaw and
                    policy.down_min_vertical_ratio <= vertical <= policy.down_max_vertical_ratio)
        guidance = "Lower your face slightly while looking toward the camera."
    if not accepted:
        raise enrollment_error("WRONG_POSE", details={
            "pose": pose, "pose_measurements": measurements, "guidance": guidance,
        })
    return {"accepted": True, "requested_pose": pose, **measurements}
