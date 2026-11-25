try:
    from supabase import create_client
except Exception:
    create_client = None
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import threading
from urllib.parse import quote
import json
from datetime import datetime
import shutil
import csv
from io import BytesIO
from recognition.model_factory import get_model
import time
import uuid
# to run API benchmark, import the following functions from evaluation.benchmark
from uuid import uuid4
from pydantic import BaseModel
from recognition.face_detector import get_shared_detector
from recognition.face_preprocessing import align_detected_face

from evaluation.benchmark import run_model_benchmark

from recognition.detection_config import (CLASSROOM_ROI, TILED_DETECTION_CONFIG, DEFAULT_CLASSROOM_DETECTION_MODE)
from recognition.thresholds import MODEL_THRESHOLDS, get_model_threshold

# load environment variables from .env file
load_dotenv()

HIKVISION_HOST = os.getenv(
    "HIKVISION_HOST",
    ""
)

HIKVISION_RTSP_PORT = int(
    os.getenv(
        "HIKVISION_RTSP_PORT",
        "554"
    )
)

HIKVISION_USERNAME = os.getenv(
    "HIKVISION_USERNAME",
    ""
)

HIKVISION_PASSWORD = os.getenv(
    "HIKVISION_PASSWORD",
    ""
)

HIKVISION_CHANNEL = int(
    os.getenv(
        "HIKVISION_CHANNEL",
        "1"
    )
)

HIKVISION_STREAM_TYPE = int(
    os.getenv(
        "HIKVISION_STREAM_TYPE",
        "1"
    )
)

HIKVISION_CAMERA_ID = os.getenv(
    "HIKVISION_CAMERA_ID",
    "hikvision_camera_01"
)

HIKVISION_CAMERA_LOCATION = os.getenv(
    "HIKVISION_CAMERA_LOCATION",
    "Unknown"
)

HIKVISION_CAMERA_TYPE = os.getenv(
    "HIKVISION_CAMERA_TYPE",
    "hikvision_rtsp"
)

RTSP_RECOGNITION_INTERVAL_SECONDS = float(
    os.getenv(
        "RTSP_RECOGNITION_INTERVAL_SECONDS",
        "1.0"
    )
)
# Function to get the RTSP URL for Hikvision camera
def build_hikvision_rtsp_url() -> str:
    if not HIKVISION_HOST:
        raise ValueError(
            "HIKVISION_HOST is missing from .env"
        )

    if not HIKVISION_USERNAME:
        raise ValueError(
            "HIKVISION_USERNAME is missing from .env"
        )

    if not HIKVISION_PASSWORD:
        raise ValueError(
            "HIKVISION_PASSWORD is missing from .env"
        )

    if HIKVISION_STREAM_TYPE not in {
        1,2,3
    }:
        raise ValueError(
            "HIKVISION_STREAM_TYPE must be 1, 2, or 3"
        )

    encoded_username = quote(
        HIKVISION_USERNAME,
        safe=""
    )

    encoded_password = quote(
        HIKVISION_PASSWORD,
        safe=""
    )

    channel_id = (
        HIKVISION_CHANNEL * 100
        + HIKVISION_STREAM_TYPE
    )

    return (
        f"rtsp://"
        f"{encoded_username}:"
        f"{encoded_password}@"
        f"{HIKVISION_HOST}:"
        f"{HIKVISION_RTSP_PORT}"
        f"/Streaming/Channels/"
        f"{channel_id}"
    )

# Draw bounding box for live camera recognition
def draw_live_camera_results(
    frame: np.ndarray,
    results: list[dict]
) -> np.ndarray:
    annotated_frame = frame.copy()

    for face in results:
        bbox = face.get("bbox")

        if not bbox:
            continue

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        confidence = float(
            face.get(
                "confidence",
                0
            )
        )

        if face.get("recognized"):
            # OpenCV default font may not render
            # Vietnamese names correctly, so use user_id.
            user_id = face.get(
                "user_id",
                "recognized"
            )

            label = (
                f"{user_id} "
                f"{confidence:.2f}"
            )

            color = (
                0,
                255,
                0
            )

        else:
            label = (
                f"Unknown "
                f"{confidence:.2f}"
            )

            color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            annotated_frame,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    cv2.putText(
        annotated_frame,
        "HIKVISION RTSP - LIVE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return annotated_frame

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
if create_client is not None and SUPABASE_URL and SUPABASE_KEY:
    try:
        # Legacy cloud endpoints remain available when configured. Local enrollment
        # never reads this client and remains usable when configuration is absent.
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def open_app():
    return FileResponse("static/index.html")


@app.get("/class-admin")
def class_admin_page():
    return FileResponse("static/class_admin.html")

os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/embeddings", exist_ok=True)

# Store for unknown faces captured during recognition
os.makedirs("storage/unknown_faces", exist_ok=True)

# store video 
os.makedirs("storage/videos", exist_ok=True)

USERS_FILE = "storage/users.json"
ATTENDANCE_FILE = "storage/attendance_logs.json"
ATTENDANCE_CANDIDATE_FILE = "storage/attendance_candidates.json"

RECOGNITION_THRESHOLD = 0.45


BLUR_THRESHOLD = 50
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

ATTENDANCE_COOLDOWN_SECONDS = 60
# face_app = FaceAnalysis(name="buffalo_l")
# face_app.prepare(ctx_id=-1)  # CPU mode
ACTIVE_MODEL_NAME = "buffalo_l" 
shared_detector = get_shared_detector()
from enrollment.api import create_enrollment_router
from enrollment.service import EnrollmentService
from attendance.api import create_attendance_router
from attendance.import_enrollment_users import upsert_enrollment_student
from attendance.repository import AttendanceRepository

local_attendance_repository = AttendanceRepository()
local_enrollment_service = EnrollmentService(
    detector=shared_detector,
    student_sync=lambda **identity: upsert_enrollment_student(
        local_attendance_repository,
        **identity,
    ),
)
app.include_router(create_enrollment_router(local_enrollment_service))
app.include_router(create_attendance_router(local_attendance_repository))
SUPPORTED_DETECTION_MODES = {
    "full",
    "roi",
    "tiled"
}


def detect_faces_by_mode(
    img,
    detection_mode: str = "full"
):
    normalized_mode = (
        detection_mode
        .strip()
        .lower()
    )

    if normalized_mode not in SUPPORTED_DETECTION_MODES:
        raise ValueError(
            "Unsupported detection mode: "
            f"{detection_mode}"
        )

    if normalized_mode == "full":
        return shared_detector.detect_faces(
            img
        )

    if normalized_mode == "roi":
        return shared_detector.detect_faces_in_roi(
            image=img,
            roi_config=CLASSROOM_ROI
        )

    return shared_detector.detect_faces_tiled(
        image=img,
        roi_config=CLASSROOM_ROI,
        tile_count=(
            TILED_DETECTION_CONFIG[
                "tile_count"
            ]
        ),
        overlap_ratio=(
            TILED_DETECTION_CONFIG[
                "overlap_ratio"
            ]
        ),
        nms_iou_threshold=(
            TILED_DETECTION_CONFIG[
                "nms_iou_threshold"
            ]
        )
    )

SUPPORTED_BENCHMARK_MODELS = [
    "buffalo_l",
    "facenet",
    "adaface",
    "arcface_r100"
]
class BenchmarkRequest(BaseModel):
    model: str
    dataset_root: str = "evaluation_dataset"
#Use Pydantic request model helping FastAPI to validate JSON body and auto create schema in Swagger 



# Helper save records
def insert_evaluation_records(
    records: list[dict],
    run_id: str,
    model_name: str,
    threshold: float
):
    rows = []

    for record in records:
        rows.append({
            "run_id": run_id,
            "model_name": model_name,
            "image_name": record["image_name"],
            "image_path": record["image_path"],
            "expected_user_id": (
                record["expected_user_id"]
            ),
            "predicted_user_id": (
                record["predicted_user_id"]
            ),
            "confidence": record["confidence"],
            "threshold": threshold,
            "is_correct": record["is_correct"],
            "processing_time_ms": (
                record["processing_time_ms"]
            ),
            "error_message": (
                record.get("error_message")
            )
        })

    batch_size = 200

    for start in range(0, len(rows), batch_size):
        batch = rows[
            start:start + batch_size
        ]

        supabase.table(
            "evaluation_results"
        ).insert(batch).execute()

# API
@app.post("/run-benchmark")
def run_benchmark_api(
    request: BenchmarkRequest
):
    model_name = request.model.strip().lower()

    supported_models = set(SUPPORTED_BENCHMARK_MODELS)

    if model_name not in supported_models:
        return {
            "success": False,
            "message": (
                f"Unsupported model: {model_name}"
            )
        }

    threshold = get_model_threshold(
        model_name
    )

    run_id = str(uuid4())

    supabase.table(
        "evaluation_runs"
    ).insert({
        "run_id": run_id,
        "model_name": model_name,
        "dataset_name": request.dataset_root,
        "threshold": threshold,
        "status": "running"
    }).execute()

    try:
        cached_users = load_users()

        cached_embeddings = (
            load_cloud_mean_embeddings(
                model_name
            )
        )

        benchmark_result = run_model_benchmark(
            model_name=model_name,
            dataset_root=request.dataset_root,
            recognition_function=(
                recognize_faces_with_model
            ),
            cached_users=cached_users,
            cached_embeddings=cached_embeddings
        )

        records = benchmark_result["records"]
        metrics = benchmark_result["metrics"]

        insert_evaluation_records(
            records=records,
            run_id=run_id,
            model_name=model_name,
            threshold=threshold
        )

        supabase.table(
            "evaluation_runs"
        ).update({
            "total_images": (
                metrics["total_images"]
            ),
            "correct_predictions": (
                metrics["correct_predictions"]
            ),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "false_accept_rate": (
                metrics["false_accept_rate"]
            ),
            "false_reject_rate": (
                metrics["false_reject_rate"]
            ),
            "wrong_identity_rate": (
                metrics["wrong_identity_rate"]
            ),
            "known_identity_accuracy": (
                metrics[
                    "known_identity_accuracy"
                ]
            ),
            "average_processing_time_ms": (
                metrics[
                    "average_processing_time_ms"
                ]
            ),
            "confusion_matrix": {
                "labels": metrics["labels"],
                "matrix": (
                    metrics["confusion_matrix"]
                )
            },
            "classification_report": (
                metrics[
                    "classification_report"
                ]
            ),
            "status": "completed",
            "completed_at": (
                datetime.now().isoformat()
            )
        }).eq(
            "run_id",
            run_id
        ).execute()

        return {
            "success": True,
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics
        }

    except Exception as error:
        supabase.table(
            "evaluation_runs"
        ).update({
            "status": "failed",
            "completed_at": (
                datetime.now().isoformat()
            )
        }).eq(
            "run_id",
            run_id
        ).execute()

        return {
            "success": False,
            "run_id": run_id,
            "model_name": model_name,
            "message": str(error)
        }
    
class RunAllBenchmarksRequest(BaseModel):
    dataset_root: str = "evaluation_dataset"

@app.post("/run-all-benchmarks")
def run_all_benchmarks(
    request: RunAllBenchmarksRequest
):
    results = []

    for model_name in SUPPORTED_BENCHMARK_MODELS:
        benchmark_request = BenchmarkRequest(
            model=model_name,
            dataset_root=request.dataset_root
        )

        result = run_benchmark_api(
            benchmark_request
        )

        results.append(result)

    successful_runs = [
        result
        for result in results
        if result.get("success", False)
    ]

    failed_runs = [
        result
        for result in results
        if not result.get("success", False)
    ]

    return {
        "success": len(failed_runs) == 0,
        "dataset_root": request.dataset_root,
        "total_models": len(results),
        "successful_models": len(successful_runs),
        "failed_models": len(failed_runs), #Know which models failed to run benchmark, to debug and fix the issue
        "results": results
    }

# api get all benchmark runs
@app.get("/evaluation-runs")
def get_evaluation_runs():
    response = (
        supabase
        .table("evaluation_runs")
        .select("*")
        .order("started_at", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "runs": response.data
    }

# api get latest benchmark run by model name
@app.get("/evaluation-latest-comparison")
def get_latest_evaluation_comparison():
    response = (
        supabase
        .table("evaluation_runs")
        .select("*")
        .eq("status", "completed")
        .order("completed_at", desc=True)
        .execute()
    )

    latest_by_model = {}

    for run in response.data:
        model_name = run.get("model_name")

        if not model_name:
            continue

        if model_name not in latest_by_model:
            latest_by_model[model_name] = run

    ordered_models = []

    for model_name in SUPPORTED_BENCHMARK_MODELS:
        if model_name in latest_by_model:
            ordered_models.append(
                latest_by_model[model_name]
            )

    return {
        "success": True,
        "total_models": len(ordered_models),
        "models": ordered_models
    }

# api get benchmark run details by run_id
@app.get("/evaluation-runs/{run_id}")
def get_evaluation_run_detail(run_id: str):
    run_response = (
        supabase
        .table("evaluation_runs")
        .select("*")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )

    if len(run_response.data) == 0:
        return {
            "success": False,
            "message": "Evaluation run not found"
        }

    results_response = (
        supabase
        .table("evaluation_results")
        .select("*")
        .eq("run_id", run_id)
        .order("image_name")
        .execute()
    )

    return {
        "success": True,
        "run": run_response.data[0],
        "total_results": len(results_response.data),
        "results": results_response.data
    }

# api get evaluation page
@app.get("/evaluation-page")
def evaluation_page():
    return FileResponse(
        "static/evaluation.html"
    )

# Create new function to recognize faces by model chosen
def recognize_faces_with_model(
    img,
    model_name: str,
    cached_users=None,
    cached_embeddings=None,
    detection_mode: str = "full"
):
    total_start = time.perf_counter()

    # ==============================
    # 1. Validate input image
    # ==============================
    if img is None:
        return {
            "success": False,
            "message": "Input image is missing.",
            "results": []
        }

    if not isinstance(img, np.ndarray):
        raise TypeError(
            "recognize_faces_with_model expects "
            "img to be a NumPy ndarray, "
            f"but received {type(img).__name__}."
        )

    if img.size == 0:
        return {
            "success": False,
            "message": "Input image is empty.",
            "results": []
        }

    # ==============================
    # 2. Normalize parameters
    # ==============================
    normalized_model_name = (
        model_name
        .strip()
        .lower()
    )

    normalized_detection_mode = (
        detection_mode
        .strip()
        .lower()
    )

    # ==============================
    # 3. Load recognition model
    # ==============================
    model = get_model(
        normalized_model_name
    )

    threshold = get_model_threshold(
        normalized_model_name
    )

    # ==============================
    # 4. Face detection timing
    # ==============================
    detection_start = time.perf_counter()

    faces = detect_faces_by_mode(
        img=img,
        detection_mode=(
            normalized_detection_mode
        )
    )

    detection_time_ms = (
        time.perf_counter()
        - detection_start
    ) * 1000

    # ==============================
    # 5. Face-size statistics
    # ==============================
    face_widths = []
    face_heights = []

    for face in faces:
        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        face_width = max(
            0,
            int(x2 - x1)
        )

        face_height = max(
            0,
            int(y2 - y1)
        )

        face_widths.append(
            face_width
        )

        face_heights.append(
            face_height
        )

    smallest_face_width = (
        min(face_widths)
        if face_widths
        else 0
    )

    smallest_face_height = (
        min(face_heights)
        if face_heights
        else 0
    )

    average_face_width = (
        sum(face_widths)
        / len(face_widths)
        if face_widths
        else 0.0
    )

    average_face_height = (
        sum(face_heights)
        / len(face_heights)
        if face_heights
        else 0.0
    )

    # ==============================
    # 6. No face detected
    # ==============================
    if len(faces) == 0:
        total_processing_time_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "success": True,
            "model_name": (
                normalized_model_name
            ),
            "detection_mode": (
                normalized_detection_mode
            ),
            "threshold": float(
                threshold
            ),
            "total_faces": 0,
            "detection_time_ms": float(
                detection_time_ms
            ),
            "recognition_time_ms": 0.0,
            "processing_time_ms": float(
                total_processing_time_ms
            ),
            "smallest_face_width": 0,
            "smallest_face_height": 0,
            "average_face_width": 0.0,
            "average_face_height": 0.0,
            "results": []
        }

    # ==============================
    # 7. Load registered users
    # ==============================
    users = (
        cached_users
        if cached_users is not None
        else load_users()
    )

    if len(users) == 0:
        total_processing_time_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "success": False,
            "message": (
                "No registered users found."
            ),
            "model_name": (
                normalized_model_name
            ),
            "detection_mode": (
                normalized_detection_mode
            ),
            "threshold": float(
                threshold
            ),
            "total_faces": len(faces),
            "detection_time_ms": float(
                detection_time_ms
            ),
            "recognition_time_ms": 0.0,
            "processing_time_ms": float(
                total_processing_time_ms
            ),
            "smallest_face_width": int(
                smallest_face_width
            ),
            "smallest_face_height": int(
                smallest_face_height
            ),
            "average_face_width": float(
                average_face_width
            ),
            "average_face_height": float(
                average_face_height
            ),
            "results": []
        }

    # ==============================
    # 8. Load mean embeddings
    # ==============================
    cloud_embeddings = (
        cached_embeddings
        if cached_embeddings is not None
        else load_cloud_mean_embeddings(
            normalized_model_name
        )
    )

    if len(cloud_embeddings) == 0:
        total_processing_time_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return {
            "success": False,
            "message": (
                f"No {normalized_model_name} "
                "embeddings found."
            ),
            "model_name": (
                normalized_model_name
            ),
            "detection_mode": (
                normalized_detection_mode
            ),
            "threshold": float(
                threshold
            ),
            "total_faces": len(faces),
            "detection_time_ms": float(
                detection_time_ms
            ),
            "recognition_time_ms": 0.0,
            "processing_time_ms": float(
                total_processing_time_ms
            ),
            "smallest_face_width": int(
                smallest_face_width
            ),
            "smallest_face_height": int(
                smallest_face_height
            ),
            "average_face_width": float(
                average_face_width
            ),
            "average_face_height": float(
                average_face_height
            ),
            "results": []
        }

    # ==============================
    # 9. Recognition timing
    # ==============================
    recognition_start = (
        time.perf_counter()
    )

    results = []

    for face in faces:
        try:
            aligned_face = align_detected_face(
                image=img,
                detected_face=face,
                output_size=112
            )

            current_embedding = (
                model.extract_embedding(
                    aligned_face
                )
            )

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            bbox_data = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }

            face_width = max(
                0,
                int(x2 - x1)
            )

            face_height = max(
                0,
                int(y2 - y1)
            )

            best_match = None
            best_score = -1.0

            for (
                user_id,
                saved_embedding
            ) in cloud_embeddings.items():
                if user_id not in users:
                    continue

                score = model.compare(
                    current_embedding,
                    saved_embedding
                )

                score = float(score)

                if score > best_score:
                    best_score = score
                    best_match = users[
                        user_id
                    ]

            recognized = (
                best_match is not None
                and best_score >= threshold
            )

            detection_score = float(
                getattr(
                    face,
                    "det_score",
                    0.0
                )
            )

            if recognized:
                results.append({
                    "recognized": True,
                    "user_id": best_match[
                        "user_id"
                    ],
                    "full_name": best_match[
                        "full_name"
                    ],
                    "role": best_match[
                        "role"
                    ],
                    "confidence": float(
                        best_score
                    ),
                    "detection_score": (
                        detection_score
                    ),
                    "face_width": (
                        face_width
                    ),
                    "face_height": (
                        face_height
                    ),
                    "bbox": bbox_data
                })

            else:
                results.append({
                    "recognized": False,
                    "message": (
                        "Unknown person"
                    ),
                    "confidence": float(
                        best_score
                    ),
                    "detection_score": (
                        detection_score
                    ),
                    "face_width": (
                        face_width
                    ),
                    "face_height": (
                        face_height
                    ),
                    "bbox": bbox_data
                })

        except Exception as error:
            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            results.append({
                "recognized": False,
                "message": (
                    "Face recognition failed."
                ),
                "error": str(error),
                "confidence": -1.0,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

    recognition_time_ms = (
        time.perf_counter()
        - recognition_start
    ) * 1000

    total_processing_time_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    # ==============================
    # 10. Final response
    # ==============================
    recognized_faces = sum(
        1
        for result in results
        if result.get("recognized")
    )

    unknown_faces = (
        len(results)
        - recognized_faces
    )

    return {
        "success": True,
        "model_name": (
            normalized_model_name
        ),
        "detection_mode": (
            normalized_detection_mode
        ),
        "threshold": float(
            threshold
        ),
        "total_faces": len(results),
        "recognized_faces": (
            recognized_faces
        ),
        "unknown_faces": unknown_faces,
        "detection_time_ms": float(
            detection_time_ms
        ),
        "recognition_time_ms": float(
            recognition_time_ms
        ),
        "processing_time_ms": float(
            total_processing_time_ms
        ),
        "smallest_face_width": int(
            smallest_face_width
        ),
        "smallest_face_height": int(
            smallest_face_height
        ),
        "average_face_width": float(
            average_face_width
        ),
        "average_face_height": float(
            average_face_height
        ),
        "results": results
    }
# Api to test 1 img
@app.post("/test-recognition-model")
async def test_recognition_model(
    model_name: str = Form(...),
    detection_mode: str = Form("full"),
    image: UploadFile = File(...)
):
    normalized_model_name = (
        model_name
        .strip()
        .lower()
    )

    normalized_detection_mode = (
        detection_mode
        .strip()
        .lower()
    )

    if (
        normalized_detection_mode
        not in SUPPORTED_DETECTION_MODES
    ):
        return {
            "success": False,
            "message": (
                "detection_mode must be "
                "full, roi, or tiled."
            ),
            "received_detection_mode": (
                detection_mode
            ),
            "supported_detection_modes": sorted(
                SUPPORTED_DETECTION_MODES
            )
        }

    if not image.filename:
        return {
            "success": False,
            "message": "Image filename is missing."
        }

    image_bytes = await image.read()

    if not image_bytes:
        return {
            "success": False,
            "message": "Uploaded image is empty."
        }

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return {
            "success": False,
            "message": (
                "Could not decode the uploaded image."
            ),
            "filename": image.filename
        }

    result = recognize_faces_with_model(
        img=img,
        model_name=normalized_model_name,
        detection_mode=(
            normalized_detection_mode
        )
    )

    return {
        "success": True,
        "filename": image.filename,
        "image_width": int(
            img.shape[1]
        ),
        "image_height": int(
            img.shape[0]
        ),
        "detection_mode": (
            normalized_detection_mode
        ),
        "model_name": (
            normalized_model_name
        ),
        "result": result
    }

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Load users from local JSON file 
# def load_users():
#     if not os.path.exists(USERS_FILE):
#         return {}

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         return json.load(f)

# Load users from Supabase
def load_users():
    response = supabase.table("users").select("*").execute()

    users = {}

    for user in response.data:
        users[user["user_id"]] = user

    return users

# def save_attendance_log(user_id, full_name, role, confidence, camera_location):
    logs = []

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

    now = datetime.now()

    for log in reversed(logs):
        if log["user_id"] == user_id:
            last_time = datetime.fromisoformat(log["timestamp"])
            diff = (now - last_time).total_seconds()

            if diff < ATTENDANCE_COOLDOWN_SECONDS:
                return False
            break

    logs.append({
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": now.isoformat(),
        "confidence": confidence,
        "camera_location": camera_location,
        "status": "present"
    })

    with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
    return True
def save_attendance_log(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    now = datetime.now()

    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if len(response.data) > 0:
        last_time = datetime.fromisoformat(response.data[0]["timestamp"])
        diff = (now - last_time).total_seconds()

        if diff < ATTENDANCE_COOLDOWN_SECONDS:
            return False

    data = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": now.isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "detected"
    }

    supabase.table("detection_logs").insert(data).execute()

    return True

@app.get("/")
def home():
    return {"message": "Face Registration API is running"}

# old api read all users, file json local
# @app.get("/users")
# def get_users():
#     if not os.path.exists(USERS_FILE):
#         return {
#             "success": True,
#             "total": 0,
#             "users": []
#         }

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         users = json.load(f)

#     return {
#         "success": True,
#         "total": len(users),
#         "users": list(users.values())
#     }

@app.get("/users")
def get_users():
    users = load_users()

    return {
        "success": True,
        "total": len(users),
        "users": list(users.values())
    }

@app.get("/users/{user_id}/registration-status")
def registration_status(
    user_id: str
):
    response = (
        supabase
        .table("face_images")
        .select(
            "pose,image_path,aligned_image_path"
        )
        .eq("user_id", user_id)
        .execute()
    )

    pose_status = {
        "front": False,
        "left": False,
        "right": False
    }

    image_records = {}

    for record in response.data:
        pose = record.get("pose")
        aligned_path = record.get(
            "aligned_image_path"
        )

        if (
            pose in pose_status
            and aligned_path
        ):
            pose_status[pose] = True
            image_records[pose] = record

    return {
        "success": True,
        "user_id": user_id,
        "poses": pose_status,
        "completed_poses": [
            pose
            for pose, exists
            in pose_status.items()
            if exists
        ],
        "missing_poses": [
            pose
            for pose, exists
            in pose_status.items()
            if not exists
        ],
        "registration_complete": all(
            pose_status.values()
        ),
        "images": image_records
    }

# api to check embedding status for a user by user_id and model_name
@app.get("/users/{user_id}/embedding-status")
def embedding_status(
    user_id: str,
    model_name: str = "buffalo_l"
):
    normalized_model_name = (
        model_name.strip().lower()
    )

    response = (
        supabase
        .table("face_embeddings")
        .select(
            "pose,model_name,embedding_path"
        )
        .eq("user_id", user_id)
        .eq(
            "model_name",
            normalized_model_name
        )
        .execute()
    )

    pose_status = {
        "front": False,
        "left": False,
        "right": False,
        "mean": False
    }

    for record in response.data:
        pose = record.get("pose")

        if pose in pose_status:
            pose_status[pose] = True

    return {
        "success": True,
        "user_id": user_id,
        "model_name": normalized_model_name,
        "pose_embeddings": {
            "front": pose_status["front"],
            "left": pose_status["left"],
            "right": pose_status["right"]
        },
        "mean_embedding_exists": (
            pose_status["mean"]
        ),
        "embedding_complete": all(
            pose_status.values()
        )
    }
def calculate_blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score

def calculate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# old function to save user profile at local json file
# def save_user_profile(user_id, full_name, role):
#     users = {}

#     if os.path.exists(USERS_FILE):
#         with open(USERS_FILE, "r", encoding="utf-8") as f:
#             users = json.load(f)

#     users[user_id] = {
#         "user_id": user_id,
#         "full_name": full_name,
#         "role": role,
#         "created_at": datetime.now().isoformat()
#     }

#     with open(USERS_FILE, "w", encoding="utf-8") as f:
#         json.dump(users, f, indent=4, ensure_ascii=False)

def save_user_profile(user_id, full_name, role):
    data = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "created_at": datetime.now().isoformat()
    }

    supabase.table("users").upsert(data, on_conflict="user_id").execute()

def validate_pose(face, pose):
    landmarks = face.kps

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance == 0:
        return False, "Invalid face landmarks"

    nose_offset = (nose[0] - eye_center_x) / eye_distance

    if pose == "front":
        if abs(nose_offset) <= 0.15:
            return True, "Valid front pose"
        return False, "Please look straight at the camera."

    if pose == "left":
        if nose_offset > 0.10:
            return True, "Valid left pose"
        return False, "Please turn your face more to the left."

    if pose == "right":
        if nose_offset < -0.10:
            return True, "Valid right pose"
        return False, "Please turn your face more to the right."

    return True, "Pose not checked"

def validate_face_position(face, img):
    bbox = face.bbox
    x1, y1, x2, y2 = bbox

    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2

    img_height, img_width = img.shape[:2]

    frame_center_x = img_width / 2
    frame_center_y = img_height / 2

    offset_x = abs(face_center_x - frame_center_x) / img_width
    offset_y = abs(face_center_y - frame_center_y) / img_height

    if offset_x > 0.18:
        return False, "Please move your face to the center of the frame."

    if offset_y > 0.20:
        return False, "Please align your face vertically inside the frame."

    return True, "Face position is valid"

# Crop face by bbox 
def crop_face_by_bbox(img, face, margin_ratio=0.25, output_size=(224, 224)):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    img_height, img_width = img.shape[:2]

    face_width = x2 - x1
    face_height = y2 - y1

    margin_x = int(face_width * margin_ratio)
    margin_y = int(face_height * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_width, x2 + margin_x)
    y2 = min(img_height, y2 + margin_y)

    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, output_size)

    return face_crop

# upload crop face img to Supabase
def upload_file_to_supabase(bucket_name, cloud_path, local_path, content_type):
    with open(local_path, "rb") as file:
        supabase.storage.from_(bucket_name).upload(
            path=cloud_path,
            file=file,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )

    return cloud_path

# Save face image metadata to Supabase
def save_face_image_metadata(
    user_id: str,
    pose: str,
    image_path: str,
    aligned_image_path: str
):
    data = {
        "user_id": user_id,
        "pose": pose,
        "image_path": image_path,
        "aligned_image_path": aligned_image_path,
        "aligned_size": 112,
        "created_at": datetime.now().isoformat()
    }

    supabase.table(
        "face_images"
    ).upsert(
        data,
        on_conflict="user_id,pose"
    ).execute()

@app.post("/register-face")
async def register_face(
    user_id: str = Form(...),
    full_name: str = Form(...),
    role: str = Form(...),
    pose: str = Form(...),
    image: UploadFile = File(...)
):
    user_id = user_id.strip()
    full_name = full_name.strip()
    role = role.strip().lower()
    pose = pose.strip().lower()

    valid_poses = {
        "front",
        "left",
        "right"
    }

    if not user_id:
        return {
            "success": False,
            "message": "User ID is required."
        }

    if not full_name:
        return {
            "success": False,
            "message": "Full name is required."
        }

    if pose not in valid_poses:
        return {
            "success": False,
            "message": (
                "Pose must be front, left, or right."
            )
        }

    # Read uploaded image
    image_bytes = await image.read()

    if not image_bytes:
        return {
            "success": False,
            "message": "Uploaded image is empty."
        }

    np_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return {
            "success": False,
            "message": (
                "Invalid or unsupported image file."
            )
        }

    # Image-quality validation
    blur_score = calculate_blur_score(
        img
    )

    if blur_score < BLUR_THRESHOLD:
        return {
            "success": False,
            "message": (
                "Image is too blurry. "
                "Please keep your face steady."
            ),
            "blur_score": float(blur_score)
        }

    brightness = calculate_brightness(
        img
    )

    if brightness < MIN_BRIGHTNESS:
        return {
            "success": False,
            "message": (
                "Image is too dark. "
                "Please move to a brighter place."
            ),
            "brightness": float(brightness)
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "success": False,
            "message": (
                "Image is too bright. "
                "Please reduce strong light."
            ),
            "brightness": float(brightness)
        }

    # Shared face detection
    faces = shared_detector.detect_faces(
        img
    )

    if len(faces) == 0:
        return {
            "success": False,
            "message": "No face detected."
        }

    if len(faces) > 1:
        return {
            "success": False,
            "message": (
                "Multiple faces detected. "
                "Only one person should be visible."
            )
        }

    face = faces[0]

    # Face-size validation
    size_valid, size_message = (
        validate_face_size(
            face,
            img
        )
    )

    if not size_valid:
        return {
            "success": False,
            "message": size_message
        }

    # Position validation
    position_valid, position_message = (
        validate_face_position(
            face,
            img
        )
    )

    if not position_valid:
        return {
            "success": False,
            "message": position_message
        }

    # Pose validation
    pose_valid, pose_message = (
        validate_pose(
            face,
            pose
        )
    )

    if not pose_valid:
        return {
            "success": False,
            "message": pose_message
        }

    # Shared landmark alignment
    try:
        aligned_face = align_detected_face(
            image=img,
            detected_face=face,
            output_size=112
        )
    except Exception as error:
        return {
            "success": False,
            "message": (
                "Face alignment failed: "
                f"{str(error)}"
            )
        }

    if (
        aligned_face is None
        or aligned_face.size == 0
    ):
        return {
            "success": False,
            "message": "Aligned face image is empty."
        }

    # Bbox crop for UI/audit
    face_crop = crop_face_by_bbox(
        img,
        face
    )

    if (
        face_crop is None
        or face_crop.size == 0
    ):
        return {
            "success": False,
            "message": "Failed to crop face image."
        }

    # Local directory
    user_image_dir = os.path.join(
        "storage",
        "images",
        user_id
    )

    os.makedirs(
        user_image_dir,
        exist_ok=True
    )

    crop_image_path = os.path.join(
        user_image_dir,
        f"{pose}.jpg"
    )

    aligned_image_path = os.path.join(
        user_image_dir,
        f"{pose}_aligned.jpg"
    )

    # Save local files
    crop_saved = cv2.imwrite(
        crop_image_path,
        face_crop
    )

    aligned_saved = cv2.imwrite(
        aligned_image_path,
        aligned_face
    )

    if not crop_saved:
        return {
            "success": False,
            "message": (
                "Failed to save cropped face image."
            )
        }

    if not aligned_saved:
        return {
            "success": False,
            "message": (
                "Failed to save aligned face image."
            )
        }

    # Cloud paths
    cloud_crop_path = (
        f"{user_id}/{pose}.jpg"
    )

    cloud_aligned_path = (
        f"{user_id}/{pose}_aligned.jpg"
    )

    # Upload crop
    upload_file_to_supabase(
        bucket_name="face-images",
        cloud_path=cloud_crop_path,
        local_path=crop_image_path,
        content_type="image/jpeg"
    )

    # Upload aligned image
    upload_file_to_supabase(
        bucket_name="face-images",
        cloud_path=cloud_aligned_path,
        local_path=aligned_image_path,
        content_type="image/jpeg"
    )

    # Save image metadata
    save_face_image_metadata(
        user_id=user_id,
        pose=pose,
        image_path=cloud_crop_path,
        aligned_image_path=cloud_aligned_path
    )

    # Save/update profile
    save_user_profile(
        user_id,
        full_name,
        role
    )

    # Determine registration progress
    completed_poses = []
    missing_poses = []

    for required_pose in [
        "front",
        "left",
        "right"
    ]:
        local_aligned_path = os.path.join(
            user_image_dir,
            f"{required_pose}_aligned.jpg"
        )

        if os.path.exists(
            local_aligned_path
        ):
            completed_poses.append(
                required_pose
            )
        else:
            missing_poses.append(
                required_pose
            )

    registration_complete = (
        len(completed_poses) == 3
    )

    return {
        "success": True,
        "message": (
            "Face image registered successfully."
        ),
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "pose": pose,
        "crop_image_path": crop_image_path,
        "aligned_image_path": (
            aligned_image_path
        ),
        "cloud_crop_path": cloud_crop_path,
        "cloud_aligned_path": (
            cloud_aligned_path
        ),
        "aligned_shape": list(
            aligned_face.shape
        ),
        "completed_poses": completed_poses,
        "missing_poses": missing_poses,
        "registration_complete": (
            registration_complete
        ),
        "embedding_created": False,
        "next_step": (
            "Generate model embeddings after "
            "registration is complete."
        )
    }
# Test supabase connection, return one user from "users" table if exists enough to verify connection
@app.get("/test-supabase")
def test_supabase():
    response = supabase.table("users").select("*").limit(1).execute()

    return {
        "success": True,
        "message": "Connected to Supabase successfully",
        "data": response.data
    }

def validate_face_size(face, img):
    bbox = face.bbox
    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    img_height, img_width = img.shape[:2]

    face_area = face_width * face_height
    img_area = img_width * img_height

    face_ratio = face_area / img_area

    if face_ratio < 0.05:
        return False, "Face is too small. Please move closer to the camera."

    return True, "Face size is valid"

# api validate user image quality
@app.post("/validate-face")
async def validate_face_api(
    pose: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    blur_score = calculate_blur_score(img)

    if blur_score < BLUR_THRESHOLD:
        return {
            "success": False,
            "message": "Image is too blurry.",
            "blur_score": blur_score
        }

    brightness = calculate_brightness(img)

    if brightness < MIN_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too dark.",
            "brightness": brightness
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too bright.",
            "brightness": brightness
        }

    # faces = face_app.get(img)
    faces = shared_detector.detect_faces(img)

    if len(faces) == 0:
        return {"success": False, "message": "No face detected."}

    if len(faces) > 1:
        return {"success": False, "message": "Multiple faces detected."}

    face = faces[0]
    try:
        align_detected_face(
            image=img,
            detected_face=face,
            output_size=112
        )
    except Exception:
        return {
            "success": False,
            "message": "Face alignment failed."
        }

    size_valid, size_message = validate_face_size(face, img)
    if not size_valid:
        return {"success": False, "message": size_message}

    position_valid, position_message = validate_face_position(face, img)
    if not position_valid:
        return {"success": False, "message": position_message}

    pose_valid, pose_message = validate_pose(face, pose)
    if not pose_valid:
        return {"success": False, "message": pose_message}

    return {
        "success": True,
        "message": "Face is valid for capture."
    }

# api to serve users.html page
@app.get("/users-page")
def users_page():
    return FileResponse("static/users.html")

# api delete user
# @app.delete("/users/{user_id}")
# def delete_user(user_id: str):
    users = {}

    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)

    if user_id not in users:
        return {
            "success": False,
            "message": "User not found"
        }

    del users[user_id]

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

    image_folder = f"storage/images/{user_id}"
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    for file in os.listdir("storage/embeddings"):
        if file.startswith(user_id):
            os.remove(f"storage/embeddings/{file}")

    return {
        "success": True,
        "message": "User deleted successfully",
        "user_id": user_id
    }

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    # Check user exists in Supabase
    user_response = (
        supabase
        .table("users")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    if len(user_response.data) == 0:
        return {
            "success": False,
            "message": "User not found in cloud database",
            "user_id": user_id
        }

    # Get face image metadata
    image_response = (
        supabase
        .table("face_images")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    #  Delete face image files from Supabase Storage
    for item in image_response.data:
        try:
            supabase.storage.from_("face-images").remove([
                item["image_path"]
            ])
        except Exception as e:
            print(f"Failed to delete image from storage: {e}")

    #  Get embedding metadata
    embedding_response = (
        supabase
        .table("face_embeddings")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    # Delete embedding files from Supabase Storage
    for item in embedding_response.data:
        try:
            supabase.storage.from_("face-embeddings").remove([
                item["embedding_path"]
            ])
        except Exception as e:
            print(f"Failed to delete embedding from storage: {e}")

    #  Delete metadata from cloud tables
    supabase.table("face_images").delete().eq("user_id", user_id).execute()
    supabase.table("face_embeddings").delete().eq("user_id", user_id).execute()
    supabase.table("users").delete().eq("user_id", user_id).execute()

    # Delete local cached images
    image_folder = f"storage/images/{user_id}"

    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    # Delete local cached embeddings
    if os.path.exists("storage/embeddings"):
        for file in os.listdir("storage/embeddings"):
            if file.startswith(user_id):
                file_path = os.path.join("storage/embeddings", file)
                os.remove(file_path)

    return {
        "success": True,
        "message": "User and related cloud data deleted successfully",
        "user_id": user_id
    }



# Save face embedding metadata to Supabase
def save_face_embedding_metadata(user_id, pose, model_name, embedding_path):
    data = {
        "user_id": user_id,
        "pose": pose,
        "model_name": model_name,
        "embedding_path": embedding_path,
        "created_at": datetime.now().isoformat()
    }
    # use upsert to avoid duplicate entries for same user and pose, will update
    supabase.table("face_embeddings").upsert(
        data,
        on_conflict="user_id,pose,model_name"
    ).execute()

# old Api check if user exists by read json local file
# @app.get("/users/{user_id}/exists")
# def check_user_exists(user_id: str):
#     if not os.path.exists(USERS_FILE):
#         return {
#             "success": True,
#             "exists": False
#         }

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         users = json.load(f)

#     return {
#         "success": True,
#         "exists": user_id in users
#     }

# Api check if user exists by query Supabase
@app.get("/users/{user_id}/exists")
def check_user_exists(user_id: str):
    response = (
        supabase
        .table("users")
        .select("user_id")
        .eq("user_id", user_id)
        .execute()
    )

    return {
        "success": True,
        "exists": len(response.data) > 0
    }
# old function to save unknown face at local folder
# def save_unknown_face(img, confidence, camera_location):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{confidence:.3f}.jpg"
    path = f"storage/unknown_faces/{filename}"

    cv2.imwrite(path, img)

    return path

# Save attendance candidate
# def save_attendance_candidate(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    candidates = []

    if os.path.exists(ATTENDANCE_CANDIDATE_FILE):
        with open(ATTENDANCE_CANDIDATE_FILE, "r", encoding="utf-8") as f:
            candidates = json.load(f)

    candidate = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "attendance_candidate",
        "sent_to_school_system": True
    }

    candidates.append(candidate)

    with open(ATTENDANCE_CANDIDATE_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4, ensure_ascii=False)

    return candidate

def save_unknown_face(img, confidence, camera_location):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{confidence:.3f}.jpg"

    local_path = f"storage/unknown_faces/{filename}"
    cloud_path = filename

    cv2.imwrite(local_path, img)

    upload_file_to_supabase(
        bucket_name="unknown-faces",
        cloud_path=cloud_path,
        local_path=local_path,
        content_type="image/jpeg"
    )

    data = {
        "filename": filename,
        "image_path": cloud_path,
        "confidence": confidence,
        "camera_location": camera_location,
        "timestamp": datetime.now().isoformat()
    }

    supabase.table("unknown_faces").insert(data).execute()

    return cloud_path

def save_attendance_candidate(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    candidate = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "attendance_candidate",
        "sent_to_school_system": True
    }

    supabase.table("attendance_candidates").insert(candidate).execute()

    return candidate

# Load embedding from Supabase
def load_embedding_from_cloud(embedding_path):
    file_bytes = supabase.storage.from_("face-embeddings").download(embedding_path)

    embedding = np.load(BytesIO(file_bytes))

    return embedding

# Load mean embeddings for all users from Supabase, return dict of user_id to embedding
def load_cloud_mean_embeddings(model_name: str):
    response = (
        supabase
        .table("face_embeddings")
        .select("user_id, pose, model_name, embedding_path")
        .eq("pose", "mean")
        .eq("model_name", model_name)
        .execute()
    )
    # model_name that guarantee to be the same as the one used for recognition, so we can filter by model_name

    embeddings = {}

    for item in response.data:
        try:
            embeddings[item["user_id"]] = (
                load_embedding_from_cloud(
                    item["embedding_path"]
                )
            )
            print(
                f"Loaded {len(embeddings)} mean embeddings "
                f"for model {model_name}"
            )
        except Exception as error:
            print(
                f"Failed to load {model_name} embedding "
                f"for {item['user_id']}: {error}"
            )

    return embeddings

# new function to recognize faces from frame
# def recognize_faces_from_frame(img):
#     # faces = face_app.get(img)
#     faces = recognition_model.detect_faces(img)

#     if len(faces) == 0:
#         return []

#     users = load_users()
#     if len(users) == 0:
#         return []
#     # put cloud embedding loading here to avoid loading if no faces detected, but only load once for all faces in this image, not load for each face to save time
#     cloud_embeddings = load_cloud_mean_embeddings(recognition_model.model_name)
    

#     results = []

#     for face in faces:
#         current_embedding = recognition_model.extract_embedding(img, face)

#         bbox = face.bbox
#         x1, y1, x2, y2 = bbox.astype(int)

#         bbox_data = {
#             "x1": int(x1),
#             "y1": int(y1),
#             "x2": int(x2),
#             "y2": int(y2)
#         }

#         best_match = None
#         best_score = -1

#         # cloud_embeddings = load_cloud_mean_embeddings()

#         for user_id, user_info in users.items():
#             if user_id not in cloud_embeddings:
#                 continue

#             saved_embedding = cloud_embeddings[user_id]
#             score = recognition_model.compare(
#                 current_embedding,
#                 saved_embedding
#             )
#             # score = cosine_similarity(current_embedding, saved_embedding)

#             if score > best_score:
#                 best_score = score
#                 best_match = user_info

#         if best_match is None:
#             results.append({
#                 "recognized": False,
#                 "message": "No registered embeddings found",
#                 "confidence": best_score,
#                 "bbox": bbox_data
#             })
#             continue

#         if best_score < RECOGNITION_THRESHOLD:
#             results.append({
#                 "recognized": False,
#                 "message": "Unknown person",
#                 "confidence": best_score,
#                 "bbox": bbox_data
#             })
#             continue


#         results.append({
#             "recognized": True,
#             "message": "User recognized",
#             "user_id": best_match["user_id"],
#             "full_name": best_match["full_name"],
#             "role": best_match["role"],
#             "confidence": best_score,
#             "bbox": bbox_data
#         })
#     return results

def recognize_faces_from_frame(img, detection_mode: str = "full"):
    recognition_result = recognize_faces_with_model(
        img=img,
        model_name=ACTIVE_MODEL_NAME,
        detection_mode=detection_mode
    )

    return recognition_result.get("results", [])

# api recognize face for attendance
@app.post("/recognize-face")
async def recognize_face(
    image: UploadFile = File(...),
    camera_id: str = Form("unknown_camera"),
    camera_location: str = Form("Unknown"),
    camera_type: str = Form("unknown")
):
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    results = recognize_faces_from_frame(img, detection_mode="full")

    if len(results) == 0:
        return {
            "success": False,
            "message": "No face detected"
        }

    final_results = []

    for face in results:
        # Case 1: recognized user
        if face.get("recognized"):
            logged = save_attendance_log(
                face["user_id"],
                face["full_name"],
                face["role"],
                face["confidence"],
                camera_id,
                camera_location,
                camera_type
            )

            attendance_candidate = None

            if face["role"] == "student":
                attendance_candidate = save_attendance_candidate(
                    face["user_id"],
                    face["full_name"],
                    face["role"],
                    face["confidence"],
                    camera_id,
                    camera_location,
                    camera_type
                )

            final_results.append({
                "recognized": True,
                "message": "User recognized",
                "user_id": face["user_id"],
                "full_name": face["full_name"],
                "role": face["role"],
                "confidence": face["confidence"],
                "status": "present",
                "logged": logged,
                "camera_id": camera_id,
                "camera_location": camera_location,
                "camera_type": camera_type,
                "bbox": face.get("bbox"),
                "attendance_candidate": attendance_candidate
            })

        # Case 2: unknown / not recognized
        else:
            unknown_path = save_unknown_face(
                img,
                face.get("confidence", -1),
                camera_location
            )

            final_results.append({
                "recognized": False,
                "message": face.get("message", "Unknown person"),
                "confidence": face.get("confidence", -1),
                "camera_id": camera_id,
                "camera_location": camera_location,
                "camera_type": camera_type,
                "unknown_image_path": unknown_path,
                "bbox": face.get("bbox")
            })

    return {
        "success": True,
        "total_faces": len(final_results),
        "results": final_results
    }

# Class RTSP camera Service 
class HikvisionRTSPService:

    def __init__(self):
        self.capture = None
        self.worker_thread = None

        self.running = False
        self.last_error = None

        self.latest_jpeg = None
        self.latest_results = []

        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.cached_users = {}
        self.cached_embeddings = {}

    def reload_recognition_data(self):
        self.cached_users = load_users()

        self.cached_embeddings = (
            load_cloud_mean_embeddings(
                ACTIVE_MODEL_NAME
            )
        )

        return {
            "registered_users": len(
                self.cached_users
            ),
            "mean_embeddings": len(
                self.cached_embeddings
            ),
            "model_name": ACTIVE_MODEL_NAME
        }

    def _open_capture(self):
        rtsp_url = (
            build_hikvision_rtsp_url()
        )

        capture = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        if not capture.isOpened():
            capture.release()

            # Fallback to OpenCV default backend
            capture = cv2.VideoCapture(
                rtsp_url
            )

        if not capture.isOpened():
            capture.release()
            return None

        # This may reduce latency when supported
        # by the selected OpenCV backend.
        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        return capture

    def start(self):
        with self.state_lock:
            if self.running:
                return {
                    "success": True,
                    "message": (
                        "Hikvision camera is "
                        "already running."
                    )
                }

            try:
                capture = self._open_capture()

                if capture is None:
                    self.last_error = (
                        "Could not open Hikvision "
                        "RTSP stream."
                    )

                    return {
                        "success": False,
                        "message": self.last_error
                    }

                cache_status = (
                    self.reload_recognition_data()
                )

                self.capture = capture
                self.running = True
                self.last_error = None
                self.latest_results = []

                self.worker_thread = (
                    threading.Thread(
                        target=self._camera_loop,
                        daemon=True
                    )
                )

                self.worker_thread.start()

                return {
                    "success": True,
                    "message": (
                        "Hikvision RTSP camera "
                        "started successfully."
                    ),
                    "camera_id": (
                        HIKVISION_CAMERA_ID
                    ),
                    "camera_location": (
                        HIKVISION_CAMERA_LOCATION
                    ),
                    "recognition_cache": (
                        cache_status
                    )
                }

            except Exception as error:
                self.running = False
                self.last_error = str(error)

                if self.capture is not None:
                    self.capture.release()
                    self.capture = None

                return {
                    "success": False,
                    "message": str(error)
                }

    def stop(self):
        with self.state_lock:
            self.running = False

            if self.capture is not None:
                self.capture.release()
                self.capture = None

            with self.frame_lock:
                self.latest_jpeg = None
                self.latest_results = []

            return {
                "success": True,
                "message": (
                    "Hikvision RTSP camera stopped."
                )
            }

    def _process_attendance(
        self,
        results: list[dict]
    ):
        for face in results:
            if not face.get("recognized"):
                continue

            logged = save_attendance_log(
                user_id=face["user_id"],
                full_name=face["full_name"],
                role=face["role"],
                confidence=face["confidence"],
                camera_id=HIKVISION_CAMERA_ID,
                camera_location=(
                    HIKVISION_CAMERA_LOCATION
                ),
                camera_type=(
                    HIKVISION_CAMERA_TYPE
                )
            )

            # Only create an attendance candidate
            # when a new detection log was accepted.
            if (
                logged
                and face["role"] == "student"
            ):
                save_attendance_candidate(
                    user_id=face["user_id"],
                    full_name=face["full_name"],
                    role=face["role"],
                    confidence=face["confidence"],
                    camera_id=(
                        HIKVISION_CAMERA_ID
                    ),
                    camera_location=(
                        HIKVISION_CAMERA_LOCATION
                    ),
                    camera_type=(
                        HIKVISION_CAMERA_TYPE
                    )
                )

    def _camera_loop(self):
        last_recognition_time = 0.0
        consecutive_read_failures = 0

        try:
            while self.running:
                if self.capture is None:
                    self.last_error = (
                        "RTSP capture is missing."
                    )
                    break

                success, frame = (
                    self.capture.read()
                )

                if not success or frame is None:
                    consecutive_read_failures += 1

                    if (
                        consecutive_read_failures
                        >= 30
                    ):
                        self.last_error = (
                            "Lost connection to "
                            "Hikvision RTSP stream."
                        )
                        break

                    time.sleep(0.1)
                    continue

                consecutive_read_failures = 0

                current_time = (
                    time.monotonic()
                )

                should_recognize = (
                    current_time
                    - last_recognition_time
                    >= RTSP_RECOGNITION_INTERVAL_SECONDS
                )

                if should_recognize:
                    try:
                        recognition_result = (
                            recognize_faces_with_model(
                                img=frame,
                                model_name=(
                                    ACTIVE_MODEL_NAME
                                ),
                                cached_users=(
                                    self.cached_users
                                ),
                                cached_embeddings=(
                                    self.cached_embeddings
                                )
                            )
                        )

                        self.latest_results = (
                            recognition_result.get(
                                "results",
                                []
                            )
                        )

                        self._process_attendance(
                            self.latest_results
                        )

                    except Exception as error:
                        self.last_error = (
                            "Recognition error: "
                            f"{error}"
                        )

                    last_recognition_time = (
                        current_time
                    )

                display_frame = (
                    draw_live_camera_results(
                        frame,
                        self.latest_results
                    )
                )

                height, width = (
                    display_frame.shape[:2]
                )

                max_display_width = 1280

                if width > max_display_width:
                    scale = (
                        max_display_width
                        / width
                    )

                    display_frame = cv2.resize(
                        display_frame,
                        (
                            max_display_width,
                            int(height * scale)
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        )
                    )

                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    display_frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        80
                    ]
                )

                if not encoded:
                    continue

                with self.frame_lock:
                    self.latest_jpeg = (
                        jpeg.tobytes()
                    )

        finally:
            self.running = False

            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def mjpeg_generator(self):
        while self.running:
            with self.frame_lock:
                jpeg = self.latest_jpeg

            if jpeg is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

            time.sleep(0.03)

    def get_status(self):
        return {
            "success": True,
            "running": self.running,
            "camera_id": (
                HIKVISION_CAMERA_ID
            ),
            "camera_location": (
                HIKVISION_CAMERA_LOCATION
            ),
            "camera_type": (
                HIKVISION_CAMERA_TYPE
            ),
            "model_name": ACTIVE_MODEL_NAME,
            "recognition_interval_seconds": (
                RTSP_RECOGNITION_INTERVAL_SECONDS
            ),
            "cached_users": len(
                self.cached_users
            ),
            "cached_embeddings": len(
                self.cached_embeddings
            ),
            "last_error": self.last_error
        }
    
# Camera API
hikvision_camera = HikvisionRTSPService()
@app.post("/camera/start")
def start_hikvision_camera():
    return hikvision_camera.start()


@app.post("/camera/stop")
def stop_hikvision_camera():
    return hikvision_camera.stop()


@app.get("/camera/status")
def get_hikvision_camera_status():
    return hikvision_camera.get_status()


@app.post("/camera/reload-recognition-data")
def reload_camera_recognition_data():
    try:
        cache_status = (
            hikvision_camera
            .reload_recognition_data()
        )

        return {
            "success": True,
            "message": (
                "Recognition data reloaded."
            ),
            **cache_status
        }

    except Exception as error:
        return {
            "success": False,
            "message": str(error)
        }


@app.get("/camera/feed")
def hikvision_camera_feed():
    if not hikvision_camera.running:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": (
                    "Camera is not running. "
                    "Call POST /camera/start first."
                )
            }
        )

    return StreamingResponse(
        hikvision_camera.mjpeg_generator(),
        media_type=(
            "multipart/x-mixed-replace;"
            "boundary=frame"
        ),
        headers={
            "Cache-Control": (
                "no-store, no-cache, "
                "must-revalidate"
            ),
            "Pragma": "no-cache"
        }
    )


# api view unknown faces
@app.get("/unknown-faces")
def get_unknown_faces():
    response = (
        supabase
        .table("unknown_faces")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    files = []

    for item in response.data:
        public_url = supabase.storage.from_("unknown-faces").get_public_url(item["image_path"])

        files.append({
            "filename": item["filename"],
            "path": item["image_path"],
            "public_url": public_url,
            "confidence": item["confidence"],
            "camera_location": item["camera_location"],
            "timestamp": item["timestamp"]
        })

    return {
        "success": True,
        "total": len(files),
        "files": files
    }
    folder = "storage/unknown_faces"

    if not os.path.exists(folder):
        return {
            "success": True,
            "total": 0,
            "files": []
        }

    files = []

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append({
                "filename": file,
                "path": f"{folder}/{file}"
            })

    return {
        "success": True,
        "total": len(files),
        "files": files
    }


# old api view attendance logs from local json file
# @app.get("/attendance-logs")
# def get_attendance_logs():
    if not os.path.exists(ATTENDANCE_FILE):
        return {
            "success": True,
            "total": 0,
            "logs": []
        }

    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    return {
        "success": True,
        "total": len(logs),
        "logs": logs
    }

@app.get("/attendance-logs")
def get_attendance_logs():
    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "logs": response.data
    }

# api attendance page
@app.get("/attendance-page")
def attendance_page():
    return FileResponse("static/attendance.html")

# api dashboard page
@app.get("/dashboard-summary")
def dashboard_summary():
    users = load_users()

    today = datetime.now().date().isoformat()

    logs_response = (
        supabase
        .table("detection_logs")
        .select("*")
        .gte("timestamp", today + "T00:00:00")
        .lte("timestamp", today + "T23:59:59")
        .execute()
    )

    unknown_response = (
        supabase
        .table("unknown_faces")
        .select("*")
        .execute()
    )

    unique_present_users = set(
        log["user_id"] for log in logs_response.data
        if log.get("user_id")
    )

    return {
        "success": True,
        "date": today,
        "registered_users_count": len(users),
        "today_total_present": len(unique_present_users),
        "unknown_detections": len(unknown_response.data),
        "latest_attendance_logs": logs_response.data[-5:][::-1]
    }
# export attendance logs as CSV
@app.get("/attendance-logs/export")
def export_attendance_logs():
    export_path = "storage/attendance_logs.csv"

    # read logs local json file
    # logs = []

    # if os.path.exists(ATTENDANCE_FILE):
    #     with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
    #         logs = json.load(f)

    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    logs = response.data
    with open(export_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "user_id",
            "full_name",
            "role",
            "timestamp",
            "camera_location",
            "confidence",
            "status"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            writer.writerow({
                "user_id": log.get("user_id", ""),
                "full_name": log.get("full_name", ""),
                "role": log.get("role", ""),
                "timestamp": log.get("timestamp", ""),
                "camera_location": log.get("camera_location", ""),
                "confidence": log.get("confidence", ""),
                "status": log.get("status", "")
            })

    return FileResponse(
        export_path,
        media_type="text/csv",
        filename="attendance_logs.csv"
    )

# api clear attendance logs
@app.delete("/attendance-logs")
def clear_attendance_logs():
    # fiter by id greater than 0 to avoid deleting any potential system metadata if exists with id 0, but in general should be safe to delete all logs
    supabase.table("detection_logs").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Detection logs cleared successfully."
    }

# api clear unknown faces
@app.delete("/unknown-faces")
def clear_unknown_faces():
    response = supabase.table("unknown_faces").select("*").execute()

    for item in response.data:
        try:
            supabase.storage.from_("unknown-faces").remove([item["image_path"]])
        except Exception as e:
            print(f"Failed to delete unknown face file: {e}")

    supabase.table("unknown_faces").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Unknown faces cleared successfully"
    }

# API view attendance candidates
@app.get("/attendance-candidates")
def get_attendance_candidates():
    response = (
        supabase
        .table("attendance_candidates")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "candidates": response.data
    }

# api clear attendance logs
@app.delete("/attendance-candidates")
def clear_attendance_candidates():
    supabase.table("attendance_candidates").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Attendance candidates cleared successfully."
    }

# api check cloud health
@app.get("/cloud-health")
def cloud_health():
    health = {
        "success": True,
        "database": {},
        "storage": {}
    }

    tables = [
        "users",
        "face_images",
        "face_embeddings",
        "detection_logs",
        "attendance_candidates",
        "unknown_faces"
    ]

    for table in tables:
        try:
            response = (
                supabase
                .table(table)
                .select("*")
                .limit(1)
                .execute()
            )

            health["database"][table] = {
                "status": "ok",
                "sample_count": len(response.data)
            }

        except Exception as e:
            health["success"] = False
            health["database"][table] = {
                "status": "error",
                "message": str(e)
            }

    buckets = [
        "face-images",
        "face-embeddings",
        "unknown-faces",
        "exports"
    ]

    for bucket in buckets:
        try:
            response = supabase.storage.from_(bucket).list()

            health["storage"][bucket] = {
                "status": "ok",
                "items_count": len(response)
            }

        except Exception as e:
            health["success"] = False
            health["storage"][bucket] = {
                "status": "error",
                "message": str(e)
            }

    return health

def insert_video_session(
    session_id: str,
    class_name: str,
    camera_id: str,
    camera_location: str,
    video_name: str,
    sample_interval_seconds: float
):
    session_data = {
        "session_id": session_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "video_name": video_name,
        "duration_seconds": 0.0,
        "total_windows": 0,
        "sample_interval_seconds": float(
            sample_interval_seconds
        )
    }

    print("[VIDEO SESSION INSERT REQUEST]")
    print(session_data)

    response = (
        supabase
        .table("video_sessions")
        .insert(session_data)
        .execute()
    )

    print(
        "[VIDEO SESSION INSERT RESPONSE]",
        response.data
    )

    if not response.data:
        raise RuntimeError(
            "Video session was not inserted into video_sessions."
        )

    return response.data[0]

def update_video_session_result(
    session_id: str,
    duration_seconds: float,
    total_windows: int
):
    update_data = {
        "duration_seconds": float(
            duration_seconds
        ),
        "total_windows": int(
            total_windows
        )
    }

    print("[VIDEO SESSION UPDATE REQUEST]")
    print("session_id:", session_id)
    print("update_data:", update_data)

    response = (
        supabase
        .table("video_sessions")
        .update(update_data)
        .eq("session_id", session_id)
        .execute()
    )

    print(
        "[VIDEO SESSION UPDATE RESPONSE]",
        response.data
    )

    if not response.data:
        raise RuntimeError(
            "Video session was not updated."
        )

    return response.data[0]
# ------------------------------------------------
# api create video session
@app.post("/video-sessions")
def create_video_session(
    class_name: str = Form(...),
    camera_id: str = Form(...),
    camera_location: str = Form(...),
    video_name: str = Form(...),
    duration_seconds: float = Form(0),
    total_windows: int = Form(0),
    sample_interval_seconds: float = Form(5)
):
    session_id = str(uuid.uuid4())

    data = {
        "session_id": session_id,
        "class_name": class_name.strip(),
        "camera_id": camera_id.strip(),
        "camera_location": camera_location.strip(),
        "video_name": video_name.strip(),
        "duration_seconds": float(duration_seconds),
        "total_windows": int(total_windows),
        "sample_interval_seconds": float(
            sample_interval_seconds
        )
    }

    response = (
        supabase
        .table("video_sessions")
        .insert(data)
        .execute()
    )

    if not response.data:
        return {
            "success": False,
            "message": "Video session was not created."
        }

    return {
        "success": True,
        "message": "Video session created successfully.",
        "session": response.data[0]
    }
# api view video sessions
@app.get("/video-sessions")
def get_video_sessions():
    response = (
        supabase
        .table("video_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "sessions": response.data
    }
# api process video upload and recognize faces
@app.post("/process-video")
async def process_video(

    class_name: str = Form(...),
    camera_id: str = Form(...),
    camera_location: str = Form(...),
    sample_interval_seconds: float = Form(5),
    detection_mode: str = Form("tiled"),
    video: UploadFile = File(...)
):
    # =========================================
    # 1. Normalize inputs
    # =========================================
    class_name = class_name.strip()
    camera_id = camera_id.strip()
    camera_location = camera_location.strip()

    normalized_detection_mode = (
        detection_mode
        .strip()
        .lower()
    )

    # =========================================
    # 2. Validate inputs
    # =========================================
    if not class_name:
        return {
            "success": False,
            "message": "class_name is required."
        }

    if not camera_id:
        return {
            "success": False,
            "message": "camera_id is required."
        }

    if not camera_location:
        return {
            "success": False,
            "message": "camera_location is required."
        }

    if (
        normalized_detection_mode
        not in SUPPORTED_DETECTION_MODES
    ):
        return {
            "success": False,
            "message": (
                "detection_mode must be "
                "full, roi, or tiled."
            ),
            "received_detection_mode": detection_mode,
            "supported_detection_modes": sorted(
                SUPPORTED_DETECTION_MODES
            )
        }

    if sample_interval_seconds <= 0:
        return {
            "success": False,
            "message": (
                "sample_interval_seconds "
                "must be greater than 0."
            )
        }

    if not video.filename:
        return {
            "success": False,
            "message": "Video filename is missing."
        }

    allowed_video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    safe_video_name = os.path.basename(
        video.filename
    )

    video_extension = os.path.splitext(
        safe_video_name
    )[1].lower()

    if (
        video_extension
        not in allowed_video_extensions
    ):
        return {
            "success": False,
            "message": "Unsupported video format.",
            "allowed_extensions": sorted(
                allowed_video_extensions
            )
        }

    # =========================================
    # 3. Generate session ID in backend
    # =========================================
    session_id = str(
        uuid.uuid4()
    )

    print(
        "[NEW VIDEO SESSION]",
        session_id
    )

    # =========================================
    # 4. Save uploaded video locally
    # =========================================
    os.makedirs(
        "storage/videos",
        exist_ok=True
    )

    video_path = os.path.join(
        "storage",
        "videos",
        f"{session_id}_{safe_video_name}"
    )

    video_bytes = await video.read()

    if not video_bytes:
        return {
            "success": False,
            "message": "Uploaded video is empty.",
            "session_id": session_id
        }

    try:
        with open(video_path, "wb") as file:
            file.write(video_bytes)

    except OSError as error:
        return {
            "success": False,
            "message": (
                "Failed to save uploaded video."
            ),
            "session_id": session_id,
            "error": str(error)
        }

    # =========================================
    # 5. Open video and read metadata
    # =========================================
    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return {
            "success": False,
            "message": "Failed to open video file.",
            "session_id": session_id,
            "video_path": video_path
        }

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        cap.release()

        return {
            "success": False,
            "message": "Invalid video FPS.",
            "session_id": session_id
        }

    duration_seconds = (
        frame_count / fps
    )

    frame_step = max(
        1,
        int(
            fps
            * sample_interval_seconds
        )
    )

    # =========================================
    # 6. Insert parent session before children
    # =========================================
    try:
        created_session = insert_video_session(
            session_id=session_id,
            class_name=class_name,
            camera_id=camera_id,
            camera_location=camera_location,
            video_name=safe_video_name,
            sample_interval_seconds=(
                sample_interval_seconds
            )
        )

    except Exception as error:
        cap.release()

        return {
            "success": False,
            "message": (
                "Failed to create video session."
            ),
            "session_id": session_id,
            "error": str(error)
        }

    total_windows = 0
    frame_index = 0
    total_detected_faces = 0
    total_recognized_faces = 0
    total_unknown_faces = 0
    evidence_save_failures = 0
    detection_log_failures = 0

    # =========================================
    # 7. Process sampled frames
    # =========================================
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if (
                frame_index
                % frame_step
                == 0
            ):
                total_windows += 1

                frame_time_seconds = (
                    frame_index / fps
                )

                results = (
                    recognize_faces_from_frame(
                        img=frame,
                        detection_mode=(
                            normalized_detection_mode
                        )
                    )
                )

                total_detected_faces += len(
                    results
                )

                recognized_count = sum(
                    1
                    for face in results
                    if face.get("recognized")
                )

                unknown_count = (
                    len(results)
                    - recognized_count
                )

                total_recognized_faces += (
                    recognized_count
                )

                total_unknown_faces += (
                    unknown_count
                )

                # Evidence is optional.
                # Failure must not terminate video processing.
                if results:
                    try:
                        save_video_evidence_frame(
                            session_id=session_id,
                            class_name=class_name,
                            camera_id=camera_id,
                            frame_time_seconds=(
                                frame_time_seconds
                            ),
                            frame=frame,
                            results=results
                        )

                    except Exception as error:
                        evidence_save_failures += 1

                        print(
                            "[WARNING] Evidence frame "
                            "save failed."
                        )

                        print(
                            "session_id:",
                            session_id
                        )

                        print(
                            "frame_time_seconds:",
                            frame_time_seconds
                        )

                        print(
                            "error:",
                            str(error)
                        )

                # Save one detection log per face.
                for face in results:
                    log_data = {
                        "session_id": session_id,
                        "class_name": class_name,
                        "camera_id": camera_id,
                        "camera_location": (
                            camera_location
                        ),
                        "frame_time_seconds": (
                            float(
                                frame_time_seconds
                            )
                        ),
                        "user_id": face.get(
                            "user_id"
                        ),
                        "full_name": face.get(
                            "full_name"
                        ),
                        "confidence": (
                            float(
                                face.get(
                                    "confidence"
                                )
                            )
                            if face.get(
                                "confidence"
                            ) is not None
                            else None
                        ),
                        "status": (
                            "recognized"
                            if face.get(
                                "recognized"
                            )
                            else "unknown"
                        )
                    }

                    try:
                        (
                            supabase
                            .table(
                                "video_detection_logs"
                            )
                            .insert(
                                log_data
                            )
                            .execute()
                        )

                    except Exception as error:
                        detection_log_failures += 1

                        print(
                            "[WARNING] Detection log "
                            "insert failed."
                        )

                        print(
                            "session_id:",
                            session_id
                        )

                        print(
                            "error:",
                            str(error)
                        )

            frame_index += 1

    except Exception as error:
        print(
            "[PROCESS VIDEO ERROR]",
            str(error)
        )

        return {
            "success": False,
            "message": (
                "Video processing failed."
            ),
            "session_id": session_id,
            "error": str(error),
            "total_windows_processed": (
                total_windows
            )
        }

    finally:
        cap.release()

    # =========================================
    # 8. Update session after processing
    # =========================================
    try:
        updated_session = (
            update_video_session_result(
                session_id=session_id,
                duration_seconds=(
                    duration_seconds
                ),
                total_windows=(
                    total_windows
                )
            )
        )

    except Exception as error:
        return {
            "success": False,
            "message": (
                "Video was processed, but the "
                "video session could not be updated."
            ),
            "session_id": session_id,
            "duration_seconds": (
                duration_seconds
            ),
            "total_windows": total_windows,
            "error": str(error)
        }

    # =========================================
    # 9. Generate attendance summary
    # =========================================
    try:
        summary_result = (
            generate_video_attendance_summary(
                session_id
            )
        )

    except Exception as error:
        print(
            "[WARNING] Attendance summary "
            "generation failed:",
            str(error)
        )

        summary_result = {
            "success": False,
            "message": (
                "Attendance summary could not "
                "be generated."
            ),
            "error": str(error)
        }

    # =========================================
    # 10. Final response
    # =========================================
    return {
        "success": True,
        "message": (
            "Video processed successfully."
        ),
        "session_id": session_id,
        "video_name": safe_video_name,
        "detection_mode": (
            normalized_detection_mode
        ),
        "duration_seconds": float(
            duration_seconds
        ),
        "frame_count": int(
            frame_count
        ),
        "fps": float(
            fps
        ),
        "frame_step": int(
            frame_step
        ),
        "sample_interval_seconds": float(
            sample_interval_seconds
        ),
        "total_windows": int(
            total_windows
        ),
        "total_detected_faces": int(
            total_detected_faces
        ),
        "total_recognized_faces": int(
            total_recognized_faces
        ),
        "total_unknown_faces": int(
            total_unknown_faces
        ),
        "evidence_save_failures": int(
            evidence_save_failures
        ),
        "detection_log_failures": int(
            detection_log_failures
        ),
        "created_session": created_session,
        "updated_session": updated_session,
        "summary_result": summary_result
    }

def generate_video_attendance_summary(session_id):
    session_response = (
        supabase
        .table("video_sessions")
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )

    if len(session_response.data) == 0:
        return {
            "success": False,
            "message": "Video session not found"
        }

    session = session_response.data[0]
    total_windows = session.get("total_windows", 0)
    class_name = session.get("class_name")

    if total_windows == 0:
        return {
            "success": False,
            "message": "Total windows is 0"
        }

    logs_response = (
        supabase
        .table("video_detection_logs")
        .select("*")
        .eq("session_id", session_id)
        .eq("status", "recognized")
        .execute()
    )

    logs = logs_response.data

    student_stats = {}

    for log in logs:
        user_id = log.get("user_id")

        if not user_id:
            continue

        if user_id not in student_stats:
            student_stats[user_id] = {
                "user_id": user_id,
                "full_name": log.get("full_name"),
                "detected_windows": set(),
                "last_seen_seconds": 0
            }

        frame_time = log.get("frame_time_seconds")

        student_stats[user_id]["detected_windows"].add(frame_time)

        if frame_time > student_stats[user_id]["last_seen_seconds"]:
            student_stats[user_id]["last_seen_seconds"] = frame_time

    summaries = []

    for user_id, stat in student_stats.items():
        detected_windows = len(stat["detected_windows"])
        presence_ratio = detected_windows / total_windows

        suggested_status = "present" if presence_ratio >= 0.5 else "absent"

        leave_status = "stayed"

        if stat["last_seen_seconds"] < session["duration_seconds"] - 60:
            leave_status = "possible_left_classroom"

        summary = {
            "session_id": session_id,
            "class_name": class_name,
            "user_id": user_id,
            "full_name": stat["full_name"],
            "detected_windows": detected_windows,
            "total_windows": total_windows,
            "presence_ratio": presence_ratio,
            "suggested_status": suggested_status,
            "last_seen_seconds": stat["last_seen_seconds"],
            "leave_status": leave_status
        }

        summaries.append(summary)

    supabase.table("video_attendance_summary").delete().eq(
        "session_id",
        session_id
    ).execute()

    if len(summaries) > 0:
        supabase.table("video_attendance_summary").insert(summaries).execute()

    return {
        "success": True,
        "session_id": session_id,
        "total_students": len(summaries),
        "summaries": summaries
    }


@app.post("/video-summary/{session_id}/generate")
def generate_video_summary_api(session_id: str):
    return generate_video_attendance_summary(session_id)

@app.get("/video-summary/{session_id}")
def get_video_summary(session_id: str):

    response = (
        supabase
        .table("video_attendance_summary")
        .select("*")
        .eq("session_id", session_id)
        .order("presence_ratio", desc=True)
        .execute()
    )

    return {
        "success": True,
        "session_id": session_id,
        "total": len(response.data),
        "summaries": response.data
    }

@app.get("/video-analysis-page")
def video_analysis_page():
    return FileResponse("static/video_analysis.html")
@app.get("/video-summary-page")
def video_summary_page():
    return FileResponse("static/video_summary.html")
@app.get("/video-evidence-page")
def video_evidence_page():
    return FileResponse("static/video_evidence.html")

# Draw bounding boxes and labels on video frames for evidence
def draw_video_evidence_frame(frame, results):
    annotated = frame.copy()

    for face in results:
        bbox = face.get("bbox")

        if not bbox:
            continue

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        if face.get("recognized"):
            label = f'{face.get("full_name")} {face.get("confidence", 0):.2f}'
            color = (0, 255, 0)
        else:
            label = f'Unknown {face.get("confidence", 0):.2f}'
            color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated

def save_video_evidence_frame(session_id, class_name, camera_id, frame_time_seconds, frame, results):
    os.makedirs("storage/video_evidence_frames", exist_ok=True)

    filename = f"{session_id}_{int(frame_time_seconds)}.jpg"
    local_path = f"storage/video_evidence_frames/{filename}"
    cloud_path = f"{session_id}/{filename}"

    annotated_frame = draw_video_evidence_frame(frame, results)

    cv2.imwrite(local_path, annotated_frame)

    upload_file_to_supabase(
        bucket_name="video-evidence-frames",
        cloud_path=cloud_path,
        local_path=local_path,
        content_type="image/jpeg"
    )

    data = {
        "session_id": session_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "frame_time_seconds": frame_time_seconds,
        "image_path": cloud_path,
        "detected_count": len(results)
    }

    supabase.table("video_evidence_frames").insert(data).execute()

    return cloud_path

@app.get("/video-evidence/{session_id}")
def get_video_evidence_frames(session_id: str):
    response = (
        supabase
        .table("video_evidence_frames")
        .select("*")
        .eq("session_id", session_id)
        .order("frame_time_seconds")
        .execute()
    )

    frames = []

    for item in response.data:
        public_url = supabase.storage.from_("video-evidence-frames").get_public_url(
            item["image_path"]
        )

        frames.append({
            "id": item["id"],
            "session_id": item["session_id"],
            "frame_time_seconds": item["frame_time_seconds"],
            "detected_count": item["detected_count"],
            "image_path": item["image_path"],
            "public_url": public_url
        })

    return {
        "success": True,
        "session_id": session_id,
        "total": len(frames),
        "frames": frames
    }

@app.get("/video-sessions")
def get_video_sessions():

    response = (
        supabase
        .table("video_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return {
        "success": True,
        "sessions": response.data
    }
