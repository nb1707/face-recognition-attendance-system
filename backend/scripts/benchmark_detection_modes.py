import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(
        0,
        BACKEND_DIR
    )


from recognition.face_detector import (
    get_shared_detector
)

from recognition.detection_config import (
    CLASSROOM_ROI,
    TILED_DETECTION_CONFIG
)


IMAGE_PATH = os.path.join(
    BACKEND_DIR,
    "test_classroom.png"
)

OUTPUT_DIR = os.path.join(
    BACKEND_DIR,
    "storage",
    "benchmarks"
)

REPEAT_COUNT = 5


def collect_face_sizes(
    faces
):
    widths = []
    heights = []

    for face in faces:
        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        width = max(
            0,
            int(x2 - x1)
        )

        height = max(
            0,
            int(y2 - y1)
        )

        widths.append(width)
        heights.append(height)

    return widths, heights


def measure_mode(
    mode_name,
    detect_function
):
    face_counts = []
    times_ms = []
    smallest_widths = []
    smallest_heights = []
    average_widths = []
    average_heights = []
    run_details = []

    print()
    print(
        f"Warm-up mode: {mode_name}"
    )

    warmup_faces = detect_function()

    print(
        f"Warm-up faces: {len(warmup_faces)}"
    )

    for run_index in range(
        1,
        REPEAT_COUNT + 1
    ):
        started_at = (
            time.perf_counter()
        )

        faces = detect_function()

        elapsed_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        widths, heights = (
            collect_face_sizes(
                faces
            )
        )

        face_count = len(faces)

        smallest_width = (
            min(widths)
            if widths
            else 0
        )

        smallest_height = (
            min(heights)
            if heights
            else 0
        )

        average_width = (
            float(np.mean(widths))
            if widths
            else 0.0
        )

        average_height = (
            float(np.mean(heights))
            if heights
            else 0.0
        )

        face_counts.append(
            face_count
        )

        times_ms.append(
            elapsed_ms
        )

        smallest_widths.append(
            smallest_width
        )

        smallest_heights.append(
            smallest_height
        )

        average_widths.append(
            average_width
        )

        average_heights.append(
            average_height
        )

        run_details.append({
            "mode": mode_name,
            "run": run_index,
            "faces": face_count,
            "time_ms": elapsed_ms,
            "smallest_face_width": (
                smallest_width
            ),
            "smallest_face_height": (
                smallest_height
            ),
            "average_face_width": (
                average_width
            ),
            "average_face_height": (
                average_height
            )
        })

        print(
            f"{mode_name:<8} "
            f"run={run_index} "
            f"faces={face_count} "
            f"time={elapsed_ms:.2f} ms "
            f"min_face="
            f"{smallest_width}x"
            f"{smallest_height}"
        )

    summary = {
        "mode": mode_name,
        "faces_mean": float(
            np.mean(face_counts)
        ),
        "faces_min": int(
            np.min(face_counts)
        ),
        "faces_max": int(
            np.max(face_counts)
        ),
        "avg_time_ms": float(
            np.mean(times_ms)
        ),
        "std_time_ms": float(
            np.std(times_ms)
        ),
        "min_time_ms": float(
            np.min(times_ms)
        ),
        "max_time_ms": float(
            np.max(times_ms)
        ),
        "smallest_face_width": int(
            min(smallest_widths)
        ),
        "smallest_face_height": int(
            min(smallest_heights)
        ),
        "average_face_width": float(
            np.mean(average_widths)
        ),
        "average_face_height": float(
            np.mean(average_heights)
        )
    }

    return summary, run_details


def save_summary_csv(
    results,
    output_path
):
    fieldnames = [
        "mode",
        "faces_mean",
        "faces_min",
        "faces_max",
        "avg_time_ms",
        "std_time_ms",
        "min_time_ms",
        "max_time_ms",
        "smallest_face_width",
        "smallest_face_height",
        "average_face_width",
        "average_face_height"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


def save_runs_csv(
    run_details,
    output_path
):
    fieldnames = [
        "mode",
        "run",
        "faces",
        "time_ms",
        "smallest_face_width",
        "smallest_face_height",
        "average_face_width",
        "average_face_height"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(run_details)


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{IMAGE_PATH}"
        )

    print(
        "Image path:",
        IMAGE_PATH
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Repeat count:",
        REPEAT_COUNT
    )

    print(
        "ROI config:",
        CLASSROOM_ROI
    )

    print(
        "Tiled config:",
        TILED_DETECTION_CONFIG
    )

    detector = get_shared_detector()

    benchmark_items = [
        (
            "full",
            lambda: detector.detect_faces(
                image
            )
        ),
        (
            "roi",
            lambda: (
                detector.detect_faces_in_roi(
                    image=image,
                    roi_config=(
                        CLASSROOM_ROI
                    )
                )
            )
        ),
        (
            "tiled",
            lambda: (
                detector.detect_faces_tiled(
                    image=image,
                    roi_config=(
                        CLASSROOM_ROI
                    ),
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
            )
        )
    ]

    summaries = []
    all_run_details = []

    for (
        mode_name,
        detect_function
    ) in benchmark_items:
        summary, run_details = (
            measure_mode(
                mode_name=mode_name,
                detect_function=(
                    detect_function
                )
            )
        )

        summaries.append(
            summary
        )

        all_run_details.extend(
            run_details
        )

    print()
    print("=" * 115)

    print(
        f"{'Mode':<10}"
        f"{'Faces':<10}"
        f"{'Avg ms':<12}"
        f"{'Std ms':<12}"
        f"{'Min ms':<12}"
        f"{'Max ms':<12}"
        f"{'Min W':<10}"
        f"{'Min H':<10}"
        f"{'Avg W':<10}"
        f"{'Avg H':<10}"
    )

    print("=" * 115)

    for result in summaries:
        print(
            f"{result['mode']:<10}"
            f"{result['faces_mean']:<10.1f}"
            f"{result['avg_time_ms']:<12.2f}"
            f"{result['std_time_ms']:<12.2f}"
            f"{result['min_time_ms']:<12.2f}"
            f"{result['max_time_ms']:<12.2f}"
            f"{result['smallest_face_width']:<10}"
            f"{result['smallest_face_height']:<10}"
            f"{result['average_face_width']:<10.2f}"
            f"{result['average_face_height']:<10.2f}"
        )

    print("=" * 115)

    full_faces = detector.detect_faces(
        image
    )

    roi_faces = detector.detect_faces_in_roi(
        image=image,
        roi_config=CLASSROOM_ROI
    )

    tiled_faces = detector.detect_faces_tiled(
        image=image,
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

    save_visualization(
        image,
        full_faces,
        "full"
    )

    save_visualization(
        image,
        roi_faces,
        "roi"
    )

    save_visualization(
        image,
        tiled_faces,
        "tiled"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        (
            "detection_summary_"
            f"{timestamp}.csv"
        )
    )

    runs_path = os.path.join(
        OUTPUT_DIR,
        (
            "detection_runs_"
            f"{timestamp}.csv"
        )
    )

    save_summary_csv(
        results=summaries,
        output_path=summary_path
    )

    save_runs_csv(
        run_details=all_run_details,
        output_path=runs_path
    )

    print()
    print(
        "Summary CSV:",
        summary_path
    )

    print(
        "Run details CSV:",
        runs_path
    )
def save_visualization(
    image,
    faces,
    mode_name
):
    output = image.copy()

    for index, face in enumerate(
        faces,
        start=1
    ):
        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        score = float(
            getattr(
                face,
                "det_score",
                0.0
            )
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            (
                f"{index} "
                f"{score:.2f}"
            ),
            (
                x1,
                max(20, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{mode_name}_visualization.jpg"
    )

    cv2.imwrite(
        output_path,
        output
    )

    print(
        f"{mode_name} visualization:",
        output_path
    )


if __name__ == "__main__":
    main()