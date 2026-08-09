import numpy as np


def create_horizontal_tiles(
    image: np.ndarray,
    tile_count: int = 3,
    overlap_ratio: float = 0.20
) -> list[tuple[np.ndarray, int, int]]:
    """
    Split an image horizontally into overlapping tiles.

    Returns:
        [
            (tile_image, offset_x, offset_y),
            ...
        ]

    offset_x and offset_y represent each tile's position
    inside the input image.
    """
    if image is None:
        raise ValueError(
            "Input image is missing."
        )

    if image.size == 0:
        raise ValueError(
            "Input image is empty."
        )

    if tile_count < 1:
        raise ValueError(
            "tile_count must be at least 1."
        )

    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            "overlap_ratio must be between 0.0 and 1.0."
        )

    height, width = image.shape[:2]

    base_tile_width = width / tile_count

    overlap_pixels = int(
        base_tile_width * overlap_ratio
    )

    tiles = []

    for index in range(tile_count):
        base_x1 = int(
            index * base_tile_width
        )

        base_x2 = int(
            (index + 1) * base_tile_width
        )

        if index == tile_count - 1:
            base_x2 = width

        x1 = max(
            0,
            base_x1 - (
                overlap_pixels
                if index > 0
                else 0
            )
        )

        x2 = min(
            width,
            base_x2 + (
                overlap_pixels
                if index < tile_count - 1
                else 0
            )
        )

        tile = image[
            0:height,
            x1:x2
        ]

        if tile.size == 0:
            continue

        tiles.append(
            (
                tile,
                x1,
                0
            )
        )

    return tiles


def calculate_iou(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray
) -> float:
    """
    Calculate Intersection over Union between two bboxes.

    bbox format:
        [x1, y1, x2, y2]
    """
    bbox_a = np.asarray(
        bbox_a,
        dtype=np.float32
    ).reshape(-1)

    bbox_b = np.asarray(
        bbox_b,
        dtype=np.float32
    ).reshape(-1)

    if bbox_a.size != 4 or bbox_b.size != 4:
        raise ValueError(
            "Each bounding box must contain 4 values."
        )

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    intersection_x1 = max(
        float(ax1),
        float(bx1)
    )

    intersection_y1 = max(
        float(ay1),
        float(by1)
    )

    intersection_x2 = min(
        float(ax2),
        float(bx2)
    )

    intersection_y2 = min(
        float(ay2),
        float(by2)
    )

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        float(ax2 - ax1)
        * float(ay2 - ay1)
    )

    area_b = max(
        0.0,
        float(bx2 - bx1)
        * float(by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return float(
        intersection_area / union_area
    )


def non_maximum_suppression_faces(
    faces: list,
    iou_threshold: float = 0.40
) -> list:
    """
    Remove duplicate detections created by overlapping tiles.

    Higher detection confidence is kept first.
    """
    if not faces:
        return []

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "iou_threshold must be between 0.0 and 1.0."
        )

    ordered_faces = sorted(
        faces,
        key=lambda face: float(
            getattr(
                face,
                "det_score",
                0.0
            )
        ),
        reverse=True
    )

    kept_faces = []

    while ordered_faces:
        selected_face = (
            ordered_faces.pop(0)
        )

        kept_faces.append(
            selected_face
        )

        remaining_faces = []

        for candidate in ordered_faces:
            iou = calculate_iou(
                selected_face.bbox,
                candidate.bbox
            )

            if iou < iou_threshold:
                remaining_faces.append(
                    candidate
                )

        ordered_faces = (
            remaining_faces
        )

    return kept_faces