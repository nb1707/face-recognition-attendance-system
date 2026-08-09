import numpy as np
from insightface.app import FaceAnalysis

from recognition.roi_utils import (
    resolve_roi
)

from recognition.tile_utils import (
    create_horizontal_tiles,
    non_maximum_suppression_faces
)


class InsightFaceDetector:
    def __init__(self):
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"]
        )

        self.face_app.prepare(
            ctx_id=-1,
            det_size=(960, 960)
        )

    def detect_faces(
        self,
        image
    ):
        if image is None:
            return []

        return self.face_app.get(
            image
        )

    def detect_faces_in_roi(
        self,
        image,
        roi_config
    ):
        if image is None:
            return []

        roi_image, offset_x, offset_y = (
            resolve_roi(
                image=image,
                roi_config=roi_config
            )
        )

        roi_faces = self.face_app.get(
            roi_image
        )

        restored_faces = []

        for face in roi_faces:
            bbox = np.asarray(
                face.bbox,
                dtype=np.float32
            ).copy()

            bbox[[0, 2]] += offset_x
            bbox[[1, 3]] += offset_y

            face.bbox = bbox

            landmarks = getattr(
                face,
                "kps",
                None
            )

            if landmarks is not None:
                landmarks = np.asarray(
                    landmarks,
                    dtype=np.float32
                ).copy()

                landmarks[:, 0] += (
                    offset_x
                )

                landmarks[:, 1] += (
                    offset_y
                )

                face.kps = landmarks

            restored_faces.append(
                face
            )

        return restored_faces

    def detect_faces_tiled(
        self,
        image,
        roi_config=None,
        tile_count: int = 3,
        overlap_ratio: float = 0.20,
        nms_iou_threshold: float = 0.40
    ):
        if image is None:
            return []

        if image.size == 0:
            return []

        roi_image, roi_offset_x, roi_offset_y = (
            resolve_roi(
                image=image,
                roi_config=roi_config
            )
        )

        tiles = create_horizontal_tiles(
            image=roi_image,
            tile_count=tile_count,
            overlap_ratio=overlap_ratio
        )

        all_faces = []

        for (
            tile,
            tile_offset_x,
            tile_offset_y
        ) in tiles:
            tile_faces = self.face_app.get(
                tile
            )

            global_offset_x = (
                roi_offset_x
                + tile_offset_x
            )

            global_offset_y = (
                roi_offset_y
                + tile_offset_y
            )

            for face in tile_faces:
                bbox = np.asarray(
                    face.bbox,
                    dtype=np.float32
                ).copy()

                bbox[[0, 2]] += (global_offset_x)

                bbox[[1, 3]] += (global_offset_y)

                face.bbox = bbox

                landmarks = getattr(face,"kps",None)

                if landmarks is not None:
                    landmarks = np.asarray(
                        landmarks,
                        dtype=np.float32
                    ).copy()

                    landmarks[:, 0] += (global_offset_x)

                    landmarks[:, 1] += (global_offset_y)
                    face.kps = landmarks

                all_faces.append(face)

        return non_maximum_suppression_faces(faces=all_faces,iou_threshold=(nms_iou_threshold)
        )


_detector_instance = None


def get_shared_detector():
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = (InsightFaceDetector())
    return _detector_instance