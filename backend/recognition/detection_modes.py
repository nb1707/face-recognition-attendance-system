"""Shared selection of existing detector paths; no model or configuration tuning."""
FULL_FRAME_DIAGNOSTIC = 'full_frame'
DESKTOP_EQUIVALENT_ROI_TILED = 'desktop_tiled'
DETECTION_MODES = (FULL_FRAME_DIAGNOSTIC, DESKTOP_EQUIVALENT_ROI_TILED)


def detect_for_mode(detector, image, mode, roi_config=None, tiled_config=None):
    if mode == FULL_FRAME_DIAGNOSTIC:
        return detector.detect_faces(image)
    if mode != DESKTOP_EQUIVALENT_ROI_TILED:
        raise ValueError('Unsupported detection mode')
    from recognition.detection_config import CLASSROOM_ROI, TILED_DETECTION_CONFIG
    roi = CLASSROOM_ROI if roi_config is None else roi_config
    tiled = TILED_DETECTION_CONFIG if tiled_config is None else tiled_config
    if not tiled.get('enabled', False):
        raise RuntimeError('Tiled face detection is disabled')
    return detector.detect_faces_tiled(image=image, roi_config=roi,
        tile_count=tiled['tile_count'], overlap_ratio=tiled['overlap_ratio'],
        nms_iou_threshold=tiled['nms_iou_threshold'])
