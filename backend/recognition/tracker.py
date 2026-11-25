"""Conservative stream-local IoU tracker using existing detections only."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    min_iou: float = .3
    ttl_seconds: float = 2.0
    max_misses: int = 2
    max_tracks: int = 64

    def __post_init__(self):
        if not 0 < self.min_iou <= 1 or not math.isfinite(self.ttl_seconds) or self.ttl_seconds <= 0:
            raise ValueError("Invalid tracker IoU/TTL")
        if self.max_misses < 0 or self.max_tracks < 1:
            raise ValueError("Invalid tracker bounds")


@dataclass
class Track:
    track_id: int
    bbox: tuple
    first_seen: float
    last_seen: float
    misses: int = 0
    observations: int = 0
    short_miss_recoveries: int = 0
    had_miss: bool = False


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / area if area > 0 else 0


class FaceTracker:
    """Only unambiguous mutual one-to-one IoU edges retain track identity.

    Multiple eligible edges retire implicated tracks (and therefore evidence).
    This deliberately favors fragmentation over identity contamination.
    Caller owns this object on one processing thread.
    """
    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.next_id = 1
        self.reset()

    def reset(self):
        self.tracks = {}
        self.last_sequence = -1
        self.last_time = -math.inf
        self.last_update = dict(created=[], retired=[], recovered=[], missed=[])

    def update(self, boxes, sequence, acquisition_time):
        if not math.isfinite(acquisition_time) or sequence <= self.last_sequence or acquisition_time <= self.last_time:
            raise ValueError("duplicate_or_out_of_order_frame")
        for box in boxes:
            if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("invalid_bbox")
        self.last_sequence, self.last_time = sequence, acquisition_time
        self.last_update = dict(created=[], retired=[], recovered=[], missed=[])
        c = self.config
        self.last_update['retired'].extend((track, 'ttl') for track in self.tracks.values()
                                          if acquisition_time-track.last_seen > c.ttl_seconds)
        self.tracks = {key: value for key, value in self.tracks.items()
                       if acquisition_time-value.last_seen <= c.ttl_seconds}
        candidates = {index: [key for key in sorted(self.tracks)
                              if bbox_iou(box, self.tracks[key].bbox) >= c.min_iou]
                      for index, box in enumerate(boxes)}
        counts = {key: sum(key in ids for ids in candidates.values()) for key in self.tracks}
        ambiguous = {key for ids in candidates.values() for key in ids
                     if len(ids) > 1 or counts[key] > 1}
        self.last_update['ambiguous_detection_count'] = sum(any(key in ambiguous for key in ids)
                                                          for ids in candidates.values())
        for key in ambiguous:
            self.last_update['retired'].append((self.tracks[key], 'ambiguity'))
            del self.tracks[key]
        assigned, used = [], set()
        for index, box in enumerate(boxes):
            ids = candidates[index]
            key = ids[0] if len(ids) == 1 and ids[0] not in ambiguous else None
            if key is None and len(self.tracks) < c.max_tracks:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = Track(key, tuple(box), acquisition_time, acquisition_time)
                self.last_update['created'].append(key)
            if key is not None:
                track = self.tracks[key]
                if track.misses:
                    track.short_miss_recoveries += 1
                    self.last_update['recovered'].append(key)
                track.observations += 1
                track.bbox, track.last_seen, track.misses = tuple(box), acquisition_time, 0
                used.add(key)
            assigned.append(key)
        for key in list(self.tracks):
            if key not in used:
                self.tracks[key].misses += 1
                self.tracks[key].had_miss = True
                self.last_update['missed'].append(key)
                if self.tracks[key].misses > c.max_misses:
                    self.last_update['retired'].append((self.tracks[key], 'max_misses'))
                    del self.tracks[key]
        return assigned
