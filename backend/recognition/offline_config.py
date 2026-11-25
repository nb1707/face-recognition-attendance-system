"""Explicit local development settings. Calibration never rewrites these."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from recognition.quality import QualityConfig
from recognition.tracker import TrackerConfig
from recognition.evidence import EvidenceConfig
from recognition.temporal import OperatingPoint, TemporalConfig


@dataclass(frozen=True)
class OfflineConfig:
    operating_point: OperatingPoint
    quality: QualityConfig = field(default_factory=QualityConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    detection_mode: str = "tiled"

    def __post_init__(self):
        if self.detection_mode != 'tiled':
            raise ValueError('Desktop/video runtime supports existing tiled detection only')


def load_offline_config(path=None):
    path = path or os.getenv("RECOGNITION_CONFIG_PATH") or Path(__file__).with_name("offline_development.json")
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return OfflineConfig(OperatingPoint(**values["operating_point"]),
                         QualityConfig(**values.get("quality", {})),
                         TrackerConfig(**values.get("tracker", {})),
                         EvidenceConfig(**values.get("evidence", {})),
                         TemporalConfig(**values.get("temporal", {})), values.get("detection_mode", "tiled"))
