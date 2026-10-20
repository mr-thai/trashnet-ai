"""Core inference and domain logic package for Trash-Net.

Independent of GUI (PyQt/PySide) and desktop presentation frameworks.
"""

from core.detector import WasteDetector, ModelLoadError, InvalidFrameError
from core.flagging import save_flagged_frame
from core.guidelines import GuidelineManager, get_waste_guideline
from core.mapper import BinMapper, map_class_to_bin
from core.ood import OODDetector
from core.smoother import TemporalSmoother, TrackedTarget, TrackState, compute_iou
from core.types import Bin, Detection

__all__ = [
    "WasteDetector",
    "ModelLoadError",
    "InvalidFrameError",
    "Bin",
    "Detection",
    "BinMapper",
    "map_class_to_bin",
    "TemporalSmoother",
    "TrackedTarget",
    "TrackState",
    "compute_iou",
    "OODDetector",
    "save_flagged_frame",
    "GuidelineManager",
    "get_waste_guideline",
]
