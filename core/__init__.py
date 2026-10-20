"""Core inference and domain logic package for Trash-Net.

Independent of GUI (PyQt/PySide) and desktop presentation frameworks.
"""

from core.detector import WasteDetector, ModelLoadError, InvalidFrameError
from core.mapper import BinMapper, map_class_to_bin
from core.types import Bin, Detection

__all__ = [
    "WasteDetector",
    "ModelLoadError",
    "InvalidFrameError",
    "Bin",
    "Detection",
    "BinMapper",
    "map_class_to_bin",
]
