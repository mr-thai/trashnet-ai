"""Background inference worker thread using QThread (US-3.3 & US-3.5).

Runs model inference asynchronously off the main GUI thread to prevent UI freezing.
"""

from pathlib import Path
import time
from typing import List, Optional
import numpy as np

from core.detector import WasteDetector
from core.types import Detection
from ui.qt_compat import QThread, Signal


class InferenceWorker(QThread):
    """Worker thread that executes WasteDetector inference in the background."""

    # Signals
    started_inference = Signal()
    finished_inference = Signal(np.ndarray, list, float)  # (frame, detections, latency_ms)
    error_occurred = Signal(str)

    def __init__(
        self,
        detector: WasteDetector,
        frame: np.ndarray,
        conf: Optional[float] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._frame = frame
        self._conf = conf

    def run(self) -> None:
        """Execute inference in background thread."""
        try:
            self.started_inference.emit()
            start_t = time.perf_counter()

            # Run detection on WasteDetector
            detections: List[Detection] = self._detector.detect(self._frame, conf=self._conf)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            self.finished_inference.emit(self._frame, detections, elapsed_ms)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
