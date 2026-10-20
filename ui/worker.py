"""Background inference worker thread using QThread (US-3.3 & US-3.4).

Runs model inference asynchronously off the main GUI thread to prevent UI freezing.
Implements a size-1 buffer (latest-frame-wins) to drop intermediate frames when
inference is slower than the camera capture rate.
"""

from pathlib import Path
import threading
import time
from typing import List, Optional
import numpy as np

from core.detector import WasteDetector
from core.smoother import TemporalSmoother
from core.types import Detection
from ui.qt_compat import QThread, Signal, Slot


class InferenceWorker(QThread):
    """Worker thread that executes WasteDetector inference asynchronously.

    Supports both single-shot mode (for static images) and continuous streaming mode
    with a size-1 buffer (for real-time webcam streams).
    """

    # Signals
    started_inference = Signal()
    finished_inference = Signal(np.ndarray, list, float)  # (frame, detections, latency_ms)
    fps_updated = Signal(float)  # current inference FPS
    error_occurred = Signal(str)

    def __init__(
        self,
        detector: WasteDetector,
        frame: Optional[np.ndarray] = None,
        conf: Optional[float] = None,
        smoother: Optional[TemporalSmoother] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._conf = conf
        self._smoother = smoother
        self._running = False
        self._single_shot: bool = (frame is not None)

        # Thread synchronization primitives for size-1 buffer (latest-frame-wins)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_frame: Optional[np.ndarray] = frame
        self._pending_conf: Optional[float] = conf

        # Performance metrics
        self._frames_processed: int = 0
        self._fps_window_start: float = time.perf_counter()

    @property
    def is_worker_running(self) -> bool:
        """Check if continuous loop is active."""
        return self._running

    def submit_frame(self, frame: np.ndarray, conf: Optional[float] = None) -> None:
        """Submit frame to size-1 buffer.

        If an unconsumed frame is already in the buffer, it is immediately
        overwritten so the worker always processes the latest frame.
        """
        with self._condition:
            self._pending_frame = frame
            if conf is not None:
                self._pending_conf = conf
                self._conf = conf
            self._condition.notify_all()

    @Slot(float)
    def set_confidence(self, conf: float) -> None:
        """Thread-safely update detection confidence threshold for subsequent inferences."""
        with self._condition:
            self._conf = float(conf)
            self._pending_conf = float(conf)

    def reset_smoother(self) -> None:
        """Reset internal temporal smoother state."""
        if self._smoother is not None:
            self._smoother.reset()

    def stop(self, timeout_ms: int = 2000) -> None:
        """Safely signal the worker to exit and wait up to timeout_ms."""
        with self._condition:
            self._running = False
            self._pending_frame = None
            self._condition.notify_all()

        if self._smoother is not None:
            self._smoother.reset()

        if self.isRunning():
            self.wait(timeout_ms)
            if self.isRunning():
                self.terminate()
                self.wait(500)

    def run(self) -> None:
        """Execute inference loop off the GUI thread."""
        self._running = True

        while self._running:
            with self._condition:
                while self._running and self._pending_frame is None:
                    # Wait for next frame without busy-waiting
                    self._condition.wait(timeout=0.1)

                if not self._running:
                    break

                # Atomically extract latest frame and clear slot
                frame = self._pending_frame
                conf = self._pending_conf if self._pending_conf is not None else self._conf
                self._pending_frame = None

            if frame is None:
                continue

            try:
                self.started_inference.emit()
                start_t = time.perf_counter()

                # Heavy CNN computation
                detections: List[Detection] = self._detector.detect(frame, conf=conf)

                # US-4.2 Temporal smoothing & anti-flickering
                if self._smoother is not None:
                    detections = self._smoother.update(detections)

                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

                self.finished_inference.emit(frame, detections, elapsed_ms)
                self.fps_updated.emit(fps)

            except Exception as exc:
                self.error_occurred.emit(str(exc))

            if self._single_shot:
                break
