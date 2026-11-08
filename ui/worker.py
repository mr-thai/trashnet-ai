"""Tiến trình chạy suy luận mô hình ngầm (Inference Worker Thread) dùng QThread.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Tách biệt hoàn toàn tính toán mô hình YOLO nặng khỏi Main GUI Thread để tránh giật đơ giao diện.
2. Thiết kế bộ đệm kích thước 1 (size-1 buffer): khung hình mới nhất luôn ghi đè khung chưa kịp xử lý.
3. Hỗ trợ cả 2 chế độ: Đơn khung hình (cho ảnh tĩnh) và Dòng liên tục (cho webcam/video).
"""

import threading
import time
from typing import List, Optional
import numpy as np

from core.detector import WasteDetector
from core.smoother import TemporalSmoother
from core.types import Detection
from ui.qt_compat import QThread, Signal, Slot


class InferenceWorker(QThread):
    """Tiến trình ngầm thực thi suy luận WasteDetector độc lập với luồng giao diện."""

    started_inference = Signal()
    finished_inference = Signal(np.ndarray, list, float)
    fps_updated = Signal(float)
    error_occurred = Signal(str)

    def __init__(
        self,
        detector: WasteDetector,
        frame: Optional[np.ndarray] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_detections: int = 9999,
        smoother: Optional[TemporalSmoother] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._conf = conf
        self._iou = iou
        self._max_detections = max_detections
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

    @Slot(float)
    def set_iou(self, iou: float) -> None:
        """Thread-safely update NMS IoU threshold for subsequent inferences."""
        with self._condition:
            self._iou = float(iou)

    @Slot(int)
    def set_max_detections(self, max_det: int) -> None:
        """Thread-safely update Top-K max detections limit."""
        with self._condition:
            self._max_detections = int(max_det)

    def reset_smoother(self) -> None:
        """Reset internal temporal smoother state."""
        if self._smoother is not None:
            self._smoother.reset()

    def stop(self, timeout_ms: int = 2000) -> None:
        """Safely signal the worker to exit and wait up to timeout_ms without abrupt termination."""
        self.requestInterruption()
        with self._condition:
            self._running = False
            self._pending_frame = None
            self._condition.notify_all()

        if self._smoother is not None:
            self._smoother.reset()

        if self.isRunning():
            if not self.wait(timeout_ms):
                with self._condition:
                    self._condition.notify_all()
                self.wait(500)

    def run(self) -> None:
        """Vòng lặp thực thi suy luận ngầm độc lập với luồng GUI."""
        self._running = True

        while self._running and not self.isInterruptionRequested():
            with self._condition:
                while self._running and not self.isInterruptionRequested() and self._pending_frame is None:
                    # Chờ khung hình mới mà không tốn CPU (Condition wait)
                    self._condition.wait(timeout=0.1)

                if not self._running or self.isInterruptionRequested():
                    break

                # Lấy khung hình mới nhất và dọn sạch slot (size-1 buffer)
                frame = self._pending_frame
                conf = self._pending_conf if self._pending_conf is not None else self._conf
                self._pending_frame = None

            if frame is None:
                continue

            try:
                self.started_inference.emit()
                start_t = time.perf_counter()

                kwargs = {}
                if hasattr(self, "_iou") and self._iou is not None:
                    kwargs["iou"] = self._iou
                detections: List[Detection] = self._detector.detect(frame, conf=conf, **kwargs)
                if hasattr(self, "_max_detections") and self._max_detections > 0 and len(detections) > self._max_detections:
                    detections = detections[:self._max_detections]

                # Áp dụng bộ làm mượt TemporalSmoother để chống rung nhấp nháy nhãn
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
