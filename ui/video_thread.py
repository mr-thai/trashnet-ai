"""Luồng phát video và suy luận bất đồng bộ từng khung hình (Inference-Paced Video Thread).

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Đọc từng khung hình video tuần tự, chạy suy luận qua mô hình WasteDetector.
2. Tự động điều tiết tốc độ phát khớp với tốc độ AI (Zero-drop: không bỏ sót khung hình).
3. Gửi khung hình kèm kết quả bounding box và độ trễ latency_ms về Main GUI Thread.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from core.detector import WasteDetector
from core.smoother import TemporalSmoother
from core.types import Detection
from ui.qt_compat import QThread, Signal, Slot


class VideoThread(QThread):
    """Luồng chạy ngầm phân tích video đồng bộ từng khung hình."""

    frame_processed = Signal(np.ndarray, list, float, int, int)
    fps_updated = Signal(float)
    state_changed = Signal(str)
    video_finished = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        detector: WasteDetector,
        video_path: Union[str, Path],
        conf: float = 0.25,
        iou: Optional[float] = None,
        max_detections: int = 9999,
        smoother: Optional[TemporalSmoother] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._video_path = Path(video_path)
        self._conf = float(conf)
        self._iou = float(iou) if iou is not None else None
        self._max_detections = int(max_detections)
        self._smoother = smoother

        self._running = False
        self._paused = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if video processing loop is active."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if video processing loop is paused."""
        return self._paused

    def pause(self) -> None:
        """Pause video scanning."""
        with self._lock:
            self._paused = True
        self.state_changed.emit("PAUSED")

    def resume(self) -> None:
        """Resume video scanning."""
        with self._lock:
            self._paused = False
        self.state_changed.emit("RUNNING")

    def stop(self, timeout_ms: int = 2000) -> None:
        """Safely signal the thread to exit and wait without blocking UI."""
        self.requestInterruption()
        with self._lock:
            self._running = False
            self._paused = False

        if self._smoother is not None:
            self._smoother.reset()

        if self.isRunning():
            if not self.wait(timeout_ms):
                self.terminate()
                self.wait(500)
        self.state_changed.emit("STOPPED")

    @Slot(float)
    def set_confidence(self, conf: float) -> None:
        """Dynamically update confidence threshold."""
        with self._lock:
            self._conf = float(conf)

    @Slot(float)
    def set_iou(self, iou: float) -> None:
        """Dynamically update NMS IoU threshold."""
        with self._lock:
            self._iou = float(iou)

    @Slot(int)
    def set_max_detections(self, max_det: int) -> None:
        """Dynamically update Top-K max detections limit."""
        with self._lock:
            self._max_detections = int(max_det)

    def run(self) -> None:
        """Scan video frames synchronously with AI model."""
        self._running = True
        self._paused = False
        self.state_changed.emit("RUNNING")

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            self._running = False
            self.error_occurred.emit(f"Không thể mở tệp video: {self._video_path.name}")
            self.state_changed.emit("STOPPED")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps is None or native_fps <= 0 or np.isnan(native_fps):
            native_fps = 30.0
        frame_interval = 1.0 / native_fps

        frame_idx = 0
        fps_start_time = time.perf_counter()
        fps_frame_count = 0

        try:
            while self._running and not self.isInterruptionRequested():
                if self._paused:
                    time.sleep(0.05)
                    continue

                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1

                with self._lock:
                    conf = self._conf
                    iou = self._iou
                    max_det = self._max_detections

                infer_start = time.perf_counter()
                kwargs = {}
                if iou is not None:
                    kwargs["iou"] = iou

                detections: List[Detection] = self._detector.detect(frame, conf=conf, **kwargs)
                if max_det > 0 and len(detections) > max_det:
                    detections = detections[:max_det]

                if self._smoother is not None:
                    detections = self._smoother.update(detections)

                latency_ms = (time.perf_counter() - infer_start) * 1000.0

                self.frame_processed.emit(frame, detections, latency_ms, frame_idx, total_frames)

                fps_frame_count += 1
                if fps_frame_count % 10 == 0:
                    elapsed_fps = time.perf_counter() - fps_start_time
                    if elapsed_fps > 0:
                        self.fps_updated.emit(10.0 / elapsed_fps)
                    fps_start_time = time.perf_counter()

                # Điều tiết tốc độ phát video đồng bộ với tốc độ AI (không bỏ sót khung hình)
                elapsed_loop = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            cap.release()
            self._running = False
            if not self.isInterruptionRequested() and total_frames > 0 and frame_idx >= total_frames:
                self.video_finished.emit()
            self.state_changed.emit("STOPPED")
