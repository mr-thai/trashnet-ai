"""Real-time Webcam Capture Thread and State Machine for Trash-Net (US-3.3 Chế độ B).

Handles V4L2 backend capture on Linux, thread-safe camera state transitions:
Idle -> Running -> Paused -> Stopped, device disconnection handling, and memory-safe streaming.
"""

from enum import Enum
import glob
from pathlib import Path
import platform
import sys
import time
from typing import List, Optional
import cv2
import numpy as np

from ui.qt_compat import QThread, Signal


class CameraState(str, Enum):
    """Camera state machine enumeration."""
    IDLE = "IDLE"          # Trạng thái chờ, chưa mở camera
    RUNNING = "RUNNING"    # Đang phát luồng và suy luận trực tiếp
    PAUSED = "PAUSED"      # Tạm dừng: giữ khung hình cuối, ngừng suy luận
    STOPPED = "STOPPED"    # Đã dừng: camera giải phóng hoàn toàn, đèn tắt


def list_available_cameras(max_indices: int = 6) -> List[int]:
    """Scan and list available video capture camera indices on the system.

    Returns:
        List of integer camera indices that can be opened.
    """
    valid_indices: List[int] = []

    # Suppress OpenCV C++ stderr warnings during camera probing
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    try:
        # On Linux, scan /dev/video* devices directly first
        if platform.system() == "Linux":
            video_devs = sorted(glob.glob("/dev/video*"))
            for dev_path in video_devs:
                try:
                    idx = int(dev_path.replace("/dev/video", ""))
                    # Quick probe without hanging
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if cap.isOpened():
                        valid_indices.append(idx)
                        cap.release()
                except Exception:
                    continue

        if not valid_indices:
            # Generic probe fallback for all operating systems
            for idx in range(max_indices):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    valid_indices.append(idx)
                    cap.release()
    except Exception:
        pass

    return sorted(list(set(valid_indices)))


class WebcamThread(QThread):
    """Asynchronous video capture thread managing webcam lifecycle and V4L2 capture."""

    # Signals
    frame_received = Signal(np.ndarray)   # Emits raw BGR frame from webcam
    state_changed = Signal(str)           # Emits state name ("IDLE", "RUNNING", "PAUSED", "STOPPED")
    camera_error = Signal(str)            # Emits human-readable error description

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state: CameraState = CameraState.IDLE
        self._camera_index: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._target_fps: int = 30
        self._running: bool = False
        self._paused: bool = False

    @property
    def current_state(self) -> CameraState:
        """Current camera state."""
        return self._state

    def _set_state(self, new_state: CameraState) -> None:
        """Update and emit state transition."""
        self._state = new_state
        self.state_changed.emit(new_state.value)

    def start_capture(self, camera_index: int = 0, target_fps: int = 30) -> bool:
        """Initiate video capture from camera index."""
        if self._state == CameraState.PAUSED:
            self.resume_capture()
            return True

        self._camera_index = camera_index
        self._target_fps = target_fps
        self._running = True
        self._paused = False

        if not self.isRunning():
            self.start()
        return True

    def pause_capture(self) -> None:
        """Pause capture: hold last frame and pause inference."""
        if self._state == CameraState.RUNNING:
            self._paused = True
            self._set_state(CameraState.PAUSED)

    def resume_capture(self) -> None:
        """Resume capture from paused state."""
        if self._state == CameraState.PAUSED:
            self._paused = False
            self._set_state(CameraState.RUNNING)

    def stop_capture(self) -> None:
        """Stop capture and completely release the hardware device."""
        self._running = False
        self._paused = False
        self._release_camera()
        if self.isRunning():
            self.wait(2000)
            if self.isRunning():
                self.terminate()
                self.wait(1000)
        self._set_state(CameraState.STOPPED)

    def _release_camera(self) -> None:
        """Release underlying OpenCV VideoCapture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def run(self) -> None:
        """Thread worker loop reading frames from camera via V4L2."""
        # 1. Open camera device
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._camera_index, backend)

        if not self._cap.isOpened():
            # Fallback to default backend if V4L2 fails
            self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            self._release_camera()
            self._running = False
            self._set_state(CameraState.STOPPED)
            self.camera_error.emit(
                f"Không thể mở camera thiết bị #{self._camera_index}. "
                "Camera có thể đang bị ứng dụng khác chiếm dụng hoặc bị rút."
            )
            return

        # Configure camera optimal frame size and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        self._set_state(CameraState.RUNNING)
        frame_interval = 1.0 / self._target_fps
        consecutive_read_failures = 0

        # 2. Main acquisition loop
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()
            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                # If 5 consecutive frame read failures occur, device was likely disconnected
                if consecutive_read_failures >= 5:
                    self._release_camera()
                    self._running = False
                    self._set_state(CameraState.STOPPED)
                    self.camera_error.emit(
                        f"Mất kết nối với camera #{self._camera_index} (thiết bị có thể đã bị rút giữa chừng)."
                    )
                    break
                time.sleep(0.02)
                continue

            consecutive_read_failures = 0
            self.frame_received.emit(frame)

            # Throttle loop to match target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._release_camera()
