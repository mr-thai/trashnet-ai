"""Main application window for Trash-Net waste classification (US-3.1, US-3.3 Static & Webcam).

Integrates VideoView, BinBanner, ResultList, ControlPanel, and WebcamThread
within a responsive QSplitter layout complying with DoD >= 1280x720.
Supports Drag & Drop, real-time Webcam state machine (Idle -> Running -> Paused -> Stopped),
V4L2 backend capture, and background inference without memory leaks.
"""

from pathlib import Path
import sys
from typing import List, Optional
import cv2
import numpy as np

from core.detector import WasteDetector
from core.types import Bin, Detection
from ui.constants import (
    DARK_PALETTE_QSS,
    DEFAULT_WINDOW_TITLE,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
)
from ui.qt_compat import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSize,
    QSplitter,
    QStatusBar,
    Qt,
    Slot,
    QVBoxLayout,
    QWidget,
)
from ui.utils import FILE_DIALOG_IMAGE_FILTER, load_image_unicode
from ui.webcam_thread import CameraState, WebcamThread, list_available_cameras
from ui.widgets.bin_banner import BinBanner
from ui.widgets.control_panel import ControlPanel
from ui.widgets.result_card import ResultList
from ui.widgets.video_view import VideoView
from ui.worker import InferenceWorker


class MainWindow(QMainWindow):
    """Primary GUI window for Trash-Net waste classification."""

    def __init__(self, model_path: Optional[str] = "yolo11n-seg.pt", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)

        # DoD: Giao diện hiển thị đúng ở độ phân giải 1280×720 trở lên
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.resize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self._current_image_path: Optional[Path] = None
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: List[Detection] = []
        self._is_inferring_webcam: bool = False

        # Load detector if weights exist
        self.detector: Optional[WasteDetector] = None
        if model_path and Path(model_path).exists():
            try:
                self.detector = WasteDetector(model_path=model_path)
            except Exception as exc:
                print(f"[!] Warning: Could not initialize WasteDetector with '{model_path}': {exc}")

        self.current_worker: Optional[InferenceWorker] = None
        self._webcam_worker: Optional[InferenceWorker] = None
        self.webcam_thread = WebcamThread(self)

        # Performance metrics (US-3.4)
        self._capture_fps: float = 0.0
        self._inference_fps: float = 0.0
        self._inference_latency_ms: float = 0.0

        self._init_ui()
        self._connect_signals()
        self._init_status_bar()

        # Discover connected cameras
        self._refresh_cameras()

        # Apply global dark styling
        self.setStyleSheet(DARK_PALETTE_QSS)

    def _init_ui(self) -> None:
        """Create and position primary UI zones with QSplitter layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(0)

        # Horizontal Splitter between Left (Video/Drop Canvas) and Right (Results & Controls)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("MainSplitter")

        # 1. Left Viewport: Aspect-Ratio Preserving Video/Image View with Drag-and-Drop
        self.video_view = VideoView(self)
        self.splitter.addWidget(self.video_view)

        # 2. Right Side: Multi-component vertical stack
        right_panel = QFrame()
        right_panel.setObjectName("RightPanel")
        right_panel.setMinimumWidth(380)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(12)

        # 2.1 Bin Banner (Top of right stack)
        self.bin_banner = BinBanner(self)
        right_layout.addWidget(self.bin_banner)

        # 2.2 Result Cards List (Middle expanding zone)
        self.result_list = ResultList(self)
        right_layout.addWidget(self.result_list, stretch=1)

        # 2.3 Control Panel (Bottom fixed actions)
        self.control_panel = ControlPanel(default_conf=0.25, parent=self)
        right_layout.addWidget(self.control_panel)

        self.splitter.addWidget(right_panel)

        # Set initial splitter stretch weights: 65% video, 35% controls
        self.splitter.setStretchFactor(0, 65)
        self.splitter.setStretchFactor(1, 35)
        self.splitter.setSizes([830, 430])

        root_layout.addWidget(self.splitter)

    def _init_status_bar(self) -> None:
        """Initialize bottom status bar."""
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        dev = self.detector.device.upper() if self.detector else "CPU"
        self.status_bar.showMessage(f"Hệ thống sẵn sàng | Thiết bị: {dev} | Chờ nguồn dữ liệu")

    def _refresh_cameras(self) -> None:
        """Scan available system cameras and populate dropdown."""
        try:
            cameras = list_available_cameras()
            self.control_panel.populate_cameras(cameras)
        except Exception:
            self.control_panel.populate_cameras([0])

    def _connect_signals(self) -> None:
        """Wire up user interaction signals from ControlPanel, VideoView, and WebcamThread."""
        self.control_panel.open_image_requested.connect(self.open_image_dialog)
        self.control_panel.open_video_requested.connect(self.open_video_dialog)
        self.control_panel.toggle_webcam_requested.connect(self.on_toggle_webcam)
        self.control_panel.confidence_changed.connect(self.on_confidence_changed)
        self.control_panel.reset_requested.connect(self.reset_all)
        self.control_panel.overlay_options_changed.connect(self.video_view.set_overlay_options)

        # Webcam State Machine Signals (US-3.3 Chế độ B)
        self.control_panel.start_webcam_requested.connect(self.on_start_webcam)
        self.control_panel.pause_webcam_requested.connect(self.on_pause_webcam)
        self.control_panel.resume_webcam_requested.connect(self.on_resume_webcam)
        self.control_panel.stop_webcam_requested.connect(self.on_stop_webcam)

        self.webcam_thread.frame_received.connect(self.on_webcam_frame_received)
        self.webcam_thread.state_changed.connect(self.on_webcam_state_changed)
        self.webcam_thread.fps_updated.connect(self.on_capture_fps_updated)
        self.webcam_thread.camera_error.connect(self.on_webcam_error)

        # Drag & Drop signals from VideoView
        self.video_view.image_dropped.connect(self.load_image_file)
        self.video_view.file_rejected.connect(self.on_file_rejected)
        self.video_view.multiple_files_warning.connect(self.on_multiple_files_warning)

    # --------------------------------------------------------------------------
    # Webcam Lifecycle Handlers (US-3.3 Chế độ B)
    # --------------------------------------------------------------------------

    @Slot(int)
    def on_start_webcam(self, camera_index: int) -> None:
        """Start streaming from chosen camera index with size-1 buffer InferenceWorker."""
        # Stop any active static file inference
        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        # Stop any existing webcam worker
        if self._webcam_worker is not None:
            self._webcam_worker.stop(timeout_ms=1000)
            self._webcam_worker = None

        # Initialize continuous InferenceWorker with size-1 buffer (latest-frame-wins)
        if self.detector is not None:
            conf = self.control_panel.confidence_threshold
            self._webcam_worker = InferenceWorker(self.detector, conf=conf, parent=self)
            self._webcam_worker.finished_inference.connect(self._on_webcam_inference_finished)
            self._webcam_worker.fps_updated.connect(self.on_inference_fps_updated)
            self._webcam_worker.error_occurred.connect(self._on_webcam_inference_error)
            self._webcam_worker.start()

        self.status_bar.showMessage(f"Đang kết nối camera #{camera_index} qua V4L2...")
        success = self.webcam_thread.start_capture(camera_index=camera_index)
        if not success:
            self.status_bar.showMessage(f"Lỗi: Không thể khởi chạy camera #{camera_index}.")

    @Slot()
    def on_pause_webcam(self) -> None:
        """Pause webcam streaming: hold last frame and pause inference."""
        self.webcam_thread.pause_capture()
        self.status_bar.showMessage("Webcam tạm dừng — Giữ khung hình hiện tại.")

    @Slot()
    def on_resume_webcam(self) -> None:
        """Resume webcam streaming."""
        self.webcam_thread.resume_capture()
        self.status_bar.showMessage("Webcam tiếp tục phát trực tiếp.")

    @Slot()
    def on_stop_webcam(self) -> None:
        """Stop webcam stream and completely release hardware device (camera LED off)."""
        self.webcam_thread.stop_capture()
        if self._webcam_worker is not None:
            self._webcam_worker.stop(timeout_ms=2000)
            self._webcam_worker = None
        self._capture_fps = 0.0
        self._inference_fps = 0.0
        self.status_bar.showMessage("Webcam đã tắt — Đã giải phóng thiết bị.")

    @Slot(str)
    def on_webcam_state_changed(self, state: str) -> None:
        """Synchronize UI buttons with camera state machine."""
        self.control_panel.set_camera_state(state)

    @Slot(str)
    def on_webcam_error(self, error_message: str) -> None:
        """Handle camera device errors or disconnection mid-stream."""
        self.status_bar.showMessage(f"Sự cố camera: {error_message}")
        self.control_panel.set_camera_state(CameraState.STOPPED.value)
        QMessageBox.warning(self, "Lỗi kết nối Camera", error_message)

    @Slot(np.ndarray)
    def on_webcam_frame_received(self, frame: np.ndarray) -> None:
        """Handle incoming frame from webcam: render at camera rate and submit to size-1 buffer."""
        self._current_frame = frame

        # Render current frame with latest detections immediately (preserves 30 FPS display)
        self.video_view.set_frame(frame, self._current_detections)

        # Submit to size-1 buffer (latest frame always overwrites unconsumed frame)
        if self._webcam_worker is not None and self._webcam_worker.is_worker_running:
            conf = self.control_panel.confidence_threshold
            self._webcam_worker.submit_frame(frame, conf=conf)

    @Slot(np.ndarray, list, float)
    def _on_webcam_inference_finished(self, frame: np.ndarray, detections: List[Detection], latency_ms: float) -> None:
        """Update detection overlay and banner from webcam inference."""
        self._current_detections = list(detections)
        self.update_results(detections)
        self._inference_latency_ms = latency_ms
        self._update_status_bar_metrics()

    @Slot(float)
    def on_capture_fps_updated(self, fps: float) -> None:
        """Update measured camera capture FPS."""
        self._capture_fps = fps
        self._update_status_bar_metrics()

    @Slot(float)
    def on_inference_fps_updated(self, fps: float) -> None:
        """Update measured AI inference FPS."""
        self._inference_fps = fps
        self._update_status_bar_metrics()

    def _update_status_bar_metrics(self) -> None:
        """Format and display real-time dual FPS and detection metrics on the status bar."""
        det_count = len(self._current_detections)
        msg = (
            f"Camera: {self._capture_fps:.1f} FPS | "
            f"AI: {self._inference_fps:.1f} FPS (Độ trễ: {self._inference_latency_ms:.1f} ms) | "
            f"Phát hiện: {det_count} vật thể"
        )
        self.status_bar.showMessage(msg)

    @Slot(str)
    def _on_webcam_inference_error(self, err_msg: str) -> None:
        """Handle inference failure during webcam stream."""
        pass

    # --------------------------------------------------------------------------
    # Static Image & Drag-Drop Handlers
    # --------------------------------------------------------------------------

    @Slot()
    def open_image_dialog(self) -> None:
        """Present file chooser dialog for static images with restricted formats."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tệp ảnh rác",
            "",
            FILE_DIALOG_IMAGE_FILTER,
        )
        if not file_path:
            return

        self.load_image_file(Path(file_path))

    def load_image_file(self, path: Path) -> bool:
        """Load and display an image from file path with Unicode support and run inference."""
        # Stop webcam if active when user opens an image
        if self.webcam_thread.current_state in (CameraState.RUNNING, CameraState.PAUSED):
            self.on_stop_webcam()

        try:
            img = load_image_unicode(path)
        except Exception as exc:
            self.status_bar.showMessage(f"Lỗi: Không thể mở ảnh '{path.name}'")
            QMessageBox.warning(self, "Lỗi đọc tệp ảnh", f"Không thể đọc tệp '{path.name}':\n{exc}")
            return False

        self._current_image_path = path
        self._current_frame = img
        self.video_view.set_frame(img)
        h, w = img.shape[:2]
        self.status_bar.showMessage(f"Đã nạp: {path.name} ({w}×{h} px). Đang chạy suy luận...")

        self._start_inference(img)
        return True

    def _start_inference(self, frame: np.ndarray) -> None:
        """Run inference asynchronously using InferenceWorker."""
        if self.detector is None:
            self.status_bar.showMessage("Cảnh báo: Chưa nạp mô hình suy luận.")
            return

        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        conf = self.control_panel.confidence_threshold
        self.current_worker = InferenceWorker(self.detector, frame, conf=conf)
        self.current_worker.finished_inference.connect(self._on_inference_finished)
        self.current_worker.error_occurred.connect(self._on_inference_error)
        self.current_worker.start()

    @Slot(np.ndarray, list, float)
    def _on_inference_finished(self, frame: np.ndarray, detections: List[Detection], latency_ms: float) -> None:
        """Handle results delivered by the background worker thread."""
        self._current_detections = list(detections)
        self.video_view.set_frame(frame, detections)
        self.update_results(detections)

        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        self.status_bar.showMessage(
            f"Hoàn tất: Phát hiện {len(detections)} vật thể | Độ trễ: {latency_ms:.1f} ms (~{fps:.1f} FPS)"
        )

    @Slot(str)
    def _on_inference_error(self, err_msg: str) -> None:
        """Handle background worker failure."""
        self.status_bar.showMessage(f"Lỗi suy luận: {err_msg}")
        QMessageBox.critical(self, "Lỗi suy luận", f"Gặp sự cố khi phân tích ảnh:\n{err_msg}")

    @Slot(str)
    def on_file_rejected(self, message: str) -> None:
        """Notify user when an invalid or corrupt file was dropped."""
        self.status_bar.showMessage("Định dạng tệp bị từ chối.")
        QMessageBox.warning(self, "Tệp không hợp lệ", message)

    @Slot(int)
    def on_multiple_files_warning(self, count: int) -> None:
        """Notify user when multiple files were dropped."""
        self.status_bar.showMessage(
            f"Lưu ý: Bạn đã kéo thả {count} tệp. Ứng dụng chỉ xử lý tệp ảnh đầu tiên."
        )

    @Slot()
    def open_video_dialog(self) -> None:
        """Present file chooser dialog for video streams."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tệp video rác",
            "",
            "Video (*.mp4 *.avi *.mkv *.mov);;Tất cả tệp (*.*)",
        )
        if not file_path:
            return

        self.status_bar.showMessage(f"Đã chọn video: {Path(file_path).name} (sẵn sàng xử lý)")

    @Slot(bool)
    def on_toggle_webcam(self, active: bool) -> None:
        """Handle legacy toggle webcam signal."""
        pass

    @Slot(float)
    def on_confidence_changed(self, threshold: float) -> None:
        """Handle confidence threshold slider changes."""
        self.status_bar.showMessage(f"Ngưỡng tin cậy cập nhật: {threshold * 100:.0f}%")
        if self._current_frame is not None and self.detector is not None and not self.webcam_thread.isRunning():
            self._start_inference(self._current_frame)

    def update_results(self, detections: List[Detection]) -> None:
        """Update detection cards and destination bin banner."""
        self._current_detections = list(detections)
        self.result_list.set_detections(self._current_detections)

        if not self._current_detections:
            self.bin_banner.reset()
            self.status_bar.showMessage("Không phát hiện rác trong khung hình")
        else:
            top_detection = self._current_detections[0]
            self.bin_banner.set_bin(top_detection.bin_type)
            self.status_bar.showMessage(
                f"Phát hiện {len(self._current_detections)} vật thể. "
                f"Đích đến: {top_detection.bin_type.display_name} ({top_detection.confidence * 100:.1f}%)"
            )

    @Slot()
    def reset_all(self) -> None:
        """Clear loaded frames, results, reset banner, and stop webcam."""
        if self.webcam_thread.isRunning():
            self.on_stop_webcam()

        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        self.video_view.clear_frame()
        self.result_list.clear()
        self.bin_banner.reset()
        self._current_image_path = None
        self._current_frame = None
        self._current_detections.clear()
        self.status_bar.showMessage("Đã đặt lại trạng thái chờ.")

    def closeEvent(self, event) -> None:
        """Ensure both CaptureThread and InferenceWorker halt cleanly within 2 seconds."""
        if hasattr(self, "_webcam_worker") and self._webcam_worker is not None:
            self._webcam_worker.stop(timeout_ms=2000)
            self._webcam_worker = None

        if hasattr(self, "current_worker") and self.current_worker is not None:
            self.current_worker.stop(timeout_ms=2000)
            self.current_worker = None

        if hasattr(self, "webcam_thread"):
            self.webcam_thread.stop_capture()

        super().closeEvent(event)


def run_app() -> None:
    """Launch Trash-Net desktop application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
