"""Cửa sổ ứng dụng chính Desktop cho hệ thống phân loại rác Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Tích hợp các widget: Khung xem hình ảnh/video (VideoView), Banner thùng rác (BinBanner),
   Danh sách kết quả (ResultList), Bảng điều khiển (ControlPanel), Thẻ hướng dẫn (GuidelineCard).
2. Hỗ trợ đa chế độ: Kéo thả ảnh tĩnh (Drag & Drop), Phát video quét từng khung hình,
   và Nhận diện trực tiếp qua Webcam thời gian thực.
3. Chạy toàn bộ suy luận mô hình ngầm qua QThread để đảm bảo giao diện luôn mượt mà.
"""

from pathlib import Path
import sys
import time
from typing import List, Optional, Union
import numpy as np

from core.audit_log import AuditLogger
from core.config import AppConfig
from core.detector import WasteDetector
from core.flagging import save_flagged_frame
from core.ood import OODDetector
from core.smoother import TemporalSmoother
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
    QSplitter,
    QStatusBar,
    Qt,
    Slot,
    QVBoxLayout,
    QWidget,
)
from ui.utils import FILE_DIALOG_IMAGE_FILTER, load_image_unicode
from ui.video_thread import VideoThread
from ui.webcam_thread import CameraState, WebcamThread, list_available_cameras
from ui.widgets.audit_dialog import AuditLogDialog
from ui.widgets.bin_banner import BinBanner
from ui.widgets.control_panel import ControlPanel
from ui.widgets.result_card import ResultList
from ui.widgets.video_view import VideoView
from ui.worker import InferenceWorker


class MainWindow(QMainWindow):
    """Primary GUI window for Trash-Net waste classification."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[AppConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.app_config = config or AppConfig.load()

        title = self.app_config.ui.window_title or DEFAULT_WINDOW_TITLE
        self.setWindowTitle(title)

        # DoD: Giao diện hiển thị đúng ở độ phân giải 1280×720 trở lên
        min_w = self.app_config.ui.min_width or MIN_WINDOW_WIDTH
        min_h = self.app_config.ui.min_height or MIN_WINDOW_HEIGHT
        self.setMinimumSize(min_w, min_h)
        self.resize(min_w, min_h)

        self._current_image_path: Optional[Path] = None
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: List[Detection] = []
        self._is_inferring_webcam: bool = False
        self._last_result_list_update_time: float = 0.0

        # Load detector if weights exist (checks models/ directory first)
        self.detector: Optional[WasteDetector] = None
        resolved_path = None

        # Resolve paths with support for PyInstaller frozen environment
        exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
        internal_dir = Path(getattr(sys, "_MEIPASS", exe_dir))

        config_weights = self.app_config.model.weights_path
        candidate_paths = [
            Path(model_path) if model_path and Path(model_path).exists() else None,
            Path(config_weights) if config_weights and Path(config_weights).exists() else None,
            exe_dir / "models" / "best.pt",
            exe_dir / "best.pt",
            internal_dir / "models" / "best.pt",
            internal_dir / "best.pt",
            Path("models/best.pt"),
            Path("best.pt"),
            exe_dir / "models" / "yolo11n-seg.pt",
            internal_dir / "models" / "yolo11n-seg.pt",
            Path("models/yolo11n-seg.pt"),
            Path("yolo11n-seg.pt"),
        ]
        for p in candidate_paths:
            if p and p.exists() and p.is_file():
                resolved_path = p
                break

        if resolved_path is not None:
            try:
                self.detector = WasteDetector(
                    model_path=resolved_path,
                    conf=self.app_config.model.conf,
                    iou=self.app_config.model.iou,
                    imgsz=self.app_config.model.imgsz,
                    device=self.app_config.model.device,
                    half=self.app_config.model.half,
                )
            except Exception as exc:
                print(f"[!] Warning: Could not initialize WasteDetector with '{resolved_path}': {exc}")

        self.current_worker: Optional[InferenceWorker] = None
        self._webcam_worker: Optional[InferenceWorker] = None
        self._video_thread: Optional[VideoThread] = None
        self.webcam_thread = WebcamThread(self)
        det_conf = self.detector.conf if hasattr(self, "detector") and self.detector is not None else 0.25
        self.ood_detector = OODDetector(conf_threshold=det_conf)
        self.audit_logger = AuditLogger()

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

        # 2.1 Bin Banner
        self.bin_banner = BinBanner(self)
        self.bin_banner.hide()

        # 2.2 Result Cards List (Middle expanding zone)
        self.result_list = ResultList(self)
        right_layout.addWidget(self.result_list, stretch=1)

        # 2.3 Control Panel (Bottom fixed actions)
        self.control_panel = ControlPanel(default_conf=0.60, parent=self)
        right_layout.addWidget(self.control_panel)

        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([920, 360])

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
            max_probe = self.app_config.camera.max_probe_index if hasattr(self, "app_config") else 6
            cameras = list_available_cameras(max_indices=max_probe)
            self.control_panel.populate_cameras(cameras)
        except Exception:
            self.control_panel.populate_cameras([0])

    def _connect_signals(self) -> None:
        """Wire up user interaction signals from ControlPanel, VideoView, and WebcamThread."""
        self.control_panel.open_image_requested.connect(self.open_image_dialog)
        self.control_panel.open_video_requested.connect(self.open_video_dialog)
        self.control_panel.toggle_webcam_requested.connect(self.on_toggle_webcam)
        self.control_panel.confidence_changed.connect(self.on_confidence_changed)
        self.control_panel.iou_changed.connect(self.on_iou_changed)
        self.control_panel.max_detections_changed.connect(self.on_max_detections_changed)
        self.control_panel.reset_requested.connect(self.reset_all)
        self.control_panel.overlay_options_changed.connect(self.video_view.set_overlay_options)
        self.control_panel.flag_requested.connect(self.on_flag_requested)
        self.control_panel.open_log_requested.connect(self.on_open_log_requested)

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
        """Start streaming from chosen camera index with size-1 buffer InferenceWorker.

        [GIẢI THÍCH NGHIỆP VỤ - KHỞI ĐỘNG LUỒNG WEBCAM THỜI GIAN THỰC]:
        [LUỒNG LÀM VIỆC]:
        1. Dừng phát video hoặc luồng phân tích tĩnh trước đó.
        2. Khởi tạo bộ làm mượt TemporalSmoother để chống rung nhấp nháy nhãn rác.
        3. Khởi tạo WebcamInferenceWorker với bộ đệm kích thước 1 (size-1 buffer)
           để luôn chỉ suy luận khung hình mới nhất, triệt tiêu độ trễ tích lũy.
        4. Bật luồng phần cứng WebcamThread đọc camera qua V4L2 backend.
        """
        # Stop any active video playback
        self.stop_video_playback()

        # Stop any active static file inference
        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        # Stop any existing webcam worker
        if self._webcam_worker is not None:
            self._webcam_worker.stop(timeout_ms=1000)
            self._webcam_worker = None

        # Initialize continuous InferenceWorker with size-1 buffer and temporal smoother
        if self.detector is not None:
            conf = self.control_panel.confidence_threshold
            iou = self.control_panel.iou_threshold
            topk = self.control_panel.max_detections
            smoother = None
            if self.app_config.smoother.enabled:
                smoother = TemporalSmoother(
                    window_size=self.app_config.smoother.window_size,
                    stability_threshold_sec=self.app_config.smoother.stability_threshold_sec,
                    disappear_timeout_sec=self.app_config.smoother.disappear_timeout_sec,
                    require_confirmation=self.app_config.smoother.require_confirmation,
                )
            self._webcam_worker = InferenceWorker(
                self.detector, conf=conf, iou=iou, max_detections=topk, smoother=smoother, parent=self
            )
            self._webcam_worker.finished_inference.connect(self._on_webcam_inference_finished)
            self._webcam_worker.fps_updated.connect(self.on_inference_fps_updated)
            self._webcam_worker.error_occurred.connect(self._on_webcam_inference_error)
            self.control_panel.confidence_changed.connect(self._webcam_worker.set_confidence)
            self.control_panel.iou_changed.connect(self._webcam_worker.set_iou)
            self.control_panel.max_detections_changed.connect(self._webcam_worker.set_max_detections)
            self._webcam_worker.start()

        self.status_bar.showMessage(f"Đang kết nối camera #{camera_index} qua V4L2...")
        success = self.webcam_thread.start_capture(camera_index=camera_index)
        if not success:
            self.status_bar.showMessage(f"Lỗi: Không thể khởi chạy camera #{camera_index}.")

    @Slot()
    def on_pause_webcam(self) -> None:
        """Pause webcam or video streaming: hold last frame and pause inference."""
        if self._video_thread is not None and self._video_thread.isRunning():
            self._video_thread.pause()
            self.status_bar.showMessage("Video tạm dừng — Giữ khung hình hiện tại.")
            return

        self.webcam_thread.pause_capture()
        self.status_bar.showMessage("Webcam tạm dừng — Giữ khung hình hiện tại.")

    @Slot()
    def on_resume_webcam(self) -> None:
        """Resume webcam or video streaming."""
        if self._video_thread is not None and self._video_thread.isRunning():
            self._video_thread.resume()
            self.status_bar.showMessage("Video tiếp tục phát.")
            return

        self.webcam_thread.resume_capture()
        self.status_bar.showMessage("Webcam tiếp tục phát trực tiếp.")

    @Slot()
    def on_stop_webcam(self) -> None:
        """Stop webcam or video stream and completely release hardware device (camera LED off)."""
        self.stop_video_playback()
        self.webcam_thread.stop_capture()
        if self._webcam_worker is not None:
            try:
                self.control_panel.confidence_changed.disconnect(self._webcam_worker.set_confidence)
                self.control_panel.iou_changed.disconnect(self._webcam_worker.set_iou)
                self.control_panel.max_detections_changed.disconnect(self._webcam_worker.set_max_detections)
            except Exception:
                pass
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
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not file_path:
            return

        self.load_image_file(Path(file_path))

    def load_image_file(self, path: Path) -> bool:
        """Load and display an image from file path with Unicode support and run inference.

        [GIẢI THÍCH NGHIỆP VỤ - NẠP ẢNH TĨNH & PHÂN TÍCH]:
        [LUỒNG LÀM VIỆC]:
        1. Dừng video hoặc webcam đang chạy nếu có để tránh xung đột GPU.
        2. Đọc file ảnh hỗ trợ đường dẫn tiếng Việt Unicode qua OpenCV (load_image_unicode).
        3. Cập nhật khung hiển thị VideoView với ảnh gốc.
        4. Kích hoạt luồng chạy ngầm (_start_inference) để mô hình YOLO dự đoán mà không đơ UI.
        """
        # Stop video playback or webcam if active when user opens an image
        self.stop_video_playback()
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
        """Run inference asynchronously using InferenceWorker.

        [GIẢI THÍCH NGHIỆP VỤ - SUY LUẬN NGẦM TRÁNH ĐƠ GIAO DIỆN]:
        [LUỒNG LÀM VIỆC]:
        1. Đọc các tham số ngưỡng hiện thời trên thanh điều khiển: conf, iou, topk.
        2. Tạo một tiến trình ngầm InferenceWorker chạy tách biệt khỏi Main GUI Thread.
        3. Kết nối tín hiệu (Signals/Slots): khi có kết quả -> _on_inference_finished.
        """
        if self.detector is None:
            self.status_bar.showMessage("Cảnh báo: Chưa nạp mô hình suy luận.")
            return

        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        conf = self.control_panel.confidence_threshold
        iou = self.control_panel.iou_threshold
        topk = self.control_panel.max_detections
        self.current_worker = InferenceWorker(self.detector, frame, conf=conf, iou=iou, max_detections=topk)
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
        """Present file chooser dialog for video streams and start inference-paced scan (Option 1)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tệp video rác",
            "",
            "Video (*.mp4 *.avi *.mkv *.mov);;Tất cả tệp (*.*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not file_path:
            return

        self.start_video_playback(Path(file_path))

    def start_video_playback(self, path: Union[str, Path]) -> None:
        """Bắt đầu quét và phân loại video từng khung hình đồng bộ theo tốc độ suy luận của AI.

        [LUỒNG LÀM VIỆC]:
        1. Dừng luồng webcam hoặc luồng phân tích ảnh tĩnh nếu đang chạy.
        2. Khởi tạo bộ làm mượt TemporalSmoother (nếu được cấu hình bật).
        3. Tạo luồng VideoThread và kết nối tín hiệu frame_processed về hàm cập nhật giao diện.
        """
        path = Path(path)
        if self.detector is None:
            self.status_bar.showMessage("Cảnh báo: Chưa nạp mô hình suy luận.")
            return

        # 1. Dừng webcam nếu đang phát
        if self.webcam_thread.isRunning():
            self.on_stop_webcam()

        # 2. Dừng luồng suy luận ảnh tĩnh nếu đang chạy
        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        # 3. Dừng video cũ nếu đang phát
        self.stop_video_playback()

        # 4. Khởi tạo bộ làm mượt nếu được bật trong cấu hình
        smoother = None
        if self.app_config.smoother.enabled:
            smoother = TemporalSmoother(
                window_size=self.app_config.smoother.window_size,
                stability_threshold_sec=self.app_config.smoother.stability_threshold_sec,
                disappear_timeout_sec=self.app_config.smoother.disappear_timeout_sec,
                require_confirmation=self.app_config.smoother.require_confirmation,
            )

        conf = self.control_panel.confidence_threshold
        iou = self.control_panel.iou_threshold
        topk = self.control_panel.max_detections

        self._video_thread = VideoThread(
            detector=self.detector,
            video_path=path,
            conf=conf,
            iou=iou,
            max_detections=topk,
            smoother=smoother,
            parent=self,
        )
        self._video_thread.frame_processed.connect(self._on_video_frame_processed)
        self._video_thread.fps_updated.connect(self.on_inference_fps_updated)
        self._video_thread.state_changed.connect(self._on_video_state_changed)
        self._video_thread.video_finished.connect(self._on_video_finished)
        self._video_thread.error_occurred.connect(self._on_video_error)

        self.control_panel.confidence_changed.connect(self._video_thread.set_confidence)
        self.control_panel.iou_changed.connect(self._video_thread.set_iou)
        self.control_panel.max_detections_changed.connect(self._video_thread.set_max_detections)

        self.status_bar.showMessage(f"Bắt đầu quét video: {path.name} (Đồng bộ nhịp AI)...")
        self._video_thread.start()

    def stop_video_playback(self) -> None:
        """Safely stop and clean up active video playback thread."""
        if self._video_thread is not None:
            try:
                self.control_panel.confidence_changed.disconnect(self._video_thread.set_confidence)
                self.control_panel.iou_changed.disconnect(self._video_thread.set_iou)
                self.control_panel.max_detections_changed.disconnect(self._video_thread.set_max_detections)
            except Exception:
                pass
            self._video_thread.stop(timeout_ms=2000)
            self._video_thread = None

    @Slot(np.ndarray, list, float, int, int)
    def _on_video_frame_processed(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        latency_ms: float,
        frame_idx: int,
        total_frames: int,
    ) -> None:
        """Handle synchronous frame-by-frame video detection result."""
        self._current_frame = frame
        self._current_detections = list(detections)
        self.video_view.set_frame(frame, detections)
        self.update_results(detections)
        self._inference_latency_ms = latency_ms

        pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        det_count = len(detections)
        self.status_bar.showMessage(
            f"Video: [{frame_idx}/{total_frames} ({pct:.0f}%)] | "
            f"AI: {self._inference_fps:.1f} FPS (Độ trễ: {latency_ms:.1f} ms) | "
            f"Phát hiện: {det_count} vật thể"
        )

    @Slot(str)
    def _on_video_state_changed(self, state: str) -> None:
        """Synchronize UI buttons with video playback state."""
        self.control_panel.set_camera_state(state)

    @Slot()
    def _on_video_finished(self) -> None:
        """Handle completion of video scanning."""
        self.status_bar.showMessage("Hoàn tất: Đã quét xong toàn bộ tệp video.")
        self.control_panel.set_camera_state("STOPPED")

    @Slot(str)
    def _on_video_error(self, err_msg: str) -> None:
        """Handle video decoding or playback failure."""
        self.status_bar.showMessage(f"Lỗi video: {err_msg}")
        self.control_panel.set_camera_state("STOPPED")
        QMessageBox.warning(self, "Lỗi đọc tệp video", f"Không thể xử lý video:\n{err_msg}")

    @Slot(bool)
    def on_toggle_webcam(self, active: bool) -> None:
        """Handle legacy toggle webcam signal."""
        pass

    @Slot(float)
    def on_confidence_changed(self, threshold: float) -> None:
        """Handle confidence threshold slider changes (US-4.1)."""
        self.status_bar.showMessage(f"Ngưỡng tin cậy cập nhật: {threshold:.2f} ({threshold * 100:.0f}%)")
        if hasattr(self, "ood_detector") and self.ood_detector is not None:
            self.ood_detector.conf_threshold = threshold
        if self._webcam_worker is not None and self._webcam_worker.is_worker_running:
            self._webcam_worker.set_confidence(threshold)
        elif self._current_frame is not None and self.detector is not None and not self.webcam_thread.isRunning():
            self._start_inference(self._current_frame)

    @Slot(float)
    def on_iou_changed(self, iou: float) -> None:
        """Handle NMS IoU threshold slider changes."""
        self.status_bar.showMessage(f"Ngưỡng IoU khử trùng lặp cập nhật: {iou:.2f}")
        if self._webcam_worker is not None and self._webcam_worker.is_worker_running:
            self._webcam_worker.set_iou(iou)
        elif self._current_frame is not None and self.detector is not None and not self.webcam_thread.isRunning():
            self._start_inference(self._current_frame)

    @Slot(int)
    def on_max_detections_changed(self, limit: int) -> None:
        """Handle Top-K maximum detections limit changes."""
        lbl = "Tất cả" if limit >= 9999 else f"Top {limit}"
        self.status_bar.showMessage(f"Giới hạn số lượng hiển thị: {lbl}")
        if self._webcam_worker is not None and self._webcam_worker.is_worker_running:
            self._webcam_worker.set_max_detections(limit)
        elif self._current_frame is not None and self.detector is not None and not self.webcam_thread.isRunning():
            self._start_inference(self._current_frame)

    def update_results(self, detections: List[Detection]) -> None:
        """Update detection cards and destination bin banner with UI throttling.

        [GIẢI THÍCH NGHIỆP VỤ - CẬP NHẬT KẾT QUẢ LÊN GIAO DIỆN]:
        [LUỒNG LÀM VIỆC]:
        1. Chạy qua bộ kiểm tra OODDetector để phát hiện các vật thể bất thường.
        2. Điều tiết tần suất cập nhật UI (Throttling tối đa 2 lần/giây khi streaming) để không lag giao diện.
        3. Cập nhật Banner thùng rác (BinBanner) theo vật thể có độ tin cậy cao nhất.
        4. Cập nhật Thẻ hướng dẫn xử lý rác (GuidelineCard) hiển thị mẹo chuẩn bị rác.
        5. Tự động ghi nhật ký kiểm toán (AuditLogger) nếu là ảnh tĩnh hoặc rác nguy hại.
        """
        checked = self.ood_detector.check_detections(detections) if hasattr(self, "ood_detector") else detections
        self._current_detections = list(checked)

        now = time.monotonic()
        is_streaming = hasattr(self, "webcam_thread") and self.webcam_thread.isRunning()

        # Throttle heavy QWidget creation for result cards to max 2 updates/sec in webcam mode
        if not is_streaming or (now - getattr(self, "_last_result_list_update_time", 0.0) >= 0.5):
            self.result_list.set_detections(self._current_detections)
            self._last_result_list_update_time = now

        if not self._current_detections:
            self.bin_banner.reset()
            self.status_bar.showMessage("Không phát hiện rác trong khung hình")
        else:
            top_detection = self._current_detections[0]
            self.bin_banner.set_bin(top_detection.bin_type)

            # US-4.5: Only log confirmed detections for static image or hazardous items
            if hasattr(self, "audit_logger") and self.audit_logger is not None:
                if not is_streaming:
                    for det in self._current_detections:
                        self.audit_logger.log_detection(
                            detection=det,
                            source="image",
                            is_hazardous=(det.bin_type == Bin.HAZARDOUS),
                        )
                else:
                    for det in self._current_detections:
                        if det.bin_type == Bin.HAZARDOUS:
                            self.audit_logger.log_detection(
                                detection=det,
                                source="webcam",
                                is_hazardous=True,
                            )

            self.status_bar.showMessage(
                f"Phát hiện {len(self._current_detections)} vật thể rác trong khung hình (cao nhất: {top_detection.class_name} {top_detection.confidence * 100:.0f}%)"
            )

    @Slot()
    def on_flag_requested(self) -> None:
        """Handle reporting and hazardous waste tagging (US-4.4).

        [GIẢI THÍCH NGHIỆP VỤ - BÁO CÁO & GẮN CỜ RÁC NGUY HẠI]:
        [LUỒNG LÀM VIỆC]:
        1. Hiển thị hộp thoại hỏi người dùng xác nhận loại rác (Nguy hại / Chưa chắc chắn).
        2. Nếu người dùng chọn 'Yes' hoặc 'No', mở tiếp hộp thoại nhập ghi chú lý do.
        3. Gọi save_flagged_frame() lưu ảnh JPEG và metadata JSON vào thư mục flagged/.
        4. Ghi nhận sự kiện vào hệ thống nhật ký kiểm toán AuditLogger.
        """
        if self._current_frame is None:
            self.status_bar.showMessage("Chưa có khung hình để báo cáo / gắn cờ.")
            return

        reply = QMessageBox.question(
            self,
            "Báo cáo / Gắn cờ",
            "Bạn có muốn gắn cờ khung hình này là RÁC NGUY HẠI (pin, hóa chất, thiết bị điện tử)?\n\n"
            "• Bấm 'Yes' để đánh dấu nguy hại (hazardous = True)\n"
            "• Bấm 'No' để đánh dấu không chắc chắn thông thường (hazardous = False)\n"
            "• Bấm 'Cancel' để hủy bỏ.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return

        is_hazardous = (reply == QMessageBox.StandardButton.Yes)
        try:
            img_path, json_path = save_flagged_frame(
                frame=self._current_frame,
                detections=self._current_detections,
                is_hazardous=is_hazardous,
                note="Báo cáo thủ công từ người dùng",
            )
            tag_name = "nguy hại (hazardous)" if is_hazardous else "không chắc chắn"
            self.status_bar.showMessage(f"Đã lưu khung hình {tag_name} vào: {img_path.name}")
            QMessageBox.information(
                self,
                "Đã lưu báo cáo",
                f"Đã lưu khung hình thành công!\n\nẢnh: {img_path.name}\nSiêu dữ liệu: {json_path.name}\nThư mục: flagged/",
            )
        except Exception as exc:
            self.status_bar.showMessage(f"Lỗi lưu báo cáo: {exc}")
            QMessageBox.critical(self, "Lỗi lưu báo cáo", str(exc))

    @Slot()
    def on_open_log_requested(self) -> None:
        """Mở cửa sổ xem nhật ký kiểm toán và xuất báo cáo CSV."""
        dialog = AuditLogDialog(self.audit_logger, parent=self)
        dialog.exec()

    @Slot()
    def reset_all(self) -> None:
        """Đặt lại toàn bộ trạng thái hệ thống: Dọn khung hình, xóa danh sách kết quả, đặt lại thùng rác."""
        self.stop_video_playback()

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
        """Sự kiện đóng cửa sổ: Giải phóng tài nguyên và dừng an toàn toàn bộ tiến trình ngầm."""
        """Ensure CaptureThread, VideoThread, and InferenceWorker halt cleanly within 2 seconds."""
        self.stop_video_playback()

        if hasattr(self, "_webcam_worker") and self._webcam_worker is not None:
            self._webcam_worker.stop(timeout_ms=2000)
            self._webcam_worker = None

        if hasattr(self, "current_worker") and self.current_worker is not None:
            self.current_worker.stop(timeout_ms=2000)
            self.current_worker = None

        if hasattr(self, "webcam_thread"):
            self.webcam_thread.stop_capture()

        super().closeEvent(event)


def run_app(argv: Optional[List[str]] = None) -> None:
    """Launch Trash-Net desktop application with dynamic CLI flag support.

    [GIẢI THÍCH NGHIỆP VỤ - CHẠY ỨNG DỤNG DESKTOP]:
    Điểm vào chính khởi chạy giao diện Desktop PyQt:
    1. Đọc cờ tham số dòng lệnh CLI (nếu có: --model, --conf, --iou, --device, --camera).
    2. Nạp file cấu hình config/app.yaml và áp dụng các giá trị ghi đè.
    3. Khởi tạo đối tượng QApplication và cửa sổ MainWindow (kích thước tối thiểu 1280x720).
    4. Hiển thị giao diện và bước vào vòng lặp sự kiện chính của ứng dụng.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Trash-Net — Hệ Thống Phân Loại Rác Thông Minh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to YOLO weights (.pt)")
    parser.add_argument("-c", "--conf", type=float, default=None, help="Detection confidence threshold (0.0, 1.0]")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (0.0, 1.0]")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference square image resolution")
    parser.add_argument("-d", "--device", type=str, default=None, help="Inference device: 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--half", action="store_true", default=None, help="Force FP16 half-precision inference")
    parser.add_argument("--config", type=str, default=None, help="Path to custom app.yaml configuration file")
    parser.add_argument("--camera", type=int, default=None, help="Default camera device index")

    cli_args = argv if argv is not None else sys.argv[1:]
    args, qt_args = parser.parse_known_args(cli_args)

    # 1. Load base configuration from YAML
    app_cfg = AppConfig.load(args.config)

    # 2. Apply CLI flag overrides dynamically
    if args.model:
        app_cfg.model.weights_path = args.model
    if args.conf is not None:
        app_cfg.model.conf = args.conf
    if args.iou is not None:
        app_cfg.model.iou = args.iou
    if args.imgsz is not None:
        app_cfg.model.imgsz = args.imgsz
    if args.device:
        app_cfg.model.device = args.device
    if args.half is not None:
        app_cfg.model.half = args.half
    if args.camera is not None:
        app_cfg.camera.default_index = args.camera

    app = QApplication([sys.argv[0]] + qt_args)
    window = MainWindow(config=app_cfg)
    window.show()
    sys.exit(app.exec())


# =============================================================================
# BÍ DANH HÀM TIẾNG VIỆT (VIETNAMESE FUNCTION ALIASES)
# =============================================================================
chay_ung_dung = run_app


if __name__ == "__main__":
    run_app()

