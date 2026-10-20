"""Unit and component integration tests for Trash-Net UI (US-3.1).

Tests MainWindow layout, BinBanner states, ResultCard formatting, and VideoView
aspect ratio preservation in headless offscreen environment.
"""

import os
import pytest
import numpy as np

# Force offscreen Qt platform for headless Linux CI/testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.types import Bin, Detection
from ui.constants import (
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    get_vietnamese_label,
)
from ui.main_window import MainWindow
from ui.qt_compat import QApplication, QSettings
from ui.widgets.bin_banner import BinBanner
from ui.widgets.control_panel import ControlPanel
from ui.widgets.result_card import ResultCard, ResultList
from ui.widgets.video_view import VideoView


@pytest.fixture(scope="session")
def qapp():
    """Ensure a singleton QApplication exists across test suite."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestBinBanner:
    """Verify BinBanner visual states and acceptance criteria."""

    def test_initial_state_is_idle(self, qapp):
        """DoD AC: Khi không có vật thể, biểu ngữ hiển thị 'Chưa phát hiện rác'."""
        banner = BinBanner()
        assert banner.is_idle is True
        assert banner.current_bin is None
        assert banner._title_label.text() == "Chưa phát hiện rác"

    def test_bin_organic_color_and_text(self, qapp):
        """Verify Organic bin style and green theme."""
        banner = BinBanner()
        banner.set_bin(Bin.ORGANIC)
        assert banner.is_idle is False
        assert banner.current_bin == Bin.ORGANIC
        assert banner._title_label.text() == "THÙNG HỮU CƠ"
        assert "#2E7D32" in banner.styleSheet()

    def test_bin_recyclable_color_and_text(self, qapp):
        """Verify Recyclable bin style and blue theme."""
        banner = BinBanner()
        banner.set_bin(Bin.RECYCLABLE)
        assert banner.is_idle is False
        assert banner.current_bin == Bin.RECYCLABLE
        assert banner._title_label.text() == "THÙNG TÁI CHẾ"
        assert "#1565C0" in banner.styleSheet()

    def test_bin_residual_color_and_text(self, qapp):
        """Verify Residual bin style and red theme."""
        banner = BinBanner()
        banner.set_bin(Bin.RESIDUAL)
        assert banner.is_idle is False
        assert banner.current_bin == Bin.RESIDUAL
        assert banner._title_label.text() == "THÙNG VÔ CƠ / KHÁC"
        assert "#C62828" in banner.styleSheet()

    def test_banner_reset(self, qapp):
        """Verify banner correctly returns to idle on reset."""
        banner = BinBanner()
        banner.set_bin(Bin.RECYCLABLE)
        banner.reset()
        assert banner.is_idle is True
        assert banner._title_label.text() == "Chưa phát hiện rác"


class TestResultCardAndList:
    """Verify detection result cards and localization."""

    def test_vietnamese_translations(self):
        """Verify Vietnamese naming for MultiTrash classes."""
        assert get_vietnamese_label("plastic") == "Nhựa"
        assert get_vietnamese_label("glass") == "Thủy tinh"
        assert get_vietnamese_label("biowaste") == "Rác hữu cơ"
        assert get_vietnamese_label("metal") == "Kim loại"
        assert get_vietnamese_label("paper") == "Giấy"
        assert get_vietnamese_label("styrofoam") == "Xốp"
        assert get_vietnamese_label("wrapper") == "Bao bì màng"

    def test_result_card_content(self, qapp):
        """DoD AC: Mỗi thẻ hiển thị tên lớp (tiếng Việt), độ tin cậy (%) và thùng đích."""
        det = Detection(
            class_id=5,
            class_name="plastic",
            confidence=0.942,
            box=(10, 20, 100, 200),
            polygon=[(10, 20), (100, 20), (100, 200), (10, 200)],
            bin_type=Bin.RECYCLABLE,
        )
        card = ResultCard(det, rank=1)
        assert card.name_label.text() == "Nhựa"
        assert card.conf_label.text() == "94.2%"
        assert "Tái chế" in card.bin_badge.text()

        # Check list insertion
        rlist = ResultList()
        assert rlist.count == 0
        rlist.set_detections([det])
        assert rlist.count == 1

    def test_result_list_empty_and_clear(self, qapp):
        """Verify ResultList empty state and clear behavior."""
        rlist = ResultList()
        det = Detection(
            class_id=0,
            class_name="biowaste",
            confidence=0.88,
            box=(0, 0, 50, 50),
            polygon=[(0, 0), (50, 50)],
            bin_type=Bin.ORGANIC,
        )
        rlist.set_detections([det])
        assert rlist.count == 1
        rlist.clear()
        assert rlist.count == 0


class TestVideoView:
    """Verify VideoView frame handling and aspect ratio logic."""

    def test_video_view_empty_state(self, qapp):
        """Test initial state without frame."""
        view = VideoView()
        assert view.has_frame is False
        assert view.frame_resolution == (0, 0)

    def test_video_view_set_frame_numpy(self, qapp):
        """Test setting OpenCV BGR numpy frame."""
        view = VideoView()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        view.set_frame(dummy_frame)
        assert view.has_frame is True
        assert view.frame_resolution == (640, 480)

        view.clear_frame()
        assert view.has_frame is False


class TestControlPanel:
    """Verify ControlPanel user interaction elements (US-3.1, US-4.1)."""

    def test_default_confidence_and_slider(self, qapp):
        """Test default confidence threshold 0.60 and slider bounds [0.10, 0.95] with step 0.05."""
        panel = ControlPanel(use_settings=False)
        assert abs(panel.confidence_threshold - 0.60) < 0.01
        assert panel._slider.minimum() == 2   # 2 * 0.05 = 0.10
        assert panel._slider.maximum() == 19  # 19 * 0.05 = 0.95
        assert panel._slider.singleStep() == 1  # 0.05 step
        assert panel._conf_val_label.text() == "0.60"

    def test_slider_change_emits_and_updates_label(self, qapp):
        """DoD AC: Khi kéo thanh trượt, ngưỡng áp dụng ngay và hiển thị cạnh thanh trượt."""
        panel = ControlPanel(use_settings=False)
        received = []
        panel.confidence_changed.connect(lambda val: received.append(val))

        # Change to 0.75 (slider index 15)
        panel._slider.setValue(15)
        assert len(received) == 1
        assert abs(received[0] - 0.75) < 0.01
        assert panel.confidence_threshold == 0.75
        assert panel._conf_val_label.text() == "0.75"

    def test_set_confidence_threshold_helper(self, qapp):
        """Verify programmatic threshold adjustment."""
        panel = ControlPanel(use_settings=False)
        panel.set_confidence_threshold(0.80)
        assert panel.confidence_threshold == 0.80
        assert panel._slider.value() == 16
        assert panel._conf_val_label.text() == "0.80"

    def test_qsettings_persistence_us41(self, qapp, tmp_path):
        """DoD AC: Ngưỡng được ghi nhớ cho lần mở sau qua QSettings."""
        settings = QSettings(str(tmp_path / "test_settings.ini"), QSettings.Format.IniFormat)
        settings.clear()

        # Session 1: user opens panel and adjusts threshold to 0.85
        panel1 = ControlPanel(settings=settings)
        panel1.set_confidence_threshold(0.85)
        assert panel1.confidence_threshold == 0.85
        assert float(settings.value("confidence_threshold")) == 0.85

        # Session 2: next app launch reads remembered threshold
        panel2 = ControlPanel(settings=settings)
        assert panel2.confidence_threshold == 0.85
        assert panel2._conf_val_label.text() == "0.85"

    def test_worker_set_confidence_thread_safe(self, qapp):
        """US-4.1: Verify InferenceWorker thread-safe confidence updating."""
        from unittest.mock import MagicMock
        from ui.worker import InferenceWorker
        detector = MagicMock()
        worker = InferenceWorker(detector, conf=0.60)
        assert worker._conf == 0.60
        assert worker._pending_conf == 0.60

        worker.set_confidence(0.85)
        assert worker._conf == 0.85
        assert worker._pending_conf == 0.85

    def test_webcam_toggle_state(self, qapp):
        """Test webcam button toggle state and state machine updates."""
        panel = ControlPanel()
        assert panel.is_webcam_active is False

        # Test start signal emission on click when inactive
        start_emitted = []
        panel.start_webcam_requested.connect(lambda idx: start_emitted.append(idx))
        panel._on_webcam_clicked()
        assert len(start_emitted) == 1

        # Simulate state transition to RUNNING
        panel.set_camera_state("RUNNING")
        assert panel.is_webcam_active is True

        # Test stop signal emission when active
        stop_emitted = []
        panel.stop_webcam_requested.connect(lambda: stop_emitted.append(True))
        panel._on_webcam_clicked()
        assert len(stop_emitted) == 1

        # Simulate state transition to STOPPED
        panel.set_camera_state("STOPPED")
        assert panel.is_webcam_active is False


class TestMainWindow:
    """Verify MainWindow satisfies US-3.1 acceptance criteria and DoD."""

    def test_window_resolution_dod(self, qapp):
        """DoD: Giao diện hiển thị đúng ở độ phân giải 1280×720 trở lên."""
        win = MainWindow()
        assert win.minimumWidth() >= 1280
        assert win.minimumHeight() >= 720
        assert win.width() >= MIN_WINDOW_WIDTH
        assert win.height() >= MIN_WINDOW_HEIGHT

    def test_child_widgets_presence(self, qapp):
        """Verify all mandatory zones are mounted in MainWindow."""
        win = MainWindow()
        assert hasattr(win, "video_view")
        assert isinstance(win.video_view, VideoView)

        assert hasattr(win, "bin_banner")
        assert isinstance(win.bin_banner, BinBanner)

        assert hasattr(win, "result_list")
        assert isinstance(win.result_list, ResultList)

        assert hasattr(win, "control_panel")
        assert isinstance(win.control_panel, ControlPanel)

    def test_update_results_updates_banner_and_list(self, qapp):
        """Verify update_results orchestrates bin banner and cards list."""
        win = MainWindow()

        # 1. Empty detections -> Banner idle
        win.update_results([])
        assert win.bin_banner.is_idle is True
        assert win.result_list.count == 0

        # 2. With detections -> Banner top-1 bin
        det1 = Detection(
            class_id=1,
            class_name="glass",
            confidence=0.91,
            box=(0, 0, 100, 100),
            polygon=[(0, 0), (100, 100)],
            bin_type=Bin.RECYCLABLE,
        )
        det2 = Detection(
            class_id=0,
            class_name="biowaste",
            confidence=0.45,
            box=(0, 0, 50, 50),
            polygon=[(0, 0), (50, 50)],
            bin_type=Bin.ORGANIC,
        )
        win.update_results([det1, det2])
        assert win.bin_banner.is_idle is False
        assert win.bin_banner.current_bin == Bin.RECYCLABLE
        assert win.result_list.count == 2


class TestImageLoadingAndDragDropUS33:
    """Test suite for US-3.3: Image Loading, Drag & Drop, and Non-blocking Worker."""

    def test_unicode_image_loading(self, tmp_path):
        """AC: Read image from path containing Vietnamese / Unicode diacritics."""
        from ui.utils import load_image_unicode
        import cv2

        unicode_file = tmp_path / "ảnh_rác_thực_nghiệm_2026.jpg"
        synthetic_img = np.zeros((200, 300, 3), dtype=np.uint8)
        synthetic_img[50:150, 50:250] = [0, 255, 0]

        _, buf = cv2.imencode(".jpg", synthetic_img)
        with open(unicode_file, "wb") as f:
            f.write(buf)

        loaded = load_image_unicode(unicode_file)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (200, 300, 3)

    def test_video_view_drag_drop_single_file(self, qapp, tmp_path):
        """AC: Dropping a single valid image emits image_dropped with Path."""
        from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PyQt6.QtGui import QDropEvent
        import cv2

        view = VideoView()
        dropped_files = []
        view.image_dropped.connect(lambda p: dropped_files.append(p))

        img_file = tmp_path / "sample_test.jpg"
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        with open(img_file, "wb") as f:
            f.write(buf)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(img_file))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(event)

        assert len(dropped_files) == 1
        assert dropped_files[0] == img_file

    def test_video_view_drag_drop_multiple_files_processes_first(self, qapp, tmp_path):
        """AC: When dropping multiple files, process the first and emit warning."""
        from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PyQt6.QtGui import QDropEvent
        import cv2

        view = VideoView()
        dropped_files = []
        warnings = []
        view.image_dropped.connect(lambda p: dropped_files.append(p))
        view.multiple_files_warning.connect(lambda count: warnings.append(count))

        f1 = tmp_path / "first.jpg"
        f2 = tmp_path / "second.png"
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        f1.write_bytes(buf)
        f2.write_bytes(buf)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(f1)), QUrl.fromLocalFile(str(f2))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(event)

        assert len(warnings) == 1
        assert warnings[0] == 2
        assert len(dropped_files) == 1
        assert dropped_files[0] == f1

    def test_video_view_drag_drop_invalid_file_emits_rejection(self, qapp, tmp_path):
        """AC: Dropping non-image file emits rejection message without crashing."""
        from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
        from PyQt6.QtGui import QDropEvent

        view = VideoView()
        rejections = []
        view.file_rejected.connect(lambda msg: rejections.append(msg))

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_text("not an image")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(pdf_file))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(event)

        assert len(rejections) == 1
        assert "không phải là ảnh hợp lệ" in rejections[0]

    def test_dod_large_image_responsiveness_4000px(self, qapp):
        """DoD: Test inference on large image (>= 4000 px) runs on worker thread without UI freeze."""
        win = MainWindow(model_path="yolo11n-seg.pt")

        large_frame = np.zeros((4000, 4000, 3), dtype=np.uint8)
        import cv2
        cv2.circle(large_frame, (2000, 2000), 400, (0, 255, 0), -1)

        completed_events = []
        orig_callback = win._on_inference_finished

        def mock_callback(frame, detections, latency_ms):
            completed_events.append((frame.shape, len(detections), latency_ms))
            orig_callback(frame, detections, latency_ms)

        win._on_inference_finished = mock_callback

        win._start_inference(large_frame)

        assert win.current_worker is not None
        win.current_worker.wait(15000)
        qapp.processEvents()

        assert len(completed_events) == 1
        shape, det_count, latency_ms = completed_events[0]
        assert shape == (4000, 4000, 3)
        assert latency_ms > 0


class TestWebcamStreamUS33:
    """Acceptance and unit tests for US-3.3 Chế độ B (Webcam thời gian thực)."""

    def test_camera_state_machine_and_lifecycle(self, qapp, monkeypatch):
        """Test Start -> Pause -> Resume -> Stop state machine and hardware release."""
        from unittest.mock import MagicMock
        from ui.webcam_thread import CameraState, WebcamThread

        thread = WebcamThread()
        assert thread.current_state == CameraState.IDLE

        # Mock cv2.VideoCapture to avoid real hardware access
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)

        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        states_observed = []
        frames_observed = []
        thread.state_changed.connect(lambda s: states_observed.append(s))
        thread.frame_received.connect(lambda f: frames_observed.append(f))

        # 1. Start capture -> transitions to RUNNING
        thread.start_capture(camera_index=0, target_fps=60)
        thread.wait(100)
        qapp.processEvents()

        assert thread.current_state == CameraState.RUNNING
        assert "RUNNING" in states_observed

        # 2. Pause capture -> transitions to PAUSED
        thread.pause_capture()
        qapp.processEvents()
        assert thread.current_state == CameraState.PAUSED
        assert "PAUSED" in states_observed

        # 3. Resume capture -> transitions back to RUNNING
        thread.resume_capture()
        qapp.processEvents()
        assert thread.current_state == CameraState.RUNNING

        # 4. Stop capture -> releases device, transitions to STOPPED
        thread.stop_capture()
        qapp.processEvents()
        assert thread.current_state == CameraState.STOPPED
        assert "STOPPED" in states_observed
        assert mock_cap.release.called

    def test_camera_open_failure_emits_error(self, qapp, monkeypatch):
        """Test error dialog and STOPPED state when camera is occupied or missing."""
        from unittest.mock import MagicMock
        from ui.webcam_thread import CameraState, WebcamThread

        thread = WebcamThread()

        # Mock unavailable camera
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        errors_emitted = []
        thread.camera_error.connect(lambda msg: errors_emitted.append(msg))

        thread.start_capture(camera_index=1)
        thread.wait(500)
        qapp.processEvents()

        assert thread.current_state == CameraState.STOPPED
        assert len(errors_emitted) == 1
        assert "Không thể mở camera thiết bị #1" in errors_emitted[0]

    def test_camera_unplugged_disconnection_mid_stream(self, qapp, monkeypatch):
        """Test graceful STOPPED transition and error emission when camera is unplugged mid-stream."""
        from unittest.mock import MagicMock
        from ui.webcam_thread import CameraState, WebcamThread

        thread = WebcamThread()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Return consecutive read failures simulating camera disconnection
        mock_cap.read.return_value = (False, None)
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        errors_emitted = []
        thread.camera_error.connect(lambda msg: errors_emitted.append(msg))

        thread.start_capture(camera_index=0)
        thread.wait(1000)
        qapp.processEvents()

        assert thread.current_state == CameraState.STOPPED
        assert len(errors_emitted) == 1
        assert "Mất kết nối với camera #0" in errors_emitted[0]
        assert mock_cap.release.called

    def test_control_panel_camera_dropdown_and_pause_resume_buttons(self, qapp):
        """Test camera index selection and Pause/Resume button triggers in ControlPanel."""
        panel = ControlPanel()

        # 1. Camera list population
        panel.populate_cameras([0, 2])
        assert panel.camera_combo.count() == 2
        assert panel.selected_camera_index == 0

        panel.camera_combo.setCurrentIndex(1)
        assert panel.selected_camera_index == 2

        # 2. Pause/Resume button behavior
        pause_emitted = []
        resume_emitted = []
        panel.pause_webcam_requested.connect(lambda: pause_emitted.append(True))
        panel.resume_webcam_requested.connect(lambda: resume_emitted.append(True))

        # Initial: disabled
        assert not panel._btn_pause.isEnabled()

        # Running state: pause enabled
        panel.set_camera_state("RUNNING")
        assert panel._btn_pause.isEnabled()
        assert panel._btn_pause.text() == "⏸️ Tạm dừng"
        panel._on_pause_clicked()
        assert len(pause_emitted) == 1

        # Paused state: resume enabled
        panel.set_camera_state("PAUSED")
        assert panel._btn_pause.isEnabled()
        assert panel._btn_pause.text() == "▶️ Tiếp tục"
        panel._on_pause_clicked()
        assert len(resume_emitted) == 1

    def test_main_window_close_event_releases_webcam(self, qapp, monkeypatch):
        """DoD: Verify closing MainWindow completely halts and releases camera stream."""
        from unittest.mock import MagicMock
        from ui.qt_compat import QCloseEvent

        win = MainWindow()
        mock_thread = MagicMock()
        win.webcam_thread = mock_thread

        close_event = QCloseEvent()
        win.closeEvent(close_event)

        assert mock_thread.stop_capture.called


class TestConcurrencyModelUS34:
    """Acceptance and unit tests for US-3.4 (Mô hình đồng thời CaptureThread và InferenceWorker)."""

    def test_size_one_buffer_latest_frame_wins(self, qapp):
        """DoD AC: Worker chỉ xử lý khung hình mới nhất; khung cũ bị ghi đè không tích tụ độ trễ."""
        from unittest.mock import MagicMock
        import time
        from ui.worker import InferenceWorker

        mock_detector = MagicMock()
        # Simulate non-zero inference time
        def mock_detect(frame, conf=None):
            time.sleep(0.04)
            return []

        mock_detector.detect.side_effect = mock_detect

        worker = InferenceWorker(detector=mock_detector)
        worker.start()

        processed_frames = []
        worker.finished_inference.connect(lambda f, det, lat: processed_frames.append(int(f[0, 0, 0])))

        # Frame 1: identifier = 10
        f1 = np.full((100, 100, 3), 10, dtype=np.uint8)
        # Frame 2: identifier = 20
        f2 = np.full((100, 100, 3), 20, dtype=np.uint8)
        # Frame 3: identifier = 30
        f3 = np.full((100, 100, 3), 30, dtype=np.uint8)

        # Submit f1 (worker picks it up)
        worker.submit_frame(f1)
        time.sleep(0.01)

        # While worker is busy with f1, rapidly submit f2 and f3
        worker.submit_frame(f2)
        worker.submit_frame(f3)  # f3 must overwrite f2 in the size-1 buffer

        # Wait for worker to finish f1 and then f3
        time.sleep(0.12)
        worker.stop(timeout_ms=1000)
        qapp.processEvents()

        # f1 was processed, f2 was dropped, f3 was processed
        assert 10 in processed_frames
        assert 30 in processed_frames
        assert 20 not in processed_frames  # Frame 2 was overwritten (latest-frame-wins)

    def test_status_bar_displays_dual_fps(self, qapp):
        """DoD AC: Đo FPS thu hình và FPS suy luận, hiển thị trên thanh trạng thái."""
        win = MainWindow()
        win.on_capture_fps_updated(29.8)
        win.on_inference_fps_updated(18.4)
        win._on_webcam_inference_finished(np.zeros((10, 10, 3), dtype=np.uint8), [], 54.3)

        status_text = win.status_bar.currentMessage()
        assert "Camera: 29.8 FPS" in status_text
        assert "AI: 18.4 FPS" in status_text
        assert "Độ trễ: 54.3 ms" in status_text

    def test_shutdown_within_two_seconds(self, qapp, monkeypatch):
        """DoD AC: Khi đóng cửa sổ, thì cả hai luồng dừng sạch trong vòng 2 giây."""
        from unittest.mock import MagicMock
        import time
        from ui.qt_compat import QCloseEvent
        from ui.webcam_thread import CaptureThread
        from ui.worker import InferenceWorker

        win = MainWindow()

        # Mock detector with fast return
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        win.detector = mock_detector

        # Start capture thread with mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        win.on_start_webcam(0)
        qapp.processEvents()

        assert win.webcam_thread.isRunning()
        assert win._webcam_worker is not None and win._webcam_worker.isRunning()

        # Trigger closeEvent and measure shutdown duration
        t_start = time.perf_counter()
        close_event = QCloseEvent()
        win.closeEvent(close_event)
        elapsed = time.perf_counter() - t_start

        # Verify stop within 2.0 seconds
        assert elapsed < 2.0
        assert not win.webcam_thread.isRunning()
        assert win._webcam_worker is None

    def test_stress_50_iterations_start_stop_no_deadlock(self, qapp, monkeypatch):
        """DoD: Kiểm thử dừng/khởi động lặp lại 50 lần để phát hiện đua luồng & không có deadlock."""
        from unittest.mock import MagicMock
        from ui.webcam_thread import CaptureThread
        from ui.worker import InferenceWorker

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        # Run 50 start/stop cycles
        for i in range(50):
            worker = InferenceWorker(detector=mock_detector)
            worker.start()

            cap = CaptureThread()
            cap.start_capture(0, target_fps=120)

            frame = np.full((100, 100, 3), i % 255, dtype=np.uint8)
            worker.submit_frame(frame)

            cap.stop_capture()
            worker.stop(timeout_ms=500)

            assert not cap.isRunning()
            assert not worker.isRunning()


class TestOverlayRenderingUS35:
    """Acceptance and unit tests for US-3.5 (Vẽ lớp phủ phân vùng)."""

    def test_control_panel_overlay_toggles(self, qapp):
        """DoD AC: Có thể bật/tắt mặt nạ, hộp bao và nhãn từ bảng điều khiển."""
        panel = ControlPanel()

        # Defaults are all True
        assert panel.show_masks is True
        assert panel.show_boxes is True
        assert panel.show_labels is True

        emitted_signals = []
        panel.overlay_options_changed.connect(
            lambda m, b, l: emitted_signals.append((m, b, l))
        )

        # Toggle masks off
        panel._chk_masks.setChecked(False)
        assert panel.show_masks is False
        assert len(emitted_signals) == 1
        assert emitted_signals[-1] == (False, True, True)

        # Toggle boxes off
        panel._chk_boxes.setChecked(False)
        assert panel.show_boxes is False
        assert len(emitted_signals) == 2
        assert emitted_signals[-1] == (False, False, True)

        # Toggle labels off
        panel._chk_labels.setChecked(False)
        assert panel.show_labels is False
        assert len(emitted_signals) == 3
        assert emitted_signals[-1] == (False, False, False)

    def test_boundary_clipping_prevention(self, qapp):
        """DoD AC: Nhãn hiển thị tên lớp và độ tin cậy, không bị cắt ở mép ảnh."""
        view = VideoView()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Detection touching top-left boundary (0, 0)
        det_top_left = Detection(
            box=(0, 0, 80, 80),
            confidence=0.92,
            class_id=0,
            class_name="Cardboard",
            bin_type=Bin.RECYCLABLE,
            polygon=[(0, 0), (80, 0), (80, 80), (0, 80)],
        )

        # 2. Detection touching bottom-right boundary (560, 400, 640, 480)
        det_bottom_right = Detection(
            box=(560, 400, 640, 480),
            confidence=0.88,
            class_id=1,
            class_name="Food Organics",
            bin_type=Bin.ORGANIC,
            polygon=[(560, 400), (640, 400), (640, 480), (560, 480)],
        )

        # Overlay render should safely execute without OpenCV assertions or clipping errors
        annotated = view._apply_overlay(frame, [det_top_left, det_bottom_right])
        assert annotated.shape == (480, 640, 3)

        # Color has been drawn on the canvas (non-zero pixels)
        assert np.count_nonzero(annotated) > 0

    def test_dod_overlapping_objects_overlay(self, qapp):
        """DoD: Hiển thị đúng với ảnh có nhiều vật thể chồng lấn."""
        view = VideoView()
        frame = np.full((400, 400, 3), 40, dtype=np.uint8)

        # 3 overlapping objects with different bins
        d1 = Detection(
            box=(50, 50, 200, 200),
            confidence=0.95,
            class_id=0,
            class_name="Cardboard",
            bin_type=Bin.RECYCLABLE,
            polygon=[(50, 50), (200, 50), (200, 200), (50, 200)],
        )
        d2 = Detection(
            box=(100, 100, 250, 250),
            confidence=0.89,
            class_id=1,
            class_name="Food Organics",
            bin_type=Bin.ORGANIC,
            polygon=[(100, 100), (250, 100), (250, 250), (100, 250)],
        )
        d3 = Detection(
            box=(150, 150, 300, 300),
            confidence=0.82,
            class_id=2,
            class_name="Miscellaneous Trash",
            bin_type=Bin.RESIDUAL,
            polygon=[(150, 150), (300, 150), (300, 300), (150, 300)],
        )

        view.set_frame(frame, [d1, d2, d3])
        assert view.has_frame is True
        assert view._pixmap is not None
        assert not view._pixmap.isNull()

        # Toggle off masks, keep boxes & labels
        view.set_overlay_options(show_masks=False, show_boxes=True, show_labels=True)
        assert view.show_masks is False
        assert view.show_boxes is True

        # Toggle off all overlays
        view.set_overlay_options(show_masks=False, show_boxes=False, show_labels=False)
        annotated_none = view._apply_overlay(frame, [d1, d2, d3])
        # When all overlays are off, frame should be identical to raw input
        np.testing.assert_array_equal(annotated_none, frame)

    def test_rendering_performance_under_10_percent_drop(self, qapp):
        """DoD AC: Việc vẽ không làm giảm FPS giao diện quá 10% (độ trễ vẽ < 5ms)."""
        import time

        view = VideoView()
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)

        det = Detection(
            box=(100, 100, 400, 400),
            confidence=0.90,
            class_id=0,
            class_name="Cardboard",
            bin_type=Bin.RECYCLABLE,
            polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
        )

        # Warm up
        view._apply_overlay(frame, [det])

        # Benchmark 50 overlay frames
        t0 = time.perf_counter()
        for _ in range(50):
            view._apply_overlay(frame, [det])
        avg_render_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

        # Rendering overlay should be blazing fast (< 5ms on CPU)
        assert avg_render_ms < 5.0



