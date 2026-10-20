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
from ui.qt_compat import QApplication
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
    """Verify ControlPanel user interaction elements."""

    def test_default_confidence_and_slider(self, qapp):
        """Test default confidence threshold and slider bounds."""
        panel = ControlPanel(default_conf=0.25)
        assert abs(panel.confidence_threshold - 0.25) < 0.01

        # Test slider emission
        received = []
        panel.confidence_changed.connect(lambda val: received.append(val))
        panel._slider.setValue(60)
        assert len(received) == 1
        assert abs(received[0] - 0.60) < 0.01

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

