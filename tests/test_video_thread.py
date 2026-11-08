"""Unit tests for VideoThread and video classification integration (Option 1)."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.types import Bin, Detection
from ui.main_window import MainWindow
from ui.qt_compat import QApplication
from ui.video_thread import VideoThread


@pytest.fixture(scope="session")
def qapp():
    """Ensure a singleton QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """Create a temporary 5-frame video for testing."""
    video_path = tmp_path / "test_waste.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (64, 64),
    )
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestVideoThread:
    """Test VideoThread lifecycle, frame synchronization, and controls."""

    def test_video_thread_initialization(self, sample_video):
        """Verify thread starts with proper defaults and parameters."""
        mock_detector = MagicMock()
        thread = VideoThread(
            detector=mock_detector,
            video_path=sample_video,
            conf=0.35,
            iou=0.5,
            max_detections=10,
        )
        assert thread._conf == 0.35
        assert thread._iou == 0.5
        assert thread._max_detections == 10
        assert not thread.is_running
        assert not thread.is_paused

    def test_video_thread_processes_frames_synchronously(self, qapp, sample_video):
        """Verify all frames are scanned without dropping and finished signal is emitted."""
        mock_detector = MagicMock()
        sample_det = Detection(
            class_id=1,
            class_name="glass",
            confidence=0.88,
            box=(10, 10, 50, 50),
            polygon=[(10, 10), (50, 50)],
            bin_type=Bin.RECYCLABLE,
        )
        mock_detector.detect.return_value = [sample_det]

        thread = VideoThread(detector=mock_detector, video_path=sample_video)
        frames_received = []
        states_received = []
        is_finished = []

        thread.frame_processed.connect(lambda f, d, l, idx, tot: frames_received.append((idx, tot, d)))
        thread.state_changed.connect(lambda s: states_received.append(s))
        thread.video_finished.connect(lambda: is_finished.append(True))

        thread.start()
        thread.wait(5000)
        qapp.processEvents()

        assert len(frames_received) == 5
        assert frames_received[0][0] == 1
        assert frames_received[-1][0] == 5
        assert frames_received[-1][1] == 5
        assert len(frames_received[0][2]) == 1
        assert frames_received[0][2][0].class_name == "glass"
        assert is_finished == [True]
        assert "STOPPED" in states_received

    def test_video_thread_pause_and_resume(self, qapp, sample_video):
        """Verify pause and resume states."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        thread = VideoThread(detector=mock_detector, video_path=sample_video)

        thread.start()
        thread.pause()
        assert thread.is_paused
        thread.resume()
        assert not thread.is_paused
        thread.stop()
        assert not thread.is_running

    def test_video_thread_invalid_file(self, qapp, tmp_path):
        """Verify error is emitted when video cannot be opened."""
        mock_detector = MagicMock()
        invalid_path = tmp_path / "non_existent.mp4"
        thread = VideoThread(detector=mock_detector, video_path=invalid_path)

        errors = []
        thread.error_occurred.connect(lambda msg: errors.append(msg))
        thread.start()
        thread.wait(3000)
        qapp.processEvents()

        assert len(errors) == 1
        assert "Không thể mở tệp video" in errors[0]


class TestMainWindowVideoIntegration:
    """Test video playback integration in MainWindow."""

    def test_start_video_playback_in_main_window(self, qapp, sample_video):
        """Verify MainWindow start_video_playback boots VideoThread and updates UI."""
        win = MainWindow()
        win.app_config.smoother.enabled = False
        mock_detector = MagicMock()
        sample_det = Detection(
            class_id=0,
            class_name="plastic",
            confidence=0.9,
            box=(0, 0, 30, 30),
            polygon=[(0, 0), (30, 30)],
            bin_type=Bin.RECYCLABLE,
        )
        mock_detector.detect.return_value = [sample_det]
        win.detector = mock_detector

        win.start_video_playback(sample_video)
        assert win._video_thread is not None
        assert win._video_thread.isRunning()

        # Wait for short video to complete
        win._video_thread.wait(5000)
        qapp.processEvents()

        assert len(win._current_detections) == 1
        assert win._current_detections[0].class_name == "plastic"

        # Cleanup
        win.reset_all()
        assert win._video_thread is None
