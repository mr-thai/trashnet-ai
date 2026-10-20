"""Main application window for Trash-Net waste classification (US-3.1).

Integrates VideoView, BinBanner, ResultList, and ControlPanel within a responsive
QSplitter layout complying with DoD >= 1280x720.
"""

from pathlib import Path
import sys
from typing import List, Optional
import cv2

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
    QSize,
    QSplitter,
    QStatusBar,
    Qt,
    Slot,
    QVBoxLayout,
    QWidget,
)
from ui.widgets.bin_banner import BinBanner
from ui.widgets.control_panel import ControlPanel
from ui.widgets.result_card import ResultList
from ui.widgets.video_view import VideoView


class MainWindow(QMainWindow):
    """Primary GUI window for Trash-Net waste classification.

    Attributes:
        video_view (VideoView): Left-hand panel aspect-ratio preserving display.
        bin_banner (BinBanner): Prominent top-right sorting bin alert.
        result_list (ResultList): Scrollable list of recognized waste categories.
        control_panel (ControlPanel): Bottom-right action and confidence sliders.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)

        # DoD: Giao diện hiển thị đúng ở độ phân giải 1280×720 trở lên
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.resize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self._current_image_path: Optional[Path] = None
        self._current_detections: List[Detection] = []

        self._init_ui()
        self._connect_signals()
        self._init_status_bar()

        # Apply global dark styling
        self.setStyleSheet(DARK_PALETTE_QSS)

    def _init_ui(self) -> None:
        """Create and position primary UI zones with QSplitter layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(0)

        # Horizontal Splitter between Left (Video) and Right (Results & Controls)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("MainSplitter")

        # 1. Left Viewport: Aspect-Ratio Preserving Video/Image View
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
        self.status_bar.showMessage("Hệ thống sẵn sàng | Độ phân giải: 1280×720 | Chờ nguồn dữ liệu")

    def _connect_signals(self) -> None:
        """Wire up user interaction signals from ControlPanel."""
        self.control_panel.open_image_requested.connect(self.open_image_dialog)
        self.control_panel.open_video_requested.connect(self.open_video_dialog)
        self.control_panel.toggle_webcam_requested.connect(self.on_toggle_webcam)
        self.control_panel.confidence_changed.connect(self.on_confidence_changed)
        self.control_panel.reset_requested.connect(self.reset_all)

    @Slot()
    def open_image_dialog(self) -> None:
        """Present file chooser dialog for static images."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tệp ảnh rác",
            "",
            "Hình ảnh (*.jpg *.jpeg *.png *.bmp *.webp);;Tất cả tệp (*.*)",
        )
        if not file_path:
            return

        self.load_image_file(Path(file_path))

    def load_image_file(self, path: Path) -> bool:
        """Load and display an image from file path.

        Args:
            path: Path to image file.

        Returns:
            bool: True if image loaded successfully, False otherwise.
        """
        if not path.exists():
            self.status_bar.showMessage(f"Lỗi: Không tìm thấy tệp '{path}'")
            return False

        # Read image using OpenCV (BGR format)
        img = cv2.imread(str(path))
        if img is None:
            self.status_bar.showMessage(f"Lỗi: Không thể giải mã ảnh '{path.name}'")
            return False

        self._current_image_path = path
        self.video_view.set_frame(img)
        h, w = img.shape[:2]
        self.status_bar.showMessage(f"Đã mở ảnh: {path.name} ({w}×{h} px)")
        return True

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
        """Handle webcam toggle events."""
        if active:
            self.status_bar.showMessage("Đang kích hoạt camera trực tiếp...")
        else:
            self.status_bar.showMessage("Đã dừng webcam.")

    @Slot(float)
    def on_confidence_changed(self, threshold: float) -> None:
        """Handle confidence threshold slider changes."""
        self.status_bar.showMessage(f"Ngưỡng tin cậy cập nhật: {threshold * 100:.0f}%")

    def update_results(self, detections: List[Detection]) -> None:
        """Update detection cards and destination bin banner.

        Acceptance Criteria:
        - When detections are empty: banner displays 'Chưa phát hiện rác'.
        - When detections exist: banner displays target bin for top confidence item.

        Args:
            detections: List of Detection objects sorted descending by confidence.
        """
        self._current_detections = list(detections)
        self.result_list.set_detections(self._current_detections)

        if not self._current_detections:
            self.bin_banner.reset()
            self.status_bar.showMessage("Không phát hiện rác trong khung hình")
        else:
            # Destination bin corresponding to highest-confidence detection
            top_detection = self._current_detections[0]
            self.bin_banner.set_bin(top_detection.bin_type)
            self.status_bar.showMessage(
                f"Phát hiện {len(self._current_detections)} vật thể. "
                f"Đích đến: {top_detection.bin_type.display_name} ({top_detection.confidence * 100:.1f}%)"
            )

    @Slot()
    def reset_all(self) -> None:
        """Clear loaded frames, results, and reset banner."""
        self.video_view.clear_frame()
        self.result_list.clear()
        self.bin_banner.reset()
        self._current_image_path = None
        self._current_detections.clear()
        self.status_bar.showMessage("Đã đặt lại trạng thái chờ.")


def run_app() -> None:
    """Launch Trash-Net desktop application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
