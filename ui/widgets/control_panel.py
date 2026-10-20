"""Application Control Panel widget (US-3.1).

Provides user controls for opening media files, starting/stopping webcam streaming,
and adjusting YOLO confidence threshold with real-time feedback.
"""

from typing import Optional
from ui.qt_compat import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    Qt,
    Signal,
    Slot,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QFrame):
    """Control panel for user input actions and inference parameters.

    Signals:
        open_image_requested: Emitted when user clicks 'Mở ảnh'.
        open_video_requested: Emitted when user clicks 'Mở video'.
        toggle_webcam_requested (bool): Emitted with True (start) or False (stop).
        confidence_changed (float): Emitted when slider value changes [0.05 - 0.95].
        reset_requested: Emitted when user clicks 'Đặt lại'.
    """

    open_image_requested = Signal()
    open_video_requested = Signal()
    toggle_webcam_requested = Signal(bool)
    confidence_changed = Signal(float)
    reset_requested = Signal()

    def __init__(self, default_conf: float = 0.25, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ControlPanel")
        self._default_conf = default_conf
        self._webcam_active = False

        self._init_ui()

    def _init_ui(self) -> None:
        """Construct buttons, slider, and status layouts."""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        # Header Title
        title_label = QLabel("Bảng điều khiển")
        title_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #D1D5DB;")
        main_layout.addWidget(title_label)

        # Action Buttons Grid (2x2)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)

        self._btn_open_img = QPushButton("📁 Mở ảnh...")
        self._btn_open_img.setToolTip("Mở tệp ảnh tĩnh (JPG, PNG, WebP) để phân loại")
        self._btn_open_img.clicked.connect(self.open_image_requested.emit)
        self._apply_button_style(self._btn_open_img, primary=False)

        self._btn_open_vid = QPushButton("🎬 Mở video...")
        self._btn_open_vid.setToolTip("Mở tệp video (MP4, AVI) để kiểm thử")
        self._btn_open_vid.clicked.connect(self.open_video_requested.emit)
        self._apply_button_style(self._btn_open_vid, primary=False)

        self._btn_webcam = QPushButton("📹 Bật Webcam")
        self._btn_webcam.setToolTip("Bật/tắt luồng camera trực tiếp")
        self._btn_webcam.clicked.connect(self._on_webcam_clicked)
        self._apply_button_style(self._btn_webcam, primary=True)

        self._btn_reset = QPushButton("🔄 Đặt lại")
        self._btn_reset.setToolTip("Xóa ảnh hiện tại và đưa về trạng thái chờ")
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._apply_button_style(self._btn_reset, primary=False)

        grid_layout.addWidget(self._btn_open_img, 0, 0)
        grid_layout.addWidget(self._btn_open_vid, 0, 1)
        grid_layout.addWidget(self._btn_webcam, 1, 0)
        grid_layout.addWidget(self._btn_reset, 1, 1)
        main_layout.addLayout(grid_layout)

        # Confidence Threshold Slider section
        slider_container = QVBoxLayout()
        slider_container.setSpacing(6)

        slider_header = QHBoxLayout()
        slider_title = QLabel("Ngưỡng tin cậy (Confidence):")
        slider_title.setStyleSheet("font-size: 12px; color: #9CA3AF;")

        pct_val = int(round(self._default_conf * 100))
        self._conf_val_label = QLabel(f"{pct_val}%")
        self._conf_val_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #3B82F6;")
        slider_header.addWidget(slider_title)
        slider_header.addStretch()
        slider_header.addWidget(self._conf_val_label)
        slider_container.addLayout(slider_header)

        # Slider: 5% to 95%
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(5, 95)
        self._slider.setValue(pct_val)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(15)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 6px;
                background: #374151;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3B82F6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #3B82F6;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #93C5FD;
            }
            """
        )
        slider_container.addWidget(self._slider)
        main_layout.addLayout(slider_container)

        # Base panel styling
        self.setStyleSheet(
            """
            QFrame#ControlPanel {
                background-color: #1F2937;
                border: 1px solid #374151;
                border-radius: 10px;
            }
            """
        )

    def _apply_button_style(self, btn: QPushButton, primary: bool = False) -> None:
        """Apply modern flat design styling to action buttons."""
        btn.setFixedHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if primary:
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2563EB;
                    color: #FFFFFF;
                    font-size: 12px;
                    font-weight: 700;
                    border: none;
                    border-radius: 6px;
                    padding: 0px 12px;
                }
                QPushButton:hover {
                    background-color: #1D4ED8;
                }
                QPushButton:pressed {
                    background-color: #1E40AF;
                }
                """
            )
        else:
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #374151;
                    color: #F3F4F6;
                    font-size: 12px;
                    font-weight: 600;
                    border: 1px solid #4B5563;
                    border-radius: 6px;
                    padding: 0px 12px;
                }
                QPushButton:hover {
                    background-color: #4B5563;
                    border-color: #6B7280;
                }
                QPushButton:pressed {
                    background-color: #1F2937;
                }
                """
            )

    @property
    def confidence_threshold(self) -> float:
        """Get the current confidence threshold fraction [0.05, 0.95]."""
        return self._slider.value() / 100.0

    @property
    def is_webcam_active(self) -> bool:
        """Check if webcam streaming is currently turned on."""
        return self._webcam_active

    @Slot(int)
    def _on_slider_changed(self, value: int) -> None:
        """Handle confidence slider position change."""
        self._conf_val_label.setText(f"{value}%")
        self.confidence_changed.emit(value / 100.0)

    @Slot()
    def _on_webcam_clicked(self) -> None:
        """Toggle webcam state and update button label."""
        self._webcam_active = not self._webcam_active
        if self._webcam_active:
            self._btn_webcam.setText("⏹️ Dừng Webcam")
            self._btn_webcam.setStyleSheet(
                """
                QPushButton {
                    background-color: #DC2626;
                    color: #FFFFFF;
                    font-size: 12px;
                    font-weight: 700;
                    border: none;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #B91C1C;
                }
                """
            )
        else:
            self._btn_webcam.setText("📹 Bật Webcam")
            self._apply_button_style(self._btn_webcam, primary=True)

        self.toggle_webcam_requested.emit(self._webcam_active)

    @Slot()
    def _on_reset_clicked(self) -> None:
        """Reset controls and webcam status."""
        if self._webcam_active:
            self._webcam_active = False
            self._btn_webcam.setText("📹 Bật Webcam")
            self._apply_button_style(self._btn_webcam, primary=True)
            self.toggle_webcam_requested.emit(False)
        self.reset_requested.emit()
