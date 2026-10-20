"""Interactive Control Panel widget for Trash-Net GUI (US-3.1 & US-3.3 Chế độ B).

Provides user controls for opening media files, webcam streaming state machine
(Start / Pause / Stop), camera device selection, and confidence threshold adjustment.
"""

from typing import List, Optional
from ui.qt_compat import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    Qt,
    Signal,
    Slot,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QFrame):
    """Panel hosting primary action buttons, camera controls, and configuration sliders."""

    # Existing Signals
    open_image_requested = Signal()
    open_video_requested = Signal()
    toggle_webcam_requested = Signal(bool)
    confidence_changed = Signal(float)
    reset_requested = Signal()

    # US-3.3 Chế độ B Signals
    start_webcam_requested = Signal(int)   # camera_index
    pause_webcam_requested = Signal()
    resume_webcam_requested = Signal()
    stop_webcam_requested = Signal()

    # US-3.5 Overlay Toggle Signals
    overlay_options_changed = Signal(bool, bool, bool)  # (show_masks, show_boxes, show_labels)

    def __init__(self, default_conf: float = 0.25, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ControlPanel")
        self._default_conf = default_conf
        self._webcam_active = False
        self._is_paused = False

        self._init_ui()

    def _init_ui(self) -> None:
        """Construct control widgets and modern dark layouts."""
        self.setStyleSheet(
            """
            #ControlPanel {
                background-color: #111827;
                border: 1px solid #1F2937;
                border-radius: 8px;
            }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # 1. Media Actions Grid
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

        self._btn_reset = QPushButton("🔄 Đặt lại")
        self._btn_reset.setToolTip("Xóa ảnh hiện tại và đưa về trạng thái chờ")
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._apply_button_style(self._btn_reset, primary=False)

        grid_layout.addWidget(self._btn_open_img, 0, 0)
        grid_layout.addWidget(self._btn_open_vid, 0, 1)
        grid_layout.addWidget(self._btn_reset, 1, 1)

        # 2. Camera Controls (US-3.3 Chế độ B)
        cam_select_layout = QHBoxLayout()
        cam_label = QLabel("Thiết bị Camera:")
        cam_label.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background: #1F2937;
                color: #E5E7EB;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
        """)
        self.camera_combo.addItem("Camera 0 (Mặc định)", 0)
        self.camera_combo.addItem("Camera 1 (Phụ)", 1)
        cam_select_layout.addWidget(cam_label)
        cam_select_layout.addWidget(self.camera_combo, stretch=1)

        main_layout.addLayout(cam_select_layout)

        # Start / Stop and Pause Buttons
        self._btn_webcam = QPushButton("▶️ Bật Webcam")
        self._btn_webcam.setToolTip("Khởi động/Dừng luồng camera trực tiếp")
        self._btn_webcam.clicked.connect(self._on_webcam_clicked)
        self._apply_button_style(self._btn_webcam, primary=True)

        self._btn_pause = QPushButton("⏸️ Tạm dừng")
        self._btn_pause.setToolTip("Tạm dừng hình ảnh và suy luận")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause_clicked)
        self._apply_button_style(self._btn_pause, primary=False)

        grid_layout.addWidget(self._btn_webcam, 1, 0)
        grid_layout.addWidget(self._btn_pause, 2, 0, 1, 2)
        main_layout.addLayout(grid_layout)

        # 3. Confidence Threshold Slider section
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
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                background: #F9FAFB;
                border: 2px solid #3B82F6;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #60A5FA;
            }
            """
        )
        slider_container.addWidget(self._slider)
        main_layout.addLayout(slider_container)

        # 4. Overlay Layer Toggles (US-3.5)
        overlay_container = QVBoxLayout()
        overlay_container.setSpacing(6)
        overlay_title = QLabel("Lớp phủ hiển thị (Overlays):")
        overlay_title.setStyleSheet("font-size: 12px; color: #9CA3AF;")
        overlay_container.addWidget(overlay_title)

        chk_layout = QHBoxLayout()
        chk_layout.setSpacing(10)

        self._chk_masks = QCheckBox("Mặt nạ")
        self._chk_masks.setChecked(True)
        self._chk_masks.setToolTip("Bật/Tắt mặt nạ đa giác phân vùng")
        self._chk_masks.toggled.connect(self._on_overlay_toggled)

        self._chk_boxes = QCheckBox("Hộp bao")
        self._chk_boxes.setChecked(True)
        self._chk_boxes.setToolTip("Bật/Tắt khung chữ nhật bao quanh vật thể")
        self._chk_boxes.toggled.connect(self._on_overlay_toggled)

        self._chk_labels = QCheckBox("Nhãn")
        self._chk_labels.setChecked(True)
        self._chk_labels.setToolTip("Bật/Tắt tên loại rác và độ tin cậy")
        self._chk_labels.toggled.connect(self._on_overlay_toggled)

        chk_style = """
            QCheckBox {
                color: #D1D5DB;
                font-size: 12px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #4B5563;
                border-radius: 3px;
                background-color: #1F2937;
            }
            QCheckBox::indicator:checked {
                background-color: #3B82F6;
                border-color: #3B82F6;
            }
        """
        self._chk_masks.setStyleSheet(chk_style)
        self._chk_boxes.setStyleSheet(chk_style)
        self._chk_labels.setStyleSheet(chk_style)

        chk_layout.addWidget(self._chk_masks)
        chk_layout.addWidget(self._chk_boxes)
        chk_layout.addWidget(self._chk_labels)
        overlay_container.addLayout(chk_layout)
        main_layout.addLayout(overlay_container)

    def populate_cameras(self, camera_indices: List[int]) -> None:
        """Update available camera items in dropdown."""
        self.camera_combo.clear()
        if not camera_indices:
            self.camera_combo.addItem("Không tìm thấy camera", -1)
            return
        for idx in camera_indices:
            self.camera_combo.addItem(f"Camera #{idx}", idx)

    def set_camera_state(self, state: str) -> None:
        """Update buttons visually to reflect CameraState machine."""
        st = state.upper()
        if st in ("IDLE", "STOPPED"):
            self._webcam_active = False
            self._is_paused = False
            self._btn_webcam.setText("▶️ Bật Webcam")
            self._apply_button_style(self._btn_webcam, primary=True)
            self._btn_pause.setText("⏸️ Tạm dừng")
            self._btn_pause.setEnabled(False)
            self.camera_combo.setEnabled(True)
        elif st == "RUNNING":
            self._webcam_active = True
            self._is_paused = False
            self._btn_webcam.setText("⏹️ Dừng Webcam")
            self._btn_webcam.setStyleSheet(
                """
                QPushButton {
                    background-color: #DC2626;
                    color: white;
                    font-weight: 600;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 14px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #B91C1C;
                }
                """
            )
            self._btn_pause.setText("⏸️ Tạm dừng")
            self._btn_pause.setEnabled(True)
            self.camera_combo.setEnabled(False)
        elif st == "PAUSED":
            self._is_paused = True
            self._btn_pause.setText("▶️ Tiếp tục")
            self._btn_pause.setEnabled(True)

    @property
    def show_masks(self) -> bool:
        """Check if mask overlay is toggled on."""
        return self._chk_masks.isChecked()

    @property
    def show_boxes(self) -> bool:
        """Check if bounding box overlay is toggled on."""
        return self._chk_boxes.isChecked()

    @property
    def show_labels(self) -> bool:
        """Check if label overlay is toggled on."""
        return self._chk_labels.isChecked()

    def set_overlay_toggles(self, masks: bool = True, boxes: bool = True, labels: bool = True) -> None:
        """Programmatically set overlay toggle states."""
        self._chk_masks.setChecked(masks)
        self._chk_boxes.setChecked(boxes)
        self._chk_labels.setChecked(labels)

    def _on_overlay_toggled(self) -> None:
        """Emit updated overlay configuration when checkboxes are clicked."""
        self.overlay_options_changed.emit(self.show_masks, self.show_boxes, self.show_labels)

    @property
    def confidence_threshold(self) -> float:
        """Get the current confidence threshold fraction [0.05, 0.95]."""
        return self._slider.value() / 100.0

    @property
    def is_webcam_active(self) -> bool:
        """Check if webcam streaming is currently turned on."""
        return self._webcam_active

    @property
    def selected_camera_index(self) -> int:
        """Return currently chosen camera device index."""
        data = self.camera_combo.currentData()
        return data if data is not None else 0

    def _on_slider_changed(self, value: int) -> None:
        """Handle confidence slider position change."""
        self._conf_val_label.setText(f"{value}%")
        self.confidence_changed.emit(value / 100.0)

    def _on_webcam_clicked(self) -> None:
        """Toggle Start / Stop webcam state."""
        if not self._webcam_active:
            cam_idx = self.selected_camera_index
            self.start_webcam_requested.emit(cam_idx)
            self.toggle_webcam_requested.emit(True)
        else:
            self.stop_webcam_requested.emit()
            self.toggle_webcam_requested.emit(False)

    def _on_pause_clicked(self) -> None:
        """Toggle Pause / Resume."""
        if not self._is_paused:
            self.pause_webcam_requested.emit()
        else:
            self.resume_webcam_requested.emit()

    def _on_reset_clicked(self) -> None:
        """Reset controls and webcam status."""
        if self._webcam_active:
            self.stop_webcam_requested.emit()
            self.toggle_webcam_requested.emit(False)
        self.reset_requested.emit()

    def _apply_button_style(self, button: QPushButton, primary: bool = False) -> None:
        """Apply sleek modern theme styling to push buttons."""
        if primary:
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: #2563EB;
                    color: white;
                    font-weight: 600;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 14px;
                    font-size: 13px;
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
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: #1F2937;
                    color: #E5E7EB;
                    font-weight: 500;
                    border: 1px solid #374151;
                    border-radius: 6px;
                    padding: 8px 14px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #374151;
                    color: #FFFFFF;
                }
                QPushButton:pressed {
                    background-color: #4B5563;
                }
                QPushButton:disabled {
                    background-color: #111827;
                    color: #4B5563;
                    border-color: #1F2937;
                }
                """
            )
