"""High-performance Video/Image display widget with aspect ratio preservation (US-3.1 & US-3.3).

Renders video feeds and still images with smooth letterboxing, drag-and-drop support,
and instance segmentation polygon mask overlays.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from core.types import Bin, Detection
from ui.qt_compat import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRect,
    QSizePolicy,
    Qt,
    Signal,
    QWidget,
)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BIN_COLORS_BGR = {
    Bin.ORGANIC: (50, 168, 82),      # Xanh lá cây (Hữu cơ)
    Bin.RECYCLABLE: (219, 114, 33),  # Xanh dương (Tái chế)
    Bin.RESIDUAL: (45, 55, 219),     # Đỏ / Cam (Vô cơ / Khác)
    Bin.UNKNOWN: (140, 140, 140),    # Xám (Không xác định)
}


class VideoView(QWidget):
    """Custom Qt widget for rendering video streams or static images with drag & drop.

    Ensures frames strictly retain their native aspect ratio during window
    resize operations, letterboxing with high-contrast neutral margins.
    Supports Drag & Drop of image files (US-3.3).
    """

    # Signals for Drag & Drop
    image_dropped = Signal(Path)             # Emitted when a valid image file is dropped
    file_rejected = Signal(str)              # Emitted when an invalid or corrupted file is dropped
    multiple_files_warning = Signal(int)     # Emitted when multiple files are dropped simultaneously

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("VideoView")
        self._pixmap: Optional[QPixmap] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._detections: List[Detection] = []
        self._original_width: int = 0
        self._original_height: int = 0

        # Enable Drag & Drop (US-3.3)
        self.setAcceptDrops(True)

        # Enable widget background fill and flexible sizing
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)

    @property
    def has_frame(self) -> bool:
        """Check if an image or video frame is currently loaded."""
        return self._pixmap is not None and not self._pixmap.isNull()

    @property
    def frame_resolution(self) -> tuple[int, int]:
        """Return native (width, height) of loaded frame."""
        return (self._original_width, self._original_height)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events containing URLs/files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Accept drag motion over widget area."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped files: validate, warn if multiple, and emit image_dropped."""
        urls = event.mimeData().urls()
        if not urls:
            return

        file_paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
        if not file_paths:
            self.file_rejected.emit("Dữ liệu kéo thả không phải là tệp hợp lệ trên máy.")
            return

        total_files = len(file_paths)
        if total_files > 1:
            self.multiple_files_warning.emit(total_files)

        # Process the first dropped file (AC requirement)
        target_file = file_paths[0]
        ext = target_file.suffix.lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            self.file_rejected.emit(
                f"Tệp '{target_file.name}' không phải là ảnh hợp lệ.\n"
                f"Chỉ chấp nhận các định dạng: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
            )
            return

        self.image_dropped.emit(target_file)
        event.acceptProposedAction()

    def set_frame(
        self,
        frame: Union[np.ndarray, QImage, QPixmap],
        detections: Optional[List[Detection]] = None,
    ) -> None:
        """Update and display a new frame with optional detection overlays.

        Args:
            frame: OpenCV ndarray (BGR or RGB), QImage, or QPixmap.
            detections: Optional list of Detection objects to draw as masks/boxes.
        """
        if frame is None:
            self.clear_frame()
            return

        if detections is not None:
            self._detections = list(detections)

        if isinstance(frame, np.ndarray):
            self._raw_frame = frame
            h, w = frame.shape[:2]
            self._original_width = w
            self._original_height = h

            # Render overlay if detections exist
            render_img = self._apply_overlay(frame, self._detections)

            if render_img.ndim == 2:  # Grayscale
                bytes_per_line = w
                q_img = QImage(
                    render_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
                ).copy()
            elif render_img.shape[2] == 3:  # BGR to RGB
                rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * w
                q_img = QImage(
                    rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()
            elif render_img.shape[2] == 4:  # BGRA to RGBA
                rgba = cv2.cvtColor(render_img, cv2.COLOR_BGRA2RGBA)
                bytes_per_line = 4 * w
                q_img = QImage(
                    rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                ).copy()
            else:
                return

            self._pixmap = QPixmap.fromImage(q_img)

        elif isinstance(frame, QImage):
            self._original_width = frame.width()
            self._original_height = frame.height()
            self._pixmap = QPixmap.fromImage(frame)

        elif isinstance(frame, QPixmap):
            self._original_width = frame.width()
            self._original_height = frame.height()
            self._pixmap = frame

        self.update()

    def set_detections(self, detections: List[Detection]) -> None:
        """Update detections on the current active frame and redraw."""
        self._detections = list(detections)
        if self._raw_frame is not None:
            self.set_frame(self._raw_frame, self._detections)

    def _apply_overlay(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Render segmentation polygon masks and bounding boxes."""
        if not detections:
            return frame

        annotated = frame.copy()
        overlay = annotated.copy()

        # 1. Semi-transparent polygon mask
        for det in detections:
            color = BIN_COLORS_BGR.get(det.bin_type, (140, 140, 140))
            if det.polygon and len(det.polygon) >= 3:
                pts = np.array(det.polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color=color)

        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        # 2. Polygon contours, bounding boxes, labels
        for det in detections:
            color = BIN_COLORS_BGR.get(det.bin_type, (140, 140, 140))
            x1, y1, x2, y2 = det.box

            # Polygon contours
            if det.polygon and len(det.polygon) >= 3:
                pts = np.array(det.polygon, dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label badge
            label_text = f"[{det.bin_type.display_name}] {det.class_name} ({det.confidence * 100:.0f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label_text, font, 0.55, 1)
            ty1 = max(0, y1 - th - 8)
            ty2 = ty1 + th + 8
            tx2 = min(annotated.shape[1], x1 + tw + 10)

            cv2.rectangle(annotated, (x1, ty1), (tx2, ty2), color=color, thickness=-1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, ty2 - 5),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def clear_frame(self) -> None:
        """Remove active frame and show placeholder screen."""
        self._pixmap = None
        self._raw_frame = None
        self._detections.clear()
        self._original_width = 0
        self._original_height = 0
        self.update()

    def paintEvent(self, event) -> None:
        """Render frame centered with strict aspect ratio, or modern placeholder."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()

        # Paint letterbox canvas background
        painter.fillRect(rect, QColor("#0B0F19"))

        if self._pixmap is not None and not self._pixmap.isNull():
            # Scale preserving native aspect ratio without distortion
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Outer subtle border
            painter.setPen(QPen(QColor("#1F2937"), 1))
            painter.drawRect(x, y, scaled.width(), scaled.height())
        else:
            self._draw_placeholder(painter, rect)

    def _draw_placeholder(self, painter: QPainter, rect: QRect) -> None:
        """Render aesthetic empty state placeholder with drag & drop guidance."""
        center_x = rect.center().x()
        center_y = rect.center().y()

        # Dashed guide border
        margin = 24
        guide_rect = rect.adjusted(margin, margin, -margin, -margin)
        pen = QPen(QColor("#1F2937"), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(guide_rect, 12, 12)

        # Main Icon
        painter.setPen(QColor("#9CA3AF"))
        font_icon = QFont(self.font())
        font_icon.setPointSize(36)
        painter.setFont(font_icon)
        painter.drawText(
            QRect(center_x - 100, center_y - 65, 200, 50),
            int(Qt.AlignmentFlag.AlignCenter),
            "📥",
        )

        # Primary label
        painter.setPen(QColor("#E5E7EB"))
        font_main = QFont(self.font())
        font_main.setPointSize(14)
        font_main.setBold(True)
        painter.setFont(font_main)
        painter.drawText(
            QRect(center_x - 250, center_y - 5, 500, 30),
            int(Qt.AlignmentFlag.AlignCenter),
            "Kéo & thả ảnh rác vào đây",
        )

        # Secondary guidance
        painter.setPen(QColor("#6B7280"))
        font_sub = QFont(self.font())
        font_sub.setPointSize(11)
        painter.setFont(font_sub)
        painter.drawText(
            QRect(center_x - 300, center_y + 30, 600, 25),
            int(Qt.AlignmentFlag.AlignCenter),
            "hoặc nhấn 'Mở ảnh' từ bảng điều khiển (.jpg, .jpeg, .png, .bmp, .webp)",
        )
