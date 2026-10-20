"""Qt Compatibility Layer for Trash-Net (PyQt6 / PySide6 Hybrid).

Allows components to seamlessly run under either PyQt6 or PySide6 without
direct vendor lock-in.
"""

from typing import Tuple

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import (
        QPoint,
        QRect,
        QSize,
        Qt,
        QThread,
        QTimer,
        pyqtSignal as Signal,
        pyqtSlot as Slot,
    )
    from PyQt6.QtGui import (
        QBrush,
        QCloseEvent,
        QColor,
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
        QFont,
        QIcon,
        QImage,
        QKeySequence,
        QPainter,
        QPen,
        QPixmap,
        QPolygon,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt6"

except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        from PySide6.QtCore import (  # type: ignore
            QPoint,
            QRect,
            QSize,
            Qt,
            QThread,
            QTimer,
            Signal,
            Slot,
        )
        from PySide6.QtGui import (  # type: ignore
            QBrush,
            QCloseEvent,
            QColor,
            QDragEnterEvent,
            QDragMoveEvent,
            QDropEvent,
            QFont,
            QIcon,
            QImage,
            QKeySequence,
            QPainter,
            QPen,
            QPixmap,
            QPolygon,
        )
        from PySide6.QtWidgets import (  # type: ignore
            QApplication,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSplitter,
            QStatusBar,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PySide6"

    except ImportError as err:
        raise ImportError(
            "Neither PyQt6 nor PySide6 is installed in the active Python environment. "
            "Please install PyQt6 with: uv pip install PyQt6 (or pip install PyQt6)"
        ) from err

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Qt",
    "Signal",
    "Slot",
    "QSize",
    "QRect",
    "QPoint",
    "QTimer",
    "QFont",
    "QColor",
    "QImage",
    "QPixmap",
    "QPainter",
    "QPen",
    "QBrush",
    "QCloseEvent",
    "QPolygon",
    "QIcon",
    "QKeySequence",
    "QApplication",
    "QMainWindow",
    "QWidget",
    "QFrame",
    "QLabel",
    "QPushButton",
    "QSlider",
    "QProgressBar",
    "QSplitter",
    "QScrollArea",
    "QFileDialog",
    "QGroupBox",
    "QStatusBar",
    "QSizePolicy",
    "QVBoxLayout",
    "QHBoxLayout",
    "QGridLayout",
    "QT_BINDING",
]
