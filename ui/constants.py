"""UI Theme, styling constants, and localization dictionary for Trash-Net (US-3.1).

Defines accessible color palettes for waste bins, translations for 8 MultiTrash classes,
and global UI geometry constraints.
"""

from typing import Dict
from core.mapper import BinMapper
from core.types import Bin

_mapper = BinMapper()

# Window geometry specifications (DoD: >= 1280x720)
MIN_WINDOW_WIDTH = 1280
MIN_WINDOW_HEIGHT = 720
DEFAULT_WINDOW_TITLE = "Trash-Net — Hệ Thống Phân Loại Rác Thông Minh"

# Vietnamese localization for MultiTrash dataset 8 classes
CLASS_TRANSLATIONS: Dict[str, str] = {
    "biowaste": "Rác hữu cơ",
    "glass": "Thủy tinh",
    "metal": "Kim loại",
    "other": "Rác khác",
    "paper": "Giấy",
    "plastic": "Nhựa",
    "styrofoam": "Xốp",
    "wrapper": "Bao bì màng",
}

# Fallback by numeric class ID (0 to 7)
CLASS_ID_TRANSLATIONS: Dict[int, str] = {
    0: "Rác hữu cơ",
    1: "Thủy tinh",
    2: "Kim loại",
    3: "Rác khác",
    4: "Giấy",
    5: "Nhựa",
    6: "Xốp",
    7: "Bao bì màng",
}


def get_vietnamese_label(class_name: str, class_id: int = -1) -> str:
    """Resolve Vietnamese label for a detected waste category.

    Args:
        class_name: Raw label string from detector/dataset.
        class_id: Numeric class identifier if available.

    Returns:
        Human-readable Vietnamese label.
    """
    normalized = class_name.strip().lower()
    if normalized in CLASS_TRANSLATIONS:
        return CLASS_TRANSLATIONS[normalized]
    if class_id in CLASS_ID_TRANSLATIONS:
        return CLASS_ID_TRANSLATIONS[class_id]
    return class_name


# Bin styling dictionary: High-contrast WCAG AAA colors
# ORGANIC (#2E7D32 Green), RECYCLABLE (#1565C0 Blue), RESIDUAL (#C62828 Red)
BIN_STYLES = {
    Bin.ORGANIC: {
        "title": "THÙNG HỮU CƠ",
        "subtitle": "Rác phân hủy: thức ăn thừa, rau củ quả, lá cây",
        "bg_color": _mapper.get_bin_color_hex(Bin.ORGANIC, "#2E7D32"),
        "text_color": "#FFFFFF",
        "badge_bg": "#1B5E20",
        "icon": "🌱",
    },
    Bin.RECYCLABLE: {
        "title": "THÙNG TÁI CHẾ",
        "subtitle": "Rác tái chế: giấy, nhựa, chai lọ, lon kim loại",
        "bg_color": _mapper.get_bin_color_hex(Bin.RECYCLABLE, "#1565C0"),
        "text_color": "#FFFFFF",
        "badge_bg": "#0D47A1",
        "icon": "♻️",
    },
    Bin.RESIDUAL: {
        "title": "THÙNG VÔ CƠ / KHÁC",
        "subtitle": "Rác còn lại: xốp, bao bì màng nilon bẩn, rác trơ",
        "bg_color": _mapper.get_bin_color_hex(Bin.RESIDUAL, "#C62828"),
        "text_color": "#FFFFFF",
        "badge_bg": "#B71C1C",
        "icon": "🗑️",
    },
    Bin.HAZARDOUS: {
        "title": "CẢNH BÁO: RÁC NGUY HẠI",
        "subtitle": "Pin, đồ điện tử, hóa chất — Cần xử lý theo quy trình riêng!",
        "bg_color": "#DC2626",
        "text_color": "#FFFFFF",
        "badge_bg": "#991B1B",
        "icon": "☣️",
    },
    Bin.UNKNOWN: {
        "title": "KHÔNG CHẮC CHẮN — CẦN KIỂM TRA THỦ CÔNG",
        "subtitle": "Độ tin cậy thấp hoặc ngoài phân phối — Không bỏ thùng thông thường!",
        "bg_color": "#D97706",
        "text_color": "#FFFFFF",
        "badge_bg": "#B45309",
        "icon": "⚠️",
    },
}

IDLE_BANNER_STYLE = {
    "title": "Chưa phát hiện rác",
    "subtitle": "Đưa rác vào trước camera hoặc mở tệp ảnh/video",
    "bg_color": "#374151",
    "text_color": "#E5E7EB",
    "icon": "🔍",
}

# Dark theme palette
DARK_PALETTE_QSS = """
QMainWindow, QDialog, QMessageBox {
    background-color: #111827;
}

QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #F3F4F6;
}

/* File Explorer (QFileDialog) & Dialog Enhancements */
QFileDialog {
    background-color: #111827;
    color: #F3F4F6;
}

QFileDialog QTreeView, QFileDialog QListView {
    background-color: #1F2937;
    color: #F3F4F6;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 4px;
}

QFileDialog QTreeView::item:selected, QFileDialog QListView::item:selected {
    background-color: #2563EB;
    color: #FFFFFF;
}

QFileDialog QLineEdit {
    background-color: #1F2937;
    color: #F3F4F6;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 5px 8px;
}

QFileDialog QPushButton {
    background-color: #374151;
    color: #F3F4F6;
    border: 1px solid #4B5563;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}

QFileDialog QPushButton:hover {
    background-color: #4B5563;
    color: #FFFFFF;
}

QFileDialog QHeaderView::section {
    background-color: #1F2937;
    color: #9CA3AF;
    padding: 4px 6px;
    border: none;
    border-bottom: 1px solid #374151;
}

QFileDialog QComboBox {
    background-color: #1F2937;
    color: #F3F4F6;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 4px 8px;
}

QFileDialog QToolButton {
    background-color: #1F2937;
    color: #F3F4F6;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 4px 6px;
    margin: 2px;
}

QFileDialog QToolButton:hover {
    background-color: #374151;
    border-color: #4B5563;
}

QFileDialog QToolButton:pressed {
    background-color: #111827;
}

QFileDialog QLabel {
    color: #D1D5DB;
    font-size: 12px;
}

QSplitter::handle {
    background-color: #374151;
    width: 3px;
    height: 3px;
}

QSplitter::handle:hover {
    background-color: #3B82F6;
}

QScrollBar:vertical {
    border: none;
    background: #1F2937;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #4B5563;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #6B7280;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1F2937;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #4B5563;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #6B7280;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QStatusBar {
    background-color: #1F2937;
    color: #9CA3AF;
    border-top: 1px solid #374151;
    font-size: 12px;
    padding: 4px 8px;
}
"""
