"""Waste detection result cards and scrollable list widget (US-3.1).

Renders Vietnamese class labels, percentage confidence scores, and destination bin badges.
"""

from typing import List, Optional
from core.types import Bin, Detection
from ui.constants import BIN_STYLES, get_vietnamese_label
from ui.qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
)


class ResultCard(QFrame):
    """Card displaying a single waste detection result.

    Attributes:
        detection (Detection): Structured detection model item.
    """

    def __init__(self, detection: Detection, rank: int = 1, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResultCard")
        self.detection = detection
        self.rank = rank

        self._init_ui()

    def _init_ui(self) -> None:
        """Construct visual elements for the detection card."""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(72)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(12)

        # 1. Rank pill badge
        rank_label = QLabel(f"#{self.rank}")
        rank_label.setObjectName("RankLabel")
        rank_label.setFixedWidth(28)
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rank_label.setStyleSheet(
            """
            QLabel#RankLabel {
                background-color: #374151;
                color: #9CA3AF;
                border-radius: 6px;
                font-weight: 700;
                font-size: 11px;
                padding: 4px;
            }
            """
        )
        main_layout.addWidget(rank_label)

        # 2. Category information (Vietnamese display name + raw name)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        vn_name = get_vietnamese_label(self.detection.class_name, self.detection.class_id)
        self.name_label = QLabel(vn_name)
        self.name_label.setObjectName("ResultCardName")
        self.name_label.setStyleSheet(
            "color: #F9FAFB; font-size: 14px; font-weight: 700; background: transparent;"
        )

        self.sub_label = QLabel(f"Gốc: {self.detection.class_name}")
        self.sub_label.setObjectName("ResultCardSub")
        self.sub_label.setStyleSheet(
            "color: #9CA3AF; font-size: 11px; background: transparent;"
        )

        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.sub_label)
        main_layout.addLayout(info_layout, stretch=1)

        # 3. Confidence meter
        conf_layout = QVBoxLayout()
        conf_layout.setSpacing(3)
        conf_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        pct = self.detection.confidence * 100.0
        self.conf_label = QLabel(f"{pct:.1f}%")
        conf_color = "#10B981" if pct >= 75.0 else ("#F59E0B" if pct >= 50.0 else "#EF4444")
        self.conf_label.setStyleSheet(
            f"color: {conf_color}; font-size: 13px; font-weight: 800; background: transparent;"
        )
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Mini confidence progress bar
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(round(pct)))
        bar.setTextVisible(False)
        bar.setFixedSize(64, 5)
        bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #374151;
                border-radius: 2px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {conf_color};
                border-radius: 2px;
            }}
            """
        )

        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(bar)
        main_layout.addLayout(conf_layout)

        # 4. Bin destination badge
        bin_style = BIN_STYLES.get(self.detection.bin_type, BIN_STYLES[Bin.UNKNOWN])
        bin_name = self.detection.bin_type.display_name
        self.bin_badge = QLabel(f"{bin_style['icon']} {bin_name}")
        self.bin_badge.setObjectName("ResultCardBinBadge")
        self.bin_badge.setStyleSheet(
            f"""
            background-color: {bin_style['bg_color']};
            color: #FFFFFF;
            font-size: 11px;
            font-weight: 700;
            border-radius: 6px;
            padding: 6px 10px;
            """
        )
        self.bin_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.bin_badge)

        # Base card styling
        self.setStyleSheet(
            """
            QFrame#ResultCard {
                background-color: #1F2937;
                border: 1px solid #374151;
                border-radius: 8px;
            }
            QFrame#ResultCard:hover {
                border-color: #4B5563;
                background-color: #242E3D;
            }
            """
        )


class ResultList(QWidget):
    """Scrollable list container for dynamic ResultCard widgets."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResultList")
        self._cards: List[ResultCard] = []

        self._init_ui()

    def _init_ui(self) -> None:
        """Set up scroll area and vertical card layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header with item counter
        self._header_label = QLabel("Danh sách phát hiện (0)")
        self._header_label.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #D1D5DB; padding: 4px 0px;"
        )
        layout.addWidget(self._header_label)

        # Scroll Area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #374151;
                border-radius: 8px;
                background-color: #111827;
            }
            """
        )

        # Container for cards
        self._container = QWidget()
        self._container.setObjectName("ResultListContainer")
        self._cards_layout = QVBoxLayout(self._container)
        self._cards_layout.setContentsMargins(8, 8, 8, 8)
        self._cards_layout.setSpacing(8)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Empty state label
        self._empty_label = QLabel("Chưa phát hiện rác trong khung hình")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6B7280; font-size: 12px; padding: 32px 0px;")
        self._cards_layout.addWidget(self._empty_label)

        self._scroll_area.setWidget(self._container)
        layout.addWidget(self._scroll_area)

    @property
    def count(self) -> int:
        """Return total number of detection cards currently listed."""
        return len(self._cards)

    def set_detections(self, detections: List[Detection]) -> None:
        """Populate list with structured detections.

        Args:
            detections: List of Detection objects sorted descending by confidence.
        """
        self.clear()

        if not detections:
            self._empty_label.setVisible(True)
            self._header_label.setText("Danh sách phát hiện (0)")
            return

        self._empty_label.setVisible(False)
        self._header_label.setText(f"Danh sách phát hiện ({len(detections)})")

        for idx, det in enumerate(detections, start=1):
            card = ResultCard(det, rank=idx)
            self._cards.append(card)
            self._cards_layout.addWidget(card)

    def clear(self) -> None:
        """Clear all cards and reset to empty state."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        self._empty_label.setVisible(True)
        self._header_label.setText("Danh sách phát hiện (0)")
