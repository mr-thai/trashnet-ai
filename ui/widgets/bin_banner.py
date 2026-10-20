"""Destination Waste Bin Banner widget (US-3.1).

Displays high-contrast prominent indicator for target sorting bin,
with distinct colors for Organic, Recyclable, and Residual bins.
"""

from typing import Optional
from core.types import Bin
from ui.constants import BIN_STYLES, IDLE_BANNER_STYLE
from ui.qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
)


class BinBanner(QFrame):
    """Visual banner that prominently directs user to destination sorting bin.

    Supports dynamic switching between 3 primary bins (ORGANIC, RECYCLABLE, RESIDUAL)
    and an idle "Chưa phát hiện rác" state when no waste items are detected.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("BinBanner")
        self._current_bin: Optional[Bin] = None

        self._init_ui()
        self.reset()

    def _init_ui(self) -> None:
        """Construct banner layout and child widgets."""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(84)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(16)

        # Large status icon
        self._icon_label = QLabel()
        self._icon_label.setObjectName("BinBannerIcon")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 34px; background: transparent;")
        self._icon_label.setFixedWidth(48)
        main_layout.addWidget(self._icon_label)

        # Text column (Title + Subtitle)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._title_label = QLabel()
        self._title_label.setObjectName("BinBannerTitle")
        self._title_label.setStyleSheet(
            "font-size: 18px; font-weight: 800; letter-spacing: 0.5px; background: transparent;"
        )

        self._subtitle_label = QLabel()
        self._subtitle_label.setObjectName("BinBannerSubtitle")
        self._subtitle_label.setStyleSheet(
            "font-size: 12px; font-weight: 500; opacity: 0.9; background: transparent;"
        )
        self._subtitle_label.setWordWrap(True)

        text_layout.addWidget(self._title_label)
        text_layout.addWidget(self._subtitle_label)
        main_layout.addLayout(text_layout)

    @property
    def current_bin(self) -> Optional[Bin]:
        """Return the currently displayed sorting bin, or None if idle."""
        return self._current_bin

    @property
    def is_idle(self) -> bool:
        """Check whether banner is in idle state."""
        return self._current_bin is None

    def set_bin(self, bin_type: Optional[Bin]) -> None:
        """Update banner appearance according to detected waste bin.

        Args:
            bin_type: Bin enum (ORGANIC, RECYCLABLE, RESIDUAL) or None for idle.
        """
        self._current_bin = bin_type

        if bin_type is None:
            self._apply_style(IDLE_BANNER_STYLE)
            return

        style_info = BIN_STYLES.get(bin_type, BIN_STYLES[Bin.UNKNOWN])
        self._apply_style(style_info)

    def reset(self) -> None:
        """Reset banner to default idle state ('Chưa phát hiện rác')."""
        self.set_bin(None)

    def _apply_style(self, style: dict) -> None:
        """Apply theme colors, title, and icon."""
        bg_color = style.get("bg_color", "#374151")
        text_color = style.get("text_color", "#FFFFFF")
        title = style.get("title", "")
        subtitle = style.get("subtitle", "")
        icon = style.get("icon", "🗑️")

        self._icon_label.setText(icon)
        self._title_label.setText(title)
        self._subtitle_label.setText(subtitle)

        # High-contrast styling with rounded corners and border
        self.setStyleSheet(
            f"""
            QFrame#BinBanner {{
                background-color: {bg_color};
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }}
            QLabel#BinBannerTitle {{
                color: {text_color};
            }}
            QLabel#BinBannerSubtitle {{
                color: {text_color};
            }}
            """
        )
