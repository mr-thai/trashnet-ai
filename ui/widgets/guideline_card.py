"""Waste preparation guidelines card widget for Trash-Net (US-4.3).

Displays actionable, user-friendly preprocessing steps for the currently
detected top waste category in the right control panel.
"""

from typing import Optional
from core.guidelines import GuidelineManager, get_waste_guideline
from ui.qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
)


class GuidelineCard(QFrame):
    """Visual card displaying preprocessing advice for detected waste items."""

    def __init__(
        self,
        manager: Optional[GuidelineManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("GuidelineCard")
        self._manager = manager if manager is not None else GuidelineManager()
        self._current_class: Optional[str] = None

        self._init_ui()
        self.reset()

    def _init_ui(self) -> None:
        """Construct card layout and labels."""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setStyleSheet(
            """
            #GuidelineCard {
                background-color: #111827;
                border: 1px solid #1F2937;
                border-left: 4px solid #F59E0B;
                border-radius: 6px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header: Icon + Category Title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)

        self._title_label = QLabel("💡 Hướng dẫn xử lý:")
        self._title_label.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #F59E0B; background: transparent;"
        )
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Quick tip / Summary
        self._tip_label = QLabel()
        self._tip_label.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #FCD34D; background: transparent;"
        )
        self._tip_label.setWordWrap(True)
        layout.addWidget(self._tip_label)

        # Bulleted action steps
        self._steps_label = QLabel()
        self._steps_label.setStyleSheet(
            "font-size: 11px; color: #D1D5DB; line-height: 140%; background: transparent;"
        )
        self._steps_label.setWordWrap(True)
        layout.addWidget(self._steps_label)

    @property
    def current_class(self) -> Optional[str]:
        """Return the waste category name currently shown, or None if idle."""
        return self._current_class

    def set_guideline(self, class_name: Optional[str]) -> None:
        """Display preparation advice for the specified waste class.

        Args:
            class_name: MultiTrash class label (e.g., 'plastic', 'glass', 'biowaste').
        """
        if not class_name:
            self.reset()
            return

        self._current_class = class_name
        info = self._manager.get_guideline(class_name)

        title = info.get("title", class_name.capitalize())
        tip = info.get("tip", "")
        steps = info.get("steps", [])

        self._title_label.setText(f"💡 Hướng dẫn: {title}")
        self._tip_label.setText(f"Mẹo nhanh: {tip}" if tip else "")
        self._tip_label.setVisible(bool(tip))

        if steps:
            bullet_text = "\n".join(f"• {s}" for s in steps)
            self._steps_label.setText(bullet_text)
            self._steps_label.setVisible(True)
        else:
            self._steps_label.setVisible(False)

    def reset(self) -> None:
        """Reset card to idle state."""
        self._current_class = None
        self._title_label.setText("💡 Lời khuyên sơ chế rác")
        self._tip_label.setText("Chờ phát hiện rác để hiển thị lời khuyên...")
        self._tip_label.setVisible(True)
        self._steps_label.setText("Đưa rác vào trước camera để nhận hướng dẫn chuẩn bị phù hợp trước khi vứt.")
        self._steps_label.setVisible(True)
