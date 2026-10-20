"""Audit log viewer dialog and table model for Trash-Net (US-4.5).

Provides a dedicated QTableView interface with custom QAbstractTableModel,
real-time category summary statistics, log clearing, and UTF-8 BOM CSV export.
"""

from typing import Any, List, Optional
from core.audit_log import AuditEntry, AuditLogger
from core.types import Bin
from ui.qt_compat import (
    QAbstractTableModel,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    Qt,
    QVBoxLayout,
    QWidget,
)


class AuditTableModel(QAbstractTableModel):
    """Custom Qt table model representing audit log entries."""

    COLUMNS = [
        "Thời điểm",
        "Loại rác",
        "Độ tin cậy",
        "Thùng đích",
        "Nguồn dữ liệu",
        "Cảnh báo nguy hại",
    ]

    def __init__(self, entries: Optional[List[AuditEntry]] = None, parent=None) -> None:
        super().__init__(parent)
        self._entries: List[AuditEntry] = entries if entries is not None else []

    def set_entries(self, entries: List[AuditEntry]) -> None:
        """Update model entries and refresh view."""
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        return len(self._entries)

    def columnCount(self, parent=None) -> int:
        return len(self.COLUMNS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None

        entry = self._entries[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            elif col == 1:
                return entry.class_name
            elif col == 2:
                return f"{entry.confidence * 100:.1f}%"
            elif col == 3:
                return entry.bin_type.display_name
            elif col == 4:
                return "Webcam" if entry.source == "webcam" else "Ảnh tĩnh"
            elif col == 5:
                return "☣️ Nguy hại" if entry.is_hazardous else "Bình thường"

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (0, 2, 4, 5):
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
        return None


class AuditLogDialog(QDialog):
    """Dialog window displaying session event history, bin metrics, and CSV export."""

    def __init__(self, logger: AuditLogger, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.logger = logger
        self.setWindowTitle("📋 Nhật Ký Phân Loại & Báo Cáo Phiên")
        self.resize(880, 520)
        self.setMinimumSize(720, 400)

        self._init_ui()
        self.refresh_data()

    def _init_ui(self) -> None:
        """Construct dialog layout, statistics cards, and actions."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #111827;
                color: #E5E7EB;
            }
            QTableView {
                background-color: #1F2937;
                color: #F9FAFB;
                border: 1px solid #374151;
                border-radius: 6px;
                gridline-color: #374151;
                selection-background-color: #2563EB;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #9CA3AF;
                font-weight: 700;
                padding: 6px;
                border: 1px solid #374151;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 1. Top Quick Summary Cards
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            """
            QFrame {
                background-color: #1F2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QLabel {
                font-size: 13px;
                font-weight: 600;
            }
            """
        )
        summary_layout = QHBoxLayout(self.summary_frame)
        summary_layout.setContentsMargins(8, 4, 8, 4)

        self._lbl_organic = QLabel("🌱 Hữu cơ: 0")
        self._lbl_organic.setStyleSheet("color: #4ADE80;")
        self._lbl_recyclable = QLabel("♻️ Tái chế: 0")
        self._lbl_recyclable.setStyleSheet("color: #60A5FA;")
        self._lbl_residual = QLabel("🗑️ Vô cơ: 0")
        self._lbl_residual.setStyleSheet("color: #F87171;")
        self._lbl_hazard = QLabel("⚠️ Nguy hại/OOD: 0")
        self._lbl_hazard.setStyleSheet("color: #FBBF24;")
        self._lbl_total = QLabel("Tổng: 0")
        self._lbl_total.setStyleSheet("color: #FFFFFF; font-weight: 700;")

        summary_layout.addWidget(self._lbl_organic)
        summary_layout.addWidget(self._lbl_recyclable)
        summary_layout.addWidget(self._lbl_residual)
        summary_layout.addWidget(self._lbl_hazard)
        summary_layout.addStretch()
        summary_layout.addWidget(self._lbl_total)

        layout.addWidget(self.summary_frame)

        # 2. Central Table View
        self.table_model = AuditTableModel(self.logger.entries, self)
        self.table_view = QTableView(self)
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)

        layout.addWidget(self.table_view, stretch=1)

        # 3. Bottom Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self._btn_clear = QPushButton("🔄 Xóa nhật ký")
        self._btn_clear.setStyleSheet(
            """
            QPushButton {
                background-color: #374151;
                color: #E5E7EB;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #4B5563; }
            """
        )
        self._btn_clear.clicked.connect(self._on_clear_clicked)

        self._btn_export = QPushButton("💾 Xuất CSV...")
        self._btn_export.setStyleSheet(
            """
            QPushButton {
                background-color: #15803D;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #16A34A; }
            """
        )
        self._btn_export.clicked.connect(self._on_export_clicked)

        self._btn_close = QPushButton("✖️ Đóng")
        self._btn_close.setStyleSheet(
            """
            QPushButton {
                background-color: #1F2937;
                color: #9CA3AF;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover { background-color: #374151; color: #FFFFFF; }
            """
        )
        self._btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self._btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_export)
        btn_layout.addWidget(self._btn_close)

        layout.addLayout(btn_layout)

    def refresh_data(self) -> None:
        """Update model records and refresh summary counters."""
        entries = self.logger.entries
        self.table_model.set_entries(entries)

        summary = self.logger.get_summary_by_bin()
        self._lbl_organic.setText(f"🌱 Hữu cơ: {summary.get(Bin.ORGANIC, 0)}")
        self._lbl_recyclable.setText(f"♻️ Tái chế: {summary.get(Bin.RECYCLABLE, 0)}")
        self._lbl_residual.setText(f"🗑️ Vô cơ: {summary.get(Bin.RESIDUAL, 0)}")
        hazard_ood = summary.get(Bin.HAZARDOUS, 0) + summary.get(Bin.UNKNOWN, 0)
        self._lbl_hazard.setText(f"⚠️ Nguy hại/OOD: {hazard_ood}")
        self._lbl_total.setText(f"Tổng: {len(entries)}")

    def _on_clear_clicked(self) -> None:
        """Prompt confirmation and clear all log entries."""
        if self.logger.count == 0:
            return

        reply = QMessageBox.question(
            self,
            "Xác nhận xóa nhật ký",
            "Bạn có chắc chắn muốn xóa toàn bộ lịch sử phân loại của phiên này không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.logger.clear()
            self.refresh_data()

    def _on_export_clicked(self) -> None:
        """Open file dialog and export CSV with UTF-8 BOM encoding."""
        if self.logger.count == 0:
            QMessageBox.information(self, "Xuất CSV", "Nhật ký hiện tại chưa có dữ liệu để xuất.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu tệp báo cáo CSV",
            "trashnet_session_report.csv",
            "Tệp CSV (*.csv);;Tất cả tệp (*.*)",
        )
        if not file_path:
            return

        try:
            saved_path = self.logger.export_csv(file_path)
            QMessageBox.information(
                self,
                "Xuất CSV thành công",
                f"Đã xuất báo cáo CSV thành công (chuẩn UTF-8 có BOM cho Excel)!\n\nĐường dẫn: {saved_path.name}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xuất CSV", f"Gặp sự cố khi lưu tệp:\n{exc}")
