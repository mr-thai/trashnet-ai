"""Unit and integration tests for session audit logging and CSV export (US-4.5).

Verifies event deduplication, summary counts, UTF-8 BOM encoding for Excel,
and AuditLogDialog Qt model operations.
"""

import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.audit_log import AuditEntry, AuditLogger
from core.types import Bin, Detection
from ui.qt_compat import QApplication, Qt
from ui.widgets.audit_dialog import AuditLogDialog, AuditTableModel


def make_test_det(
    class_id: int = 5,
    class_name: str = "plastic",
    confidence: float = 0.88,
    bin_type: Bin = Bin.RECYCLABLE,
    box=(20, 20, 120, 120),
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box=box,
        polygon=[(20, 20), (120, 120)],
        bin_type=bin_type,
    )


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestAuditLogger:
    """Verify in-memory logging, anti-duplication, and CSV export."""

    def test_log_detection_anti_duplication(self):
        """DoD AC: Cùng một vật thể ổn định không bị ghi lặp lại hàng trăm lần."""
        logger = AuditLogger(cooldown_seconds=3.0)
        det = make_test_det()
        t0 = datetime(2026, 10, 20, 12, 0, 0)

        # First record: accepted
        ok1 = logger.log_detection(det, track_id=1, timestamp=t0)
        assert ok1 is True
        assert logger.count == 1

        # Second record within 1.0s (under 3.0s cooldown): rejected / deduplicated
        t1 = t0 + timedelta(seconds=1.0)
        ok2 = logger.log_detection(det, track_id=1, timestamp=t1)
        assert ok2 is False
        assert logger.count == 1

        # Third record after 3.5s (past 3.0s cooldown): accepted
        t2 = t0 + timedelta(seconds=3.5)
        ok3 = logger.log_detection(det, track_id=1, timestamp=t2)
        assert ok3 is True
        assert logger.count == 2

    def test_summary_by_bin(self):
        """DoD AC: Có thống kê nhanh số lượng theo thùng."""
        logger = AuditLogger()
        logger.log_detection(make_test_det(bin_type=Bin.ORGANIC), track_id=1)
        logger.log_detection(make_test_det(bin_type=Bin.ORGANIC), track_id=2)
        logger.log_detection(make_test_det(bin_type=Bin.RECYCLABLE), track_id=3)
        logger.log_detection(make_test_det(bin_type=Bin.HAZARDOUS), track_id=4, is_hazardous=True)

        summary = logger.get_summary_by_bin()
        assert summary[Bin.ORGANIC] == 2
        assert summary[Bin.RECYCLABLE] == 1
        assert summary[Bin.HAZARDOUS] == 1
        assert summary[Bin.RESIDUAL] == 0

    def test_export_csv_with_utf8_bom(self, tmp_path):
        """DoD AC: Tệp mã hóa UTF-8 có BOM để Excel/LibreOffice đọc đúng tiếng Việt."""
        logger = AuditLogger()
        t = datetime(2026, 10, 20, 15, 30, 45)
        logger.log_detection(
            make_test_det(class_name="plastic", confidence=0.915, bin_type=Bin.RECYCLABLE),
            source="webcam",
            timestamp=t,
        )
        logger.log_detection(
            make_test_det(class_name="biowaste", confidence=0.840, bin_type=Bin.ORGANIC),
            source="image",
            timestamp=t,
        )

        csv_file = tmp_path / "test_report.csv"
        saved = logger.export_csv(csv_file)
        assert saved.exists()

        # 1. Verify binary BOM header: \xef\xbb\xbf
        with open(saved, "rb") as f:
            raw_bytes = f.read(3)
            assert raw_bytes == b"\xef\xbb\xbf", "Missing UTF-8 Byte Order Mark (BOM)"

        # 2. Read with utf-8-sig and verify rows
        with open(saved, "r", encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))

        headers = reader[0]
        assert "Thời điểm" in headers
        assert "Loại rác" in headers
        assert "Thùng đích" in headers

        row1 = reader[1]
        assert "2026-10-20 15:30:45" in row1[0]
        assert "plastic" in row1[1]
        assert "91.5%" in row1[2]
        assert "Tái chế" in row1[3]
        assert "Webcam" in row1[4]

        row2 = reader[2]
        assert "Hữu cơ" in row2[3]
        assert "Ảnh tĩnh" in row2[4]

    def test_clear_logger(self):
        logger = AuditLogger()
        logger.log_detection(make_test_det())
        assert logger.count == 1
        logger.clear()
        assert logger.count == 0


class TestAuditDialogAndModel:
    """Verify Qt table model and dialog widget."""

    def test_audit_table_model(self, qapp):
        logger = AuditLogger()
        logger.log_detection(make_test_det(class_name="plastic", confidence=0.92))

        model = AuditTableModel(logger.entries)
        assert model.rowCount() == 1
        assert model.columnCount() == 6
        assert model.headerData(0, Qt.Orientation.Horizontal) == "Thời điểm"
        assert model.data(model.index(0, 1)) == "plastic"
        assert model.data(model.index(0, 2)) == "92.0%"

    def test_audit_dialog_refresh_and_clear(self, qapp):
        logger = AuditLogger()
        logger.log_detection(make_test_det(bin_type=Bin.ORGANIC))
        logger.log_detection(make_test_det(bin_type=Bin.RECYCLABLE, class_name="metal", box=(50, 50, 100, 100)))

        dialog = AuditLogDialog(logger)
        assert "Tổng: 2" in dialog._lbl_total.text()
        assert "Hữu cơ: 1" in dialog._lbl_organic.text()
        assert "Tái chế: 1" in dialog._lbl_recyclable.text()
