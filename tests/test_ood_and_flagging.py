"""Unit tests for OOD detection and manual hazardous flagging (US-4.4).

Verifies Out-of-Distribution uncertainty rules, Bin enum expansion,
and flagged dataset persistence with JSON metadata.
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.flagging import save_flagged_frame
from core.ood import OODDetector
from core.types import Bin, Detection
from ui.qt_compat import QApplication
from ui.widgets.bin_banner import BinBanner


@pytest.fixture(scope="session")
def qapp():
    """Ensure a singleton QApplication exists across test suite."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_det(confidence: float = 0.85, bin_type: Bin = Bin.RECYCLABLE) -> Detection:
    return Detection(
        class_id=5,
        class_name="plastic",
        confidence=confidence,
        box=(10, 10, 100, 100),
        polygon=[(10, 10), (100, 100)],
        bin_type=bin_type,
    )


class TestOODDetection:
    """Verify OOD detection rules and uncertainty thresholds."""

    def test_low_confidence_marked_unknown(self):
        """DoD AC: Khi độ tin cậy dưới ngưỡng OOD (0.50), gắn cờ không chắc chắn."""
        detector = OODDetector(conf_threshold=0.50)
        det_low = make_det(confidence=0.42, bin_type=Bin.RECYCLABLE)
        checked = detector.check_detection(det_low)
        assert checked.bin_type == Bin.UNKNOWN

    def test_high_confidence_retains_bin(self):
        """High confidence detections remain unchanged."""
        detector = OODDetector(conf_threshold=0.50)
        det_high = make_det(confidence=0.88, bin_type=Bin.RECYCLABLE)
        checked = detector.check_detection(det_high)
        assert checked.bin_type == Bin.RECYCLABLE

    def test_margin_dispersion_triggers_ood(self):
        """DoD AC: Xác định quy tắc OOD dựa trên độ phân tán phiếu bầu (margin)."""
        detector = OODDetector(conf_threshold=0.50, margin_threshold=0.15)
        det_ambiguous = make_det(confidence=0.65, bin_type=Bin.ORGANIC)
        # Margin between top 1 and top 2 is only 0.05 (< 0.15)
        checked = detector.check_detection(det_ambiguous, margin=0.05)
        assert checked.bin_type == Bin.UNKNOWN


class TestFlaggingSystem:
    """Verify manual reporting and hazardous flagging into flagged/ directory."""

    def test_save_hazardous_frame_and_metadata(self, tmp_path):
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = [make_det(confidence=0.38, bin_type=Bin.UNKNOWN)]

        img_path, json_path = save_flagged_frame(
            frame=dummy_frame,
            detections=detections,
            is_hazardous=True,
            note="Phát hiện pin đồ chơi rò rỉ",
            flagged_dir=tmp_path,
        )

        assert img_path.exists()
        assert json_path.exists()
        assert img_path.suffix == ".jpg"
        assert json_path.suffix == ".json"

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["hazardous"] is True
        assert "pin" in meta["user_note"]
        assert meta["detection_count"] == 1
        assert meta["detections"][0]["confidence"] == 0.38

    def test_save_non_hazardous_frame(self, tmp_path):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path, json_path = save_flagged_frame(
            frame=dummy_frame,
            detections=[],
            is_hazardous=False,
            note="Vật thể lạ chưa rõ",
            flagged_dir=tmp_path,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["hazardous"] is False


class TestBinBannerOODAndHazardous:
    """Verify visual presentation for UNKNOWN and HAZARDOUS states."""

    def test_banner_unknown_ood_state(self, qapp):
        banner = BinBanner()
        banner.set_bin(Bin.UNKNOWN)
        assert banner.current_bin == Bin.UNKNOWN
        assert "KHÔNG CHẮC CHẮN" in banner._title_label.text()
        assert "#D97706" in banner.styleSheet()

    def test_banner_hazardous_state(self, qapp):
        banner = BinBanner()
        banner.set_bin(Bin.HAZARDOUS)
        assert banner.current_bin == Bin.HAZARDOUS
        assert "RÁC NGUY HẠI" in banner._title_label.text()
        assert "#DC2626" in banner.styleSheet()
