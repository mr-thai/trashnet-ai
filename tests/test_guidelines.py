"""Unit tests for waste preparation guidelines module (US-4.3).

Verifies that all 8 MultiTrash classes have valid advice, YAML configuration loading,
and GuidelineCard UI presentation.
"""

import os
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.guidelines import GuidelineManager
from ui.qt_compat import QApplication
from ui.widgets.guideline_card import GuidelineCard


ALL_8_CLASSES = [
    "biowaste",
    "glass",
    "metal",
    "other",
    "paper",
    "plastic",
    "styrofoam",
    "wrapper",
]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestGuidelineManager:
    """Verify loading from YAML and coverage for all 8 MultiTrash classes."""

    def test_all_8_classes_covered(self):
        manager = GuidelineManager()
        for cname in ALL_8_CLASSES:
            info = manager.get_guideline(cname)
            assert "title" in info, f"Missing title for {cname}"
            assert "tip" in info, f"Missing tip for {cname}"
            assert "steps" in info, f"Missing steps for {cname}"
            assert len(info["steps"]) >= 1, f"Expected at least 1 step for {cname}"
            assert len(info["tip"]) > 0, f"Expected non-empty tip for {cname}"

    def test_unknown_fallback(self):
        manager = GuidelineManager()
        info = manager.get_guideline("completely_random_unknown_item")
        assert "steps" in info
        assert "title" in info

    def test_custom_yaml_path(self, tmp_path):
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(
            """
guidelines:
  plastic:
    title: "Nhựa Tái Chế"
    tip: "Mẹo tùy chỉnh"
    steps:
      - "Bước 1 tùy chỉnh"
""",
            encoding="utf-8",
        )
        mgr = GuidelineManager(config_path=custom_yaml)
        info = mgr.get_guideline("plastic")
        assert info["title"] == "Nhựa Tái Chế"
        assert info["tip"] == "Mẹo tùy chỉnh"


class TestGuidelineCardWidget:
    """Verify visual presentation of advice on GuidelineCard."""

    def test_initial_and_reset_state(self, qapp):
        card = GuidelineCard()
        assert card.current_class is None
        assert "Lời khuyên" in card._title_label.text()

    def test_set_guideline_updates_ui(self, qapp):
        card = GuidelineCard()
        card.set_guideline("plastic")
        assert card.current_class == "plastic"
        assert "Nhựa" in card._title_label.text()
        assert len(card._steps_label.text()) > 0
        assert "•" in card._steps_label.text()

    def test_reset_clears_content(self, qapp):
        card = GuidelineCard()
        card.set_guideline("paper")
        assert card.current_class == "paper"
        card.reset()
        assert card.current_class is None
