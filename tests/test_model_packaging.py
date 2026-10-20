"""Unit and integration tests for model packaging and verification (US-1.5).

Verifies that models/best.pt is in place, has the correct SHA-256 integrity hash,
contains complete metadata in models/MODEL_CARD.md, and executes real inference
on sample test images.
"""

import hashlib
from pathlib import Path
import cv2
import pytest
from ultralytics import YOLO

from core.detector import WasteDetector


class TestModelPackaging:
    """Acceptance criteria test suite for US-1.5."""

    def test_best_weights_exist_at_models_path(self):
        """DoD AC: best.pt đặt đúng tại models/best.pt."""
        weights_path = Path("models/best.pt")
        assert weights_path.exists(), "models/best.pt does not exist!"
        assert weights_path.is_file()
        assert weights_path.stat().st_size > 1_000_000, "models/best.pt is too small or corrupt!"

    def test_model_card_contents_and_sha256(self):
        """DoD AC: Có tệp models/MODEL_CARD.md ghi ngày, ultralytics, mAP, danh sách lớp, SHA-256."""
        model_card = Path("models/MODEL_CARD.md")
        assert model_card.exists(), "Missing models/MODEL_CARD.md"

        content = model_card.read_text(encoding="utf-8")

        # Verify SHA-256 matches actual file
        weights_path = Path("models/best.pt")
        sha256_actual = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        assert sha256_actual in content, f"SHA256 {sha256_actual} not recorded in MODEL_CARD.md"

        # Verify required metadata sections
        assert "ultralytics" in content.lower()
        assert "map" in content.lower()
        assert "biowaste" in content
        assert "plastic" in content
        assert "metal" in content

    def test_yolo_model_inference_on_sample_image(self):
        """DoD AC: Khi nạp bằng YOLO('models/best.pt'), suy luận chạy được trên một ảnh mẫu."""
        model = YOLO("models/best.pt")
        assert len(model.names) == 8, f"Expected 8 classes, got {len(model.names)}"

        sample_img_path = Path("tests/data/sample_waste.jpg")
        assert sample_img_path.exists()

        img = cv2.imread(str(sample_img_path))
        assert img is not None

        results = model.predict(source=img, conf=0.3, imgsz=320, verbose=False)
        assert len(results) > 0
        assert results[0].boxes is not None

    def test_waste_detector_integration_with_best_weights(self):
        """Verify WasteDetector seamlessly integrates with models/best.pt."""
        detector = WasteDetector(model_path="models/best.pt", conf=0.3, imgsz=320)
        sample_img = cv2.imread("tests/data/sample_waste.jpg")
        detections = detector.detect(sample_img)
        assert isinstance(detections, list)
        assert len(detections) > 0
        assert hasattr(detections[0], "bin_type")
