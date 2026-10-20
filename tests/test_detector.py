"""Unit tests for WasteDetector model wrapper (US-2.1 & US-2.3).

Verifies model loading, error handling, device selection, input immutability,
structured Detection outputs (box, polygon, bin_type), descending confidence sorting,
and strict independence from PyQt/GUI frameworks.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from core.detector import WasteDetector, ModelLoadError, InvalidFrameError
from core.types import Bin, Detection


@pytest.fixture(scope="session")
def pretrained_weights_path() -> Path:
    """Fixture providing path to yolo11n-seg.pt (downloads if necessary)."""
    from ultralytics import YOLO

    weights_path = Path("yolo11n-seg.pt")
    if not weights_path.exists():
        YOLO("yolo11n-seg.pt")
    return weights_path


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture providing a sample BGR image from tests/data or synthetic image."""
    img_path = Path("tests/data/sample_waste.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            return img
    # Fallback to synthetic image
    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic[100:300, 150:450] = [0, 255, 0]
    return synthetic


class TestWasteDetector:
    """Test suite for WasteDetector."""

    def test_init_with_valid_weights(self, pretrained_weights_path: Path):
        """Test WasteDetector loads weights and initializes attributes properly."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            conf=0.3,
            imgsz=320,
            warmup=True,
        )
        assert detector.model_path == pretrained_weights_path
        assert detector.conf == 0.3
        assert detector.imgsz == 320
        assert detector.device in ("cpu", "cuda")
        assert len(detector.names) > 0

    def test_load_nonexistent_weights_raises_file_not_found(self):
        """Test that a non-existent weights path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model weights file not found"):
            WasteDetector(model_path="non_existent_weights_file.pt")

    def test_load_corrupted_weights_raises_model_load_error(self):
        """Test that a corrupted/invalid weights file raises ModelLoadError."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"corrupted binary data not a valid torch checkpoint")
            corrupted_path = Path(f.name)

        try:
            with pytest.raises(ModelLoadError, match="Failed to load YOLO model"):
                WasteDetector(model_path=corrupted_path)
        finally:
            if corrupted_path.exists():
                corrupted_path.unlink()

    def test_device_auto_detection_and_override(self, pretrained_weights_path: Path):
        """Test device can be explicitly set or queried."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            device="cpu",
            warmup=False,
        )
        assert detector.device == "cpu"

    def test_detect_input_immutability(
        self,
        pretrained_weights_path: Path,
        sample_image: np.ndarray,
    ):
        """Test that detect() does NOT mutate the input frame."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            imgsz=320,
            warmup=False,
        )
        original_copy = sample_image.copy()

        _ = detector.detect(sample_image)

        assert np.array_equal(sample_image, original_copy), "Input frame was modified in-place!"

    def test_detect_invalid_frame_types(self, pretrained_weights_path: Path):
        """Test that invalid frame arguments raise InvalidFrameError."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            imgsz=320,
            warmup=False,
        )

        with pytest.raises(InvalidFrameError, match="must be a valid numpy.ndarray"):
            detector.detect(None)

        with pytest.raises(InvalidFrameError, match="must be a valid numpy.ndarray"):
            detector.detect("not_an_image")

        gray_2d = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(InvalidFrameError, match="must have 3 dimensions"):
            detector.detect(gray_2d)

    def test_detect_returns_structured_detections(
        self,
        pretrained_weights_path: Path,
        sample_image: np.ndarray,
    ):
        """Test US-2.3: Returns typed Detection objects with box, polygon, bin."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            conf=0.1,  # Lower threshold to ensure detections on test image
            imgsz=320,
            warmup=False,
        )
        detections = detector.detect(sample_image)

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.box) == 4
            assert isinstance(det.box[0], int)
            assert isinstance(det.polygon, list)
            assert isinstance(det.bin_type, Bin)

    def test_detect_sorting_descending_by_confidence(
        self,
        pretrained_weights_path: Path,
        sample_image: np.ndarray,
    ):
        """Test US-2.3: Detections are sorted descending by confidence score."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            conf=0.01,
            imgsz=320,
            warmup=False,
        )
        detections = detector.detect(sample_image)

        if len(detections) > 1:
            scores = [d.confidence for d in detections]
            assert scores == sorted(scores, reverse=True), "Detections are not sorted descending by confidence!"

    def test_detect_empty_result_returns_empty_list_not_none(
        self,
        pretrained_weights_path: Path,
    ):
        """Test US-2.3: When no objects are detected, returns [] (never None)."""
        detector = WasteDetector(
            model_path=pretrained_weights_path,
            conf=0.9999,  # Very high threshold -> no detections
            imgsz=320,
            warmup=False,
        )
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(blank_image)

        assert detections == []
        assert detections is not None

    def test_no_pyqt_dependency_in_core(self):
        """DoD verification: core package MUST NOT import or depend on PyQt/PySide."""
        core_dir = Path("core")
        forbidden_modules = ["PyQt5", "PyQt6", "PySide2", "PySide6"]

        for py_file in core_dir.glob("**/*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                for mod in forbidden_modules:
                    assert f"import {mod}" not in content, f"Found forbidden import '{mod}' in {py_file}"
                    assert f"from {mod}" not in content, f"Found forbidden from-import '{mod}' in {py_file}"
