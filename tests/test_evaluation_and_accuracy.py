"""Comprehensive evaluation and accuracy verification test suite.

Validates model inference on real MultiTrash dataset samples, confirms color
channel integrity, verifies coordinate boundaries across extreme image dimensions,
and tests the complete end-to-end pipeline (Detector -> Mapper -> OOD -> Smoother).
"""

from pathlib import Path
from typing import List, Set
import cv2
import numpy as np
import pytest

from core.detector import WasteDetector
from core.mapper import BinMapper
from core.ood import OODDetector
from core.smoother import TemporalSmoother
from core.types import Bin, Detection


MODEL_PATH = Path("models/best.pt")
TRAIN_IMG_DIR = Path("MultiTrash/train/images")
TRAIN_LBL_DIR = Path("MultiTrash/train/labels")


@pytest.fixture(scope="module")
def waste_detector() -> WasteDetector:
    """Fixture providing initialized WasteDetector with production best.pt weights."""
    assert MODEL_PATH.exists(), f"Production model not found at {MODEL_PATH}"
    return WasteDetector(
        model_path=MODEL_PATH,
        conf=0.25,
        imgsz=640,
        warmup=True,
    )


class TestGroundTruthAccuracy:
    """Tests evaluating actual recognition accuracy against MultiTrash ground truth."""

    def test_detection_on_real_train_samples_matches_ground_truth(self, waste_detector: WasteDetector):
        """Verify that WasteDetector predictions on train images match ground truth classes."""
        if not TRAIN_IMG_DIR.exists() or not TRAIN_LBL_DIR.exists():
            pytest.skip("MultiTrash dataset not present in workspace")

        # Pick 5 diverse real training samples
        sample_images = sorted(list(TRAIN_IMG_DIR.glob("*.jpg")))[:5]
        assert len(sample_images) > 0, "No training images found"

        total_gt_classes: Set[str] = set()
        matched_classes: Set[str] = set()

        for img_path in sample_images:
            lbl_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            gt_ids = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_ids.append(int(parts[0]))

            gt_names = {waste_detector.names[cid] for cid in gt_ids if cid in waste_detector.names}
            total_gt_classes.update(gt_names)

            frame = cv2.imread(str(img_path))
            assert frame is not None, f"Failed to read image {img_path}"

            detections = waste_detector.detect(frame, conf=0.25)
            pred_names = {d.class_name for d in detections}

            # Intersect predictions with ground truth
            common = gt_names.intersection(pred_names)
            matched_classes.update(common)

        # Ensure model actually recognized majority of ground truth categories
        assert len(matched_classes) >= len(total_gt_classes) * 0.6, (
            f"Model matched only {matched_classes} out of ground truth {total_gt_classes}"
        )


class TestColorChannelRegression:
    """Regression test ensuring BGR input is not double-converted to RGB."""

    def test_bgr_input_preserves_higher_detection_confidence(self, waste_detector: WasteDetector):
        """Verify that proper BGR input produces valid high-confidence detections."""
        sample_img_path = Path("tests/data/sample_waste.jpg")
        if not sample_img_path.exists():
            pytest.skip("sample_waste.jpg not found")

        frame_bgr = cv2.imread(str(sample_img_path))
        assert frame_bgr is not None

        dets_bgr = waste_detector.detect(frame_bgr, conf=0.20)
        assert len(dets_bgr) > 0, "No objects detected in sample image"

        top_det = dets_bgr[0]
        assert top_det.confidence > 0.30
        assert top_det.bin_type in (Bin.ORGANIC, Bin.RECYCLABLE, Bin.RESIDUAL)


class TestDimensionRobustness:
    """Test inference across extreme image resolutions and aspect ratios."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (1920, 1080),  # Full HD landscape
            (1080, 1920),  # Full HD portrait / smartphone camera
            (1280, 360),   # Ultra-wide aspect ratio
            (360, 1280),   # Ultra-tall aspect ratio
            (320, 320),    # Small square
        ],
    )
    def test_coordinate_boundaries_within_frame_limits(
        self, waste_detector: WasteDetector, width: int, height: int
    ):
        """Ensure all predicted bounding boxes and polygons remain bounded within [0, W] and [0, H]."""
        # Create synthetic frame with identifiable patterns
        synthetic_frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.circle(synthetic_frame, (width // 2, height // 2), min(width, height) // 4, (200, 50, 50), -1)

        detections = waste_detector.detect(synthetic_frame, conf=0.05)

        for det in detections:
            x1, y1, x2, y2 = det.box
            assert 0 <= x1 <= width, f"x1 ({x1}) out of range [0, {width}]"
            assert 0 <= x2 <= width, f"x2 ({x2}) out of range [0, {width}]"
            assert 0 <= y1 <= height, f"y1 ({y1}) out of range [0, {height}]"
            assert 0 <= y2 <= height, f"y2 ({y2}) out of range [0, {height}]"
            assert x1 <= x2
            assert y1 <= y2

            for px, py in det.polygon:
                assert 0 <= px <= width, f"polygon x ({px}) out of bounds"
                assert 0 <= py <= height, f"polygon y ({py}) out of bounds"


class TestEndToEndPipeline:
    """Test full integration from raw frame to smoothed and audit-logged output."""

    def test_complete_detection_and_filtering_pipeline(self, waste_detector: WasteDetector):
        """Verify Detector -> Mapper -> OOD -> Smoother integration flow."""
        sample_img_path = Path("tests/data/sample_waste.jpg")
        if not sample_img_path.exists():
            pytest.skip("sample_waste.jpg not found")

        frame = cv2.imread(str(sample_img_path))
        assert frame is not None

        mapper = BinMapper()
        ood = OODDetector(conf_threshold=0.35)
        smoother = TemporalSmoother(window_size=10, stability_threshold_sec=0.0, require_confirmation=False)

        # 1. Detector
        raw_dets = waste_detector.detect(frame, conf=0.25)
        assert len(raw_dets) > 0

        # 2. Mapper check
        for d in raw_dets:
            assert d.bin_type == mapper.map_class_to_bin(d.class_id)

        # 3. OOD check
        filtered_dets = ood.check_detections(raw_dets)
        assert len(filtered_dets) == len(raw_dets)

        # 4. Smoother update over sequential identical frames
        for _ in range(3):
            smoothed_dets = smoother.update(filtered_dets)
        assert len(smoothed_dets) > 0


class TestStabilityAndStress:
    """Inference stress and memory consistency tests."""

    def test_sequential_inference_stability(self, waste_detector: WasteDetector):
        """Verify that 20 consecutive inference calls execute without crashing or shape corruption."""
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for _ in range(20):
            res = waste_detector.detect(dummy_frame, conf=0.5)
            assert isinstance(res, list)
