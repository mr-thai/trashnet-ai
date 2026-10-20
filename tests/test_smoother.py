"""Unit tests for TemporalSmoother and IoU tracker (US-4.2).

Verifies sliding window voting, 1.5-second stability threshold,
anti-flickering, multi-object IoU assignment, and disappearance timeouts.
"""

import pytest
from core.smoother import TemporalSmoother, TrackState, TrackedTarget, compute_iou
from core.types import Bin, Detection


def make_detection(
    class_id: int = 5,
    class_name: str = "plastic",
    confidence: float = 0.85,
    box=(50, 50, 150, 150),
    bin_type: Bin = Bin.RECYCLABLE,
) -> Detection:
    """Helper to generate mock Detection instance."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box=box,
        polygon=[(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])],
        bin_type=bin_type,
    )


class TestComputeIoU:
    """Verify IoU calculation edge cases."""

    def test_identical_boxes(self):
        box = (10, 10, 100, 100)
        assert compute_iou(box, box) == 1.0

    def test_disjoint_boxes(self):
        box_a = (0, 0, 50, 50)
        box_b = (100, 100, 150, 150)
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = (0, 0, 100, 100)   # Area 10000
        box_b = (50, 0, 150, 100)  # Area 10000
        # Intersection: (50, 0, 100, 100) -> 50 * 100 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU: 5000 / 15000 = 1/3
        iou = compute_iou(box_a, box_b)
        assert abs(iou - (1.0 / 3.0)) < 1e-4

    def test_zero_area_or_invalid(self):
        box_a = (50, 50, 50, 50)
        box_b = (0, 0, 100, 100)
        assert compute_iou(box_a, box_b) == 0.0


class TestTemporalSmootherStabilityAndFlickering:
    """Verify acceptance criteria for 1.5s stability and anti-flickering."""

    def test_new_object_requires_1_point_5_seconds(self):
        """DoD AC: Phải xuất hiện ổn định đủ 1.5s mới xác nhận nhãn."""
        smoother = TemporalSmoother(stability_threshold_sec=1.5, require_confirmation=True)
        det_plastic = make_detection(class_id=5, class_name="plastic", confidence=0.90)

        # Frame at t=0.0s -> Tentative, suppressed
        results_t0 = smoother.update([det_plastic], timestamp=0.0)
        assert len(results_t0) == 0

        # Frame at t=1.0s -> Still < 1.5s, suppressed
        results_t1 = smoother.update([det_plastic], timestamp=1.0)
        assert len(results_t1) == 0

        # Frame at t=1.5s -> Elapsed == 1.5s, now CONFIRMED!
        results_t15 = smoother.update([det_plastic], timestamp=1.5)
        assert len(results_t15) == 1
        assert results_t15[0].class_name == "plastic"
        assert results_t15[0].bin_type == Bin.RECYCLABLE

    def test_brief_flicker_does_not_switch_label(self):
        """DoD AC: Khi một lớp chỉ xuất hiện chớp nhoáng (dưới 1.5s), không đổi nhãn đang hiển thị."""
        smoother = TemporalSmoother(stability_threshold_sec=1.5, require_confirmation=True)
        det_plastic = make_detection(class_id=5, class_name="plastic", confidence=0.90)
        det_glass = make_detection(class_id=1, class_name="glass", confidence=0.85, bin_type=Bin.RECYCLABLE)

        # 1. Establish stable "plastic" track
        smoother.update([det_plastic], timestamp=0.0)
        smoother.update([det_plastic], timestamp=1.0)
        confirmed = smoother.update([det_plastic], timestamp=1.5)
        assert len(confirmed) == 1
        assert confirmed[0].class_name == "plastic"

        # 2. Flash noisy "glass" for 0.4s (< 1.5s)
        res_flash1 = smoother.update([det_glass], timestamp=1.6)
        res_flash2 = smoother.update([det_glass], timestamp=1.9)
        assert res_flash1[0].class_name == "plastic"
        assert res_flash2[0].class_name == "plastic"

        # 3. Returns to "plastic"
        res_back = smoother.update([det_plastic], timestamp=2.0)
        assert res_back[0].class_name == "plastic"

    def test_persistent_change_switches_label_after_1_point_5_seconds(self):
        """Verify that genuine new class replaces old class once consistent >= 1.5s."""
        smoother = TemporalSmoother(stability_threshold_sec=1.5, require_confirmation=True)
        det_plastic = make_detection(class_id=5, class_name="plastic", confidence=0.90)
        det_bio = make_detection(class_id=0, class_name="biowaste", confidence=0.92, bin_type=Bin.ORGANIC)

        # Confirm plastic at t=1.5s
        smoother.update([det_plastic], timestamp=0.0)
        smoother.update([det_plastic], timestamp=1.5)

        # New object/class biowaste starts at t=2.0s
        smoother.update([det_bio], timestamp=2.0)
        smoother.update([det_bio], timestamp=2.5)
        smoother.update([det_bio], timestamp=3.0)

        # At t=3.0s (1.0s elapsed): still displays plastic
        res_t3 = smoother.update([det_bio], timestamp=3.0)
        assert res_t3[0].class_name == "plastic"

        # At t=3.5s (1.5s elapsed): switches to biowaste and ORGANIC bin!
        res_t35 = smoother.update([det_bio], timestamp=3.5)
        assert res_t35[0].class_name == "biowaste"
        assert res_t35[0].bin_type == Bin.ORGANIC

    def test_disappearance_timeout_resets_to_empty(self):
        """DoD AC: Khi vật thể rời khỏi khung hình quá thời gian chờ, trạng thái trở về rỗng."""
        smoother = TemporalSmoother(
            stability_threshold_sec=1.5,
            disappear_timeout_sec=1.0,
            require_confirmation=True,
        )
        det = make_detection()

        # Confirm object
        smoother.update([det], timestamp=0.0)
        smoother.update([det], timestamp=1.5)
        assert smoother.active_tracks_count == 1

        # Object is missing for 0.5s (< timeout 1.0s)
        res_missing_05 = smoother.update([], timestamp=2.0)
        # Note: missing track is LOST, but if it has not yet timed out it can be kept in active tracks
        assert smoother.active_tracks_count == 1

        # Object is missing for 1.2s (> timeout 1.0s)
        res_missing_12 = smoother.update([], timestamp=2.7)
        assert smoother.active_tracks_count == 0
        assert len(res_missing_12) == 0


class TestDoDFlickeringReductionSimulation:
    """DoD: Trong kiểm thử mô phỏng, số lần đổi nhãn giảm rõ rệt so với không làm mượt."""

    def test_flickering_reduction_comparison(self):
        smoother = TemporalSmoother(
            window_size=30,
            stability_threshold_sec=1.0,
            require_confirmation=False,  # Measure label switches directly
        )

        # Simulate 30 frames: True class is plastic, but corrupted by glass alternatingly
        noisy_sequence = []
        for i in range(30):
            if i % 2 == 0:
                noisy_sequence.append(make_detection(class_name="plastic", confidence=0.85))
            else:
                noisy_sequence.append(make_detection(class_name="glass", confidence=0.40))

        # Count raw switches
        raw_switches = 0
        for i in range(1, len(noisy_sequence)):
            if noisy_sequence[i].class_name != noisy_sequence[i - 1].class_name:
                raw_switches += 1

        # Count smoothed switches
        smoothed_labels = []
        for idx, det in enumerate(noisy_sequence):
            # 30 FPS simulated: dt = 0.033s
            timestamp = idx * 0.033
            res = smoother.update([det], timestamp=timestamp)
            if res:
                smoothed_labels.append(res[0].class_name)

        smoothed_switches = 0
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] != smoothed_labels[i - 1]:
                smoothed_switches += 1

        # Verify massive reduction in label switches
        assert raw_switches >= 20
        assert smoothed_switches <= 1
        assert smoothed_switches < raw_switches


class TestMultiObjectTracking:
    """Verify independent multi-object smoothing based on IoU."""

    def test_two_distinct_objects_tracked_separately(self):
        smoother = TemporalSmoother(stability_threshold_sec=1.0, require_confirmation=True)

        obj_left = make_detection(class_name="plastic", box=(10, 10, 100, 100))
        obj_right = make_detection(class_name="metal", box=(400, 400, 500, 500), bin_type=Bin.RECYCLABLE)

        # Feed both objects for 1.0s
        for step in range(11):
            t = step * 0.1
            smoother.update([obj_left, obj_right], timestamp=t)

        results = smoother.update([obj_left, obj_right], timestamp=1.0)
        assert len(results) == 2
        classes = {r.class_name for r in results}
        assert classes == {"plastic", "metal"}
        assert smoother.active_tracks_count == 2
