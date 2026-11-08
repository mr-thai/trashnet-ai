"""Module làm mượt theo thời gian và khử rung nhấp nháy nhãn rác (Temporal Smoother).

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Theo dõi vật thể liên tục qua chuỗi khung hình video dựa trên mức độ chồng khớp bounding box (IoU tracking).
2. Áp dụng cơ chế bỏ phiếu theo trọng số độ tin cậy trong cửa sổ trượt (Sliding window voting).
3. Đòi hỏi nhãn rác phải ổn định liên tục tối thiểu 1.5 giây trước khi xác nhận đổi nhãn (State Machine: TENTATIVE -> CONFIRMED).
4. Tự động loại bỏ các nhiễu nhấp nháy tức thời (glitches) và giải phóng các vật thể đã rời khỏi camera.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple

from core.mapper import BinMapper
from core.types import Bin, Detection


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2).

    [GIẢI THÍCH NGHIỆP VỤ - TÍNH DIỆN TÍCH GIAO TRÊN DIỆN TÍCH HỢP (IoU)]:
    Đo mức độ trùng khớp giữa 2 khung bounding box:
    - Giá trị = 1.0: Trùng khít hoàn toàn.
    - Giá trị = 0.0: Không chạm nhau.
    Dùng để nhận biết xem vật thể ở khung hình hiện tại có trùng với vật thể ở khung trước hay không.

    Args:
        box_a: (x1, y1, x2, y2) of the first box.
        box_b: (x1, y1, x2, y2) of the second box.

    Returns:
        IoU score in range [0.0, 1.0].
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


class TrackState(str, Enum):
    """Lifecycle state of an individual tracked waste object."""

    TENTATIVE = "TENTATIVE"    # Accumulating consistent detections (< 1.5s)
    CONFIRMED = "CONFIRMED"    # Consistently identified (>= 1.5s)
    LOST = "LOST"              # Temporarily missing in latest frame


@dataclass
class TrackedTarget:
    """Represents a single waste object tracked across video frames."""

    track_id: int
    first_seen: float
    last_seen: float
    state: TrackState = TrackState.TENTATIVE
    history: deque = field(default_factory=deque)  # Stores (timestamp, Detection)
    current_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    current_polygon: List[Tuple[int, int]] = field(default_factory=list)

    # Nhãn rác đã xác nhận ổn định (Confirmed)
    confirmed_class_id: Optional[int] = None
    confirmed_class_name: Optional[str] = None
    confirmed_bin_type: Optional[Bin] = None

    # Nhãn rác ứng viên đang trong thời gian theo dõi ổn định (Candidate)
    candidate_class_id: Optional[int] = None
    candidate_class_name: Optional[str] = None
    candidate_bin_type: Optional[Bin] = None
    candidate_first_seen: float = 0.0

    def update_detection(
        self,
        det: Detection,
        timestamp: float,
        stability_threshold_sec: float = 1.5,
    ) -> None:
        """Cập nhật vết theo dõi với phát hiện mới và tính toán nhãn làm mượt."""
        self.history.append((timestamp, det))
        self.current_box = det.box
        self.current_polygon = det.polygon
        self.last_seen = timestamp

        # 1. Bỏ phiếu số đông có trọng số độ tin cậy trong cửa sổ trượt
        scores: Dict[Tuple[int, str, Bin], float] = defaultdict(float)
        for _, sample in self.history:
            key = (sample.class_id, sample.class_name, sample.bin_type)
            scores[key] += float(sample.confidence)

        if not scores:
            return

        best_key = max(scores.keys(), key=lambda k: scores[k])
        best_class_id, best_class_name, best_bin_type = best_key

        # 2. Máy trạng thái tính ổn định theo thời gian
        if self.confirmed_class_name is None:
            # Trạng thái TENTATIVE: Cần xuất hiện ổn định liên tục >= 1.5 giây
            if self.candidate_class_name != best_class_name:
                self.candidate_class_id = best_class_id
                self.candidate_class_name = best_class_name
                self.candidate_bin_type = best_bin_type
                self.candidate_first_seen = timestamp
            else:
                elapsed = timestamp - self.candidate_first_seen
                if elapsed >= stability_threshold_sec:
                    self.confirmed_class_id = self.candidate_class_id
                    self.confirmed_class_name = self.candidate_class_name
                    self.confirmed_bin_type = self.candidate_bin_type
                    self.state = TrackState.CONFIRMED
        else:
            # Đã ở trạng thái CONFIRMED
            self.state = TrackState.CONFIRMED

            if det.class_name == self.confirmed_class_name:
                self.candidate_class_id = None
                self.candidate_class_name = None
                self.candidate_bin_type = None
                self.candidate_first_seen = timestamp
            else:
                # Xuất hiện nhãn rác thay thế khác
                if self.candidate_class_name != det.class_name:
                    self.candidate_class_id = det.class_id
                    self.candidate_class_name = det.class_name
                    self.candidate_bin_type = det.bin_type
                    self.candidate_first_seen = timestamp
                else:
                    elapsed = timestamp - self.candidate_first_seen
                    if elapsed >= stability_threshold_sec:
                        # Đã duy trì liên tục >= 1.5 giây -> Cho phép chuyển nhãn
                        self.confirmed_class_id = self.candidate_class_id
                        self.confirmed_class_name = self.candidate_class_name
                        self.confirmed_bin_type = self.candidate_bin_type
                        self.candidate_class_id = None
                        self.candidate_class_name = None
                        self.candidate_bin_type = None

    def mark_missed(self) -> None:
        """Đánh dấu vết theo dõi bị tạm mất trong khung hình hiện tại."""
        self.state = TrackState.LOST

    def get_smoothed_detection(self, require_confirmation: bool = True) -> Optional[Detection]:
        """Tạo đối tượng Detection đã được làm mượt phục vụ hiển thị và xuất báo cáo."""
        if require_confirmation and self.state != TrackState.CONFIRMED:
            return None

        if self.confirmed_class_name is not None:
            cid = self.confirmed_class_id
            cname = self.confirmed_class_name
            btype = self.confirmed_bin_type
        elif not require_confirmation and self.candidate_class_name is not None:
            cid = self.candidate_class_id
            cname = self.candidate_class_name
            btype = self.candidate_bin_type
        else:
            return None

        if cid is None or btype is None:
            return None

        # Compute average confidence for the active class in the window
        matching_confs = [d.confidence for _, d in self.history if d.class_name == cname]
        avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.50

        return Detection(
            class_id=int(cid),
            class_name=str(cname),
            confidence=float(round(avg_conf, 3)),
            box=self.current_box,
            polygon=self.current_polygon,
            bin_type=btype,
        )


class TemporalSmoother:
    """Multi-object temporal smoother and anti-flicker tracker (US-4.2).

    [GIẢI THÍCH NGHIỆP VỤ - BỘ LÀM MƯỢT KHUNG HÌNH CHỐNG NHẤP NHÁY]:
    Theo dõi vật thể qua chuỗi khung hình video liên tục (Tracking by IoU).
    Áp dụng cửa sổ trượt (sliding window) và yêu cầu nhận diện nhất quán liên tục trong 1.5 giây
    trước khi xác nhận đổi nhãn hiển thị rác, giúp triệt tiêu hiện tượng nhấp nháy (flickering).
    """

    def __init__(
        self,
        window_size: int = 30,
        stability_threshold_sec: float = 1.5,
        disappear_timeout_sec: float = 1.0,
        iou_threshold: float = 0.30,
        require_confirmation: bool = True,
        bin_mapper: Optional[BinMapper] = None,
    ) -> None:
        """Initialize TemporalSmoother.

        Args:
            window_size: Sliding window capacity (20-30 frames). Defaults to 30.
            stability_threshold_sec: Time in seconds required for consistent class detection. Defaults to 1.5.
            disappear_timeout_sec: Time in seconds before an unseen object is purged. Defaults to 1.0.
            iou_threshold: Minimum IoU overlap required to match tracks between frames. Defaults to 0.30.
            require_confirmation: Whether to suppress tentative detections (<1.5s). Defaults to True.
            bin_mapper: Optional BinMapper for mapping class IDs to bins.
        """
        self.window_size = max(5, int(window_size))
        self.stability_threshold_sec = float(stability_threshold_sec)
        self.disappear_timeout_sec = float(disappear_timeout_sec)
        self.iou_threshold = float(iou_threshold)
        self.require_confirmation = require_confirmation
        self.bin_mapper = bin_mapper

        self._tracks: Dict[int, TrackedTarget] = {}
        self._next_track_id: int = 1

    @property
    def active_tracks_count(self) -> int:
        """Return the number of currently tracked objects."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all active tracks and reset smoother state."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: List[Detection],
        timestamp: Optional[float] = None,
    ) -> List[Detection]:
        """Update tracker with detections from current frame and return smoothed results.

        Args:
            detections: List of raw Detection objects detected in the current frame.
            timestamp: Optional monotonic timestamp in seconds. If None, uses time.perf_counter().

        Returns:
            List of smoothed Detection objects sorted by confidence descending.
        """
        curr_time = float(time.perf_counter()) if timestamp is None else float(timestamp)

        matched_track_ids = set()
        unmatched_detections: List[Detection] = []

        # Bước 1: Khớp tham lam (Greedy IoU matching) giữa các vết đang theo dõi và phát hiện mới
        if self._tracks and detections:
            candidate_pairs = []
            for track_id, track in self._tracks.items():
                for det_idx, det in enumerate(detections):
                    iou = compute_iou(track.current_box, det.box)
                    if iou >= self.iou_threshold:
                        candidate_pairs.append((iou, track_id, det_idx))

            candidate_pairs.sort(key=lambda x: x[0], reverse=True)

            used_det_indices = set()
            for _, track_id, det_idx in candidate_pairs:
                if track_id in matched_track_ids or det_idx in used_det_indices:
                    continue

                det = detections[det_idx]
                self._tracks[track_id].update_detection(
                    det=det,
                    timestamp=curr_time,
                    stability_threshold_sec=self.stability_threshold_sec,
                )
                matched_track_ids.add(track_id)
                used_det_indices.add(det_idx)

            for det_idx, det in enumerate(detections):
                if det_idx not in used_det_indices:
                    unmatched_detections.append(det)
        else:
            unmatched_detections = list(detections)

        # Bước 2: Đánh dấu các vết không tìm thấy trong khung hình hiện tại là tạm mất (LOST)
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.mark_missed()

        # Bước 3: Khởi tạo vết theo dõi mới cho các phát hiện chưa được khớp
        for det in unmatched_detections:
            new_track = TrackedTarget(
                track_id=self._next_track_id,
                first_seen=curr_time,
                last_seen=curr_time,
                state=TrackState.TENTATIVE,
                history=deque(maxlen=self.window_size),
                current_box=det.box,
                current_polygon=det.polygon,
            )
            new_track.update_detection(
                det=det,
                timestamp=curr_time,
                stability_threshold_sec=self.stability_threshold_sec,
            )
            self._tracks[self._next_track_id] = new_track
            self._next_track_id += 1

        # Bước 4: Xóa bỏ các vết đã biến mất quá thời gian chờ (disappear_timeout_sec)
        expired_track_ids = [
            tid
            for tid, trk in self._tracks.items()
            if (curr_time - trk.last_seen) > self.disappear_timeout_sec
        ]
        for tid in expired_track_ids:
            del self._tracks[tid]

        # Bước 5: Trích xuất danh sách các đối tượng nhận diện đã làm mượt
        smoothed_results: List[Detection] = []
        for track in self._tracks.values():
            smoothed_det = track.get_smoothed_detection(
                require_confirmation=self.require_confirmation
            )
            if smoothed_det is not None:
                smoothed_results.append(smoothed_det)

        smoothed_results.sort(key=lambda d: d.confidence, reverse=True)
        return smoothed_results

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # =========================================================================
    cap_nhat = update
    lam_muot = update
    dat_lai = reset


# =============================================================================
# BÍ DANH HÀM & LỚP TIẾNG VIỆT (VIETNAMESE FUNCTION & CLASS ALIASES)
# =============================================================================
BoLamMuotTheoThoiGian = TemporalSmoother
tinh_chi_so_iou = compute_iou
TrangThaiTheoDoi = TrackState
DoiTuongTheoDoi = TrackedTarget

