"""Out-of-Distribution (OOD) and uncertain waste detection rules (US-4.4).

NOTE: The model is NOT trained to detect hazardous waste directly.
This module provides a safety fallback rule based on confidence thresholds
and vote margin dispersion to flag unknown/uncertain objects for manual inspection.
"""

from dataclasses import replace
from typing import List, Optional
from core.types import Bin, Detection

DEFAULT_OOD_CONF_THRESHOLD: float = 0.25  # Aligned with default WasteDetector confidence (0.25)
DEFAULT_OOD_MARGIN_THRESHOLD: float = 0.15


class OODDetector:
    """Detects Out-of-Distribution and uncertain predictions requiring manual inspection.

    [GIẢI THÍCH NGHIỆP VỤ - BỘ PHÁT HIỆN VẬT THỂ LẠ / NGOÀI PHÂN PHỐI (OOD)]:
    Cung cấp lớp phòng vệ an toàn khi mô hình gặp rác lạ chưa từng học:
    1. Kiểm tra độ tin cậy (confidence) so với ngưỡng tối thiểu (conf_threshold).
    2. Kiểm tra khoảng cách chênh lệch điểm xác suất (margin) giữa 2 nhãn cao nhất.
    3. Nếu không chắc chắn, tự động chuyển phân loại sang Thùng KHÔNG XÁC ĐỊNH (Bin.UNKNOWN)
       để người dùng kiểm tra thủ công, tránh phân loại sai gây nguy hiểm.
    """

    def __init__(
        self,
        conf_threshold: float = DEFAULT_OOD_CONF_THRESHOLD,
        margin_threshold: float = DEFAULT_OOD_MARGIN_THRESHOLD,
    ) -> None:
        """Initialize OODDetector."""
        self.conf_threshold = float(conf_threshold)
        self.margin_threshold = float(margin_threshold)

    def is_ood(self, confidence: float, margin: Optional[float] = None) -> bool:
        """Determine if a prediction falls outside acceptable distribution bounds.

        [LUỒNG XỬ LÝ]:
        - Nếu confidence < ngưỡng quy định -> Trả về True (Ngoài phân phối).
        - Nếu margin (độ chênh 2 nhãn đầu) < ngưỡng biên -> Trả về True (Không chắc chắn).
        - Ngược lại -> Trả về False (Tin cậy).
        """
        if confidence < self.conf_threshold:
            return True
        if margin is not None and margin < self.margin_threshold:
            return True
        return False

    def check_detection(
        self,
        detection: Detection,
        margin: Optional[float] = None,
    ) -> Detection:
        """Inspect Detection and convert to Bin.UNKNOWN if uncertain/OOD.

        [LUỒNG XỬ LÝ]:
        - Kiểm tra đối tượng bằng hàm is_ood().
        - Nếu bất thường, đổi bin_type thành Bin.UNKNOWN.
        - Giữ nguyên toàn bộ tọa độ bounding box và polygon.
        """
        if self.is_ood(detection.confidence, margin=margin):
            return replace(
                detection,
                bin_type=Bin.UNKNOWN,
            )
        return detection

    def check_detections(
        self,
        detections: List[Detection],
        margins: Optional[List[float]] = None,
    ) -> List[Detection]:
        """Apply OOD safety check to a list of detections."""
        results: List[Detection] = []
        for idx, det in enumerate(detections):
            m = margins[idx] if (margins and idx < len(margins)) else None
            results.append(self.check_detection(det, margin=m))
        return results

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # =========================================================================
    kiem_tra_ngoai_phan_phoi = is_ood
    kiem_tra_nhan_dien = check_detection
    kiem_tra_danh_sach_nhan_dien = check_detections


# =============================================================================
# BÍ DANH LỚP TIẾNG VIỆT (VIETNAMESE CLASS ALIASES)
# =============================================================================
BoKiemTraVatTheLa = OODDetector

