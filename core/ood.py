"""Out-of-Distribution (OOD) and uncertain waste detection rules (US-4.4).

NOTE: The model is NOT trained to detect hazardous waste directly.
This module provides a safety fallback rule based on confidence thresholds
and vote margin dispersion to flag unknown/uncertain objects for manual inspection.
"""

from dataclasses import replace
from typing import List, Optional
from core.types import Bin, Detection

DEFAULT_OOD_CONF_THRESHOLD: float = 0.50  # Acceptance criteria: default in range 0.35–0.60
DEFAULT_OOD_MARGIN_THRESHOLD: float = 0.15


class OODDetector:
    """Detects Out-of-Distribution and uncertain predictions requiring manual inspection."""

    def __init__(
        self,
        conf_threshold: float = DEFAULT_OOD_CONF_THRESHOLD,
        margin_threshold: float = DEFAULT_OOD_MARGIN_THRESHOLD,
    ) -> None:
        """Initialize OODDetector.

        Args:
            conf_threshold: Maximum confidence below which an object is considered uncertain (0.35–0.60).
            margin_threshold: Minimum margin between top-1 and top-2 class probabilities/votes.
        """
        self.conf_threshold = float(conf_threshold)
        self.margin_threshold = float(margin_threshold)

    def is_ood(self, confidence: float, margin: Optional[float] = None) -> bool:
        """Determine if a prediction falls outside acceptable distribution bounds."""
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
        """Inspect Detection and convert to Bin.UNKNOWN if uncertain/OOD."""
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
