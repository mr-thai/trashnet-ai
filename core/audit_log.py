"""Session audit logging and CSV export system for Trash-Net (US-4.5).

Records confirmed detection events with deduplication, bin summaries,
and exports UTF-8 with BOM (utf-8-sig) CSV reports compatible with Excel.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.types import Bin, Detection


@dataclass
class AuditEntry:
    """Individual recorded sorting event."""

    timestamp: datetime
    class_name: str
    confidence: float
    bin_type: Bin
    source: str  # 'webcam' or 'image'
    is_hazardous: bool = False
    track_id: Optional[int] = None


class AuditLogger:
    """Manages in-memory session audit logs with deduplication and CSV exporting."""

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        """Initialize AuditLogger.

        Args:
            cooldown_seconds: Minimum time between duplicate records for the same track.
        """
        self.cooldown_seconds = float(cooldown_seconds)
        self._entries: List[AuditEntry] = []
        self._logged_tracks: Dict[int, float] = {}
        self._recent_signatures: Dict[str, float] = {}

    @property
    def entries(self) -> List[AuditEntry]:
        """Return shallow copy of recorded entries."""
        return list(self._entries)

    @property
    def count(self) -> int:
        """Return total number of recorded events in this session."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all session entries and reset deduplication state."""
        self._entries.clear()
        self._logged_tracks.clear()
        self._recent_signatures.clear()

    def log_detection(
        self,
        detection: Detection,
        source: str = "webcam",
        is_hazardous: bool = False,
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record a confirmed detection event with anti-duplication protection.

        Args:
            detection: Confirmed Detection object.
            source: 'webcam' or 'image'.
            is_hazardous: Flag indicating dangerous or unknown material.
            track_id: Optional tracking identifier for persistent objects.
            timestamp: Optional custom datetime for deterministic testing.

        Returns:
            True if newly recorded, False if deduplicated (skipped).
        """
        now = timestamp if timestamp is not None else datetime.now()
        now_ts = now.timestamp()

        # 1. Anti-duplication by persistent track_id
        if track_id is not None:
            if track_id in self._logged_tracks:
                last_time = self._logged_tracks[track_id]
                if (now_ts - last_time) < self.cooldown_seconds:
                    return False
            self._logged_tracks[track_id] = now_ts
        else:
            # 2. Anti-duplication by spatial signature (class + box coordinates)
            sig = f"{detection.class_name}_{detection.box}"
            if sig in self._recent_signatures:
                last_time = self._recent_signatures[sig]
                if (now_ts - last_time) < self.cooldown_seconds:
                    return False
            self._recent_signatures[sig] = now_ts

        entry = AuditEntry(
            timestamp=now,
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            bin_type=detection.bin_type,
            source=str(source),
            is_hazardous=bool(is_hazardous or detection.bin_type == Bin.HAZARDOUS),
            track_id=track_id,
        )
        self._entries.append(entry)
        return True

    def get_summary_by_bin(self) -> Dict[Bin, int]:
        """Compute distribution count grouped by destination waste bin."""
        summary: Dict[Bin, int] = {
            Bin.ORGANIC: 0,
            Bin.RECYCLABLE: 0,
            Bin.RESIDUAL: 0,
            Bin.HAZARDOUS: 0,
            Bin.UNKNOWN: 0,
        }
        for entry in self._entries:
            summary[entry.bin_type] = summary.get(entry.bin_type, 0) + 1
        return summary

    def export_csv(self, file_path: Union[str, Path]) -> Path:
        """Export session audit log to CSV with UTF-8 BOM encoding for Excel compatibility.

        Args:
            file_path: Destination output CSV file path.

        Returns:
            Path object of the created CSV file.
        """
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "Thời điểm",
            "Loại rác",
            "Độ tin cậy",
            "Thùng đích",
            "Nguồn dữ liệu",
            "Cảnh báo nguy hại",
        ]

        # Use utf-8-sig to write the UTF-8 Byte Order Mark (BOM: \ufeff)
        with open(out_path, mode="w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for e in self._entries:
                source_label = "Webcam" if e.source == "webcam" else "Ảnh tĩnh"
                hazard_label = "Nguy hại" if e.is_hazardous else "Bình thường"
                writer.writerow(
                    [
                        e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        e.class_name,
                        f"{e.confidence * 100:.1f}%",
                        e.bin_type.display_name,
                        source_label,
                        hazard_label,
                    ]
                )

        return out_path
