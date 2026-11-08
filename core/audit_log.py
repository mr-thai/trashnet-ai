"""Hệ thống ghi nhật ký kiểm toán và xuất báo cáo CSV cho Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Ghi nhận các sự kiện nhận diện rác thải đã xác nhận (Confirmed) với cơ chế chống ghi trùng lặp (Cooldown deduplication).
2. Thống kê tổng hợp số lượng rác theo từng loại thùng chứa.
3. Xuất file báo cáo định dạng CSV chuẩn mã hóa UTF-8 BOM (utf-8-sig) mở trực tiếp trên Excel không bị lỗi phông tiếng Việt.
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
    """Manages in-memory session audit logs with deduplication and CSV exporting.

    [GIẢI THÍCH NGHIỆP VỤ - HỆ THỐNG NHẬT KÝ KIỂM TOÁN]:
    Ghi nhận lại toàn bộ các sự kiện nhận diện rác thải đã được xác nhận (Confirmed),
    có cơ chế chống ghi trùng lặp (cooldown deduplication), thống kê số lượng theo loại thùng,
    và hỗ trợ xuất báo cáo định dạng CSV mã hóa UTF-8 BOM hiển thị chuẩn tiếng Việt trên Excel.
    """

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
        """Ghi nhận sự kiện phân loại rác kèm cơ chế chống trùng lặp theo thời gian (Cooldown)."""
        now = timestamp if timestamp is not None else datetime.now()
        now_ts = now.timestamp()

        # Cơ chế 1: Chống ghi trùng theo mã định danh vết theo dõi (track_id)
        if track_id is not None:
            if track_id in self._logged_tracks:
                last_time = self._logged_tracks[track_id]
                if (now_ts - last_time) < self.cooldown_seconds:
                    return False
            self._logged_tracks[track_id] = now_ts
        else:
            # Cơ chế 2: Chống ghi trùng theo chữ ký không gian (tên lớp + tọa độ bounding box)
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
        """Thống kê tổng số lượng vật thể đã ghi nhận theo từng loại thùng chứa."""
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
        """Xuất toàn bộ nhật ký phiên làm việc ra file CSV hỗ trợ tiếng Việt trên Excel."""
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

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # =========================================================================
    ghi_nhat_ky = log_detection
    thong_ke_theo_thung = get_summary_by_bin
    xuat_bao_cao_csv = export_csv
    xoa_trang = clear


# =============================================================================
# BÍ DANH LỚP TIẾNG VIỆT (VIETNAMESE CLASS ALIASES)
# =============================================================================
NhatKyHeThong = AuditLogger
MucKiemToan = AuditEntry

