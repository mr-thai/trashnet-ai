"""Các kiểu dữ liệu và Enum cốt lõi cho hệ thống Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Định nghĩa phân loại thùng rác đích (Bin: Hữu cơ, Tái chế, Vô cơ, Nguy hại, Không xác định).
2. Định nghĩa cấu trúc kết quả nhận diện bất biến (Detection dataclass: Tọa độ, Đa giác mặt nạ, Độ tin cậy).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Bin(str, Enum):
    """Phân loại thùng rác đích cho từng loại phế thải."""

    RECYCLABLE = "RECYCLABLE"  # Thùng Tái chế (giấy, nhựa, kim loại, thủy tinh)
    ORGANIC = "ORGANIC"        # Thùng Hữu cơ (thực phẩm, rác hữu cơ)
    RESIDUAL = "RESIDUAL"      # Thùng Vô cơ / Khác (xốp, bao bì màng, rác tạp)
    HAZARDOUS = "HAZARDOUS"    # Rác Nguy hại (pin, đồ điện tử, thiết bị rò rỉ)
    UNKNOWN = "UNKNOWN"        # Không xác định / OOD (ngoài phân phối, cần kiểm tra thủ công)

    @property
    def display_name(self) -> str:
        """Nhãn tiếng Việt thân thiện hiển thị trên giao diện người dùng."""
        names = {
            Bin.RECYCLABLE: "Tái chế",
            Bin.ORGANIC: "Hữu cơ",
            Bin.RESIDUAL: "Vô cơ / Khác",
            Bin.HAZARDOUS: "Rác nguy hại",
            Bin.UNKNOWN: "Không xác định",
        }
        return names.get(self, "Không xác định")


@dataclass(frozen=True)
class Detection:
    """Cấu trúc dữ liệu bất biến biểu diễn một vật thể rác được phát hiện."""

    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    polygon: List[Tuple[int, int]]
    bin_type: Bin


# =============================================================================
# BÍ DANH LỚP TIẾNG VIỆT (VIETNAMESE CLASS ALIASES)
# =============================================================================
ThungRac = Bin
KetQuaNhanDien = Detection

