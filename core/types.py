"""Structured data types and enums for Trash-Net (US-2.2 & US-2.3).

Contains Bin classification categories and immutable Detection dataclass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Bin(str, Enum):
    """Destination waste sorting bin category."""

    RECYCLABLE = "RECYCLABLE"  # Thùng Tái chế (giấy, nhựa, kim loại, thủy tinh)
    ORGANIC = "ORGANIC"        # Thùng Hữu cơ (thực phẩm, rác hữu cơ)
    RESIDUAL = "RESIDUAL"      # Thùng Vô cơ / Khác (xốp, bao bì màng, rác tạp)
    UNKNOWN = "UNKNOWN"        # Không xác định (ngoài danh mục huấn luyện)

    @property
    def display_name(self) -> str:
        """User-friendly Vietnamese display label."""
        names = {
            Bin.RECYCLABLE: "Tái chế",
            Bin.ORGANIC: "Hữu cơ",
            Bin.RESIDUAL: "Vô cơ / Khác",
            Bin.UNKNOWN: "Không xác định",
        }
        return names.get(self, "Không xác định")


@dataclass(frozen=True)
class Detection:
    """Immutable structured representation of an object detection/segmentation result.

    Attributes:
        class_id (int): Zero-indexed category identifier (0-7).
        class_name (str): Label name of the detected waste class.
        confidence (float): Model confidence score in range [0.0, 1.0].
        box (Tuple[int, int, int, int]): Bounding box in (x1, y1, x2, y2) integer pixel coordinates.
        polygon (List[Tuple[int, int]]): Ordered list of contour (x, y) integer pixel coordinates.
        bin_type (Bin): Destination sorting bin category.
    """

    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    polygon: List[Tuple[int, int]]
    bin_type: Bin
