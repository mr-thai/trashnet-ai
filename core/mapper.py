"""Bộ ánh xạ lớp phân loại rác thải vào thùng chứa (Bin Mapper) cho Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Ánh xạ 8 lớp rác của tập dữ liệu MultiTrash vào 3 loại thùng chính (Hữu cơ, Tái chế, Vô cơ/Khác).
2. Đọc cấu hình từ file config/bin_mapping.yaml; nếu lỗi hoặc thiếu file sẽ tự động dùng bảng dự phòng an toàn.
3. Cung cấp màu sắc định dạng HEX và BGR phục vụ việc vẽ bounding box và giao diện.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import yaml
from core.types import Bin

DEFAULT_MAPPING_PATH = Path("config/bin_mapping.yaml")


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Chuyển đổi chuỗi mã màu hex (ví dụ: '#2E7D32') thành bộ 3 số nguyên BGR cho OpenCV."""
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) != 6:
        return (140, 140, 140)
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (b, g, r)
    except ValueError:
        return (140, 140, 140)

# Bảng ánh xạ dự phòng an toàn khi thiếu file cấu hình YAML
FALLBACK_CLASS_TO_BIN: Dict[int, Bin] = {
    0: Bin.ORGANIC,     # biowaste (rác hữu cơ)
    1: Bin.RECYCLABLE,  # glass (thủy tinh)
    2: Bin.RECYCLABLE,  # metal (kim loại)
    3: Bin.RESIDUAL,    # other (vô cơ khác)
    4: Bin.RECYCLABLE,  # paper (giấy / carton)
    5: Bin.RECYCLABLE,  # plastic (nhựa)
    6: Bin.RESIDUAL,    # styrofoam (xốp)
    7: Bin.RESIDUAL,    # wrapper (bao bì màng)
}


class BinMapper:
    """Manages mapping between dataset class IDs and destination sorting bins.

    [GIẢI THÍCH NGHIỆP VỤ - BỘ ÁNH XẠ THÙNG RÁC]:
    Quản lý việc phân loại 8 loại rác của MultiTrash vào các thùng tương ứng:
    - Rác hữu cơ (biowaste) -> Thùng HỮU CƠ (ORGANIC).
    - Chai lọ, kim loại, giấy, nhựa (glass, metal, paper, plastic) -> Thùng TÁI CHẾ (RECYCLABLE).
    - Hộp xốp, bao bì màng, rác tạp (styrofoam, wrapper, other) -> Thùng VÔ CƠ / KHÁC (RESIDUAL).
    Hỗ trợ nạp cấu hình động từ file config/bin_mapping.yaml hoặc dùng giá trị dự phòng an toàn.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize BinMapper with YAML configuration.

        Args:
            config_path: Path to bin_mapping.yaml file. Defaults to config/bin_mapping.yaml.
        """
        self._class_to_bin: Dict[int, Bin] = dict(FALLBACK_CLASS_TO_BIN)
        self._bin_info: Dict[str, dict] = {}

        path = Path(config_path) if config_path else DEFAULT_MAPPING_PATH
        if path.exists() and path.is_file():
            self._load_from_yaml(path)

    def _load_from_yaml(self, path: Path) -> None:
        """Parse mapping from YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "bins" not in data:
                return

            new_mapping: Dict[int, Bin] = {}
            for bin_key, bin_data in data["bins"].items():
                try:
                    bin_enum = Bin[bin_key.upper()]
                except KeyError:
                    bin_enum = Bin.UNKNOWN

                classes = bin_data.get("classes", [])
                for cid in classes:
                    new_mapping[int(cid)] = bin_enum

            if new_mapping:
                self._class_to_bin = new_mapping
                self._bin_info = data.get("bins", {})
        except Exception:
            # Tự động chuyển sang cấu hình dự phòng khi đọc file YAML gặp sự cố
            self._class_to_bin = dict(FALLBACK_CLASS_TO_BIN)

    def map_class_to_bin(self, class_id: int) -> Bin:
        """Ánh xạ mã lớp số nguyên (0-7) sang phân loại Thùng rác (Bin) tương ứng."""
        return self._class_to_bin.get(class_id, Bin.UNKNOWN)

    def get_bin_color_hex(self, bin_type: Bin, default: str = "#8C8C8C") -> str:
        """Lấy chuỗi mã màu hex đại diện cho loại thùng rác."""
        info = self._bin_info.get(bin_type.name, {})
        return str(info.get("color", default))

    def get_bin_color_bgr(self, bin_type: Bin) -> Tuple[int, int, int]:
        """Lấy bộ 3 số nguyên màu BGR dùng để vẽ viền khung trong OpenCV."""
        if bin_type == Bin.HAZARDOUS:
            return (38, 38, 220)  # Màu đỏ cảnh báo
        if bin_type == Bin.UNKNOWN:
            return (140, 140, 140)  # Màu xám trung tính
        hex_color = self.get_bin_color_hex(bin_type, "")
        if hex_color:
            return hex_to_bgr(hex_color)
        return (140, 140, 140)

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # =========================================================================
    anh_xa = map_class_to_bin
    lay_mau_thung_hex = get_bin_color_hex
    lay_mau_thung_bgr = get_bin_color_bgr


# Global singleton instance for convenient import
_default_mapper = BinMapper()


def map_class_to_bin(class_id: int) -> Bin:
    """Functional convenience wrapper for class-to-bin mapping.

    [GIẢI THÍCH NGHIỆP VỤ - ÁNH XẠ LỚP RÁC THÀNH THÙNG CHỨA]:
    Nhận vào mã lớp số nguyên class_id (0-7), trả về Enum Bin tương ứng
    (ORGANIC, RECYCLABLE, RESIDUAL, UNKNOWN).
    """
    return _default_mapper.map_class_to_bin(class_id)


# =============================================================================
# BÍ DANH HÀM & LỚP TIẾNG VIỆT (VIETNAMESE FUNCTION & CLASS ALIASES)
# =============================================================================
AnhXaThungRac = BinMapper
anh_xa_thung_rac = map_class_to_bin
chuyen_hex_sang_bgr = hex_to_bgr

