"""Quản lý cẩm nang hướng dẫn sơ chế và phân loại rác thải Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Cung cấp hướng dẫn sơ chế rác trực quan (rửa sạch, bóp bẹp, tháo nắp...) cho 8 loại rác.
2. Tải dữ liệu từ config/guidelines.yaml; nếu lỗi tự động sử dụng cấu hình dự phòng an toàn.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

DEFAULT_GUIDELINES_PATH = Path("config/guidelines.yaml")

FALLBACK_GUIDELINES: Dict[str, Dict[str, Any]] = {
    "biowaste": {
        "title": "Rác hữu cơ",
        "tip": "Chắt ráo nước và tách riêng đồ nhựa",
        "steps": [
            "Chắt ráo hết nước và chất lỏng thừa trước khi bỏ vào thùng.",
            "Không để lẫn túi nilon, thìa nĩa nhựa hoặc tăm tre.",
        ],
    },
    "glass": {
        "title": "Chai lọ thủy tinh",
        "tip": "Tráng sạch cặn bẩn, tháo nắp và tránh làm vỡ",
        "steps": [
            "Đổ hết chất lỏng và tráng sơ bằng nước sạch.",
            "Tháo rời nắp chai kim loại hoặc nhựa.",
            "Tránh va đập mạnh làm vỡ vụn gây nguy hiểm.",
        ],
    },
    "metal": {
        "title": "Kim loại / Vỏ lon",
        "tip": "Đổ sạch nước và bẹp vỏ lon để tiết kiệm chỗ",
        "steps": [
            "Đổ sạch đồ uống thừa và tráng sơ lòng lon.",
            "Bóp bẹp lon nhôm để tiết kiệm diện tích chứa.",
        ],
    },
    "other": {
        "title": "Rác vô cơ khác",
        "tip": "Gói gọn gàng và không bỏ lẫn rác nguy hại",
        "steps": [
            "Buộc kín trong túi đựng rác sinh hoạt thông thường.",
            "Không bỏ lẫn pin hoặc thiết bị điện tử.",
        ],
    },
    "paper": {
        "title": "Giấy / Bìa các-tông",
        "tip": "Giữ khô ráo và gỡ bỏ băng dính",
        "steps": [
            "Giữ giấy khô ráo, không để dính nước hoặc dầu mỡ.",
            "Bóc gỡ băng dính nhựa và ghim sắt bấm.",
            "Gấp phẳng thùng các-tông trước khi xếp gọn.",
        ],
    },
    "plastic": {
        "title": "Nhựa / Chai nhựa",
        "tip": "Đổ hết nước, tháo nắp và bóp bẹp chai",
        "steps": [
            "Đổ sạch hoàn toàn chất lỏng thừa bên trong.",
            "Tháo rời nắp chai và vòng đệm nhựa.",
            "Bóp bẹp thân chai nhựa để tiết kiệm không gian.",
        ],
    },
    "styrofoam": {
        "title": "Xốp / Hộp xốp",
        "tip": "Gạt sạch dầu mỡ thức ăn thừa",
        "steps": [
            "Gạt bỏ toàn bộ thức ăn thừa bám dính trên hộp xốp.",
            "Bẻ nhỏ hoặc xếp lồng các hộp vào nhau.",
        ],
    },
    "wrapper": {
        "title": "Bao bì màng / Túi nilon",
        "tip": "Giũ sạch vụn bánh kẹo và cuộn chặt",
        "steps": [
            "Giũ sạch vụn thức ăn và gia vị bên trong bao bì.",
            "Cuộn chặt hoặc nhét vào túi nilon lớn để tránh bay rác.",
        ],
    },
    "unknown": {
        "title": "Vật thể lạ / Cần kiểm tra",
        "tip": "Kiểm tra kỹ nhãn mác trước khi phân loại",
        "steps": [
            "Đọc kỹ ký hiệu tái chế in trên bao bì sản phẩm.",
            "Không vứt pin hoặc thiết bị điện tử vào thùng thông thường.",
        ],
    },
}


class GuidelineManager:
    """Manages waste preparation advice for all waste categories.

    [GIẢI THÍCH NGHIỆP VỤ - QUẢN LÝ HƯỚNG DẪN XỬ LÝ RÁC]:
    Cung cấp lời khuyên xử lý sơ bộ và các bước chuẩn bị rác (súc rửa, tháo nắp, bóp bẹp...)
    cho 8 loại rác thải MultiTrash, tải cấu hình từ config/guidelines.yaml hoặc dùng bộ dự phòng.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize GuidelineManager with external YAML file."""
        self._guidelines: Dict[str, Dict[str, Any]] = dict(FALLBACK_GUIDELINES)
        path = Path(config_path) if config_path else DEFAULT_GUIDELINES_PATH
        if path.exists() and path.is_file():
            self._load_from_yaml(path)

    def _load_from_yaml(self, path: Path) -> None:
        """Parse instructions from YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "guidelines" in data and isinstance(data["guidelines"], dict):
                for key, val in data["guidelines"].items():
                    if isinstance(val, dict):
                        self._guidelines[key.strip().lower()] = val
        except Exception:
            self._guidelines = dict(FALLBACK_GUIDELINES)

    def get_guideline(self, class_name: str) -> Dict[str, Any]:
        """Truy xuất từ điển các bước chuẩn bị và lời khuyên theo tên lớp rác."""
        normalized = class_name.strip().lower()
        return self._guidelines.get(normalized, self._guidelines["unknown"])

    def get_steps(self, class_name: str) -> List[str]:
        """Lấy danh sách các bước hành động cụ thể cho từng loại rác."""
        info = self.get_guideline(class_name)
        return info.get("steps", [])

    def get_quick_tip(self, class_name: str) -> str:
        """Lấy câu mẹo tóm tắt nhanh một dòng cho loại rác."""
        info = self.get_guideline(class_name)
        return info.get("tip", "")

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # =========================================================================
    lay_huong_dan = get_guideline
    lay_cac_buoc = get_steps
    lay_meo_nhanh = get_quick_tip


_default_guideline_manager = GuidelineManager()


def get_waste_guideline(class_name: str) -> Dict[str, Any]:
    """Convenience helper to retrieve waste advice.

    [GIẢI THÍCH NGHIỆP VỤ - LẤY HƯỚNG DẪN XỬ LÝ RÁC]:
    Truy vấn hướng dẫn phân loại và các bước xử lý sơ bộ theo tên lớp rác (ví dụ: 'plastic', 'glass').
    """
    return _default_guideline_manager.get_guideline(class_name)


# =============================================================================
# BÍ DANH HÀM & LỚP TIẾNG VIỆT (VIETNAMESE FUNCTION & CLASS ALIASES)
# =============================================================================
QuanLyHuongDan = GuidelineManager
lay_huong_dan_phan_loai = get_waste_guideline

