"""Waste preparation guidelines manager for Trash-Net (US-4.3).

Provides accessible preprocessing advice for 8 MultiTrash classes
loaded from external YAML configuration.
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
    """Manages waste preparation advice for all waste categories."""

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
        """Fetch preparation steps and advice dictionary for a given waste class name."""
        normalized = class_name.strip().lower()
        return self._guidelines.get(normalized, self._guidelines["unknown"])

    def get_steps(self, class_name: str) -> List[str]:
        """Get bulleted actionable steps for a class."""
        info = self.get_guideline(class_name)
        return info.get("steps", [])

    def get_quick_tip(self, class_name: str) -> str:
        """Get brief one-liner tip for a class."""
        info = self.get_guideline(class_name)
        return info.get("tip", "")


_default_guideline_manager = GuidelineManager()


def get_waste_guideline(class_name: str) -> Dict[str, Any]:
    """Convenience helper to retrieve waste advice."""
    return _default_guideline_manager.get_guideline(class_name)
