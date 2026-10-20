"""Class-to-bin category mapper for Trash-Net (US-2.2).

Maps 8 MultiTrash classes to 3 sorting bins based on external YAML configuration.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import yaml
from core.types import Bin

DEFAULT_MAPPING_PATH = Path("config/bin_mapping.yaml")

# Fallback mapping in case YAML file is missing or unreadable
FALLBACK_CLASS_TO_BIN: Dict[int, Bin] = {
    0: Bin.ORGANIC,     # biowaste
    1: Bin.RECYCLABLE,  # glass
    2: Bin.RECYCLABLE,  # metal
    3: Bin.RESIDUAL,    # other
    4: Bin.RECYCLABLE,  # paper
    5: Bin.RECYCLABLE,  # plastic
    6: Bin.RESIDUAL,    # styrofoam
    7: Bin.RESIDUAL,    # wrapper
}


class BinMapper:
    """Manages mapping between dataset class IDs and destination sorting bins."""

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
            # Fall back to hardcoded safe defaults
            self._class_to_bin = dict(FALLBACK_CLASS_TO_BIN)

    def map_class_to_bin(self, class_id: int) -> Bin:
        """Map a numeric class_id (0-7) to its designated Bin category.

        Args:
            class_id: Model prediction class index.

        Returns:
            Bin enum value: ORGANIC, RECYCLABLE, RESIDUAL, or UNKNOWN if unmapped.
        """
        return self._class_to_bin.get(class_id, Bin.UNKNOWN)


# Global singleton instance for convenient import
_default_mapper = BinMapper()


def map_class_to_bin(class_id: int) -> Bin:
    """Functional convenience wrapper for class-to-bin mapping."""
    return _default_mapper.map_class_to_bin(class_id)
