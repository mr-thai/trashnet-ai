"""Unit tests for waste class to bin mapping (US-2.2).

Verifies 100% coverage of the 8 MultiTrash classes, out-of-range class handling,
and external YAML configuration loading with fallback.
"""

from pathlib import Path
import tempfile
import pytest
from core.mapper import BinMapper, map_class_to_bin
from core.types import Bin


class TestBinMapper:
    """Test suite for BinMapper and map_class_to_bin."""

    def test_all_eight_classes_mapped_correctly(self):
        """AC: Verify exact mapping for all 8 MultiTrash v8 classes."""
        expected_mappings = {
            0: Bin.ORGANIC,     # biowaste -> Hữu cơ
            1: Bin.RECYCLABLE,  # glass -> Tái chế
            2: Bin.RECYCLABLE,  # metal -> Tái chế
            3: Bin.RESIDUAL,    # other -> Vô cơ / Khác
            4: Bin.RECYCLABLE,  # paper -> Tái chế
            5: Bin.RECYCLABLE,  # plastic -> Tái chế
            6: Bin.RESIDUAL,    # styrofoam -> Vô cơ / Khác
            7: Bin.RESIDUAL,    # wrapper -> Vô cơ / Khác
        }

        mapper = BinMapper()
        for class_id, expected_bin in expected_mappings.items():
            assert mapper.map_class_to_bin(class_id) == expected_bin, (
                f"Class {class_id} mapped to {mapper.map_class_to_bin(class_id)}, expected {expected_bin}"
            )

    def test_out_of_range_classes_return_unknown(self):
        """AC: When class_id is not in mapping, return UNKNOWN without raising errors."""
        mapper = BinMapper()
        for invalid_id in [-1, 8, 99, 1000]:
            assert mapper.map_class_to_bin(invalid_id) == Bin.UNKNOWN

    def test_functional_convenience_wrapper(self):
        """Test map_class_to_bin functional wrapper."""
        assert map_class_to_bin(0) == Bin.ORGANIC
        assert map_class_to_bin(5) == Bin.RECYCLABLE
        assert map_class_to_bin(99) == Bin.UNKNOWN

    def test_custom_yaml_configuration(self):
        """Test that BinMapper dynamically loads a custom YAML config file."""
        custom_yaml = """
bins:
  ORGANIC:
    classes: [10]
  RECYCLABLE:
    classes: [20, 21]
  RESIDUAL:
    classes: [30]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(custom_yaml)
            yaml_path = Path(f.name)

        try:
            custom_mapper = BinMapper(config_path=yaml_path)
            assert custom_mapper.map_class_to_bin(10) == Bin.ORGANIC
            assert custom_mapper.map_class_to_bin(20) == Bin.RECYCLABLE
            assert custom_mapper.map_class_to_bin(21) == Bin.RECYCLABLE
            assert custom_mapper.map_class_to_bin(30) == Bin.RESIDUAL
            assert custom_mapper.map_class_to_bin(0) == Bin.UNKNOWN  # Not in custom config
        finally:
            if yaml_path.exists():
                yaml_path.unlink()

    def test_missing_or_corrupt_yaml_falls_back_safely(self):
        """Test graceful fallback to built-in default mapping if YAML is missing/invalid."""
        corrupt_mapper = BinMapper(config_path="non_existent_file.yaml")
        # Should still safely return default mappings
        assert corrupt_mapper.map_class_to_bin(0) == Bin.ORGANIC
        assert corrupt_mapper.map_class_to_bin(5) == Bin.RECYCLABLE

    def test_bin_display_names(self):
        """Test user-friendly Vietnamese display labels of Bin enum."""
        assert Bin.RECYCLABLE.display_name == "Tái chế"
        assert Bin.ORGANIC.display_name == "Hữu cơ"
        assert Bin.RESIDUAL.display_name == "Vô cơ / Khác"
        assert Bin.UNKNOWN.display_name == "Không xác định"
