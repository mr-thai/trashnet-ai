"""Unit tests for centralized AppConfig and environment overrides."""

import os
from pathlib import Path
import tempfile
import pytest

from core.config import AppConfig, ModelConfig, SmootherConfig, CameraConfig, UIConfig


class TestAppConfig:
    """Test suite for AppConfig YAML loader and environment overrides."""

    def test_load_default_config(self):
        """Verify loading default config/app.yaml file."""
        cfg = AppConfig.load()
        assert isinstance(cfg, AppConfig)
        assert cfg.model.conf == 0.25
        assert cfg.model.iou == 0.45
        assert cfg.model.imgsz == 640
        assert cfg.smoother.enabled is True
        assert cfg.smoother.window_size == 30
        assert cfg.camera.default_index == 0
        assert cfg.camera.max_probe_index == 6
        assert "Trash-Net" in cfg.ui.window_title

    def test_load_custom_yaml(self):
        """Verify loading custom YAML config."""
        custom_yaml = """
model:
  weights_path: "custom/path.pt"
  conf: 0.70
  iou: 0.60
  imgsz: 1280
  device: "cpu"
smoother:
  enabled: false
  window_size: 15
camera:
  default_index: 2
ui:
  window_title: "Custom Title"
  min_width: 1920
  min_height: 1080
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(custom_yaml)
            path = Path(f.name)

        try:
            cfg = AppConfig.load(path)
            assert cfg.model.weights_path == "custom/path.pt"
            assert cfg.model.conf == 0.70
            assert cfg.model.iou == 0.60
            assert cfg.model.imgsz == 1280
            assert cfg.model.device == "cpu"
            assert cfg.smoother.enabled is False
            assert cfg.smoother.window_size == 15
            assert cfg.camera.default_index == 2
            assert cfg.ui.window_title == "Custom Title"
            assert cfg.ui.min_width == 1920
            assert cfg.ui.min_height == 1080
        finally:
            if path.exists():
                path.unlink()

    def test_environment_variable_overrides(self, monkeypatch):
        """Verify environment variables override YAML values."""
        monkeypatch.setenv("TRASHNET_MODEL_PATH", "env/weights.pt")
        monkeypatch.setenv("TRASHNET_CONF", "0.85")
        monkeypatch.setenv("TRASHNET_IOU", "0.30")
        monkeypatch.setenv("TRASHNET_DEVICE", "cuda:0")
        monkeypatch.setenv("TRASHNET_IMGSZ", "800")
        monkeypatch.setenv("TRASHNET_CAMERA_INDEX", "3")

        cfg = AppConfig.load()
        assert cfg.model.weights_path == "env/weights.pt"
        assert cfg.model.conf == 0.85
        assert cfg.model.iou == 0.30
        assert cfg.model.device == "cuda:0"
        assert cfg.model.imgsz == 800
        assert cfg.camera.default_index == 3

    def test_missing_yaml_falls_back_gracefully(self):
        """Verify missing YAML loads safe defaults without exception."""
        cfg = AppConfig.load("non_existent_app_config.yaml")
        assert cfg.model.conf == 0.25
        assert cfg.model.iou == 0.45
        assert cfg.model.imgsz == 640
        assert cfg.smoother.enabled is True
