"""Centralized configuration loader for Trash-Net application.

Parses config/app.yaml, applies environment variable overrides,
and provides strongly typed configuration dataclasses decoupled from GUI frameworks.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

DEFAULT_CONFIG_PATH = Path("config/app.yaml")


@dataclass
class ModelConfig:
    """YOLO segmentation model configuration parameters."""
    weights_path: str = "models/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None
    half: Optional[bool] = None


@dataclass
class SmootherConfig:
    """Temporal smoothing and anti-flickering configuration."""
    enabled: bool = True
    window_size: int = 30
    stability_threshold_sec: float = 1.5
    disappear_timeout_sec: float = 1.0
    require_confirmation: bool = True


@dataclass
class CameraConfig:
    """Camera streaming and video capture configuration."""
    default_index: int = 0
    max_probe_index: int = 6
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class UIConfig:
    """Desktop user interface configuration."""
    window_title: str = "Trash-Net — Hệ Thống Phân Loại Rác Thông Minh"
    min_width: int = 1280
    min_height: int = 720
    theme: str = "dark"


@dataclass
class AppConfig:
    """Root application configuration aggregating all operational settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    config_path: Optional[Path] = None

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "AppConfig":
        """Load configuration from a YAML file with environment variable overrides.

        Args:
            config_path: Optional path to app.yaml. Defaults to config/app.yaml.

        Returns:
            AppConfig instance initialized with merged values.
        """
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        data: Dict[str, Any] = {}

        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    parsed = yaml.safe_load(f)
                    if isinstance(parsed, dict):
                        data = parsed
            except Exception:
                data = {}

        # Parse nested sections with safe fallbacks
        model_data = data.get("model", {})
        smoother_data = data.get("smoother", {})
        camera_data = data.get("camera", {})
        ui_data = data.get("ui", {})

        model_cfg = ModelConfig(
            weights_path=str(model_data.get("weights_path", "models/best.pt")),
            conf=float(model_data.get("conf", 0.25)),
            iou=float(model_data.get("iou", 0.45)),
            imgsz=int(model_data.get("imgsz", 640)),
            device=model_data.get("device", None),
            half=model_data.get("half", None),
        )

        smoother_cfg = SmootherConfig(
            enabled=bool(smoother_data.get("enabled", True)),
            window_size=int(smoother_data.get("window_size", 30)),
            stability_threshold_sec=float(smoother_data.get("stability_threshold_sec", 1.5)),
            disappear_timeout_sec=float(smoother_data.get("disappear_timeout_sec", 1.0)),
            require_confirmation=bool(smoother_data.get("require_confirmation", True)),
        )

        camera_cfg = CameraConfig(
            default_index=int(camera_data.get("default_index", 0)),
            max_probe_index=int(camera_data.get("max_probe_index", 6)),
            width=int(camera_data.get("width", 640)),
            height=int(camera_data.get("height", 480)),
            fps=int(camera_data.get("fps", 30)),
        )

        ui_cfg = UIConfig(
            window_title=str(ui_data.get("window_title", "Trash-Net — Hệ Thống Phân Loại Rác Thông Minh")),
            min_width=int(ui_data.get("min_width", 1280)),
            min_height=int(ui_data.get("min_height", 720)),
            theme=str(ui_data.get("theme", "dark")),
        )

        # Environment variable overrides
        if os.getenv("TRASHNET_MODEL_PATH"):
            model_cfg.weights_path = os.environ["TRASHNET_MODEL_PATH"]
        if os.getenv("TRASHNET_CONF"):
            try:
                model_cfg.conf = float(os.environ["TRASHNET_CONF"])
            except ValueError:
                pass
        if os.getenv("TRASHNET_IOU"):
            try:
                model_cfg.iou = float(os.environ["TRASHNET_IOU"])
            except ValueError:
                pass
        if os.getenv("TRASHNET_IMGSZ"):
            try:
                model_cfg.imgsz = int(os.environ["TRASHNET_IMGSZ"])
            except ValueError:
                pass
        if os.getenv("TRASHNET_DEVICE"):
            model_cfg.device = os.environ["TRASHNET_DEVICE"]
        if os.getenv("TRASHNET_CAMERA_INDEX"):
            try:
                camera_cfg.default_index = int(os.environ["TRASHNET_CAMERA_INDEX"])
            except ValueError:
                pass

        return cls(
            model=model_cfg,
            smoother=smoother_cfg,
            camera=camera_cfg,
            ui=ui_cfg,
            config_path=path if path.exists() else None,
        )
