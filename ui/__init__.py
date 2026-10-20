"""UI presentation layer for Trash-Net Desktop application.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import MainWindow

from ui.utils import FILE_DIALOG_IMAGE_FILTER, VALID_IMAGE_EXTENSIONS, load_image_unicode
from ui.worker import InferenceWorker


def __getattr__(name: str):
    if name == "MainWindow":
        from ui.main_window import MainWindow
        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MainWindow",
    "InferenceWorker",
    "load_image_unicode",
    "VALID_IMAGE_EXTENSIONS",
    "FILE_DIALOG_IMAGE_FILTER",
]
