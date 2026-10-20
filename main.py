"""Main entry point for Trash-Net Desktop application."""

import sys
# Pre-import torchvision to register C++ custom operators (e.g. torchvision::nms) in PyInstaller runtime
try:
    import torchvision
    import torchvision.ops
except Exception:
    pass

from ui.main_window import run_app

if __name__ == "__main__":
    run_app()
