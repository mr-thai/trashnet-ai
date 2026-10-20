"""Top-level executable module entry point for python -m trashnet (US-5.2)."""

import sys
# Pre-import torchvision to register C++ custom operators in PyInstaller runtime
try:
    import torchvision
    import torchvision.ops
except Exception:
    pass

from ui.main_window import run_app

if __name__ == "__main__":
    run_app()
