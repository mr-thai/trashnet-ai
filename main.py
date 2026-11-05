"""Main entry point for Trash-Net Desktop application."""

# Pre-import torchvision to register C++ custom operators (e.g. torchvision::nms) in PyInstaller runtime
try:
    import torchvision  # noqa: F401
    import torchvision.ops  # noqa: F401
except Exception:
    pass

from ui.main_window import run_app

if __name__ == "__main__":
    run_app()
