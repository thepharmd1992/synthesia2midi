#!/usr/bin/env python3
"""
GUI entry point for Synthesia2MIDI.

This script configures startup logging, ensures the `synthesia2midi` package is
importable, creates the Qt application, and launches the main window.
"""
import os
import sys
import logging
import traceback

# Ensure the local package is importable (run.py lives next to the package directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Centralized logging (single folder + single file per run)
from synthesia2midi.core.logging_config import LoggingConfig

log_file = LoggingConfig.setup_logging(
    log_to_file=True,
    log_to_console=True,
    log_level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("Synthesia2MIDI Starting")
logger.info("=" * 80)
logger.info("Python version: %s", sys.version)
logger.info("Platform: %s", sys.platform)
logger.info("Current directory: %s", os.getcwd())
logger.info("Script path: %s", os.path.abspath(__file__))
logger.info("Log file: %s", log_file)

def _missing_dep_instructions(missing_module: str) -> str:
    setup_hint = (
        "Setup instructions:\n"
        "- Windows: run setup_windows.bat\n"
        "- macOS/Linux: run ./setup.sh\n"
    )
    if missing_module in {"PySide6"}:
        return (
            f"Missing required dependency: {missing_module}\n\n"
            f"{setup_hint}\n"
            "If you already ran setup, re-run it to reinstall dependencies."
        )
    return (
        f"Missing required dependency: {missing_module}\n\n"
        f"{setup_hint}\n"
        "If you already ran setup, re-run it to reinstall dependencies."
    )


def _show_missing_dependency(missing_module: str) -> None:
    message = _missing_dep_instructions(missing_module)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Missing Dependency", message)
        app.processEvents()
    except Exception:
        print(message)


try:
    # Add the current directory to the Python path so we can import synthesia2midi package
    logger.info(f"Adding to Python path: {current_dir}")

    logger.info("Importing Qt modules...")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QLoggingCategory, QCoreApplication
    from PySide6.QtGui import QFont
    logger.info("Qt modules imported successfully")
    
    # Log Qt version info
    logger.info(f"Qt version: {QCoreApplication.applicationVersion()}")
    
    logger.info("Importing Video2MidiApp...")
    from synthesia2midi.main import Video2MidiApp
    logger.info("Video2MidiApp imported successfully")
    
    
except ModuleNotFoundError as e:
    missing_module = getattr(e, "name", None) or "unknown"
    logger.error(f"Missing dependency: {missing_module}")
    logger.error(traceback.format_exc())
    _show_missing_dependency(missing_module)
    sys.exit(1)
except Exception as e:
    logger.error(f"Import error: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

if __name__ == "__main__":
    try:
        # High DPI scaling is handled automatically in PySide6/Qt6
        
        # Suppress Qt CSS parser warnings before creating QApplication
        os.environ['QT_LOGGING_RULES'] = 'qt.widgets.stylesheet.warning=false'
        
        logger.info("Creating QApplication...")
        qapp = QApplication(sys.argv)
        logger.info("QApplication created successfully")
        
        
        # Set default font for the entire application - larger for readability
        logger.info("Setting default font...")
        default_font = qapp.font()
        logger.info(f"Current font: {default_font.family()}, size: {default_font.pointSize()}")
        # Set absolute font size for better readability at 100% display scaling
        default_font.setPointSize(14)  # Absolute size instead of multiplier
        qapp.setFont(default_font)
        logger.info(f"Font size set to 14pt for improved readability")
        
        
        logger.info("Creating Video2MidiApp instance...")
        app = Video2MidiApp()
        logger.info("Video2MidiApp instance created successfully")
        
        logger.info("Showing application window...")
        app.show()
        logger.info("Application window shown")
        
        logger.info("Starting Qt event loop...")
        exit_code = qapp.exec()
        logger.info(f"Application exited with code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Runtime error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # On Windows, pause to keep the console window open
        if sys.platform == "win32":
            print("\nPress Enter to exit...")
            input()
        
        sys.exit(1)
    
    finally:
        # Always show where the log file is
        if sys.platform == "win32":
            print(f"\nLog file saved to: {log_file}")
            print("Press Enter to exit...")
            input()
