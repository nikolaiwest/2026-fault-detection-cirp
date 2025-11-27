"""
Centralized logging configuration with colorized console output.

This module provides a factory function to create consistently configured loggers
across all modules. It supports:
- Colorized console output (INFO and above)
- Detailed file logging (DEBUG and above)
- Hierarchical logger naming based on module structure
- Custom section() and subsection() methods for visual separation

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.section("DATA PROCESSING")
    logger.info("Processing started")
    logger.debug("Detailed information for troubleshooting")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False
    print("Warning: colorlog not installed. Install with: pip install colorlog")


# Global configuration
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "logger.log"
CONSOLE_LEVEL = logging.INFO
FILE_LEVEL = logging.DEBUG

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Track if root logger has been configured
_ROOT_CONFIGURED = False


class CustomLogger(logging.Logger):
    """
    Extended Logger class with additional formatting methods.

    Adds:
        - section(): For major section headers with visual separation
        - subsection(): For sub-section headers with lighter separation
    """

    # Fixed width for all separators to ensure visual consistency
    SEPARATOR_WIDTH = 80

    def section(self, msg: str) -> None:
        """
        Log a major section header with visual separation.

        Creates a visually distinct block for major sections.
        Uses fixed-width separators for consistent alignment.

        Args:
            msg: Section title

        Example:
            >>> logger.section("INITIALIZATION")

            ================================================================================
                                        INITIALIZATION
            ================================================================================
        """
        separator = "=" * self.SEPARATOR_WIDTH

        self.info("")
        self.info(separator)
        self.info(msg.center(self.SEPARATOR_WIDTH))
        self.info(separator)

    def subsection(self, msg: str) -> None:
        """
        Log a sub-section header with lighter visual separation.

        For sections within major sections.

        Args:
            msg: Subsection title

        Example:
            >>> logger.subsection("Loading Configuration")

            --------------------------------------------------------------------------------
            Loading Configuration
            --------------------------------------------------------------------------------
        """
        separator = "-" * self.SEPARATOR_WIDTH

        self.info("")
        self.info(separator)
        self.info(msg)
        self.info(separator)


# Set custom logger class globally
logging.setLoggerClass(CustomLogger)


def get_logger(
    name: str, console_level: Optional[int] = None, file_level: Optional[int] = None
) -> CustomLogger:
    """
    Get or create a logger with consistent configuration.

    This function returns a logger with the specified name, automatically configuring
    the root logger on first call to ensure all loggers share the same handlers.

    Args:
        name: Logger name (typically __name__ from calling module)
        console_level: Override default console log level (default: INFO)
        file_level: Override default file log level (default: DEBUG)

    Returns:
        Configured CustomLogger instance with section() and subsection() methods

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.section("MAIN PROCESS")
        >>> logger.info("Starting analysis")
        >>> logger.debug("Detailed parameter values: x=42, y=13.7")
    """
    global _ROOT_CONFIGURED

    # Use provided levels or defaults
    console_lvl = console_level if console_level is not None else CONSOLE_LEVEL
    file_lvl = file_level if file_level is not None else FILE_LEVEL

    # Configure root logger on first call
    if not _ROOT_CONFIGURED:
        _configure_root_logger(console_lvl, file_lvl)
        _ROOT_CONFIGURED = True

    # Return logger for this module (will be CustomLogger due to setLoggerClass)
    return logging.getLogger(name)


def _configure_root_logger(console_level: int, file_level: int) -> None:
    """
    Configure the root logger with console and file handlers.

    This is called automatically by get_logger() on first use.

    Args:
        console_level: Minimum level for console output
        file_level: Minimum level for file output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

    # Remove any existing handlers (prevents duplicates)
    root_logger.handlers.clear()

    # === Silence noisy third-party libraries ===
    # These libraries spam DEBUG logs, set them to WARNING
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # === Console Handler (colorized) ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    if HAS_COLORLOG:
        # Colorized format with 3-char level names
        console_formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname).3s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            reset=True,
            style="%",
        )
    else:
        # Fallback to plain format if colorlog not available
        console_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname).3s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # === File Handler (detailed, plain text) ===
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)

    # More detailed format for file (includes module, function, line number)
    # Keeps full level names for better searchability
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log initialization message
    root_logger.info("=" * 80)
    root_logger.info("Logging initialized")
    root_logger.info(f"Console level: {logging.getLevelName(console_level)}")
    root_logger.info(f"File level: {logging.getLevelName(file_level)}")
    root_logger.info(f"Log file: {LOG_FILE.absolute()}")
    root_logger.info("=" * 80)


def set_level(level: int) -> None:
    """
    Change the logging level for all loggers dynamically.

    Useful for temporarily enabling debug logging without restarting.

    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> from src.utils.logger import set_level
        >>> import logging
        >>> set_level(logging.DEBUG)  # Enable verbose logging
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(level)
    root_logger.info(f"Logging level changed to: {logging.getLevelName(level)}")


def get_log_file_path() -> Path:
    """
    Get the path to the current log file.

    Returns:
        Path object pointing to the log file

    Example:
        >>> from src.utils.logger import get_log_file_path
        >>> print(f"Logs saved to: {get_log_file_path()}")
    """
    return LOG_FILE.absolute()


# Example usage (only runs when module is executed directly)
if __name__ == "__main__":
    # Demonstrate logging functionality
    logger = get_logger(__name__)

    # Test section headers
    logger.section("MAIN PROCESS")

    logger.info("This is an INFO message (console + file)")
    logger.debug("This is a DEBUG message (only in file)")
    logger.warning("This is a WARNING message")

    logger.subsection("Data Loading")
    logger.info("Processing component A")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    # Demonstrate hierarchical logging
    sub_logger = get_logger("myapp.data.loader")
    sub_logger.section("CONFIGURATION")
    sub_logger.info("Message from a submodule")

    # Show log file location
    print(f"\nLog file saved to: {get_log_file_path()}")
