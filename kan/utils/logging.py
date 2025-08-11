from contextlib import contextmanager

from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.traceback import install

console = Console()


class _ConsoleWrapper:
    """Provide a file-like interface for a Rich :class:`Console`."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def write(self, message: str) -> None:  # pragma: no cover - simple passthrough
        """Forward loguru messages to :meth:`Console.print`.

        Loguru expects sinks to implement a ``write`` method. Rich's
        :class:`Console` does not provide this API directly, so we wrap it and
        trim empty messages (loguru sometimes sends ``"\n"``).
        """

        message = message.rstrip()
        if message:
            self._console.print(message, end="", markup=False)

    def flush(self) -> None:  # pragma: no cover - interface compliance
        """Compatibility stub for file-like API."""

        return None


install(console=console, show_locals=True)

logger.remove()
logger.add(
    _ConsoleWrapper(console),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)


def progress(iterable, description: str = "Processing"):
    """Wrap an iterable with a Rich progress bar."""
    return track(iterable, description=description)


@contextmanager
def status(message: str):
    """Display a status message using Rich's console."""
    with console.status(message) as status:
        yield status


__all__ = ["logger", "console", "progress", "status"]
