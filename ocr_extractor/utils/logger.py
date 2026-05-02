"""Rich-powered terminal logging and preview helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


APP_NAME = "PDF & Image -> Structured Excel Extractor"
BANNER_STYLE = "bold white on blue"
INFO_STYLE = "cyan"
SUCCESS_STYLE = "green"
WARNING_STYLE = "yellow"
ERROR_STYLE = "bold red"
TABLE_PREVIEW_LIMIT = 5
MAX_PREVIEW_CELL_LENGTH = 80


class RichLogger:
    """Small facade around Rich console primitives used across the project."""

    def __init__(self, console: Console | None = None) -> None:
        """Create a logger bound to a Rich console instance."""
        self.console = console or Console()

    def banner(self) -> None:
        """Print the application banner."""
        title = Text(APP_NAME, style="bold white")
        subtitle = Text("OCR, intelligent parsing, Excel, and JSON export", style="white")
        self.console.print(Panel.fit(f"{title}\n{subtitle}", style=BANNER_STYLE, border_style="blue"))

    def info(self, message: str) -> None:
        """Print an informational message."""
        self.console.print(f"[{INFO_STYLE}]INFO[/] {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[{SUCCESS_STYLE}]OK[/] {message}")

    def warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[{WARNING_STYLE}]WARN[/] {message}")

    def error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[{ERROR_STYLE}]ERROR[/] {message}")

    def rule(self, title: str) -> None:
        """Print a horizontal rule with a title."""
        self.console.rule(f"[bold blue]{title}[/]")

    def path(self, path_value: Path | str) -> str:
        """Return a Rich-safe path string."""
        return f"[bold]{Path(path_value)}[/]"

    def build_progress(self) -> Progress:
        """Create a reusable progress bar configured for file and page work."""
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    def preview_dataframe(self, dataframe: Any, title: str, max_rows: int = TABLE_PREVIEW_LIMIT) -> None:
        """Render the first rows of a pandas DataFrame in the terminal."""
        table = Table(title=title, show_lines=False, header_style="bold white on blue")

        if dataframe is None or getattr(dataframe, "empty", True):
            table.add_column("Note")
            table.add_row("No table rows detected")
            self.console.print(table)
            return

        for column in dataframe.columns:
            table.add_column(str(column), overflow="fold", max_width=MAX_PREVIEW_CELL_LENGTH)

        for _, row in dataframe.head(max_rows).iterrows():
            table.add_row(*[self._format_cell(value) for value in row.tolist()])

        self.console.print(table)

    def preview_key_values(self, key_values: dict[str, str], title: str = "Key-Value Preview") -> None:
        """Render detected key-value pairs in a compact Rich table."""
        table = Table(title=title, show_lines=False, header_style="bold white on blue")
        table.add_column("Key", style="bold")
        table.add_column("Value")

        if not key_values:
            table.add_row("No key-value pairs detected", "")
            self.console.print(table)
            return

        for key, value in key_values.items():
            table.add_row(self._truncate(key), self._truncate(value))

        self.console.print(table)

    def summary(self, rows: Iterable[tuple[str, Any]]) -> None:
        """Print a final run summary panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        for label, value in rows:
            table.add_row(str(label), str(value))
        self.console.print(Panel(table, title="Extraction Summary", border_style="green"))

    def _format_cell(self, value: Any) -> str:
        """Format a preview cell while keeping terminal output readable."""
        if value is None:
            return ""
        return self._truncate(str(value))

    def _truncate(self, value: str, max_length: int = MAX_PREVIEW_CELL_LENGTH) -> str:
        """Shorten long preview values without changing exported content."""
        normalized = " ".join(value.split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max_length - 3]}..."


def get_logger() -> RichLogger:
    """Return a new Rich logger instance."""
    return RichLogger()
