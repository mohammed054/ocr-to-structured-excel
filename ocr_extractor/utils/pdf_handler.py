"""PDF conversion helpers backed by pdf2image."""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image

try:
    from .logger import RichLogger, get_logger
except ImportError:  # pragma: no cover - supports direct script execution
    from logger import RichLogger, get_logger


DEFAULT_DPI = 300
PDF_SUFFIX = ".pdf"
DEFAULT_THREAD_COUNT = 2
PDF_OUTPUT_FORMAT = "png"
ENCRYPTED_PDF_MARKERS = ("incorrect password", "encrypted", "requires a password")


class PDFConversionError(RuntimeError):
    """Raised when a PDF cannot be converted for a recoverable setup reason."""


def is_pdf_file(path: Path) -> bool:
    """Return True when the provided path points to a PDF file."""
    return path.suffix.lower() == PDF_SUFFIX


def convert_pdf_to_images(
    pdf_path: Path,
    dpi: int = DEFAULT_DPI,
    logger: RichLogger | None = None,
) -> list[Image.Image]:
    """Convert a PDF into high-resolution PIL images, one image per page."""
    active_logger = logger or get_logger()
    resolved_path = Path(pdf_path).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"PDF file not found: {resolved_path}")

    try:
        pages = convert_from_path(
            str(resolved_path),
            dpi=dpi,
            fmt=PDF_OUTPUT_FORMAT,
            thread_count=DEFAULT_THREAD_COUNT,
        )
    except PDFInfoNotInstalledError as exc:
        raise PDFConversionError(
            "Poppler is required for PDF conversion but was not found. "
            "Install Poppler and make sure its bin directory is on PATH."
        ) from exc
    except PDFPageCountError as exc:
        if _looks_encrypted(str(exc)):
            active_logger.warning(f"Encrypted PDF skipped: {resolved_path.name}")
            return []
        raise PDFConversionError(f"Could not read page count for PDF: {resolved_path}") from exc
    except PDFSyntaxError as exc:
        if _looks_encrypted(str(exc)):
            active_logger.warning(f"Encrypted PDF skipped: {resolved_path.name}")
            return []
        raise PDFConversionError(f"PDF syntax error while reading: {resolved_path}") from exc

    converted_pages = [page.convert("RGB") for page in pages]
    active_logger.info(f"Converted {len(converted_pages)} page(s) from {resolved_path.name} at {dpi} DPI")
    return converted_pages


def _looks_encrypted(message: str) -> bool:
    """Return True when a pdf2image error message indicates encryption."""
    normalized = message.lower()
    return any(marker in normalized for marker in ENCRYPTED_PDF_MARKERS)
