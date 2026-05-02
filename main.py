"""Convenience launcher for the OCR extractor CLI."""

from __future__ import annotations

from ocr_extractor.main import cli


def main() -> None:
    """Run the Click command-line interface."""
    cli()


if __name__ == "__main__":
    main()
