"""Click command-line interface for the OCR structured Excel extractor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from PIL import Image, ImageSequence, UnidentifiedImageError
from rich.progress import Progress

if __package__:
    from .core.exporter import BulkExportItem, DocumentExporter, ExportResult
    from .core.ocr_engine import OCREngineError, PaddleOCREngine
    from .core.parser import DocumentParser, ParsedDocument
    from .core.preprocessor import ImagePreprocessor
    from .utils.logger import RichLogger, get_logger
    from .utils.pdf_handler import PDFConversionError, convert_pdf_to_images, is_pdf_file
else:  # pragma: no cover - supports `python main.py` inside ocr_extractor
    from core.exporter import BulkExportItem, DocumentExporter, ExportResult
    from core.ocr_engine import OCREngineError, PaddleOCREngine
    from core.parser import DocumentParser, ParsedDocument
    from core.preprocessor import ImagePreprocessor
    from utils.logger import RichLogger, get_logger
    from utils.pdf_handler import PDFConversionError, convert_pdf_to_images, is_pdf_file


ParseMode = Literal["auto", "table", "keyvalue"]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_MODE: ParseMode = "auto"
IMAGE_LOAD_MODE = "RGB"
SUPPORTED_FORMATS_MESSAGE = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))


class UnsupportedFormatError(ValueError):
    """Raised when an input file is not a supported document type."""


@dataclass(frozen=True)
class ProcessingResult:
    """Parsed pages and export paths for one processed input file."""

    input_path: Path
    parsed_pages: list[ParsedDocument]
    export_result: ExportResult


@click.group()
def cli() -> None:
    """PDF and image OCR extraction commands."""


@cli.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path, exists=False, file_okay=True, dir_okay=True),
    help="Path to a PDF/image file, or a folder when --bulk is used.",
)
@click.option(
    "--output",
    "output_dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Directory where Excel and JSON files will be saved.",
)
@click.option(
    "--mode",
    default=DEFAULT_MODE,
    show_default=True,
    type=click.Choice(["auto", "table", "keyvalue"], case_sensitive=False),
    help="Extraction mode.",
)
@click.option("--bulk", is_flag=True, help="Process every supported file in the input folder.")
@click.option("--preview", is_flag=True, help="Print extracted tables and key-values before saving.")
@click.option("--no-preprocess", is_flag=True, help="Skip OpenCV preprocessing for clean scans.")
def extract(
    input_path: Path,
    output_dir: Path,
    mode: ParseMode,
    bulk: bool,
    preview: bool,
    no_preprocess: bool,
) -> None:
    """Extract structured tables and key-values from PDFs or images."""
    logger = get_logger()
    logger.banner()

    try:
        if bulk:
            process_bulk_input(
                input_path=input_path,
                output_dir=output_dir,
                mode=mode,
                preview=preview,
                no_preprocess=no_preprocess,
                logger=logger,
            )
        else:
            process_single_input(
                input_path=input_path,
                output_dir=output_dir,
                mode=mode,
                preview=preview,
                no_preprocess=no_preprocess,
                logger=logger,
            )
    except (FileNotFoundError, UnsupportedFormatError, PDFConversionError, OCREngineError, ValueError) as exc:
        logger.error(str(exc))
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - CLI boundary must surface all failures clearly
        logger.error(f"Unexpected extraction error: {exc}")
        raise click.ClickException(f"Unexpected extraction error: {exc}") from exc


def process_single_input(
    input_path: Path,
    output_dir: Path,
    mode: ParseMode,
    preview: bool,
    no_preprocess: bool,
    logger: RichLogger,
) -> ProcessingResult:
    """Process one PDF or image file and save individual outputs."""
    resolved_input = validate_single_input(input_path)
    engine_cache: dict[str, PaddleOCREngine] = {}

    result = process_file(
        input_path=resolved_input,
        output_dir=output_dir,
        mode=mode,
        preview=preview,
        no_preprocess=no_preprocess,
        logger=logger,
        engine_cache=engine_cache,
        progress=None,
    )

    logger.summary(
        [
            ("Input", result.input_path.name),
            ("Pages", result.export_result.page_count),
            ("Tables Found", result.export_result.table_count),
            ("Key-Value Pairs", result.export_result.key_value_count),
            ("Excel", result.export_result.excel_path),
            ("JSON", result.export_result.json_path),
        ]
    )
    return result


def process_bulk_input(
    input_path: Path,
    output_dir: Path,
    mode: ParseMode,
    preview: bool,
    no_preprocess: bool,
    logger: RichLogger,
) -> list[ProcessingResult]:
    """Process every supported file in a folder and create a master workbook."""
    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input folder not found: {resolved_input}")
    if not resolved_input.is_dir():
        raise ValueError("--bulk requires --input to be a folder path")

    files = collect_bulk_files(resolved_input)
    if not files:
        raise FileNotFoundError(f"No supported files found in {resolved_input}. Supported: {SUPPORTED_FORMATS_MESSAGE}")

    logger.info(f"Bulk mode found {len(files)} supported file(s)")
    engine_cache: dict[str, PaddleOCREngine] = {}
    results: list[ProcessingResult] = []

    with logger.build_progress() as progress:
        file_task = progress.add_task("Files", total=len(files))
        for index, file_path in enumerate(files, start=1):
            progress.update(file_task, description=f"File {index}/{len(files)}: {file_path.name}")
            try:
                result = process_file(
                    input_path=file_path,
                    output_dir=output_dir,
                    mode=mode,
                    preview=preview,
                    no_preprocess=no_preprocess,
                    logger=logger,
                    engine_cache=engine_cache,
                    progress=progress,
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - bulk mode should continue after per-file failures
                logger.error(f"Failed to process {file_path.name}: {exc}")
            finally:
                progress.update(file_task, advance=1)

    if not results:
        raise RuntimeError("Bulk mode finished, but no files were processed successfully")

    exporter = DocumentExporter()
    master_items = [
        BulkExportItem(
            input_path=result.input_path,
            parsed_pages=result.parsed_pages,
            export_result=result.export_result,
        )
        for result in results
    ]
    master_path = exporter.export_bulk_master(master_items, output_dir=output_dir)

    logger.summary(
        [
            ("Files Processed", len(results)),
            ("Total Tables", sum(result.export_result.table_count for result in results)),
            ("Total Key-Value Pairs", sum(result.export_result.key_value_count for result in results)),
            ("Master Excel", master_path),
            ("Output Directory", Path(output_dir).expanduser().resolve()),
        ]
    )
    return results


def process_file(
    input_path: Path,
    output_dir: Path,
    mode: ParseMode,
    preview: bool,
    no_preprocess: bool,
    logger: RichLogger,
    engine_cache: dict[str, PaddleOCREngine],
    progress: Progress | None,
) -> ProcessingResult:
    """Process one supported file from image loading through export."""
    resolved_input = validate_single_input(input_path)
    logger.rule(f"Processing {resolved_input.name}")

    images = load_input_images(resolved_input, logger=logger)
    parsed_pages = parse_images(
        images=images,
        file_name=resolved_input.name,
        mode=mode,
        preview=preview,
        no_preprocess=no_preprocess,
        logger=logger,
        engine_cache=engine_cache,
        progress=progress,
    )

    exporter = DocumentExporter()
    export_result = exporter.export_document(
        input_path=resolved_input,
        parsed_pages=parsed_pages,
        output_dir=output_dir,
    )

    logger.success(f"Saved Excel: {export_result.excel_path}")
    logger.success(f"Saved JSON: {export_result.json_path}")

    return ProcessingResult(
        input_path=resolved_input,
        parsed_pages=parsed_pages,
        export_result=export_result,
    )


def validate_single_input(input_path: Path) -> Path:
    """Validate one input file path and return its resolved path."""
    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_input}")
    if resolved_input.is_dir():
        raise ValueError("Folder input requires --bulk")
    if not is_supported_file(resolved_input):
        raise UnsupportedFormatError(
            f"Unsupported file format: {resolved_input.suffix or '[none]'}. "
            f"Supported formats: {SUPPORTED_FORMATS_MESSAGE}"
        )
    return resolved_input


def collect_bulk_files(folder_path: Path) -> list[Path]:
    """Collect supported PDF and image files from a folder."""
    return sorted(
        [path for path in folder_path.iterdir() if path.is_file() and is_supported_file(path)],
        key=lambda path: path.name.lower(),
    )


def is_supported_file(path: Path) -> bool:
    """Return True when a file extension is supported."""
    return path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS


def load_input_images(input_path: Path, logger: RichLogger) -> list[Image.Image]:
    """Load a PDF or image input as a list of PIL images."""
    if is_pdf_file(input_path):
        return convert_pdf_to_images(input_path, logger=logger)
    return load_image_file(input_path)


def load_image_file(image_path: Path) -> list[Image.Image]:
    """Load a raster image, including multi-frame TIFF images."""
    try:
        with Image.open(image_path) as image:
            frames = [frame.convert(IMAGE_LOAD_MODE).copy() for frame in ImageSequence.Iterator(image)]
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as exc:
        raise UnsupportedFormatError(f"Could not identify image file: {image_path}") from exc

    return frames


def parse_images(
    images: list[Image.Image],
    file_name: str,
    mode: ParseMode,
    preview: bool,
    no_preprocess: bool,
    logger: RichLogger,
    engine_cache: dict[str, PaddleOCREngine],
    progress: Progress | None,
) -> list[ParsedDocument]:
    """Run preprocessing, OCR, parsing, and optional previews for all pages."""
    if not images:
        logger.warning(f"No pages/images available for OCR in {file_name}; exporting an empty workbook")
        return []

    if progress is None:
        with logger.build_progress() as local_progress:
            return parse_images_with_progress(
                images=images,
                file_name=file_name,
                mode=mode,
                preview=preview,
                no_preprocess=no_preprocess,
                logger=logger,
                engine_cache=engine_cache,
                progress=local_progress,
                remove_task_when_done=False,
            )

    return parse_images_with_progress(
        images=images,
        file_name=file_name,
        mode=mode,
        preview=preview,
        no_preprocess=no_preprocess,
        logger=logger,
        engine_cache=engine_cache,
        progress=progress,
        remove_task_when_done=True,
    )


def parse_images_with_progress(
    images: list[Image.Image],
    file_name: str,
    mode: ParseMode,
    preview: bool,
    no_preprocess: bool,
    logger: RichLogger,
    engine_cache: dict[str, PaddleOCREngine],
    progress: Progress,
    remove_task_when_done: bool,
) -> list[ParsedDocument]:
    """Parse pages while updating a Rich page progress task."""
    preprocessor = ImagePreprocessor()
    parser = DocumentParser()
    engine = get_or_create_engine(engine_cache, logger)
    parsed_pages: list[ParsedDocument] = []
    page_task = progress.add_task(f"Pages: {file_name}", total=len(images))

    for page_index, image in enumerate(images, start=1):
        progress.update(page_task, description=f"Page {page_index}/{len(images)}: {file_name}")
        ocr_image = image if no_preprocess else preprocessor.preprocess(image)
        ocr_items = engine.extract(ocr_image)

        if not ocr_items:
            logger.warning(f"No OCR text detected on {file_name} page {page_index}")

        parsed_page = parser.parse(ocr_items, page_number=page_index, mode=mode)
        parsed_pages.append(parsed_page)

        if preview:
            preview_parsed_page(parsed_page, logger)

        progress.update(page_task, advance=1)

    if remove_task_when_done:
        progress.remove_task(page_task)

    return parsed_pages


def get_or_create_engine(engine_cache: dict[str, PaddleOCREngine], logger: RichLogger) -> PaddleOCREngine:
    """Return a cached PaddleOCR engine, creating it on first use."""
    if "engine" not in engine_cache:
        logger.info("Initializing PaddleOCR engine")
        engine_cache["engine"] = PaddleOCREngine(logger=logger)
    return engine_cache["engine"]


def preview_parsed_page(parsed_page: ParsedDocument, logger: RichLogger) -> None:
    """Print terminal previews for tables and key-values on one parsed page."""
    if not parsed_page.tables and not parsed_page.key_values:
        logger.warning(f"Page {parsed_page.page_number}: no structured data detected")
        return

    for table_index, dataframe in enumerate(parsed_page.tables, start=1):
        logger.preview_dataframe(
            dataframe=dataframe,
            title=f"Page {parsed_page.page_number} Table {table_index}",
        )

    if parsed_page.key_values:
        logger.preview_key_values(
            parsed_page.key_values,
            title=f"Page {parsed_page.page_number} Key-Values",
        )


if __name__ == "__main__":
    cli()
