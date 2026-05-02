"""Intelligent parsing for OCR text blocks, tables, and key-value pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Literal, Mapping

import pandas as pd


ParseMode = Literal["auto", "table", "keyvalue"]

SUPPORTED_PARSE_MODES: set[str] = {"auto", "table", "keyvalue"}
MIN_TABLE_ROWS = 2
MIN_TABLE_COLUMNS = 2
MIN_AVERAGE_CELLS_PER_TABLE_ROW = 1.6
MIN_ROW_Y_TOLERANCE = 10.0
ROW_Y_TOLERANCE_RATIO = 0.65
MIN_COLUMN_X_TOLERANCE = 34.0
COLUMN_X_TOLERANCE_RATIO = 0.75
TABLE_VERTICAL_GAP_RATIO = 2.8
MIN_TABLE_VERTICAL_GAP = 30.0
KEY_VALUE_MIN_GAP = 12.0
MAX_KEY_LENGTH = 80
MAX_VALUE_LENGTH = 250
EMPTY_COLUMN_PREFIX = "Column"
TABLE_SCORE_WEIGHT = 1.0
KEY_VALUE_SCORE_WEIGHT = 2.0

INLINE_KEY_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?P<key>[A-Za-z][\w\s/&().#-]{1,80}?)\s*[:=]\s*(?P<value>.+?)\s*$"),
    re.compile(r"^\s*(?P<key>[A-Za-z][\w\s/&().#-]{1,80}?)\s*\.{2,}\s*(?P<value>.+?)\s*$"),
    re.compile(
        r"^\s*(?P<key>[A-Za-z][\w\s/&().#-]{1,80}?)\s{2,}"
        r"(?P<value>(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[$]?\s?\d[\d,]*(?:\.\d{2})?|"
        r"[A-Z0-9][\w\s,./#-]{1,120}))\s*$"
    ),
)
VALUE_LIKE_PATTERN = re.compile(
    r"^\s*(?:[$]?\s?\d[\d,]*(?:\.\d+)?|\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Z]{0,4}\d{2,})\s*$",
    re.IGNORECASE,
)
KEY_CLEANUP_PATTERN = re.compile(r"[:=\.\s_]+$")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TextBox:
    """OCR text with derived geometric properties."""

    text: str
    bbox: list[list[float]]
    confidence: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ParsedDocument:
    """Structured content extracted from one page of an input document."""

    tables: list[pd.DataFrame]
    key_values: dict[str, str]
    raw_text: str
    page_number: int
    mode: str = "auto"
    chosen_mode: str = "empty"
    structure_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the parsed page."""
        return {
            "page_number": self.page_number,
            "mode": self.mode,
            "chosen_mode": self.chosen_mode,
            "structure_score": self.structure_score,
            "raw_text": self.raw_text,
            "key_values": self.key_values,
            "tables": [_dataframe_to_dict(table) for table in self.tables],
        }


class DocumentParser:
    """Parse OCR output into tables, key-value pairs, and raw text."""

    def parse(
        self,
        ocr_items: Iterable[Mapping[str, Any]],
        page_number: int,
        mode: ParseMode = "auto",
    ) -> ParsedDocument:
        """Parse one page of OCR items according to the requested mode."""
        if mode not in SUPPORTED_PARSE_MODES:
            raise ValueError(f"Unsupported parse mode '{mode}'. Choose one of: auto, table, keyvalue.")

        boxes = self._normalise_boxes(ocr_items)
        raw_text = self._build_raw_text(boxes)

        detected_tables = self.detect_tables(boxes) if mode in {"auto", "table"} else []
        detected_key_values = self.detect_key_values(boxes) if mode in {"auto", "keyvalue"} else {}

        table_score = self._score_tables(detected_tables)
        key_value_score = self._score_key_values(detected_key_values)
        chosen_mode = self._choose_mode(mode=mode, table_score=table_score, key_value_score=key_value_score)

        return ParsedDocument(
            tables=detected_tables,
            key_values=detected_key_values,
            raw_text=raw_text,
            page_number=page_number,
            mode=mode,
            chosen_mode=chosen_mode,
            structure_score=max(table_score, key_value_score),
        )

    def detect_tables(self, boxes: list[TextBox]) -> list[pd.DataFrame]:
        """Detect tables by grouping OCR boxes into rows and inferred columns."""
        if not boxes:
            return []

        row_groups = self._group_boxes_into_rows(boxes)
        table_blocks = self._split_rows_into_table_blocks(row_groups)
        tables: list[pd.DataFrame] = []

        for block in table_blocks:
            if not self._is_table_like(block):
                continue

            matrix = self._rows_to_matrix(block)
            dataframe = self._matrix_to_dataframe(matrix)
            if not dataframe.empty:
                tables.append(dataframe)

        return tables

    def detect_key_values(self, boxes: list[TextBox]) -> dict[str, str]:
        """Detect key-value pairs using regex and positional row heuristics."""
        key_values: dict[str, str] = {}
        row_groups = self._group_boxes_into_rows(boxes)

        for line in self._rows_to_text_lines(row_groups):
            for key, value in self._extract_inline_key_values(line):
                self._add_key_value(key_values, key, value)

        for row in row_groups:
            positional_pair = self._extract_positional_key_value(row)
            if positional_pair is not None:
                key, value = positional_pair
                self._add_key_value(key_values, key, value)

        return key_values

    def _normalise_boxes(self, ocr_items: Iterable[Mapping[str, Any]]) -> list[TextBox]:
        """Convert OCR dictionaries into TextBox records sorted by page position."""
        boxes: list[TextBox] = []

        for item in ocr_items:
            text = str(item.get("text", "")).strip()
            bbox = item.get("bbox", [])
            confidence = self._safe_float(item.get("confidence", 0.0))

            if not text or not bbox:
                continue

            try:
                points = [(float(point[0]), float(point[1])) for point in bbox]
            except (TypeError, ValueError, IndexError):
                continue

            if len(points) < 4:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width = max(1.0, x_max - x_min)
            height = max(1.0, y_max - y_min)

            boxes.append(
                TextBox(
                    text=text,
                    bbox=[[x, y] for x, y in points[:4]],
                    confidence=confidence,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    x_center=x_min + width / 2.0,
                    y_center=y_min + height / 2.0,
                    width=width,
                    height=height,
                )
            )

        return sorted(boxes, key=lambda box: (box.y_center, box.x_min))

    def _build_raw_text(self, boxes: list[TextBox]) -> str:
        """Build plain text from OCR boxes while preserving line order."""
        rows = self._group_boxes_into_rows(boxes)
        return "\n".join(self._rows_to_text_lines(rows))

    def _group_boxes_into_rows(self, boxes: list[TextBox]) -> list[list[TextBox]]:
        """Group OCR boxes into text rows using Y-axis proximity."""
        if not boxes:
            return []

        tolerance = self._row_tolerance(boxes)
        rows: list[list[TextBox]] = []
        row_centers: list[float] = []

        for box in sorted(boxes, key=lambda item: (item.y_center, item.x_min)):
            row_index = self._nearest_row_index(row_centers, box.y_center, tolerance)
            if row_index is None:
                rows.append([box])
                row_centers.append(box.y_center)
            else:
                rows[row_index].append(box)
                row_centers[row_index] = self._average([item.y_center for item in rows[row_index]])

        return [sorted(row, key=lambda item: item.x_min) for row in rows]

    def _row_tolerance(self, boxes: list[TextBox]) -> float:
        """Calculate a row grouping tolerance from OCR box heights."""
        heights = [box.height for box in boxes if box.height > 0]
        if not heights:
            return MIN_ROW_Y_TOLERANCE
        return max(MIN_ROW_Y_TOLERANCE, median(heights) * ROW_Y_TOLERANCE_RATIO)

    def _nearest_row_index(self, row_centers: list[float], y_center: float, tolerance: float) -> int | None:
        """Find the nearest row center within the configured Y tolerance."""
        if not row_centers:
            return None

        distances = [abs(center - y_center) for center in row_centers]
        nearest_index = min(range(len(distances)), key=distances.__getitem__)
        return nearest_index if distances[nearest_index] <= tolerance else None

    def _split_rows_into_table_blocks(self, rows: list[list[TextBox]]) -> list[list[list[TextBox]]]:
        """Split row groups into candidate table blocks based on vertical gaps."""
        if not rows:
            return []

        sorted_rows = sorted(rows, key=self._row_center)
        threshold = self._vertical_gap_threshold(sorted_rows)
        blocks: list[list[list[TextBox]]] = []
        current_block: list[list[TextBox]] = [sorted_rows[0]]

        for previous_row, current_row in zip(sorted_rows, sorted_rows[1:]):
            gap = self._row_center(current_row) - self._row_center(previous_row)
            if gap > threshold:
                blocks.append(current_block)
                current_block = [current_row]
            else:
                current_block.append(current_row)

        blocks.append(current_block)
        return blocks

    def _vertical_gap_threshold(self, rows: list[list[TextBox]]) -> float:
        """Estimate when a vertical row gap should split table candidates."""
        heights = [self._row_height(row) for row in rows if row]
        if not heights:
            return MIN_TABLE_VERTICAL_GAP
        return max(MIN_TABLE_VERTICAL_GAP, median(heights) * TABLE_VERTICAL_GAP_RATIO)

    def _is_table_like(self, rows: list[list[TextBox]]) -> bool:
        """Return True when grouped rows have enough repeated column structure."""
        if len(rows) < MIN_TABLE_ROWS:
            return False

        cell_counts = [len(row) for row in rows]
        average_cells = sum(cell_counts) / len(cell_counts)
        max_cells = max(cell_counts)

        return average_cells >= MIN_AVERAGE_CELLS_PER_TABLE_ROW and max_cells >= MIN_TABLE_COLUMNS

    def _rows_to_matrix(self, rows: list[list[TextBox]]) -> list[list[str]]:
        """Convert table rows into a rectangular matrix of cell text."""
        columns = self._cluster_columns([box for row in rows for box in row])
        if len(columns) < MIN_TABLE_COLUMNS:
            return []

        matrix: list[list[str]] = []
        for row in rows:
            cells = ["" for _ in columns]
            for box in row:
                column_index = self._nearest_column_index(columns, box.x_center)
                existing = cells[column_index]
                cells[column_index] = f"{existing} {box.text}".strip() if existing else box.text
            matrix.append(cells)

        return self._trim_sparse_columns(matrix)

    def _cluster_columns(self, boxes: list[TextBox]) -> list[float]:
        """Cluster text boxes into inferred column centers using X coordinates."""
        if not boxes:
            return []

        tolerance = self._column_tolerance(boxes)
        clusters: list[list[float]] = []

        for box in sorted(boxes, key=lambda item: item.x_center):
            if not clusters or abs(self._average(clusters[-1]) - box.x_center) > tolerance:
                clusters.append([box.x_center])
            else:
                clusters[-1].append(box.x_center)

        return [self._average(cluster) for cluster in clusters]

    def _column_tolerance(self, boxes: list[TextBox]) -> float:
        """Calculate a column clustering tolerance from OCR box widths."""
        widths = [box.width for box in boxes if box.width > 0]
        if not widths:
            return MIN_COLUMN_X_TOLERANCE
        return max(MIN_COLUMN_X_TOLERANCE, median(widths) * COLUMN_X_TOLERANCE_RATIO)

    def _nearest_column_index(self, columns: list[float], x_center: float) -> int:
        """Return the closest inferred column index for an OCR box."""
        return min(range(len(columns)), key=lambda index: abs(columns[index] - x_center))

    def _trim_sparse_columns(self, matrix: list[list[str]]) -> list[list[str]]:
        """Remove fully empty inferred columns from a matrix."""
        if not matrix:
            return []

        column_count = max(len(row) for row in matrix)
        padded = [row + [""] * (column_count - len(row)) for row in matrix]
        keep_indexes = [
            index
            for index in range(column_count)
            if any(str(row[index]).strip() for row in padded)
        ]

        return [[row[index] for index in keep_indexes] for row in padded]

    def _matrix_to_dataframe(self, matrix: list[list[str]]) -> pd.DataFrame:
        """Convert a table matrix into a clean pandas DataFrame."""
        if not matrix or len(matrix) < MIN_TABLE_ROWS:
            return pd.DataFrame()

        column_count = max(len(row) for row in matrix)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in matrix]

        if self._first_row_looks_like_header(normalized_rows):
            columns = self._make_unique_columns(normalized_rows[0])
            data_rows = normalized_rows[1:]
        else:
            columns = [f"{EMPTY_COLUMN_PREFIX}_{index + 1}" for index in range(column_count)]
            data_rows = normalized_rows

        dataframe = pd.DataFrame(data_rows, columns=columns)
        return dataframe.dropna(how="all").replace("", pd.NA).dropna(how="all").fillna("")

    def _first_row_looks_like_header(self, matrix: list[list[str]]) -> bool:
        """Return True when the first table row appears to be a header."""
        first_row = [cell.strip() for cell in matrix[0]]
        non_empty = [cell for cell in first_row if cell]
        if len(non_empty) < MIN_TABLE_COLUMNS:
            return False

        alpha_count = sum(any(char.isalpha() for char in cell) for cell in non_empty)
        unique_count = len({cell.lower() for cell in non_empty})
        return alpha_count >= max(1, len(non_empty) // 2) and unique_count == len(non_empty)

    def _make_unique_columns(self, values: list[str]) -> list[str]:
        """Create unique DataFrame column names from OCR header cells."""
        columns: list[str] = []
        seen: dict[str, int] = {}

        for index, value in enumerate(values):
            base = self._normalize_spaces(value) or f"{EMPTY_COLUMN_PREFIX}_{index + 1}"
            count = seen.get(base, 0) + 1
            seen[base] = count
            columns.append(base if count == 1 else f"{base}_{count}")

        return columns

    def _rows_to_text_lines(self, rows: list[list[TextBox]]) -> list[str]:
        """Convert grouped rows into human-readable text lines."""
        lines: list[str] = []
        for row in rows:
            line = " ".join(box.text for box in sorted(row, key=lambda item: item.x_min)).strip()
            if line:
                lines.append(self._normalize_spaces(line))
        return lines

    def _extract_inline_key_values(self, line: str) -> list[tuple[str, str]]:
        """Extract key-value pairs that appear in one OCR text line."""
        pairs: list[tuple[str, str]] = []
        for pattern in INLINE_KEY_VALUE_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            key = self._clean_key(match.group("key"))
            value = self._clean_value(match.group("value"))
            if self._valid_key_value(key, value):
                pairs.append((key, value))
        return pairs

    def _extract_positional_key_value(self, row: list[TextBox]) -> tuple[str, str] | None:
        """Extract key-value pairs from neighboring OCR boxes on the same row."""
        if len(row) < 2:
            return None

        sorted_row = sorted(row, key=lambda item: item.x_min)
        key_candidate = sorted_row[0].text
        value_candidate = " ".join(item.text for item in sorted_row[1:])
        horizontal_gap = sorted_row[1].x_min - sorted_row[0].x_max

        key = self._clean_key(key_candidate)
        value = self._clean_value(value_candidate)

        if horizontal_gap < KEY_VALUE_MIN_GAP and not key_candidate.strip().endswith((':', '=')):
            return None
        if not self._looks_like_label(key):
            return None
        if not self._valid_key_value(key, value):
            return None
        return key, value

    def _add_key_value(self, key_values: dict[str, str], key: str, value: str) -> None:
        """Add a key-value pair while preserving duplicate keys with suffixes."""
        clean_key = self._clean_key(key)
        clean_value = self._clean_value(value)
        if not self._valid_key_value(clean_key, clean_value):
            return

        if clean_key not in key_values:
            key_values[clean_key] = clean_value
            return

        suffix = 2
        while f"{clean_key} ({suffix})" in key_values:
            suffix += 1
        key_values[f"{clean_key} ({suffix})"] = clean_value

    def _clean_key(self, value: str) -> str:
        """Normalize a key label extracted from OCR text."""
        normalized = self._normalize_spaces(value)
        return KEY_CLEANUP_PATTERN.sub("", normalized).strip()

    def _clean_value(self, value: str) -> str:
        """Normalize a value extracted from OCR text."""
        return self._normalize_spaces(value).strip(" .:_")

    def _valid_key_value(self, key: str, value: str) -> bool:
        """Return True when a candidate key-value pair is usable."""
        if not key or not value:
            return False
        if len(key) > MAX_KEY_LENGTH or len(value) > MAX_VALUE_LENGTH:
            return False
        if key.lower() == value.lower():
            return False
        return any(char.isalpha() for char in key)

    def _looks_like_label(self, value: str) -> bool:
        """Return True when text resembles a form or invoice label."""
        normalized = value.strip()
        if not normalized or len(normalized) > MAX_KEY_LENGTH:
            return False
        if VALUE_LIKE_PATTERN.match(normalized):
            return False
        return any(char.isalpha() for char in normalized)

    def _score_tables(self, tables: list[pd.DataFrame]) -> float:
        """Score table quality from dimensions and filled-cell density."""
        score = 0.0
        for dataframe in tables:
            if dataframe.empty:
                continue
            total_cells = max(1, dataframe.shape[0] * dataframe.shape[1])
            filled_cells = int(dataframe.astype(str).map(lambda value: bool(value.strip())).sum().sum())
            density = filled_cells / total_cells
            score += dataframe.shape[0] * dataframe.shape[1] * density * TABLE_SCORE_WEIGHT
        return score

    def _score_key_values(self, key_values: dict[str, str]) -> float:
        """Score key-value quality from the number of detected pairs."""
        return float(len(key_values) * KEY_VALUE_SCORE_WEIGHT)

    def _choose_mode(self, mode: str, table_score: float, key_value_score: float) -> str:
        """Choose the best extraction style for auto mode."""
        if mode in {"table", "keyvalue"}:
            return mode
        if table_score <= 0 and key_value_score <= 0:
            return "empty"
        return "table" if table_score >= key_value_score else "keyvalue"

    def _row_center(self, row: list[TextBox]) -> float:
        """Return the vertical center for a grouped OCR row."""
        return self._average([box.y_center for box in row])

    def _row_height(self, row: list[TextBox]) -> float:
        """Return the median text height for a grouped OCR row."""
        heights = [box.height for box in row if box.height > 0]
        return float(median(heights)) if heights else 1.0

    def _safe_float(self, value: Any) -> float:
        """Convert a value to float, returning zero on invalid input."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _average(self, values: Iterable[float]) -> float:
        """Return the arithmetic mean for a non-empty iterable of floats."""
        value_list = list(values)
        if not value_list:
            return 0.0
        return sum(value_list) / len(value_list)

    def _normalize_spaces(self, value: str) -> str:
        """Collapse repeated whitespace in OCR text."""
        return MULTISPACE_PATTERN.sub(" ", str(value)).strip()


def _dataframe_to_dict(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame into JSON-friendly columns and rows."""
    safe_dataframe = dataframe.astype(object).where(pd.notna(dataframe), None)
    return {
        "columns": [str(column) for column in safe_dataframe.columns],
        "rows": safe_dataframe.values.tolist(),
    }
