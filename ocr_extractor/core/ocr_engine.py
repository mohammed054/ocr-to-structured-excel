"""PaddleOCR wrapper that returns normalized text blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from ..utils.logger import RichLogger, get_logger
except ImportError:  # pragma: no cover - supports direct script execution
    from utils.logger import RichLogger, get_logger


DEFAULT_LANGUAGE = "en"
DEFAULT_USE_ANGLE_CLASSIFIER = True
DEFAULT_CONFIDENCE_FLOOR = 0.0
PADDLE_SHOW_LOG = False
BBOX_POINTS = 4


class OCREngineError(RuntimeError):
    """Raised when the OCR engine cannot be initialized or executed."""


@dataclass(frozen=True)
class OCRTextBlock:
    """Normalized OCR line with geometry and confidence."""

    text: str
    bbox: list[list[float]]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the OCR text block into a JSON-friendly dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }


class PaddleOCREngine:
    """Thin production wrapper around PaddleOCR."""

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        use_angle_cls: bool = DEFAULT_USE_ANGLE_CLASSIFIER,
        logger: RichLogger | None = None,
    ) -> None:
        """Initialize PaddleOCR with English recognition and angle classification."""
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.logger = logger or get_logger()
        self._ocr = self._create_engine()

    def extract(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run OCR on a PIL image and return normalized dictionaries."""
        blocks = self.extract_blocks(image)
        return [block.to_dict() for block in blocks]

    def extract_blocks(self, image: Image.Image) -> list[OCRTextBlock]:
        """Run OCR on a PIL image and return normalized OCR text blocks."""
        cv_image = self._pil_to_bgr(image)

        try:
            raw_result = self._ocr.ocr(cv_image, cls=self.use_angle_cls)
        except Exception as exc:  # noqa: BLE001 - PaddleOCR raises mixed exception types
            raise OCREngineError(f"PaddleOCR failed while reading an image: {exc}") from exc

        blocks = self._parse_raw_result(raw_result)
        if not blocks:
            self.logger.warning("OCR returned no text for this page")
        return blocks

    def _create_engine(self) -> Any:
        """Create the PaddleOCR instance lazily so import errors are friendly."""
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCREngineError(
                "PaddleOCR is not installed. Run `pip install -r requirements.txt` "
                "inside the ocr_extractor project folder."
            ) from exc

        try:
            return PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.language,
                show_log=PADDLE_SHOW_LOG,
            )
        except Exception as exc:  # noqa: BLE001 - PaddleOCR initialization can fail broadly
            raise OCREngineError(f"Could not initialize PaddleOCR: {exc}") from exc

    def _pil_to_bgr(self, image: Image.Image) -> np.ndarray:
        """Convert PIL RGB data to OpenCV BGR format for PaddleOCR."""
        rgb_array = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _parse_raw_result(self, raw_result: Any) -> list[OCRTextBlock]:
        """Parse PaddleOCR output into normalized OCR text blocks."""
        if not raw_result:
            return []

        dict_blocks = self._parse_dict_style_result(raw_result)
        if dict_blocks:
            return dict_blocks

        lines = self._flatten_legacy_lines(raw_result)
        blocks: list[OCRTextBlock] = []

        for line in lines:
            parsed = self._parse_legacy_line(line)
            if parsed is not None and parsed.confidence >= DEFAULT_CONFIDENCE_FLOOR:
                blocks.append(parsed)

        return blocks

    def _parse_dict_style_result(self, raw_result: Any) -> list[OCRTextBlock]:
        """Parse dictionary-style PaddleOCR outputs when present."""
        pages = raw_result if isinstance(raw_result, list) else [raw_result]
        blocks: list[OCRTextBlock] = []

        for page in pages:
            if not isinstance(page, dict):
                continue

            texts = page.get("rec_texts") or page.get("texts") or []
            scores = page.get("rec_scores") or page.get("scores") or []
            boxes = page.get("rec_polys") or page.get("dt_polys") or page.get("boxes") or []

            for text, score, bbox in zip(texts, scores, boxes):
                normalized = self._build_block(text=text, confidence=score, bbox=bbox)
                if normalized is not None:
                    blocks.append(normalized)

        return blocks

    def _flatten_legacy_lines(self, raw_result: Any) -> list[Any]:
        """Flatten PaddleOCR 2.x nested page results into line records."""
        if self._looks_like_legacy_line(raw_result):
            return [raw_result]

        flattened: list[Any] = []
        if not isinstance(raw_result, list):
            return flattened

        for item in raw_result:
            if item is None:
                continue
            if self._looks_like_legacy_line(item):
                flattened.append(item)
                continue
            if isinstance(item, list):
                for nested in item:
                    if self._looks_like_legacy_line(nested):
                        flattened.append(nested)

        return flattened

    def _looks_like_legacy_line(self, value: Any) -> bool:
        """Return True when a value resembles a PaddleOCR 2.x line."""
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        bbox, recognition = value[0], value[1]
        return self._looks_like_bbox(bbox) and self._looks_like_recognition_pair(recognition)

    def _parse_legacy_line(self, line: Any) -> OCRTextBlock | None:
        """Parse one PaddleOCR 2.x line record."""
        try:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
        except (IndexError, TypeError):
            return None
        return self._build_block(text=text, confidence=confidence, bbox=bbox)

    def _build_block(self, text: Any, confidence: Any, bbox: Any) -> OCRTextBlock | None:
        """Build a normalized OCR block after validating text and geometry."""
        normalized_text = str(text).strip()
        if not normalized_text:
            return None

        normalized_bbox = self._normalize_bbox(bbox)
        if not normalized_bbox:
            return None

        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError):
            normalized_confidence = DEFAULT_CONFIDENCE_FLOOR

        return OCRTextBlock(
            text=normalized_text,
            bbox=normalized_bbox,
            confidence=normalized_confidence,
        )

    def _normalize_bbox(self, bbox: Any) -> list[list[float]]:
        """Normalize OCR bounding boxes into four point coordinates."""
        if bbox is None:
            return []

        try:
            points = np.array(bbox, dtype=float).reshape(-1, 2).tolist()
        except (TypeError, ValueError):
            return []
        return [[float(x), float(y)] for x, y in points[:BBOX_POINTS]]

    def _looks_like_bbox(self, value: Any) -> bool:
        """Return True when a value resembles four OCR polygon points."""
        if not isinstance(value, (list, tuple)) or len(value) < BBOX_POINTS:
            return False

        try:
            points = np.array(value, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            return False

        return len(points) >= BBOX_POINTS

    def _looks_like_recognition_pair(self, value: Any) -> bool:
        """Return True when a value resembles a PaddleOCR text/confidence pair."""
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False

        text, confidence = value[0], value[1]
        if not isinstance(text, str):
            return False

        try:
            float(confidence)
        except (TypeError, ValueError):
            return False

        return True
