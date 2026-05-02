"""OpenCV image cleanup pipeline for OCR-ready scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_ADAPTIVE_BLOCK_SIZE = 35
DEFAULT_ADAPTIVE_C = 15
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)
DEFAULT_DENOISE_H = 12
DEFAULT_DENOISE_TEMPLATE_WINDOW = 7
DEFAULT_DENOISE_SEARCH_WINDOW = 21
MAX_DESKEW_ANGLE = 15.0
MORPH_KERNEL_SIZE = (1, 1)
ROTATION_SCALE = 1.0
WHITE_PIXEL = 255


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the OpenCV preprocessing pipeline."""

    adaptive_block_size: int = DEFAULT_ADAPTIVE_BLOCK_SIZE
    adaptive_c: int = DEFAULT_ADAPTIVE_C
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT
    clahe_tile_grid_size: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID_SIZE
    denoise_h: int = DEFAULT_DENOISE_H
    denoise_template_window_size: int = DEFAULT_DENOISE_TEMPLATE_WINDOW
    denoise_search_window_size: int = DEFAULT_DENOISE_SEARCH_WINDOW
    max_deskew_angle: float = MAX_DESKEW_ANGLE


class ImagePreprocessor:
    """Clean scanned images before they are passed to PaddleOCR."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Create a preprocessor with optional custom settings."""
        self.config = config or PreprocessingConfig()

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Run grayscale, denoise, contrast, threshold, and deskew steps."""
        cv_image = self._pil_to_cv(image)
        grayscale = self._to_grayscale(cv_image)
        denoised = self._denoise(grayscale)
        enhanced = self._enhance_contrast(denoised)
        thresholded = self._adaptive_threshold(enhanced)
        deskewed = self._deskew(thresholded)
        cleaned = self._morphological_cleanup(deskewed)
        return self._cv_gray_to_pil_rgb(cleaned)

    def _pil_to_cv(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL image into an OpenCV-compatible RGB array."""
        return np.array(image.convert("RGB"))

    def _cv_gray_to_pil_rgb(self, image: np.ndarray) -> Image.Image:
        """Convert a grayscale OpenCV array into an RGB PIL image."""
        return Image.fromarray(image).convert("RGB")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an RGB or BGR image array to grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove scanner noise using OpenCV non-local means denoising."""
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=self.config.denoise_h,
            templateWindowSize=self.config.denoise_template_window_size,
            searchWindowSize=self.config.denoise_search_window_size,
        )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Improve local contrast using CLAHE."""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        return clahe.apply(image)

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for uneven scan backgrounds."""
        block_size = self._ensure_odd(self.config.adaptive_block_size)
        return cv2.adaptiveThreshold(
            image,
            WHITE_PIXEL,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c,
        )

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct small scan rotations when the detected angle is reliable."""
        inverted = cv2.bitwise_not(image)
        coordinates = np.column_stack(np.where(inverted > 0))

        if coordinates.size == 0:
            return image

        angle = cv2.minAreaRect(coordinates)[-1]
        correction_angle = self._normalize_skew_angle(float(angle))

        if abs(correction_angle) < 0.1 or abs(correction_angle) > self.config.max_deskew_angle:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, ROTATION_SCALE)

        return cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=WHITE_PIXEL,
        )

    def _normalize_skew_angle(self, angle: float) -> float:
        """Convert OpenCV rectangle angles into a page rotation correction."""
        if angle < -45.0:
            return -(90.0 + angle)
        return -angle

    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Run a light morphological cleanup without damaging characters."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def _ensure_odd(self, value: int) -> int:
        """Return an odd integer accepted by adaptive thresholding."""
        adjusted = max(3, int(value))
        return adjusted if adjusted % 2 == 1 else adjusted + 1


def preprocess_image(image: Image.Image, config: PreprocessingConfig | None = None) -> Image.Image:
    """Convenience function for preprocessing a single PIL image."""
    return ImagePreprocessor(config=config).preprocess(image)
