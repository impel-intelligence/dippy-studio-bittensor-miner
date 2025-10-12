"""Image validation utilities to detect incomplete or corrupted images."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    """Result of image validation check."""

    is_valid: bool
    reason: Optional[str] = None
    file_size: Optional[int] = None
    dimensions: Optional[tuple[int, int]] = None
    blank_percentage: Optional[float] = None
    variance: Optional[float] = None


class ImageValidator:
    """Validates generated images for completeness and quality."""

    def __init__(
        self,
        expected_width: int = 1024,
        expected_height: int = 1024,
        min_file_size: int = 100_000,  # ~100KB minimum for 1024x1024 PNG
        max_file_size: int = 10_000_000,  # ~10MB maximum
        max_blank_percentage: float = 0.5,  # 50% blank/black is suspicious
        min_variance: float = 100.0,  # Minimum pixel variance (detects solid colors)
    ):
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.max_blank_percentage = max_blank_percentage
        self.min_variance = min_variance

    def validate(self, image_path: Path | str) -> ValidationResult:
        """
        Validate image for completeness and quality.

        Checks performed:
        1. File exists and is readable
        2. File size is within expected range
        3. PIL can open and verify the image
        4. Dimensions match expected size
        5. Image is not mostly blank/black (indicates partial render)
        6. Pixel variance is reasonable (not solid color)

        Args:
            image_path: Path to image file

        Returns:
            ValidationResult with is_valid and diagnostic information
        """
        image_path = Path(image_path)

        # Check 1: File exists
        if not image_path.exists():
            return ValidationResult(
                is_valid=False,
                reason=f"File does not exist: {image_path}"
            )

        # Check 2: File size
        file_size = image_path.stat().st_size
        if file_size < self.min_file_size:
            return ValidationResult(
                is_valid=False,
                reason=f"File too small ({file_size} bytes, expected >{self.min_file_size})",
                file_size=file_size
            )

        if file_size > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                reason=f"File too large ({file_size} bytes, expected <{self.max_file_size})",
                file_size=file_size
            )

        # Check 3: PIL can open
        try:
            with Image.open(image_path) as img:
                # Verify image integrity
                img.verify()
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"PIL verification failed: {e}",
                file_size=file_size
            )

        # Reopen for analysis (verify() closes the file)
        try:
            with Image.open(image_path) as img:
                img.load()  # Force load pixel data

                # Check 4: Dimensions
                width, height = img.size
                if width != self.expected_width or height != self.expected_height:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Dimension mismatch: expected {self.expected_width}x{self.expected_height}, got {width}x{height}",
                        file_size=file_size,
                        dimensions=(width, height)
                    )

                # Convert to numpy for analysis
                img_array = np.array(img.convert('RGB'))

                # Check 5: Blank/black region detection
                # Count pixels that are very dark (RGB < 10)
                dark_pixels = np.all(img_array < 10, axis=2)
                blank_percentage = np.sum(dark_pixels) / (width * height)

                if blank_percentage > self.max_blank_percentage:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Image has {blank_percentage*100:.1f}% blank/black pixels (threshold: {self.max_blank_percentage*100:.1f}%)",
                        file_size=file_size,
                        dimensions=(width, height),
                        blank_percentage=blank_percentage
                    )

                # Check 6: Pixel variance (detects solid colors)
                variance = float(np.var(img_array))

                if variance < self.min_variance:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Image variance too low ({variance:.1f}, threshold: {self.min_variance}), may be solid color or corrupted",
                        file_size=file_size,
                        dimensions=(width, height),
                        blank_percentage=blank_percentage,
                        variance=variance
                    )

                # All checks passed
                return ValidationResult(
                    is_valid=True,
                    file_size=file_size,
                    dimensions=(width, height),
                    blank_percentage=blank_percentage,
                    variance=variance
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"Image analysis failed: {e}",
                file_size=file_size
            )

    def validate_bytes(self, image_bytes: bytes) -> ValidationResult:
        """
        Validate image from bytes (e.g., from callback).

        Args:
            image_bytes: Raw image file bytes

        Returns:
            ValidationResult with is_valid and diagnostic information
        """
        # Check file size
        file_size = len(image_bytes)
        if file_size < self.min_file_size:
            return ValidationResult(
                is_valid=False,
                reason=f"File too small ({file_size} bytes, expected >{self.min_file_size})",
                file_size=file_size
            )

        if file_size > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                reason=f"File too large ({file_size} bytes, expected <{self.max_file_size})",
                file_size=file_size
            )

        # Open from bytes
        try:
            img_buffer = io.BytesIO(image_bytes)
            with Image.open(img_buffer) as img:
                img.verify()
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"PIL verification failed: {e}",
                file_size=file_size
            )

        # Reopen for analysis
        try:
            img_buffer = io.BytesIO(image_bytes)
            with Image.open(img_buffer) as img:
                img.load()

                # Check dimensions
                width, height = img.size
                if width != self.expected_width or height != self.expected_height:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Dimension mismatch: expected {self.expected_width}x{self.expected_height}, got {width}x{height}",
                        file_size=file_size,
                        dimensions=(width, height)
                    )

                # Convert to numpy for analysis
                img_array = np.array(img.convert('RGB'))

                # Check blank regions
                dark_pixels = np.all(img_array < 10, axis=2)
                blank_percentage = np.sum(dark_pixels) / (width * height)

                if blank_percentage > self.max_blank_percentage:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Image has {blank_percentage*100:.1f}% blank/black pixels (threshold: {self.max_blank_percentage*100:.1f}%)",
                        file_size=file_size,
                        dimensions=(width, height),
                        blank_percentage=blank_percentage
                    )

                # Check variance
                variance = float(np.var(img_array))

                if variance < self.min_variance:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Image variance too low ({variance:.1f}, threshold: {self.min_variance}), may be solid color or corrupted",
                        file_size=file_size,
                        dimensions=(width, height),
                        blank_percentage=blank_percentage,
                        variance=variance
                    )

                # All checks passed
                return ValidationResult(
                    is_valid=True,
                    file_size=file_size,
                    dimensions=(width, height),
                    blank_percentage=blank_percentage,
                    variance=variance
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"Image analysis failed: {e}",
                file_size=file_size
            )


def detect_partial_render(image_path: Path | str, threshold: float = 0.3) -> tuple[bool, Optional[str]]:
    """
    Quick check specifically for partial renders (horizontal cutoff).

    Detects if the bottom portion of the image is blank/black, which indicates
    the generation was interrupted mid-render.

    Args:
        image_path: Path to image file
        threshold: Percentage of bottom rows that must be blank to trigger (default: 30%)

    Returns:
        Tuple of (is_partial, reason)
        - is_partial: True if image appears to be partially rendered
        - reason: Human-readable explanation
    """
    try:
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            img_array = np.array(img.convert('RGB'))

            # Check bottom 30% of image for blank rows
            bottom_start = int(height * (1 - threshold))
            bottom_region = img_array[bottom_start:, :, :]

            # Count blank/dark rows (where all pixels are < 10)
            dark_pixels_per_row = np.all(img_array < 10, axis=(1, 2))
            dark_rows_in_bottom = np.sum(dark_pixels_per_row[bottom_start:])
            total_bottom_rows = height - bottom_start

            blank_row_percentage = dark_rows_in_bottom / total_bottom_rows if total_bottom_rows > 0 else 0

            if blank_row_percentage > 0.8:  # 80% of bottom rows are blank
                return True, f"Partial render detected: {blank_row_percentage*100:.1f}% of bottom rows are blank"

            return False, None

    except Exception as e:
        return False, f"Could not analyze: {e}"
