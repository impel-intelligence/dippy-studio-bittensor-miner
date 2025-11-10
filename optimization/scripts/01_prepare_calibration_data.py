#!/usr/bin/env python3
"""
Calibration Dataset Preparation Script

This script prepares a calibration dataset for quantizing the FLUX Kontext model.
Calibration helps find optimal scale factors for FP8/FP4 quantization.

Usage:
    python3 optimization/scripts/01_prepare_calibration_data.py
"""

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationDatasetPreparer:
    """Prepares calibration dataset for quantization."""

    def __init__(
        self,
        output_dir: str = "optimization/calibration/data",
        num_samples: int = 100,
        seed: int = 42
    ):
        """
        Initialize calibration dataset preparer.

        Args:
            output_dir: Directory to save calibration data
            num_samples: Number of calibration samples to generate
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.seed = seed

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        random.seed(seed)

        # Base test images
        self.base_images = [
            Path("character_1.png"),
            Path("character_2.png"),
        ]

        # Diverse prompts for calibration
        # Cover various editing operations to get good statistical coverage
        self.prompt_templates = [
            # Color modifications
            "Change the {item} color to {color}",
            "Make the {item} {color}",
            "Add a {color} tint to the image",

            # Object additions
            "Add a {object} to the scene",
            "Place a {object} in the background",
            "Put a {object} near the person",

            # Style modifications
            "Make the image look like {style}",
            "Apply a {style} filter",
            "Transform the style to {style}",

            # Lighting changes
            "Make the lighting {lighting}",
            "Change the lighting to {lighting}",
            "Add {lighting} lighting",

            # Background modifications
            "Change the background to {background}",
            "Replace the background with {background}",
            "Set the background as {background}",

            # Detail modifications
            "Add {detail} to the image",
            "Include {detail} in the scene",

            # Remove/modify
            "Remove the {item}",
            "Blur the {item}",
            "Enhance the {item}",
        ]

        # Vocabulary for prompt generation
        self.vocab = {
            "item": ["background", "foreground", "person", "face", "clothing", "hair"],
            "color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "golden"],
            "object": ["hat", "sunglasses", "scarf", "tree", "flowers", "clouds", "moon", "stars", "building"],
            "style": ["watercolor painting", "oil painting", "sketch", "cartoon", "anime", "pixel art", "impressionist painting"],
            "lighting": ["brighter", "darker", "more dramatic", "softer", "warmer", "cooler", "sunset", "dawn"],
            "background": ["beach", "mountains", "forest", "city", "desert", "ocean", "space", "garden"],
            "detail": ["shadows", "highlights", "reflections", "texture", "depth", "blur", "sharpness"],
        }

    def generate_prompt(self) -> str:
        """
        Generate a random prompt from templates and vocabulary.

        Returns:
            Generated prompt string
        """
        template = random.choice(self.prompt_templates)

        # Fill in template placeholders
        prompt = template
        for placeholder, options in self.vocab.items():
            if f"{{{placeholder}}}" in prompt:
                prompt = prompt.replace(f"{{{placeholder}}}", random.choice(options))

        return prompt

    def create_calibration_dataset(self) -> List[Dict]:
        """
        Create calibration dataset with diverse prompts and images.

        Returns:
            List of calibration samples with prompts and image paths
        """
        dataset = []

        logger.info(f"Generating {self.num_samples} calibration samples...")

        for i in range(self.num_samples):
            # Cycle through base images
            base_image = self.base_images[i % len(self.base_images)]

            # Generate prompt
            prompt = self.generate_prompt()

            # Copy image to calibration directory
            img_filename = f"calib_{i:04d}.png"
            img_path = self.output_dir / img_filename

            if not img_path.exists():
                shutil.copy(base_image, img_path)

            # Create sample
            sample = {
                "id": i,
                "prompt": prompt,
                "image_path": str(img_path),
                "seed": self.seed + i,  # Unique seed per sample
                "base_image": str(base_image),
            }

            dataset.append(sample)

            if (i + 1) % 20 == 0:
                logger.info(f"Generated {i + 1}/{self.num_samples} samples")

        logger.info(f"✓ Created {len(dataset)} calibration samples")

        return dataset

    def save_dataset(self, dataset: List[Dict]):
        """
        Save calibration dataset to JSON file.

        Args:
            dataset: List of calibration samples
        """
        output_file = self.output_dir / "calibration_dataset.json"

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"✓ Saved calibration dataset to {output_file}")

        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("Calibration Dataset Statistics")
        logger.info("="*60)
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Unique base images: {len(self.base_images)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60 + "\n")

        # Print sample prompts
        logger.info("Sample prompts:")
        for i in range(min(5, len(dataset))):
            logger.info(f"  {i+1}. {dataset[i]['prompt']}")
        logger.info("")

    def verify_images(self):
        """Verify that base images exist and are valid."""
        logger.info("Verifying base images...")

        for img_path in self.base_images:
            if not img_path.exists():
                raise FileNotFoundError(f"Base image not found: {img_path}")

            # Try to load image
            try:
                img = Image.open(img_path)
                logger.info(f"✓ {img_path.name}: {img.size} {img.mode}")
            except Exception as e:
                raise ValueError(f"Failed to load image {img_path}: {e}")

        logger.info("✓ All base images verified\n")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("FLUX Kontext Calibration Dataset Preparation")
    logger.info("="*60 + "\n")

    # Create preparer
    preparer = CalibrationDatasetPreparer(
        output_dir="optimization/calibration/data",
        num_samples=100,  # Start with 100, can increase to 500 later
        seed=42
    )

    # Verify images exist
    preparer.verify_images()

    # Create dataset
    dataset = preparer.create_calibration_dataset()

    # Save dataset
    preparer.save_dataset(dataset)

    logger.info("✓ Calibration dataset preparation complete!")
    logger.info("Next step: Run baseline performance measurement (02_measure_baseline.py)")


if __name__ == "__main__":
    main()
