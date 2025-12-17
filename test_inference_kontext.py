#!/usr/bin/env python3
"""
Simple Kontext inference test script.

Usage:
    python test_inference_kontext.py
    python test_inference_kontext.py --prompt "Your custom prompt"
    python test_inference_kontext.py --image path/to/image.png --seed 123
"""
import argparse
import time
from pathlib import Path

from PIL import Image

from kontext_pipeline import KontextInferenceManager

# Default prompt for testing
DEFAULT_PROMPT = "Transform into a powerful fire mage with glowing orange eyes, flames dancing around the hands, and a flowing crimson cape billowing in magical wind"


def main():
    parser = argparse.ArgumentParser(description="Run Kontext image editing inference")
    parser.add_argument("--image", type=str, default="ninja.png", help="Input image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Edit prompt")
    parser.add_argument("--seed", type=int, default=618854, help="Random seed for determinism")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=4, help="Guidance scale")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    args = parser.parse_args()

    # Validate input image exists
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1

    print("=" * 60)
    print("Kontext Inference Test")
    print("=" * 60)
    print(f"Input:    {args.image}")
    print(f"Prompt:   {args.prompt}")
    print(f"Seed:     {args.seed}")
    print(f"Steps:    {args.steps}")
    print(f"Guidance: {args.guidance}")
    print(f"Output:   {args.output}")
    print("=" * 60)

    # Initialize manager
    print("\nInitializing KontextInferenceManager...")
    manager = KontextInferenceManager()

    # Load input image
    print(f"Loading input image: {args.image}")
    input_image = Image.open(args.image)
    print(f"  Size: {input_image.size}, Mode: {input_image.mode}")

    # Run inference
    print(f"\nRunning inference (this may take a while on first run)...")
    start_time = time.time()

    output_image = manager.generate(
        prompt=args.prompt,
        image=input_image,
        seed=args.seed,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps
    )

    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.1f} seconds")

    # Save output
    output_image.save(args.output)
    print(f"\nSaved output to: {args.output}")
    print(f"  Size: {output_image.size}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
