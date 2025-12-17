#!/usr/bin/env python3
"""
Torch Compile Determinism Test

Tests that torch.compile() preserves determinism:
- Same seed should produce identical outputs
- Original vs Compiled should produce identical outputs

Generates a visual grid suitable for sharing with non-technical stakeholders.

Usage:
    python test_torch_compile_determinism.py
"""
import gc
import hashlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Test parameters
TEST_SEED = 618854
TEST_GUIDANCE_SCALE = 4
TEST_NUM_INFERENCE_STEPS = 20
TEST_PROMPT = "Transform into a powerful fire mage with glowing orange eyes, flames dancing around the hands, and a flowing crimson cape billowing in magical wind"

OUTPUT_DIR = Path("test_output/compile_determinism")


def get_image_hash(img: Image.Image) -> str:
    """Compute SHA256 hash of image pixels."""
    arr = np.array(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def images_identical(img1: Image.Image, img2: Image.Image) -> tuple[bool, float]:
    """
    Check if two images are identical.
    Returns (is_identical, difference_percentage).
    """
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    if arr1.shape != arr2.shape:
        return False, 100.0

    identical = np.array_equal(arr1, arr2)
    diff_pixels = np.sum(arr1 != arr2) / arr1.size * 100

    return identical, diff_pixels


def create_results_grid(results: list, output_path: Path, title: str):
    """
    Create a visual grid of results for stakeholder review.

    Each cell shows:
    - The generated image
    - Test name
    - Parameters (seed, steps, guidance, mode)
    - Timing information
    - Hash (for determinism verification)
    """
    if not results:
        print("No results to visualize")
        return

    print(f"\n{'='*60}")
    print(f"Creating Visual Grid: {title}")
    print(f"{'='*60}")

    # Grid configuration
    cell_width = 540
    cell_height = 680
    header_height = 140  # Space for text above image
    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols

    # Create canvas with title area
    title_height = 80
    grid_width = cell_width * cols
    grid_height = title_height + (cell_height * rows)
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_title = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw main title
    draw.text((20, 15), title, fill=(0, 0, 0), font=font_title)

    # Draw timestamp and test info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_text = f"Generated: {timestamp} | Prompt: \"{TEST_PROMPT[:60]}...\""
    draw.text((20, 50), info_text, fill=(100, 100, 100), font=font_small)

    # Draw each result cell
    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = title_height + (row * cell_height)

        # Draw cell border
        border_color = (200, 200, 200)
        if result.get("is_match") is True:
            border_color = (0, 180, 0)  # Green for match
        elif result.get("is_match") is False:
            border_color = (220, 0, 0)  # Red for mismatch
        draw.rectangle([x+2, y+2, x + cell_width - 3, y + cell_height - 3],
                      outline=border_color, width=3)

        # Draw test name (header)
        test_name = result.get("test", "Test")
        draw.text((x + 10, y + 8), test_name, fill=(0, 0, 0), font=font_bold)

        # Draw mode badge
        mode = result.get("mode", "unknown")
        mode_color = (0, 100, 200) if mode == "compiled" else (100, 100, 100)
        mode_text = f"[{mode.upper()}]"
        draw.text((x + cell_width - 100, y + 8), mode_text, fill=mode_color, font=font_bold)

        # Draw parameters
        y_text = y + 32
        params = [
            f"Seed: {result.get('seed', 'N/A')}",
            f"Steps: {result.get('steps', 'N/A')}",
            f"Guidance: {result.get('guidance', 'N/A')}",
        ]
        draw.text((x + 10, y_text), " | ".join(params), fill=(60, 60, 60), font=font_normal)

        # Draw timing
        y_text += 20
        duration = result.get("duration", 0)
        timing_text = f"Duration: {duration:.2f}s"
        timing_color = (0, 130, 0) if duration < 30 else (200, 100, 0)
        draw.text((x + 10, y_text), timing_text, fill=timing_color, font=font_normal)

        # Draw hash (first 16 chars)
        img_hash = result.get("hash", "N/A")
        hash_text = f"Hash: {img_hash[:16]}..."
        draw.text((x + 200, y_text), hash_text, fill=(100, 100, 100), font=font_small)

        # Draw determinism result if applicable
        y_text += 20
        if "is_match" in result:
            if result["is_match"]:
                match_text = "✓ DETERMINISTIC - Identical to reference"
                match_color = (0, 150, 0)
            else:
                diff_pct = result.get("diff_pct", 0)
                match_text = f"✗ MISMATCH - {diff_pct:.4f}% pixels differ"
                match_color = (200, 0, 0)
            draw.text((x + 10, y_text), match_text, fill=match_color, font=font_bold)

        # Draw run number if applicable
        if "run" in result:
            run_text = f"Run #{result['run']}"
            draw.text((x + cell_width - 70, y_text), run_text, fill=(100, 100, 100), font=font_small)

        # Paste the image
        img = result["image"].copy()
        img_size = cell_height - header_height - 20
        img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)

        # Center image horizontally
        img_x = x + (cell_width - img.width) // 2
        img_y = y + header_height
        grid.paste(img, (img_x, img_y))

    # Add summary footer
    footer_y = grid_height - 30
    total_tests = len(results)
    matches = sum(1 for r in results if r.get("is_match") is True)
    mismatches = sum(1 for r in results if r.get("is_match") is False)

    if matches > 0 or mismatches > 0:
        summary = f"Summary: {matches} identical, {mismatches} mismatches out of {total_tests} tests"
        summary_color = (0, 150, 0) if mismatches == 0 else (200, 0, 0)
        # Draw at bottom of last row

    # Save grid
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {grid_width}x{grid_height}")

    return output_path


def run_inference(manager, name: str, mode: str, image: Image.Image, run_num: int = 1) -> dict:
    """Run a single inference and collect results."""
    print(f"  Running {name} (run {run_num})...")

    start = time.time()
    output = manager.generate(
        prompt=TEST_PROMPT,
        image=image,
        seed=TEST_SEED,
        guidance_scale=TEST_GUIDANCE_SCALE,
        num_inference_steps=TEST_NUM_INFERENCE_STEPS
    )
    duration = time.time() - start

    img_hash = get_image_hash(output)
    print(f"    Duration: {duration:.2f}s | Hash: {img_hash[:16]}...")

    return {
        "test": name,
        "mode": mode,
        "image": output,
        "hash": img_hash,
        "duration": duration,
        "seed": TEST_SEED,
        "steps": TEST_NUM_INFERENCE_STEPS,
        "guidance": TEST_GUIDANCE_SCALE,
        "run": run_num,
    }


def test_original_determinism(image: Image.Image) -> list:
    """Test that original (non-compiled) pipeline is deterministic."""
    print("\n" + "="*60)
    print("TEST 1: Original Pipeline Determinism")
    print("="*60)
    print("Running same inference 3x with same seed - should be identical")

    from kontext_pipeline import KontextInferenceManager
    manager = KontextInferenceManager()

    results = []
    reference_hash = None

    for i in range(3):
        result = run_inference(manager, f"Original Run {i+1}", "original", image, i+1)

        if reference_hash is None:
            reference_hash = result["hash"]
            result["is_match"] = True  # Reference
        else:
            is_identical, diff_pct = images_identical(results[0]["image"], result["image"])
            result["is_match"] = is_identical
            result["diff_pct"] = diff_pct

        results.append(result)

    # Summary
    all_match = all(r.get("is_match", False) for r in results)
    print(f"\n{'✓' if all_match else '✗'} Original determinism: {'PASS' if all_match else 'FAIL'}")

    # Cleanup
    del manager
    torch.cuda.empty_cache()
    gc.collect()

    return results


def test_compiled_determinism(image: Image.Image) -> list:
    """Test that compiled pipeline is deterministic."""
    print("\n" + "="*60)
    print("TEST 2: Compiled Pipeline Determinism")
    print("="*60)
    print("Running same inference 3x with same seed - should be identical")
    print("NOTE: First run includes compilation time")

    from kontext_pipeline_fast import KontextFastInferenceManager
    manager = KontextFastInferenceManager(compile_transformer=True)

    results = []
    reference_hash = None

    for i in range(3):
        result = run_inference(manager, f"Compiled Run {i+1}", "compiled", image, i+1)

        if reference_hash is None:
            reference_hash = result["hash"]
            result["is_match"] = True  # Reference
        else:
            is_identical, diff_pct = images_identical(results[0]["image"], result["image"])
            result["is_match"] = is_identical
            result["diff_pct"] = diff_pct

        results.append(result)

    # Summary
    all_match = all(r.get("is_match", False) for r in results)
    print(f"\n{'✓' if all_match else '✗'} Compiled determinism: {'PASS' if all_match else 'FAIL'}")

    # Cleanup
    del manager
    torch.cuda.empty_cache()
    gc.collect()

    return results


def test_original_vs_compiled(image: Image.Image) -> list:
    """Test that original and compiled produce identical outputs."""
    print("\n" + "="*60)
    print("TEST 3: Original vs Compiled Comparison")
    print("="*60)
    print("Comparing outputs between original and compiled pipelines")

    results = []

    # Run original
    print("\nRunning Original pipeline...")
    from kontext_pipeline import KontextInferenceManager
    original_manager = KontextInferenceManager()
    original_result = run_inference(original_manager, "Original (Reference)", "original", image)
    original_result["is_match"] = True  # Reference
    results.append(original_result)

    del original_manager
    torch.cuda.empty_cache()
    gc.collect()

    # Run compiled
    print("\nRunning Compiled pipeline...")
    from kontext_pipeline_fast import KontextFastInferenceManager
    compiled_manager = KontextFastInferenceManager(compile_transformer=True)
    compiled_result = run_inference(compiled_manager, "Compiled", "compiled", image)

    # Compare
    is_identical, diff_pct = images_identical(original_result["image"], compiled_result["image"])
    compiled_result["is_match"] = is_identical
    compiled_result["diff_pct"] = diff_pct
    results.append(compiled_result)

    # Summary
    print(f"\n{'✓' if is_identical else '✗'} Original vs Compiled: {'IDENTICAL' if is_identical else 'DIFFERENT'}")
    if not is_identical:
        print(f"  Difference: {diff_pct:.4f}% of pixels")

    del compiled_manager
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    """Run all determinism tests and generate visual grids."""
    print("="*60)
    print("TORCH.COMPILE DETERMINISM TEST SUITE")
    print("="*60)
    print(f"Seed: {TEST_SEED}")
    print(f"Steps: {TEST_NUM_INFERENCE_STEPS}")
    print(f"Guidance: {TEST_GUIDANCE_SCALE}")
    print(f"Prompt: {TEST_PROMPT[:50]}...")
    print(f"Output: {OUTPUT_DIR.absolute()}")

    # Load test image
    print("\nLoading test image...")
    image = Image.open("ninja.png")
    print(f"Image size: {image.size}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Test 1: Original determinism
    original_results = test_original_determinism(image)
    create_results_grid(
        original_results,
        OUTPUT_DIR / "1_original_determinism.png",
        "Test 1: Original Pipeline Determinism"
    )
    all_results.extend(original_results)

    # Test 2: Compiled determinism
    compiled_results = test_compiled_determinism(image)
    create_results_grid(
        compiled_results,
        OUTPUT_DIR / "2_compiled_determinism.png",
        "Test 2: Compiled Pipeline Determinism"
    )
    all_results.extend(compiled_results)

    # Test 3: Original vs Compiled
    comparison_results = test_original_vs_compiled(image)
    create_results_grid(
        comparison_results,
        OUTPUT_DIR / "3_original_vs_compiled.png",
        "Test 3: Original vs Compiled Comparison"
    )
    all_results.extend(comparison_results)

    # Create combined grid
    create_results_grid(
        all_results,
        OUTPUT_DIR / "FULL_DETERMINISM_REPORT.png",
        "Full Determinism Test Report: Original vs torch.compile()"
    )

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    total_tests = len(all_results)
    matches = sum(1 for r in all_results if r.get("is_match") is True)
    mismatches = sum(1 for r in all_results if r.get("is_match") is False)

    print(f"Total tests: {total_tests}")
    print(f"Identical:   {matches}")
    print(f"Mismatches:  {mismatches}")

    if mismatches == 0:
        print("\n✅ ALL TESTS PASSED - Determinism preserved!")
    else:
        print(f"\n❌ {mismatches} TESTS FAILED - Determinism may be broken!")

    print(f"\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f}")

    print(f"\nShare this with stakeholders:")
    print(f"  {OUTPUT_DIR.absolute()}/FULL_DETERMINISM_REPORT.png")


if __name__ == "__main__":
    main()
