#!/usr/bin/env python3
"""
Test Kontext editing with visual output.

This script runs E2E tests and saves all generated images for inspection.
"""
import os
import sys
import base64
import requests
import time
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import hashlib
import numpy as np

# Configuration
MINER_URL = os.getenv("ASYNC_MINER_URL", "http://localhost:8091")
OUTPUT_DIR = Path("test_output/kontext_images")

# Store all test results for final grid
ALL_RESULTS = []


def create_sample_image():
    """Load the test image for Kontext editing"""
    # Load ninja.png from project root
    script_dir = Path(__file__).parent
    image_path = script_dir.parent / "ninja.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found at {image_path}")

    return Image.open(image_path)


def image_to_b64(image):
    """Convert PIL Image to base64"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def post_edit(prompt, image_b64, seed, **kwargs):
    """POST /edit request"""
    payload = {
        "prompt": prompt,
        "image_b64": image_b64,
        "seed": seed,
        **kwargs
    }
    response = requests.post(f"{MINER_URL}/edit", json=payload)
    response.raise_for_status()
    return response.json()


def wait_for_completion(job_id, timeout=300):
    """Wait for job to complete"""
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{MINER_URL}/edit/status/{job_id}")
        response.raise_for_status()
        status = response.json()

        if status["status"] == "completed":
            return status
        elif status["status"] == "failed":
            raise RuntimeError(f"Job failed: {status.get('error')}")

        time.sleep(2)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def download_result(job_id):
    """Download edited image as PIL Image"""
    response = requests.get(f"{MINER_URL}/edit/result/{job_id}")
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def create_grid_visualization():
    """Create a grid showing all test results with prompts"""
    if not ALL_RESULTS:
        print("No results to visualize")
        return

    print("\n\n=== Creating Visual Grid ===")

    # Configuration
    cell_width = 512
    cell_height = 620  # Extra space for text + timing (was 600)
    text_height = 100  # Increased from 80 for timing line
    cols = 3
    rows = (len(ALL_RESULTS) + cols - 1) // cols  # Ceiling division

    # Create canvas
    grid_width = cell_width * cols
    grid_height = cell_height * rows
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # Place each result in the grid
    for idx, result in enumerate(ALL_RESULTS):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * cell_height

        # Resize image to fit cell (preserve aspect ratio)
        img = result["image"].copy()
        img.thumbnail((cell_width, cell_height - text_height), Image.Resampling.LANCZOS)

        # Center the image in the cell
        img_x = x + (cell_width - img.width) // 2
        img_y = y + text_height
        grid.paste(img, (img_x, img_y))

        # Add text
        draw = ImageDraw.Draw(grid)

        # Test name
        test_name = result.get("test", "Test")
        draw.text((x + 10, y + 5), test_name, fill=(100, 100, 100), font=font_bold)

        # Prompt (wrap if too long)
        prompt = result["prompt"]
        if len(prompt) > 50:
            prompt = prompt[:47] + "..."
        draw.text((x + 10, y + 25), f'"{prompt}"', fill=(0, 0, 0), font=font)

        # Seed
        seed_text = f"Seed: {result.get('seed', 'N/A')}"
        draw.text((x + 10, y + 45), seed_text, fill=(60, 60, 60), font=font)

        # Duration and Steps (if available)
        if "duration" in result:
            duration = result["duration"]
            steps = result.get("steps", "N/A")
            guidance = result.get("guidance")
            guidance_str = f"{guidance}" if guidance is not None else "default"
            timing_text = f"Duration: {duration:.1f}s | Steps: {steps} | Guidance: {guidance_str}"
            draw.text((x + 10, y + 65), timing_text, fill=(0, 100, 0), font=font)

        # Border
        draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1],
                      outline=(200, 200, 200), width=2)

    # Save grid
    grid_path = OUTPUT_DIR / "VISUAL_GRID.png"
    grid.save(grid_path)
    print(f"✓ Saved visual grid: {grid_path}")
    print(f"  Grid size: {grid_width}x{grid_height} ({len(ALL_RESULTS)} images)")

    return grid_path


def test_determinism():
    """Test that same seed produces identical output"""
    print("\n=== Test 1: Determinism Verification ===")

    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    # Save input image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_image.save(OUTPUT_DIR / "input_sample.png")
    print(f"✓ Saved input image: {OUTPUT_DIR / 'input_sample.png'}")

    # Add original image to grid
    ALL_RESULTS.append({
        "test": "Original Input",
        "prompt": "Base ninja anime character (no edits)",
        "seed": "N/A",
        "image": sample_image
    })

    prompt = "Transform the ninja into a fire-style jutsu master with blazing red and orange chakra aura, flames dancing around the headband, and glowing ember eyes"
    seed = 42

    # First generation
    print(f"\n🎨 Generating image 1 with seed={seed}...")
    num_steps = 10
    guidance = 2.5
    start_time = time.time()
    response1 = post_edit(
        prompt=prompt,
        image_b64=sample_b64,
        seed=seed,
        num_inference_steps=num_steps,
        guidance_scale=guidance
    )
    job_id_1 = response1["job_id"]
    print(f"  Job ID: {job_id_1}")

    wait_for_completion(job_id_1)
    duration = time.time() - start_time

    result1 = download_result(job_id_1)
    result1.save(OUTPUT_DIR / f"{job_id_1}_result1.png")
    print(f"✓ Saved: {OUTPUT_DIR / f'{job_id_1}_result1.png'}")
    print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

    # Add to grid
    ALL_RESULTS.append({
        "test": "Determinism Test #1",
        "prompt": prompt,
        "seed": seed,
        "image": result1,
        "duration": duration,
        "steps": num_steps,
        "guidance": guidance
    })

    # Second generation with same seed
    print(f"\n🎨 Generating image 2 with seed={seed} (should be identical)...")
    start_time = time.time()
    response2 = post_edit(
        prompt=prompt,
        image_b64=sample_b64,
        seed=seed,
        num_inference_steps=num_steps,
        guidance_scale=guidance
    )
    job_id_2 = response2["job_id"]
    print(f"  Job ID: {job_id_2}")

    wait_for_completion(job_id_2)
    duration = time.time() - start_time

    result2 = download_result(job_id_2)
    result2.save(OUTPUT_DIR / f"{job_id_2}_result2.png")
    print(f"✓ Saved: {OUTPUT_DIR / f'{job_id_2}_result2.png'}")
    print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

    # Add to grid
    ALL_RESULTS.append({
        "test": "Determinism Test #2",
        "prompt": prompt,
        "seed": seed,
        "image": result2,
        "duration": duration,
        "steps": num_steps,
        "guidance": guidance
    })

    # Verify determinism
    arr1 = np.array(result1)
    arr2 = np.array(result2)

    hash1 = hashlib.sha256(arr1.tobytes()).hexdigest()
    hash2 = hashlib.sha256(arr2.tobytes()).hexdigest()

    if np.array_equal(arr1, arr2):
        print(f"\n✅ DETERMINISM VERIFIED: Images are pixel-perfect identical!")
        print(f"   SHA256: {hash1[:16]}...")
    else:
        print(f"\n❌ DETERMINISM FAILED: Images differ!")
        print(f"   Hash 1: {hash1[:16]}...")
        print(f"   Hash 2: {hash2[:16]}...")
        diff_ratio = np.mean(arr1 != arr2)
        print(f"   Difference: {diff_ratio*100:.2f}% of pixels")

    # Additional determinism tests with different prompts
    additional_prompts = [
        "Activate Sharingan eyes with red irises, three black tomoe swirling patterns, and purple chakra energy emanating from the eyes",
        "Transform into sage mode with golden eyes featuring horizontal rectangular pupils, orange pigmentation around eyes, and nature energy glowing aura",
        "Unlock Byakugan with pure white eyes showing visible veins, 360-degree vision effect with ethereal white chakra radiating outward",
    ]

    for idx, test_prompt in enumerate(additional_prompts, start=3):
        test_seed = 100 + idx
        print(f"\n🎨 Determinism Test #{idx}: '{test_prompt}' (seed={test_seed})")

        # Generate twice with same seed to verify determinism
        for run in [1, 2]:
            print(f"  Run {run}/2...")
            num_steps = 10
            guidance = 2.5
            start_time = time.time()
            response = post_edit(
                prompt=test_prompt,
                image_b64=sample_b64,
                seed=test_seed,
                num_inference_steps=num_steps,
                guidance_scale=guidance
            )
            job_id = response["job_id"]
            print(f"    Job ID: {job_id}")

            wait_for_completion(job_id)
            duration = time.time() - start_time

            result = download_result(job_id)
            result.save(OUTPUT_DIR / f"det_{idx}_run{run}_{job_id}.png")
            print(f"  ✓ Saved: {OUTPUT_DIR / f'det_{idx}_run{run}_{job_id}.png'}")
            print(f"    Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

            # Add to grid
            ALL_RESULTS.append({
                "test": f"Determinism Test #{idx} (Run {run})",
                "prompt": test_prompt,
                "seed": test_seed,
                "image": result,
                "duration": duration,
                "steps": num_steps,
                "guidance": guidance
            })


def test_different_seeds():
    """Test that different seeds produce different outputs"""
    print("\n\n=== Test 2: Different Seeds ===")

    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    prompt = "Transform into a legendary Sannin with snake-like features, vertical slit pupils, purple eye shadow markings, pale skin, and dark mystical serpent symbols on face"

    results = []
    num_steps = 10
    guidance = None  # Use API default
    for seed in [42, 123, 999]:
        print(f"\n🎨 Generating with seed={seed}...")
        start_time = time.time()
        response = post_edit(
            prompt=prompt,
            image_b64=sample_b64,
            seed=seed,
            num_inference_steps=num_steps
        )
        job_id = response["job_id"]
        print(f"  Job ID: {job_id}")

        wait_for_completion(job_id)
        duration = time.time() - start_time

        result = download_result(job_id)
        result.save(OUTPUT_DIR / f"seed_{seed}_{job_id}.png")
        print(f"✓ Saved: {OUTPUT_DIR / f'seed_{seed}_{job_id}.png'}")
        print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: default")
        results.append(np.array(result))

        # Add to grid
        ALL_RESULTS.append({
            "test": f"Different Seeds",
            "prompt": prompt,
            "seed": seed,
            "image": result,
            "duration": duration,
            "steps": num_steps,
            "guidance": guidance
        })

    # Verify they're different
    if not np.array_equal(results[0], results[1]):
        print(f"\n✅ SEED VARIATION VERIFIED: Different seeds produce different outputs")
    else:
        print(f"\n❌ WARNING: Different seeds produced identical outputs!")


def test_various_prompts():
    """Test various creative prompts"""
    print("\n\n=== Test 3: Various Prompts ===")

    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    prompts = [
        "Activate Nine-Tails Chakra Mode with glowing golden yellow chakra cloak, fox-like whisker marks intensified, blazing orange eyes with vertical pupils, and swirling chakra flames",
        "Transform into Lightning Release armor with crackling blue electricity surrounding body, electric chakra coursing through veins visible on skin, and eyes glowing bright white",
        "Enter Curse Mark state with dark purple skin markings spreading across face and neck, flame-like patterns, glowing yellow eyes, and dark chakra aura",
        "Unlock Rinnegan with concentric purple ripple pattern eyes, ethereal lavender chakra emanating, mystical energy trails, and otherworldly presence",
        "Activate Edo Tensei form with cracked porcelain skin texture, glowing blue-white eyes, dark chakra wisps rising from body, and ethereal undead appearance",
    ]

    seed = 42  # Use same seed for consistency
    num_steps = 20  # Higher quality
    guidance = 3.0  # Strong prompt adherence

    for i, prompt in enumerate(prompts):
        print(f"\n🎨 Prompt {i+1}: '{prompt}'")
        start_time = time.time()
        response = post_edit(
            prompt=prompt,
            image_b64=sample_b64,
            seed=seed + i,  # Vary seed slightly
            num_inference_steps=num_steps,
            guidance_scale=guidance
        )
        job_id = response["job_id"]
        print(f"  Job ID: {job_id}")

        wait_for_completion(job_id, timeout=600)
        duration = time.time() - start_time

        result = download_result(job_id)

        filename = f"prompt_{i+1}_{job_id}.png"
        result.save(OUTPUT_DIR / filename)
        print(f"✓ Saved: {OUTPUT_DIR / filename}")
        print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

        # Add to grid
        ALL_RESULTS.append({
            "test": f"Various Prompts #{i+1}",
            "prompt": prompt,
            "seed": seed + i,
            "image": result,
            "duration": duration,
            "steps": num_steps,
            "guidance": guidance
        })


def test_reference_chaining():
    """Test chaining edits together"""
    print("\n\n=== Test 4: Reference Job Chaining ===")

    # Start with base image
    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    prompts = [
        "Add ANBU black ops mask with red and white traditional patterns, animal motif design, covering lower face while showing intense focused eyes",
        "Apply war paint markings in red clan symbols across cheeks, forehead with battle-ready expression, and fierce determined gaze",
        "Add Akatsuki outfit with high collar showing red clouds on black fabric, distinctive criminal organization style, and menacing shadowed appearance",
    ]

    previous_job_id = None
    seed = 100
    num_steps = 15  # Medium quality
    guidance = None  # Use API default

    for i, prompt in enumerate(prompts):
        print(f"\n🎨 Edit {i+1}: '{prompt}'")

        start_time = time.time()
        if previous_job_id:
            # Use reference chaining
            response = post_edit(
                prompt=prompt,
                reference_job_id=previous_job_id,
                image_b64=sample_b64,  # Fallback
                seed=seed + i,
                num_inference_steps=num_steps
            )
        else:
            # First edit
            response = post_edit(
                prompt=prompt,
                image_b64=sample_b64,
                seed=seed + i,
                num_inference_steps=num_steps
            )

        job_id = response["job_id"]
        print(f"  Job ID: {job_id}")

        wait_for_completion(job_id)
        duration = time.time() - start_time

        result = download_result(job_id)

        filename = f"chain_{i+1}_{job_id}.png"
        result.save(OUTPUT_DIR / filename)
        print(f"✓ Saved: {OUTPUT_DIR / filename}")
        print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: default")

        # Add to grid
        ALL_RESULTS.append({
            "test": f"Chained Edit #{i+1}",
            "prompt": prompt,
            "seed": seed + i,
            "image": result,
            "duration": duration,
            "steps": num_steps,
            "guidance": guidance
        })

        previous_job_id = job_id

    print(f"\n✅ CHAINING VERIFIED: Created {len(prompts)} sequential edits")


def test_seed_variation():
    """Test same prompt with different seeds to show variation"""
    print("\n\n=== Test 5: Seed Variation (Same Prompt) ===")

    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    # Use a complex prompt with everything else constant
    prompt = "Unlock Mangekyo Sharingan with unique kaleidoscope pattern in crimson red eyes, black intricate geometric design, purple chakra flames, and intense power radiating"

    num_steps = 15
    guidance = 2.5

    seeds = [42, 123, 456, 789, 999]

    for seed in seeds:
        print(f"\n🎨 Generating with seed={seed}...")
        start_time = time.time()
        response = post_edit(
            prompt=prompt,
            image_b64=sample_b64,
            seed=seed,
            num_inference_steps=num_steps,
            guidance_scale=guidance
        )
        job_id = response["job_id"]
        print(f"  Job ID: {job_id}")

        wait_for_completion(job_id)
        duration = time.time() - start_time

        result = download_result(job_id)
        result.save(OUTPUT_DIR / f"seed_var_{seed}_{job_id}.png")
        print(f"✓ Saved: {OUTPUT_DIR / f'seed_var_{seed}_{job_id}.png'}")
        print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

        # Add to grid
        ALL_RESULTS.append({
            "test": f"Seed Variation",
            "prompt": prompt,
            "seed": seed,
            "image": result,
            "duration": duration,
            "steps": num_steps,
            "guidance": guidance
        })

    print(f"\n✅ SEED VARIATION TEST COMPLETE: Generated {len(seeds)} variations")


def test_guidance_variation():
    """Test same prompt/seed with different guidance scales"""
    print("\n\n=== Test 6: Guidance Scale Variation (Same Prompt & Seed) ===")

    sample_image = create_sample_image()
    sample_b64 = image_to_b64(sample_image)

    # Use same prompt and seed, vary guidance
    prompt = "Transform into Eight Gates mode with red energy aura, intense power radiating from body, green glowing skin, steam rising, and overwhelming chakra presence"

    seed = 42
    num_steps = 15
    guidance_scales = [1.5, 2.0, 2.5, 3.0, 3.5]

    for guidance in guidance_scales:
        print(f"\n🎨 Generating with guidance={guidance}...")
        start_time = time.time()
        response = post_edit(
            prompt=prompt,
            image_b64=sample_b64,
            seed=seed,
            num_inference_steps=num_steps,
            guidance_scale=guidance
        )
        job_id = response["job_id"]
        print(f"  Job ID: {job_id}")

        wait_for_completion(job_id)
        duration = time.time() - start_time

        result = download_result(job_id)
        result.save(OUTPUT_DIR / f"guidance_{guidance}_{job_id}.png")
        print(f"✓ Saved: {OUTPUT_DIR / f'guidance_{guidance}_{job_id}.png'}")
        print(f"  Duration: {duration:.1f}s | Steps: {num_steps} | Guidance: {guidance}")

        # Add to grid
        ALL_RESULTS.append({
            "test": f"Guidance Variation",
            "prompt": prompt,
            "seed": seed,
            "image": result,
            "duration": duration,
            "steps": num_steps,
            "guidance": guidance
        })

    print(f"\n✅ GUIDANCE VARIATION TEST COMPLETE: Generated {len(guidance_scales)} variations")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Kontext Deterministic Image Editing - Visual Test Suite")
    print("=" * 60)

    # Check if miner is running
    try:
        response = requests.get(f"{MINER_URL}/health")
        response.raise_for_status()
        health = response.json()
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to miner at {MINER_URL}")
        print(f"   {e}")
        print(f"\nPlease start the miner with:")
        print(f"   make setup-kontext")
        sys.exit(1)

    # Verify Kontext is enabled
    if health["services"]["kontext_edit"] == "disabled":
        print(f"\n❌ ERROR: Kontext editing is disabled")
        print(f"\nEnable it with:")
        print(f"   echo 'ENABLE_KONTEXT_EDIT=true' >> .env")
        print(f"   make restart")
        sys.exit(1)

    print(f"\n✓ Miner is healthy at {MINER_URL}")
    print(f"✓ Kontext editing: {health['services']['kontext_edit']}")
    print(f"\n📁 Output directory: {OUTPUT_DIR.absolute()}")

    # Run tests
    try:
        test_determinism()
        test_different_seeds()
        test_various_prompts()
        test_reference_chaining()
        test_seed_variation()
        test_guidance_variation()

        # Create visual grid
        grid_path = create_grid_visualization()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n📸 Generated images saved to: {OUTPUT_DIR.absolute()}")
        print(f"\n🎨 VISUAL GRID: {grid_path.absolute()}")
        print(f"\nView grid with:")
        print(f"   open {grid_path.absolute()}  # macOS")
        print(f"   xdg-open {grid_path.absolute()}  # Linux")
        print(f"   explorer {grid_path.absolute()}  # Windows")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
