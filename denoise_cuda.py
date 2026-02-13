#!/usr/bin/env python3
"""
Cryo-EM Micrograph Denoiser - CUDA Fortran Engine

Thin Python orchestrator that calls the CUDA Fortran denoising engine
for each MRC file. The engine handles MRC reading, frame averaging,
CNN inference (cuDNN), contrast stretching, and PGM output. This
script handles file discovery, range selection, and PGM-to-PNG conversion.

Usage:
    python denoise_cuda.py /path/to/movies/           # Denoise all, output to sample_data/
    python denoise_cuda.py /path/to/movie.mrc         # Denoise single file
    python denoise_cuda.py /path/to/movies/ -n 10     # First 10 files
    python denoise_cuda.py /path/to/movies/ -r 6:10   # Files 6 through 10
    python denoise_cuda.py /path/to/movies/ --skip-existing
    python denoise_cuda.py /path/to/movies/ -o /custom/dir/  # Custom output

Requirements:
    - cryo_denoise_engine binary (run 'make' to build)
    - Pillow (for PGM to PNG conversion)
    - Pre-trained model weights in models/ directory
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Denoise cryo-EM micrographs using CUDA Fortran engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python denoise_cuda.py /data/movies/                  # All .mrc files
  python denoise_cuda.py /data/movies/sample.mrc        # Single file
  python denoise_cuda.py /data/movies/ -n 5             # First 5 files
  python denoise_cuda.py /data/movies/ -r 6:10          # Files 6 through 10
  python denoise_cuda.py /data/movies/ --skip-existing  # Skip already processed
        """,
    )
    parser.add_argument("input", type=str, help="Input .mrc file or directory")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./sample_data for viewer)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=None,
        help="Number of files to process (default: all)",
    )
    parser.add_argument(
        "-r",
        "--range",
        type=str,
        default=None,
        help="File range, 1-indexed (e.g. 2:5 for files 2-5)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help="Path to model weights directory"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have output PNGs",
    )

    args = parser.parse_args()

    # Find engine binary
    script_dir = Path(__file__).parent.resolve()
    engine = script_dir / "cryo_denoise_engine"

    # Default output to sample_data/ so viewer picks up new images
    if args.output is None:
        args.output = str(script_dir / "sample_data")
    if not engine.exists():
        print(f"Error: Engine binary not found at {engine}")
        print("Run 'make' in the project directory to build it.")
        sys.exit(1)

    # Find model directory
    if args.model:
        model_dir = Path(args.model)
    elif (script_dir / "models").exists():
        model_dir = script_dir / "models"
    else:
        print("Error: Model weights not found. Specify with -m or place in ./models/")
        sys.exit(1)

    if not (model_dir / "conv01_weights.bin").exists():
        print(f"Error: Model weights not found in {model_dir}")
        sys.exit(1)

    # Find input files
    input_path = Path(args.input)
    if input_path.is_file():
        mrc_files = [input_path]
    elif input_path.is_dir():
        mrc_files = sorted(input_path.glob("*.mrc"))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not mrc_files:
        print(f"No .mrc files found in {input_path}")
        sys.exit(1)

    # Apply range selection
    if args.range:
        parts = args.range.split(":")
        start = int(parts[0]) - 1  # Convert to 0-indexed
        end = int(parts[1]) if len(parts) > 1 else start + 1
        mrc_files = mrc_files[start:end]
    elif args.num:
        mrc_files = mrc_files[: args.num]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-processed files
    if args.skip_existing:
        original_count = len(mrc_files)
        mrc_files = [
            f
            for f in mrc_files
            if not (
                output_dir / (f.stem.replace(".frames", "") + "_denoised.png")
            ).exists()
        ]
        skipped = original_count - len(mrc_files)
        if skipped > 0:
            print(f"Skipping {skipped} already-processed file(s)")

    if not mrc_files:
        print("No files to process.")
        return

    print(f"Processing {len(mrc_files)} micrograph(s) with CUDA Fortran engine")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")

    total_start = time.time()

    for i, mrc_path in enumerate(mrc_files):
        print(f"\n[{i + 1}/{len(mrc_files)}] {mrc_path.name}")

        out_stem = mrc_path.stem.replace(".frames", "") + "_denoised"
        pgm_path = output_dir / (out_stem + ".pgm")
        png_path = output_dir / (out_stem + ".png")

        # Run CUDA Fortran engine
        t0 = time.time()
        result = subprocess.run(
            [str(engine), str(model_dir), str(mrc_path), str(pgm_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: Engine failed (exit code {result.returncode})")
            if result.stderr:
                print(f"  {result.stderr.strip()}")
            continue

        # Print engine output (indented)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("==="):
                print(f"  {line}")

        # Convert PGM to PNG
        if pgm_path.exists():
            img = Image.open(pgm_path)
            img.save(png_path)
            pgm_path.unlink()  # Remove intermediate PGM
            size_mb = png_path.stat().st_size / 1024 / 1024
            elapsed = time.time() - t0
            print(f"  Saved: {png_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        else:
            print(f"  ERROR: PGM output not created")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done! {len(mrc_files)} image(s) in {total_elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
