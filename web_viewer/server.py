#!/usr/bin/env python3
"""
FastAPI server for Cryo-EM 3D Image Viewer

Serves denoised micrograph images for the THREE.js viewer.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

app = FastAPI(title="Cryo-EM 3D Image Viewer")

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()

cached_data = {}


def find_data_dir():
    """Find directory with images."""
    possible_dirs = [
        PROJECT_DIR / "sample_data",
        PROJECT_DIR / "data" / "denoised",
        PROJECT_DIR / "data" / "particles",
    ]
    
    for d in possible_dirs:
        if d.exists():
            pngs = list(d.glob("*.png"))
            if pngs:
                return d
    
    # Fallback
    fallback = PROJECT_DIR / "sample_data"
    fallback.mkdir(exist_ok=True)
    return fallback


def load_data():
    """Find micrograph images."""
    global cached_data
    
    data_dir = find_data_dir()
    micrograph_files = sorted(data_dir.glob("*.png"))
    
    cached_data['data_dir'] = data_dir
    cached_data['micrograph_files'] = [str(f) for f in micrograph_files]
    cached_data['n_micrographs'] = len(micrograph_files)
    
    print(f"Found {len(micrograph_files)} micrograph images in {data_dir}:")
    for f in micrograph_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.1f} MB)")


@app.on_event("startup")
async def startup():
    load_data()


@app.get("/")
async def root():
    return FileResponse(SCRIPT_DIR / "index.html")


@app.get("/metadata")
async def get_metadata():
    return {
        "n_micrographs": cached_data.get('n_micrographs', 0),
        "data_dir": str(cached_data.get('data_dir', '')),
    }


@app.get("/image_names")
async def get_image_names():
    return JSONResponse(cached_data.get('micrograph_files', []))


@app.get("/micrograph/{index}")
async def get_micrograph(index: int):
    """Serve micrograph image by index."""
    files = cached_data.get('micrograph_files', [])
    if 0 <= index < len(files):
        filepath = Path(files[index])
        if filepath.exists():
            return FileResponse(filepath, media_type="image/png")
    return JSONResponse({"error": f"Image {index} not found"}, status_code=404)


if __name__ == "__main__":
    load_data()  # Load data before printing
    
    print("\n" + "="*60)
    print("  CRYO-EM 3D IMAGE VIEWER")
    print("="*60)
    print(f"\nData directory: {cached_data.get('data_dir')}")
    print(f"Images found: {cached_data.get('n_micrographs')}")
    print(f"\nStarting server at http://localhost:8000")
    print("\nControls:")
    print("  Scroll: Zoom in/out (move through 3D space)")
    print("  Left drag: Rotate view")
    print("  Right drag: Pan")
    print("  Left/Right arrows: Navigate frames")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
