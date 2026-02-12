#!/usr/bin/env python3
"""
FastAPI server for Cryo-EM 3D Image Viewer

Serves denoised micrograph images for the THREE.js viewer.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Cryo-EM 3D Image Viewer")

# Data paths - check multiple locations
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Priority: sample_data > data/denoised > data/particles
POSSIBLE_DIRS = [
    PROJECT_DIR / "sample_data",
    PROJECT_DIR / "data" / "denoised", 
    PROJECT_DIR / "data" / "particles",
]

DATA_DIR = None
for d in POSSIBLE_DIRS:
    if d.exists() and list(d.glob("*.png")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    DATA_DIR = PROJECT_DIR / "sample_data"
    DATA_DIR.mkdir(exist_ok=True)

# Cache
cached_data = {}


def load_data():
    """Find micrograph images."""
    global cached_data
    
    micrograph_files = sorted(DATA_DIR.glob("*.png"))
    
    cached_data['micrograph_files'] = [str(f) for f in micrograph_files]
    cached_data['n_micrographs'] = len(micrograph_files)
    
    print(f"Found {len(micrograph_files)} micrograph images in {DATA_DIR}:")
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
        "data_dir": str(DATA_DIR),
    }


@app.get("/image_names")
async def get_image_names():
    return JSONResponse(cached_data.get('micrograph_files', []))


@app.get("/micrograph/{index}")
async def get_micrograph(index: int):
    """Serve micrograph image by index."""
    files = cached_data.get('micrograph_files', [])
    if 0 <= index < len(files):
        return FileResponse(files[index], media_type="image/png")
    return JSONResponse({"error": "Image not found"}, status_code=404)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CRYO-EM 3D IMAGE VIEWER")
    print("="*60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"\nStarting server at http://localhost:8000")
    print("\nControls:")
    print("  Scroll: Zoom in/out (move through 3D space)")
    print("  Left drag: Rotate view")
    print("  Right drag: Pan")
    print("  Left/Right arrows: Navigate frames")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
