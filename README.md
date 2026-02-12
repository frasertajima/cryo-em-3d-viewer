# Cryo-EM 3D Micrograph Viewer

An interactive 3D viewer for denoised cryo-EM micrographs, enabling intuitive exploration of protein structures through depth perception and motion parallax.

![Cryo-EM Viewer](docs/viewer_screenshot.png)

## Key Achievement

This project demonstrates that **viewing cryo-EM data in 3D space significantly enhances structure visibility** - even for inherently 2D projection images. By moving the image through 3D space, the visual cortex engages differently than with static 2D viewing, making it easier to distinguish signal from noise and identify protein structures.

### What We Built

1. **Noise2Noise Denoising Pipeline**: A 12-layer residual CNN (93,089 parameters) trained using the Noise2Noise framework on real cryo-EM data from EMPIAR-10025 (T20S Proteasome).

2. **Interactive 3D Viewer**: A THREE.js-based web application that renders full-resolution micrographs (7676×7420 pixels) as 3D planes, enabling:
   - **Depth-based exploration**: Scroll to move closer/further from the image
   - **3D rotation**: Drag to rotate the viewing angle
   - **Motion parallax**: The movement through 3D space acts as a natural filter, making structures "pop" visually
   - **Multi-frame navigation**: Browse through multiple micrographs with arrow keys

3. **Streamlined Workflow**: Simple command-line tools to denoise new micrographs and view them immediately.

### Structures Visible

In the denoised T20S Proteasome micrographs, several distinct structures become clearly visible:
- **Salt/ice crystals**: Rectangular block-like structures
- **Proteasome particles**: Barrel-shaped complexes (~150 pixels diameter)
- **Various conformations**: Different orientations and states of the protein complex

## Quick Start

### View Sample Data

```bash
# Clone the repository
git clone https://github.com/yourusername/cryo-em-3d-viewer.git
cd cryo-em-3d-viewer

# Install dependencies
pip install torch mrcfile pillow fastapi uvicorn

# Start the viewer (uses sample_data/ by default)
python web_viewer/server.py

# Open http://localhost:8000 in your browser
```

### Denoise Your Own Data

```bash
# Denoise MRC movie files
python denoise.py /path/to/movies/ -o ./my_denoised/

# Denoise specific files
python denoise.py /path/to/movie.mrc

# Denoise first N files
python denoise.py /path/to/movies/ -n 5

# View results
cp my_denoised/*.png sample_data/
python web_viewer/server.py
```

## Controls

| Action | Control |
|--------|---------|
| Zoom in/out | Scroll wheel |
| Rotate view | Left-click + drag |
| Pan | Right-click + drag |
| Next frame | Right arrow / Next button |
| Previous frame | Left arrow / Prev button |
| Adjust brightness | Slider in control panel |
| Adjust contrast | Slider in control panel |

## Project Structure

```
cryo-em-3d-viewer/
├── denoise.py              # Main denoising script
├── web_viewer/
│   ├── server.py           # FastAPI server
│   └── index.html          # THREE.js viewer
├── sample_data/            # Sample full-resolution images (~53MB each)
├── models/                 # Pre-trained model weights
│   ├── conv01_weights.bin
│   ├── conv01_bias.bin
│   └── ...
└── tools/
    └── generate_denoised_images.py  # Batch processing script
```

## Technical Details

### Denoising Model

- **Architecture**: 12-layer residual CNN
- **Training**: Noise2Noise framework (no clean targets needed)
- **Parameters**: 93,089
- **Input**: 64×64 patches with 48-pixel overlap
- **Normalization**: Mean/std normalization per image
- **Inference**: Batched GPU processing (~220K patches per 7676×7420 image)

### Data

- **Source**: EMPIAR-10025 (T20S Proteasome)
- **Resolution**: 7676 × 7420 pixels per micrograph
- **Pixel size**: ~1.5 Å/pixel
- **Particle size**: ~150 pixels diameter (~700 kDa complex)

### Viewer

- **Framework**: THREE.js with OrbitControls
- **Backend**: FastAPI + Uvicorn
- **Features**: Real-time brightness/contrast adjustment, smooth 3D navigation
- **Performance**: Handles 50+ MB images at full resolution

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- mrcfile
- Pillow
- FastAPI
- Uvicorn

```bash
pip install torch mrcfile pillow fastapi uvicorn
```

## Why 3D Viewing Works

The human visual system is remarkably good at detecting patterns through motion. When viewing a 2D image in 3D space:

1. **Motion parallax**: Moving the viewpoint creates subtle depth cues that help separate foreground (signal) from background (noise)
2. **Active perception**: The act of moving through the data engages different visual processing pathways
3. **Variable viewing angles**: Slight rotations can reveal structures that are invisible from a single fixed viewpoint
4. **Depth-based filtering**: Zooming in/out naturally adjusts the spatial frequency content being examined

This is similar to how cryo-EM practitioners often "rock" physical micrographs or use stereo viewers to better see structures.

## Acknowledgments

- EMPIAR-10025 dataset (T20S Proteasome) from the Bharat Lab
- Noise2Noise framework: Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data"

## License

MIT License - See LICENSE file for details.
