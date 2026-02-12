# Cryo-EM 3D Explorer

Interactive 3D visualization of cryo-EM particle data using THREE.js.

## Quick Start

```bash
cd web_viewer
./START.sh
```

Then open http://localhost:8000 in your browser.

## Features

### Visualization Modes

1. **Particle Cloud** - See all 385 extracted particles as colored points in 3D space
   - Colors indicate particle class (from 2D classification)
   - Adjust particle size, opacity, and Z-spread
   - Semi-transparent planes show micrograph layers

2. **Volume Slices** - View cross-sections through the pseudo-3D reconstruction
   - Scrub through Z slices
   - Adjust density threshold

3. **Isosurface** - 3D density surface rendering
   - Adjust iso-level to show different density thresholds
   - Control surface opacity

4. **Micrograph Stack** - Combined particle + layer view

### Navigation

- **Left drag**: Rotate view (orbit around center)
- **Right drag**: Pan (move camera)
- **Scroll**: Zoom in/out
- **Double-click**: Focus on point

### Camera Presets

- Reset View: Return to isometric view
- Top View: Look down Z axis
- Front View: Look along Y axis  
- Side View: Look along X axis

### Animation

- Toggle auto-rotation
- Adjust rotation speed

## Data

The viewer loads:
- `../data/particles/particle_stack.npy` - 385 particles (256×256 each)
- `../data/3d_reconstruction/pseudo_3d_volume.npy` - 96³ pseudo-reconstruction
- `../data/3d_reconstruction/class_averages.npy` - 8 class averages

## Technical Details

- Frontend: THREE.js with OrbitControls
- Backend: FastAPI with WebSocket streaming
- Volume data sent as binary Float32Array for efficiency

## Architecture

```
Browser (THREE.js)
    ↕ WebSocket
FastAPI Server
    ↓
NumPy arrays (.npy files)
```

Based on the emergency response visualization framework (v43_interactive_studio).
