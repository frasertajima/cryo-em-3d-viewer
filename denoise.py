#!/usr/bin/env python3
"""
Cryo-EM Micrograph Denoiser

Simple workflow to denoise cryo-EM micrographs using a pre-trained
Noise2Noise residual CNN model.

Usage:
    python denoise.py /path/to/movies/           # Denoise all .mrc files
    python denoise.py /path/to/movie.mrc         # Denoise single file
    python denoise.py /path/to/movies/ -n 10     # Denoise first 10 files
    python denoise.py /path/to/movies/ -o /output/dir/

Requirements:
    - PyTorch with CUDA support
    - mrcfile
    - PIL/Pillow
    - Pre-trained model weights in models/ directory
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Optional imports with helpful error messages
try:
    import mrcfile
except ImportError:
    print("Error: mrcfile not installed. Run: pip install mrcfile")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


class ResidualCNN12(nn.Module):
    """12-layer residual CNN for cryo-EM denoising (93,089 parameters)."""
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=1)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1) for _ in range(10)
        ])
        self.conv12 = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        for conv in self.conv_layers:
            out = self.relu(conv(out))
        noise_pred = self.conv12(out)
        return identity - noise_pred


def load_model(model_dir, device):
    """Load model weights from binary files."""
    model = ResidualCNN12().to(device)
    model_dir = Path(model_dir)
    
    # Load conv1
    w = np.fromfile(model_dir / 'conv01_weights.bin', dtype=np.float32).reshape(32, 1, 3, 3)
    model.conv1.weight.data = torch.from_numpy(w).to(device)
    b = np.fromfile(model_dir / 'conv01_bias.bin', dtype=np.float32)
    model.conv1.bias.data = torch.from_numpy(b).to(device)
    
    # Load middle layers
    for i, conv in enumerate(model.conv_layers):
        layer_num = i + 2
        w = np.fromfile(model_dir / f'conv{layer_num:02d}_weights.bin', dtype=np.float32).reshape(32, 32, 3, 3)
        conv.weight.data = torch.from_numpy(w).to(device)
        b = np.fromfile(model_dir / f'conv{layer_num:02d}_bias.bin', dtype=np.float32)
        conv.bias.data = torch.from_numpy(b).to(device)
    
    # Load conv12
    w = np.fromfile(model_dir / 'conv12_weights.bin', dtype=np.float32).reshape(1, 32, 3, 3)
    model.conv12.weight.data = torch.from_numpy(w).to(device)
    b = np.fromfile(model_dir / 'conv12_bias.bin', dtype=np.float32)
    model.conv12.bias.data = torch.from_numpy(b).to(device)
    
    model.eval()
    return model


def denoise_micrograph(model, device, image, patch_size=64, overlap=48, batch_size=512):
    """Denoise a full micrograph using batched patch inference."""
    h, w = image.shape
    stride = patch_size - overlap
    
    # Normalize using mean/std
    mu = image.mean()
    std = image.std() + 1e-8
    normalized = (image - mu) / std
    
    output = np.zeros_like(normalized)
    weights = np.zeros_like(normalized)
    
    # Collect patches
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(normalized[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))
    
    # Handle edges
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patches.append(normalized[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))
    
    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patches.append(normalized[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))
    
    # Corner
    if (w - patch_size) % stride != 0 and (h - patch_size) % stride != 0:
        patches.append(normalized[h-patch_size:h, w-patch_size:w])
        coords.append((h - patch_size, w - patch_size))
    
    print(f"    Processing {len(patches)} patches...")
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            batch_coords = coords[i:i+batch_size]
            
            batch_tensor = torch.from_numpy(np.array(batch)).float().unsqueeze(1).to(device)
            denoised = model(batch_tensor).cpu().numpy()[:, 0]
            
            for j, (y, x) in enumerate(batch_coords):
                output[y:y+patch_size, x:x+patch_size] += denoised[j]
                weights[y:y+patch_size, x:x+patch_size] += 1
            
            if (i + batch_size) % 10000 < batch_size:
                print(f"      {min(i+batch_size, len(patches))}/{len(patches)}")
    
    weights = np.maximum(weights, 1)
    output = output / weights
    output = output * std + mu
    
    return output


def contrast_stretch(img, low_pct=0.5, high_pct=99.5):
    """Apply percentile-based contrast stretching."""
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)
    return (np.clip((img - low) / (high - low + 1e-8), 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Denoise cryo-EM micrographs using Noise2Noise CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python denoise.py /data/movies/                    # All .mrc files
  python denoise.py /data/movies/sample.mrc          # Single file
  python denoise.py /data/movies/ -n 5               # First 5 files
  python denoise.py /data/movies/ -o ./denoised/     # Custom output
        """
    )
    parser.add_argument('input', type=str, help='Input .mrc file or directory containing .mrc files')
    parser.add_argument('-o', '--output', type=str, default='./denoised', help='Output directory (default: ./denoised)')
    parser.add_argument('-n', '--num', type=int, default=None, help='Number of files to process (default: all)')
    parser.add_argument('-m', '--model', type=str, default=None, help='Path to model weights directory')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for GPU inference (default: 512)')
    
    args = parser.parse_args()
    
    # Find model directory
    script_dir = Path(__file__).parent
    if args.model:
        model_dir = Path(args.model)
    elif (script_dir / 'models').exists():
        model_dir = script_dir / 'models'
    else:
        print("Error: Model weights not found. Please specify with -m or place in ./models/")
        print("Download from: [repository releases]")
        sys.exit(1)
    
    # Check for model files
    if not (model_dir / 'conv01_weights.bin').exists():
        print(f"Error: Model weights not found in {model_dir}")
        sys.exit(1)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {model_dir}...")
    model = load_model(model_dir, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Find input files
    input_path = Path(args.input)
    if input_path.is_file():
        mrc_files = [input_path]
    elif input_path.is_dir():
        mrc_files = sorted(input_path.glob('*.mrc'))
        if args.num:
            mrc_files = mrc_files[:args.num]
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    if not mrc_files:
        print(f"No .mrc files found in {input_path}")
        sys.exit(1)
    
    print(f"\nProcessing {len(mrc_files)} micrograph(s)...")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for i, mrc_path in enumerate(mrc_files):
        print(f"\n[{i+1}/{len(mrc_files)}] {mrc_path.name}")
        
        # Load movie and average frames
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
        
        # Average frames if movie, otherwise use as-is
        if data.ndim == 3:
            avg = data.mean(axis=0)
            print(f"  Averaged {data.shape[0]} frames -> {avg.shape}")
        else:
            avg = data
            print(f"  Shape: {avg.shape}")
        
        # Denoise
        print("  Denoising...")
        denoised = denoise_micrograph(model, device, avg, batch_size=args.batch_size)
        
        # Save as PNG
        img = contrast_stretch(denoised)
        pil_img = Image.fromarray(img, mode='L')
        
        out_name = mrc_path.stem.replace('.frames', '') + '_denoised.png'
        out_path = output_dir / out_name
        pil_img.save(out_path)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print(f"\n{'='*60}")
    print(f"Done! Denoised images saved to: {output_dir}")
    print(f"\nTo view in 3D:")
    print(f"  1. Copy images to sample_data/ (or update server.py DATA_DIR)")
    print(f"  2. Run: python web_viewer/server.py")
    print(f"  3. Open: http://localhost:8000")


if __name__ == '__main__':
    main()
