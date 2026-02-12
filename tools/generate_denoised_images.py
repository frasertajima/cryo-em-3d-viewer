#!/usr/bin/env python3
"""
Generate denoised micrograph images using the trained N2N model.
FULL RESOLUTION - Uses GPU with batched inference for speed.
Exact same model architecture as the evaluation notebook.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import mrcfile
from PIL import Image

CRYO_V2_DIR = Path("/var/home/fraser/cryo_em_v2")
MODEL_DIR = CRYO_V2_DIR / "v33_deep_residual" / "saved_models" / "n2n_real" / "epoch_0009"
MOVIE_DIR = CRYO_V2_DIR / "14sep05c_raw_196"
OUTPUT_DIR = Path("/var/home/fraser/cryo_em_3d_reconstruction/data/denoised")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResidualCNN12(nn.Module):
    """Exact same architecture as training/notebook."""
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


def load_fortran_weights(model, checkpoint_dir):
    """Load weights from Fortran binary files - exact same as notebook."""
    checkpoint_dir = Path(checkpoint_dir)

    w = np.fromfile(checkpoint_dir / 'conv01_weights.bin', dtype=np.float32).reshape(32, 1, 3, 3)
    model.conv1.weight.data = torch.from_numpy(w).to(device)
    b = np.fromfile(checkpoint_dir / 'conv01_bias.bin', dtype=np.float32)
    model.conv1.bias.data = torch.from_numpy(b).to(device)

    for i, conv in enumerate(model.conv_layers):
        layer_num = i + 2
        w = np.fromfile(checkpoint_dir / f'conv{layer_num:02d}_weights.bin', dtype=np.float32).reshape(32, 32, 3, 3)
        conv.weight.data = torch.from_numpy(w).to(device)
        b = np.fromfile(checkpoint_dir / f'conv{layer_num:02d}_bias.bin', dtype=np.float32)
        conv.bias.data = torch.from_numpy(b).to(device)

    w = np.fromfile(checkpoint_dir / 'conv12_weights.bin', dtype=np.float32).reshape(1, 32, 3, 3)
    model.conv12.weight.data = torch.from_numpy(w).to(device)
    b = np.fromfile(checkpoint_dir / 'conv12_bias.bin', dtype=np.float32)
    model.conv12.bias.data = torch.from_numpy(b).to(device)


def denoise_full_resolution(model, image, patch_size=64, overlap=48, batch_size=512):
    """Denoise at FULL RESOLUTION using batched GPU inference."""
    h, w = image.shape
    stride = patch_size - overlap

    # Normalize
    mu = image.mean()
    std = image.std() + 1e-8
    normalized = (image - mu) / std

    output = np.zeros_like(normalized)
    weights = np.zeros_like(normalized)

    # Collect all patches and coordinates
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

    print(f"    {len(patches)} patches, batch_size={batch_size}")

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

            if (i + batch_size) % 5000 < batch_size:
                print(f"      {min(i+batch_size, len(patches))}/{len(patches)}")

    weights = np.maximum(weights, 1)
    output = output / weights
    output = output * std + mu

    return output


def contrast_stretch(img, low_pct=0.5, high_pct=99.5):
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)
    return (np.clip((img - low) / (high - low + 1e-8), 0, 1) * 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading model...")

    model = ResidualCNN12().to(device)
    load_fortran_weights(model, MODEL_DIR)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    movie_files = sorted(MOVIE_DIR.glob("*.mrc"))[:5]
    print(f"\nProcessing {len(movie_files)} movies at FULL RESOLUTION...")

    for i, mrc_path in enumerate(movie_files):
        print(f"\n[{i+1}/{len(movie_files)}] {mrc_path.name}")

        with mrcfile.open(mrc_path, permissive=True) as mrc:
            movie = mrc.data.astype(np.float32)

        avg = movie.mean(axis=0)
        print(f"  Shape: {avg.shape} (FULL RESOLUTION)")

        print("  Denoising...")
        denoised = denoise_full_resolution(model, avg, patch_size=64, overlap=48, batch_size=512)

        img = contrast_stretch(denoised)
        pil_img = Image.fromarray(img, mode='L')

        out_name = mrc_path.stem.replace('.frames', '') + '_denoised.png'
        out_path = OUTPUT_DIR / out_name
        pil_img.save(out_path)
        print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nDone! Full resolution images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
