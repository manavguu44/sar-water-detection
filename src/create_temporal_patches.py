import rasterio
import numpy as np
from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

PATCH_DIR = PROJECT_DIR / "temporal_patches"
IMAGE_PATCH_DIR = PATCH_DIR / "images"
MASK_PATCH_DIR = PATCH_DIR / "masks"

if PATCH_DIR.exists():
    shutil.rmtree(PATCH_DIR)

IMAGE_PATCH_DIR.mkdir(parents=True, exist_ok=True)
MASK_PATCH_DIR.mkdir(parents=True, exist_ok=True)

sar_path = DATA_DIR / "delhi_temporal_vv_vh.tif"
mask_path = DATA_DIR / "water_mask.tif"

PATCH_SIZE = 128
STRIDE = 128

print("Loading temporal SAR and mask...")

with rasterio.open(sar_path) as src:
    sar = src.read()
    print("Bands:", src.count)
    print("Width:", src.width)
    print("Height:", src.height)

with rasterio.open(mask_path) as src:
    mask = src.read(1)

if sar.shape[0] != 8:
    raise ValueError(f"Expected 8 bands, but got {sar.shape[0]} bands")

mask = (mask > 0).astype(np.uint8)

sar = np.nan_to_num(sar, nan=-25)

sar = (sar + 25) / 25
sar = np.clip(sar, 0, 1).astype(np.float32)

channels, height, width = sar.shape

count = 0

for y in range(0, height - PATCH_SIZE, STRIDE):
    for x in range(0, width - PATCH_SIZE, STRIDE):
        sar_patch = sar[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        np.save(IMAGE_PATCH_DIR / f"image_{count}.npy", sar_patch)
        np.save(MASK_PATCH_DIR / f"mask_{count}.npy", mask_patch)

        count += 1

print("Created temporal patches:", count)
print("Saved to:", PATCH_DIR)