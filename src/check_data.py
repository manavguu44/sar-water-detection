import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sar_path = DATA_DIR / "sar_image.tif"
mask_path = DATA_DIR / "water_mask.tif"

# Check files exist
if not sar_path.exists():
    raise FileNotFoundError(f"SAR image not found: {sar_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Water mask not found: {mask_path}")
print("Files found. Starting to read images...")
# Load SAR image
# Load only a small preview window of SAR image
with rasterio.open(sar_path) as src:
    print("SAR width:", src.width)
    print("SAR height:", src.height)
    sar = src.read(1, window=((0, 1000), (0, 1000)))
    sar_profile = src.profile

# Load water mask
# Load only a small preview window of water mask
with rasterio.open(mask_path) as src:
    print("Mask width:", src.width)
    print("Mask height:", src.height)
    mask = src.read(1, window=((0, 1000), (0, 1000)))
    mask_profile = src.profile

print("SAR shape:", sar.shape)
print("Mask shape:", mask.shape)

print("SAR min:", np.nanmin(sar))
print("SAR max:", np.nanmax(sar))

print("Mask unique values:", np.unique(mask))

# Show both images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("SAR Image")
plt.imshow(sar, cmap="gray")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Water Mask")
plt.imshow(mask, cmap="gray")
plt.colorbar()

plt.tight_layout()

OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / "check_data.png", dpi=150)
plt.show()