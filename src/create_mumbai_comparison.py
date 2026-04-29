import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

paths = {
    "SAR VV": DATA_DIR / "mumbai_vv_vh.tif",
    "Mask": DATA_DIR / "mumbai_mask.tif",
    "RF": OUTPUT_DIR / "mumbai_rf_prediction.tif",
    "UNet": OUTPUT_DIR / "mumbai_unet_prediction.tif",
    "Temporal UNet": OUTPUT_DIR / "mumbai_temporal_unet_prediction.tif",
}

images = {}

with rasterio.open(paths["SAR VV"]) as src:
    images["SAR VV"] = src.read(1)

for key in ["Mask", "RF", "UNet", "Temporal UNet"]:
    with rasterio.open(paths[key]) as src:
        images[key] = src.read(1)

plt.figure(figsize=(22, 5))

for i, key in enumerate(images.keys(), start=1):
    plt.subplot(1, 5, i)
    plt.title(key)
    plt.imshow(images[key], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mumbai_model_comparison.png", dpi=180)
plt.close()

print("Saved:", OUTPUT_DIR / "mumbai_model_comparison.png")