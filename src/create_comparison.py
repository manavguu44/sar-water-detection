import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sar_path = DATA_DIR / "sar_vv_vh.tif"
rf_path = OUTPUT_DIR / "prediction_mask.tif"
unet_vv_vh_path = OUTPUT_DIR / "unet_full_prediction.tif"

with rasterio.open(sar_path) as src:
    sar = src.read(1)

with rasterio.open(rf_path) as src:
    rf = src.read(1)

with rasterio.open(unet_vv_vh_path) as src:
    unet_vv_vh = src.read(1)

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.title("Sentinel-1 SAR")
plt.imshow(sar, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Random Forest Prediction")
plt.imshow(rf, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("UNet VV + VH Prediction")
plt.imshow(unet_vv_vh, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
plt.close()

print("Saved:", OUTPUT_DIR / "model_comparison.png")