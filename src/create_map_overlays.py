import rasterio
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

items = {
    "delhi": OUTPUT_DIR / "unet_full_prediction.tif",
    "mumbai": OUTPUT_DIR / "mumbai_unet_prediction.tif",
}

for name, path in items.items():
    with rasterio.open(path) as src:
        pred = src.read(1)

    pred = (pred > 0).astype(np.uint8)

    rgba = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)

    # Water = blue transparent
    rgba[pred == 1] = [0, 120, 255, 140]

    img = Image.fromarray(rgba, mode="RGBA")
    out_path = OUTPUT_DIR / f"{name}_water_overlay.png"
    img.save(out_path)

    print("Saved:", out_path)