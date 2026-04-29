import rasterio
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

model = joblib.load(OUTPUT_DIR / "random_forest_model.joblib")

with rasterio.open(DATA_DIR / "mumbai_vv_vh.tif") as src:
    sar = src.read(1)
    profile = src.profile

with rasterio.open(DATA_DIR / "mumbai_mask.tif") as src:
    mask = src.read(1)

mask = (mask > 0).astype(np.uint8)

sar = np.nan_to_num(sar, nan=-25)
sar = (sar + 25) / 25
sar = np.clip(sar, 0, 1)

X = sar.reshape(-1, 1)
y = mask.reshape(-1)

pred = model.predict(X).reshape(sar.shape)

print("RF Mumbai Accuracy:", accuracy_score(y, pred.reshape(-1)))
print("RF Mumbai F1:", f1_score(y, pred.reshape(-1)))
print("RF Mumbai IoU:", jaccard_score(y, pred.reshape(-1)))

with rasterio.open(OUTPUT_DIR / "mumbai_rf_prediction.tif", "w", **profile) as dst:
    dst.write(pred.astype(np.uint8), 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Mumbai SAR VV")
plt.imshow(sar, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mumbai Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("RF Prediction")
plt.imshow(pred, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mumbai_rf_result.png", dpi=160)
plt.close()

print("Saved Mumbai RF result.")