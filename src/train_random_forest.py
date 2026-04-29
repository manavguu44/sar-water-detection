print("TRAIN FILE STARTED")
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sar_path = DATA_DIR / "sar_image.tif"
mask_path = DATA_DIR / "water_mask.tif"

print("Loading data...")

# Load full SAR image
with rasterio.open(sar_path) as src:
    sar = src.read(1)
    profile = src.profile

# Load full mask
with rasterio.open(mask_path) as src:
    mask = src.read(1)

print("SAR shape:", sar.shape)
print("Mask shape:", mask.shape)

# Make sure mask is 0 and 1
mask = (mask > 0).astype(np.uint8)

# Remove invalid SAR pixels
valid_pixels = np.isfinite(sar)

sar_valid = sar[valid_pixels]
mask_valid = mask[valid_pixels]

# Normalize SAR values
sar_norm = (sar_valid + 25) / 25
sar_norm = np.clip(sar_norm, 0, 1)

X = sar_norm.reshape(-1, 1)
y = mask_valid.reshape(-1)

print("Total valid pixels:", len(y))
print("Water pixels:", np.sum(y == 1))
print("Non-water pixels:", np.sum(y == 0))

# Use sample so training is not too slow
sample_size = min(200000, len(y))
indices = np.random.choice(len(y), sample_size, replace=False)

X_sample = X[indices]
y_sample = y[indices]

print("Training sample size:", len(y_sample))

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating...")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("IoU:", jaccard_score(y_test, y_pred))

print("Predicting full image...")

sar_full_norm = (sar + 25) / 25
sar_full_norm = np.clip(sar_full_norm, 0, 1)

X_full = sar_full_norm.reshape(-1, 1)
pred_full = model.predict(X_full).reshape(sar.shape)

# Save model
joblib.dump(model, OUTPUT_DIR / "random_forest_model.joblib")

# Save prediction GeoTIFF
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open(OUTPUT_DIR / "prediction_mask.tif", "w", **profile) as dst:
    dst.write(pred_full.astype(np.uint8), 1)

# Save visual result
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("SAR Image")
plt.imshow(sar, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Original Water Mask")
plt.imshow(mask, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("RF Prediction")
plt.imshow(pred_full, cmap="gray")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "random_forest_result.png", dpi=150)
plt.show()

print("Done.")
print("Saved model:", OUTPUT_DIR / "random_forest_model.joblib")
print("Saved prediction:", OUTPUT_DIR / "prediction_mask.tif")
print("Saved image:", OUTPUT_DIR / "random_forest_result.png")