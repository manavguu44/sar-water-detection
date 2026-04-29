import numpy as np
import rasterio
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

# -------- Paths --------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sar_path = DATA_DIR / "delhi_temporal_vv_vh.tif"

MODEL_PATH = OUTPUT_DIR / "temporal_unet_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Same UNet definition --------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(8, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        mid = self.middle(p3)

        u3 = self.up3(mid)
        x = torch.cat([u3, c3], dim=1)
        x = self.conv3(x)

        u2 = self.up2(x)
        x = torch.cat([u2, c2], dim=1)
        x = self.conv2(x)

        u1 = self.up1(x)
        x = torch.cat([u1, c1], dim=1)
        x = self.conv1(x)

        return self.out(x)


# -------- Load model --------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")

# -------- Load SAR --------
with rasterio.open(sar_path) as src:
    sar = src.read()
    profile = src.profile

sar = np.nan_to_num(sar, nan=-25)

sar = (sar + 25) / 25
sar = np.clip(sar, 0, 1)

channels, height, width = sar.shape

PATCH_SIZE = 128

prediction = np.zeros((height, width))

print("Running full image inference...")

for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
    for x in range(0, width - PATCH_SIZE, PATCH_SIZE):
        patch = sar[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        patch_tensor = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(model(patch_tensor)).cpu().numpy()[0, 0]

        pred_mask = (pred > 0.5).astype(np.uint8)

        prediction[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_mask


# -------- Save result --------
        pred_mask = (pred > 0.5).astype(np.uint8)

        prediction[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_mask


# -------- Post-processing --------
import cv2

print("Applying post-processing...")

prediction = prediction.astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)

# Remove small noise
prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)

# Fill small gaps
prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)


# -------- Save result --------
profile.update(dtype=rasterio.uint8, count=1)

output_path = OUTPUT_DIR / "temporal_unet_full_prediction.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(prediction.astype(np.uint8), 1)

print("Saved full prediction:", output_path)


# -------- Visualization --------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("SAR Image")
plt.imshow(sar[0], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Temporal UNet Full Prediction")
plt.imshow(prediction, cmap="gray")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "unet_full_result.png", dpi=150)
plt.close()