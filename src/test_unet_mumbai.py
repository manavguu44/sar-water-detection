import numpy as np
import rasterio
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sar_path = DATA_DIR / "mumbai_vv_vh.tif"
mask_path = DATA_DIR / "mumbai_mask.tif"
MODEL_PATH = OUTPUT_DIR / "unet_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 128


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
        self.down1 = DoubleConv(2, 32)
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


model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with rasterio.open(sar_path) as src:
    sar = src.read()
    profile = src.profile

with rasterio.open(mask_path) as src:
    mask = src.read(1)

mask = (mask > 0).astype(np.uint8)

sar = np.nan_to_num(sar, nan=-25)
sar = (sar + 25) / 25
sar = np.clip(sar, 0, 1).astype(np.float32)

channels, height, width = sar.shape
prediction = np.zeros((height, width), dtype=np.uint8)

for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
    for x in range(0, width - PATCH_SIZE, PATCH_SIZE):
        patch = sar[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        patch_tensor = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(model(patch_tensor)).cpu().numpy()[0, 0]

        prediction[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = (pred > 0.5).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)

y_true = mask.reshape(-1)
y_pred = prediction.reshape(-1)

print("UNet Mumbai Accuracy:", accuracy_score(y_true, y_pred))
print("UNet Mumbai F1:", f1_score(y_true, y_pred))
print("UNet Mumbai IoU:", jaccard_score(y_true, y_pred))

profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open(OUTPUT_DIR / "mumbai_unet_prediction.tif", "w", **profile) as dst:
    dst.write(prediction.astype(np.uint8), 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Mumbai SAR VV")
plt.imshow(sar[0], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mumbai Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("UNet Prediction")
plt.imshow(prediction, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mumbai_unet_result.png", dpi=160)
plt.close()

print("Saved Mumbai UNet result.")