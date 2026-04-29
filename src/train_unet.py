import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_DIR = PROJECT_DIR / "patches"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_DIR = PATCH_DIR / "images"
MASK_DIR = PATCH_DIR / "masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


class SARDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(image_dir.glob("*.npy"))
        self.mask_paths = sorted(mask_dir.glob("*.npy"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.float32)

        image = torch.tensor(image)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


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


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

    return 1.0 - dice


def iou_score(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


dataset = SARDataset(IMAGE_DIR, MASK_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = UNet().to(DEVICE)

bce_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 15

best_val_iou = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)

        loss = bce_loss(logits, masks) + dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            val_iou += iou_score(logits, masks)

    train_loss = train_loss / len(train_loader)
    val_iou = val_iou / len(val_loader)

    print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val IoU={val_iou:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), OUTPUT_DIR / "unet_model.pth")
        print("Saved best UNet model.")

print("Training finished. Best Val IoU:", best_val_iou)


# Save one sample prediction image
model.load_state_dict(torch.load(OUTPUT_DIR / "unet_model.pth", map_location=DEVICE))
model.eval()

image, mask = val_dataset[0]
image_input = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = torch.sigmoid(model(image_input)).cpu().squeeze().numpy()

pred_mask = (pred > 0.5).astype(np.uint8)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("SAR Patch")
plt.imshow(image[0].numpy(), cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask.squeeze().numpy(), cmap="gray")

plt.subplot(1, 3, 3)
plt.title("UNet Prediction")
plt.imshow(pred_mask, cmap="gray")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "unet_sample_result.png", dpi=150)
plt.show()