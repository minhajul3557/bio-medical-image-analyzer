"""
src/unet.py
------------
Task 3: PyTorch U-Net Segmentation Network for Biomedical Image Analysis.
Implements:
- Custom NucleiDataset (PyTorch Dataset)
- Lightweight U-Net Architecture (Encoder, Bottleneck, Decoder with Skip Connections)
- Dice Loss, IoU, and Combined BCE+Dice Loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import imageio.v2 as imageio

class NucleiDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, filename)
        
        rgb = imageio.imread(img_path)
        # Blue channel intensity normalized to [0, 1]
        gray = rgb[..., 2].astype(np.float32) / 255.0
        
        mask = imageio.imread(mask_path)
        binary_mask = (mask > 128).astype(np.float32)
        
        # Convert to Tensors: [1, H, W]
        img_tensor = torch.from_numpy(gray).unsqueeze(0)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
        
        return img_tensor, mask_tensor, filename

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.poolers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Encoder
        curr_ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(curr_ch, feat))
            self.poolers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_ch = feat
            
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        rev_features = list(reversed(features))
        curr_ch = features[-1] * 2
        for feat in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(curr_ch, feat, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feat * 2, feat))
            curr_ch = feat
            
        # Output Head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Contracting Path
        for encoder, pooler in zip(self.encoders, self.poolers):
            x = encoder(x)
            skip_connections.append(x)
            x = pooler(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Expansive Path
        for idx in range(len(self.decoders)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            # Concatenate along channel dimension (dim=1)
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)
            
        return self.final_conv(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1. - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

def calculate_metrics(pred_mask, gt_mask, threshold=0.5, smooth=1e-6):
    """
    Computes Dice coefficient and IoU for binary segmentation.
    pred_mask: tensor/array probabilities or binary mask
    gt_mask: tensor/array binary mask
    """
    pred_bin = (pred_mask >= threshold).float()
    gt_bin = (gt_mask >= 0.5).float()
    
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    
    dice = (2. * intersection + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.item(), iou.item()
