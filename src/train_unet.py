"""
src/train_unet.py
------------------
Task 3: Train and evaluate U-Net segmentation network on nuclei_dataset.
- Trains U-Net on 80 train images, validates on 20 val images.
- Implements loss ablation: BCE vs Dice vs Combined (BCE + Dice).
- Evaluates Mean Dice and Mean IoU.
- Generates validation side-by-side panel figures and Otsu comparison.
"""
import os
os.environ["MPLCONFIGDIR"] = os.path.abspath("results/.matplotlib")
import json
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import binary_fill_holes

from unet import NucleiDataset, UNet, DiceLoss, BCEDiceLoss, calculate_metrics

DATASET_DIR = "nuclei_dataset"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks, _ in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(logits)
            for i in range(imgs.size(0)):
                d, iou = calculate_metrics(probs[i], masks[i])
                dices.append(d)
                ious.append(iou)
    return total_loss / len(dataloader.dataset), np.mean(dices), np.mean(ious)

def run_training_ablation(epochs=25, lr=1e-3, batch_size=8):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")
    
    train_dataset = NucleiDataset(DATASET_DIR, split="train")
    val_dataset = NucleiDataset(DATASET_DIR, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    loss_configs = {
        "BCE_Loss": torch.nn.BCEWithLogitsLoss(),
        "Dice_Loss": DiceLoss(),
        "Combined_BCE_Dice": BCEDiceLoss(bce_weight=0.5)
    }
    
    ablation_results = {}
    best_overall_dice = -1.0
    best_overall_model_state = None
    best_loss_name = ""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("U-Net Loss Function Ablation & Validation Metrics", fontsize=14, fontweight='bold')
    
    for loss_name, criterion in loss_configs.items():
        print(f"\n================ Training U-Net with {loss_name} ================")
        model = UNet(in_channels=1, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
        
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)
            
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
                
        ablation_results[loss_name] = {
            "final_val_loss": float(history["val_loss"][-1]),
            "final_val_dice": float(history["val_dice"][-1]),
            "final_val_iou": float(history["val_iou"][-1]),
            "max_val_dice": float(max(history["val_dice"])),
            "history": history
        }
        
        if max(history["val_dice"]) > best_overall_dice:
            best_overall_dice = max(history["val_dice"])
            best_overall_model_state = model.state_dict()
            best_loss_name = loss_name
            
        ax1.plot(range(1, epochs + 1), history["val_loss"], label=f"Val Loss ({loss_name})")
        ax2.plot(range(1, epochs + 1), history["val_dice"], label=f"Val Dice ({loss_name})")
        
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Validation Dice Coefficient")
    ax2.set_title("Validation Dice Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(FIGURES_DIR, "unet_training_curves.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training curves to {curves_path}")
    
    # Save best model
    best_model_path = os.path.join(RESULTS_DIR, "unet_best_model.pth")
    torch.save(best_overall_model_state, best_model_path)
    print(f"Saved best U-Net model ({best_loss_name}, Best Dice: {best_overall_dice:.4f}) to {best_model_path}")
    
    # Save ablation summary JSON
    summary_path = os.path.join(RESULTS_DIR, "unet_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved ablation summary to {summary_path}")
    
    # Visualize Side-by-Side Predictions for Validation Images
    visualize_val_predictions(best_overall_model_state, device, val_dataset)
    compare_otsu_vs_unet(best_overall_model_state, device, val_dataset)
    
    return ablation_results

def visualize_val_predictions(model_state, device, val_dataset):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Pick 4 representative validation images across sparse, normal, dense, clustered
    indices = [0, 5, 10, 15] if len(val_dataset) >= 16 else list(range(min(4, len(val_dataset))))
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.5 * len(indices)))
    fig.suptitle("U-Net Validation Predictions: Input | Ground Truth Mask | Predicted Mask", fontsize=14, fontweight='bold')
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, mask_tensor, filename = val_dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)
            logit = model(img_input)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_mask = (prob >= 0.5).astype(np.float32)
            
            gt = mask_tensor.squeeze().numpy()
            raw_img = img_tensor.squeeze().numpy()
            
            dice, iou = calculate_metrics(torch.from_numpy(pred_mask), torch.from_numpy(gt))
            
            axes[i, 0].imshow(raw_img, cmap='gray')
            axes[i, 0].set_title(f"Input Image\n{filename}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt, cmap='Blues')
            axes[i, 1].set_title(f"Ground Truth Mask")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='Blues')
            axes[i, 2].set_title(f"Predicted Mask\nDice: {dice:.3f} | IoU: {iou:.3f}")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    val_fig_path = os.path.join(FIGURES_DIR, "unet_val_predictions.png")
    plt.savefig(val_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved validation prediction panels to {val_fig_path}")

def compare_otsu_vs_unet(model_state, device, val_dataset):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Pick 2 representative images: one where Otsu struggles (clustered), one where Otsu does well
    indices = [1, 11] if len(val_dataset) > 11 else [0, 1]
    
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    fig.suptitle("Segmentation Comparison: Raw Image | Ground Truth | Classical Otsu | Learned U-Net", fontsize=14, fontweight='bold')
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, mask_tensor, filename = val_dataset[idx]
            raw_gray = img_tensor.squeeze().numpy()
            gt = mask_tensor.squeeze().numpy()
            
            # Classical Otsu
            thresh = threshold_otsu(raw_gray)
            otsu_bin = raw_gray > thresh
            otsu_bin = binary_opening(otsu_bin, disk(1))
            otsu_bin = binary_fill_holes(otsu_bin).astype(np.float32)
            otsu_dice, otsu_iou = calculate_metrics(torch.from_numpy(otsu_bin), torch.from_numpy(gt))
            
            # Learned U-Net
            img_input = img_tensor.unsqueeze(0).to(device)
            logit = model(img_input)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            unet_bin = (prob >= 0.5).astype(np.float32)
            unet_dice, unet_iou = calculate_metrics(torch.from_numpy(unet_bin), torch.from_numpy(gt))
            
            axes[i, 0].imshow(raw_gray, cmap='gray')
            axes[i, 0].set_title(f"Raw Input: {filename}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt, cmap='Blues')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(otsu_bin, cmap='Blues')
            axes[i, 2].set_title(f"Classical Otsu\nDice: {otsu_dice:.3f} | IoU: {otsu_iou:.3f}")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(unet_bin, cmap='Blues')
            axes[i, 3].set_title(f"Learned U-Net\nDice: {unet_dice:.3f} | IoU: {unet_iou:.3f}")
            axes[i, 3].axis('off')
            
    plt.tight_layout()
    comp_fig_path = os.path.join(FIGURES_DIR, "otsu_vs_unet_comparison.png")
    plt.savefig(comp_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Otsu vs U-Net comparison to {comp_fig_path}")

if __name__ == "__main__":
    run_training_ablation(epochs=15)
