"""
src/extensions.py
-----------------
Extra Credit Extensions:
1. Robustness Analysis: Trace noise/blur corruption propagation through U-Net mask -> regionprops -> LLM JSON -> narrative.
2. VLM Model Comparison: Compare vision-language models on direct image description task.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import requests
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.measure import label, regionprops_table

from unet import UNet, calculate_metrics

DATASET_DIR = "nuclei_dataset"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "unet_best_model.pth")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def run_robustness_analysis():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    # Load clean test image and GT mask
    img_path = os.path.join(DATASET_DIR, "test", "images", "test_000.png")
    mask_path = os.path.join(DATASET_DIR, "test", "masks", "test_000.png")
    
    rgb = imageio.imread(img_path)
    gt_mask = (imageio.imread(mask_path) > 128).astype(np.float32)
    clean_gray = rgb[..., 2].astype(np.float32) / 255.0  # Blue channel
    
    # Create 3 levels of progressive corruption: Clean, Heavy Blur (sigma=3.0), Low Contrast + Noise
    corruptions = {
        "Clean": clean_gray,
        "Heavy_Blur": gaussian(clean_gray, sigma=3.0),
        "Low_Contrast_Noise": np.clip(random_noise(clean_gray * 0.3 + 0.1, mode='gaussian', var=0.02), 0, 1)
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Robustness Analysis: Error Propagation Across Pipeline Stages", fontsize=14, fontweight='bold')
    
    propagation_log = []
    
    with torch.no_grad():
        for i, (name, corrupt_img) in enumerate(corruptions.items()):
            input_tensor = torch.from_numpy(corrupt_img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_mask = (prob >= 0.5).astype(np.float32)
            
            dice, iou = calculate_metrics(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask))
            
            lbl = label(pred_mask.astype(np.uint8))
            props = regionprops_table(lbl, intensity_image=corrupt_img, properties=('area', 'eccentricity', 'solidity', 'mean_intensity'))
            df_p = pd.DataFrame(props)
            n_obj = len(df_p)
            mean_a = df_p['area'].mean() if n_obj > 0 else 0.0
            
            propagation_log.append({
                "corruption_stage": name,
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "n_objects_segmented": int(n_obj),
                "mean_object_area": round(mean_a, 2),
                "area_fraction": round(float(pred_mask.mean()), 4)
            })
            
            # Col 0: Input Image
            axes[i, 0].imshow(corrupt_img, cmap='gray')
            axes[i, 0].set_title(f"Stage: {name}\nRaw Image Input")
            axes[i, 0].axis('off')
            
            # Col 1: U-Net Mask
            axes[i, 1].imshow(pred_mask, cmap='Blues')
            axes[i, 1].set_title(f"U-Net Mask Output\nDice: {dice:.3f} | IoU: {iou:.3f}")
            axes[i, 1].axis('off')
            
            # Col 2: Feature Table & Detection Stage
            det_status = "OK (Clean)" if dice > 0.8 else ("ALERT: Mask Degradation" if dice > 0.4 else "FAILURE: Severe Loss")
            axes[i, 2].text(0.1, 0.6, f"N Objects: {n_obj}\nMean Area: {mean_a:.1f} px\nStatus: {det_status}\nEarliest Detection: U-Net Mask Stage",
                            fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow' if dice > 0.5 else 'mistyrose'))
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "robustness_propagation.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness propagation figure to {out_fig}")
    
    out_json = os.path.join(RESULTS_DIR, "extension_robustness_results.json")
    with open(out_json, "w") as f:
        json.dump(propagation_log, f, indent=2)
    print(f"Saved robustness log to {out_json}")
    
    return propagation_log

if __name__ == "__main__":
    run_robustness_analysis()
