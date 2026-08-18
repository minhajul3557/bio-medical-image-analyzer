"""
src/eda.py
-----------
Task 1 Data Preparation & Exploratory Data Analysis (EDA).
- Loads synthetic stained-nuclei dataset images and masks.
- Converts images to grayscale / blue intensity channel.
- Generates representative sample plots & intensity histograms across density regimes.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

DATASET_DIR = "nuclei_dataset"
FIGURES_DIR = "results/figures"

def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata.csv with {len(df)} entries.")
    
    # Dataset statistics by split and density
    print("\n--- Split Summary ---")
    print(df.groupby('split')['n_objects'].describe())
    print("\n--- Density Regime Summary ---")
    print(df.groupby('density')['n_objects'].describe())
    
    # Select representative samples for 4 regimes: sparse, normal, dense, clustered
    regimes = ['sparse', 'normal', 'dense', 'clustered']
    sample_rows = {}
    for r in regimes:
        sample_rows[r] = df[(df['split'] == 'train') & (df['density'] == r)].iloc[0]
        
    # Plot 4x4 Grid: Raw Image, Grayscale/Blue Channel, GT Mask, Intensity Histogram
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    fig.suptitle("Exploratory Data Analysis: Stained-Nuclei Fluorescence Dataset", fontsize=16, fontweight='bold')
    
    all_fg_intensities = []
    all_bg_intensities = []
    
    for idx, r in enumerate(regimes):
        row = sample_rows[r]
        img_id = row['image_id']
        img_path = os.path.join(DATASET_DIR, row['split'], 'images', f"{img_id}.png")
        mask_path = os.path.join(DATASET_DIR, row['split'], 'masks', f"{img_id}.png")
        
        rgb = imageio.imread(img_path)
        mask = imageio.imread(mask_path)
        
        # Grayscale / Blue-channel extraction (DAPI stain signal)
        gray = rgb[..., 2] / 255.0  # Normalized blue channel
        binary_mask = (mask > 128).astype(np.uint8)
        
        fg_vals = gray[binary_mask == 1]
        bg_vals = gray[binary_mask == 0]
        all_fg_intensities.extend(fg_vals)
        all_bg_intensities.extend(bg_vals)
        
        # Column 0: RGB Input Image
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"Regime: {r.capitalize()}\nImage: {img_id} ({row['n_objects']} nuclei)")
        axes[idx, 0].axis('off')
        
        # Column 1: Grayscale (Blue Channel)
        axes[idx, 1].imshow(gray, cmap='gray')
        axes[idx, 1].set_title(f"Grayscale (Blue Ch)\n256x256")
        axes[idx, 1].axis('off')
        
        # Column 2: Ground Truth Mask
        axes[idx, 2].imshow(binary_mask, cmap='Blues')
        axes[idx, 2].set_title(f"GT Binary Mask\nArea Frac: {row['area_fraction']:.3f}")
        axes[idx, 2].axis('off')
        
        # Column 3: Intensity Histogram
        axes[idx, 3].hist(bg_vals, bins=30, alpha=0.6, color='gray', label='Background', density=True)
        axes[idx, 3].hist(fg_vals, bins=30, alpha=0.7, color='blue', label='Nuclei (FG)', density=True)
        axes[idx, 3].set_title(f"Intensity Distribution")
        axes[idx, 3].set_xlabel("Pixel Intensity")
        axes[idx, 3].set_ylabel("Density")
        axes[idx, 3].legend(loc='upper right', fontsize=8)
        
    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "eda_samples_and_histograms.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved EDA plot to {out_png}")
    
    # Save overall dataset summary stats to JSON
    summary_stats = {
        "total_images": len(df),
        "splits": df['split'].value_counts().to_dict(),
        "regimes": df['density'].value_counts().to_dict(),
        "mean_nuclei_per_image": float(df['n_objects'].mean()),
        "min_nuclei": int(df['n_objects'].min()),
        "max_nuclei": int(df['n_objects'].max()),
        "mean_area_fraction": float(df['area_fraction'].mean())
    }
    print("Summary Stats:", summary_stats)
    return summary_stats

if __name__ == "__main__":
    run_eda()
