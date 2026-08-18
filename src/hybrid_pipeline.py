"""
src/hybrid_pipeline.py
-----------------------
Task 4: End-to-End Hybrid Pipeline Batch Execution on Unseen Test Images.
Pipeline Flow:
Unseen Test Image -> U-Net Mask -> regionprops_table -> LLM Structured JSON -> Narrative Report.
Aggregates JSON records across all test images into a pandas DataFrame and exports to CSV.
"""
import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import requests
from skimage.measure import label, regionprops_table

from unet import UNet

DATASET_DIR = "nuclei_dataset"
RESULTS_DIR = "results"
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test", "images")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "unet_best_model.pth")
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "test_pipeline_summary.csv")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEXT_MODEL = "llama3.2:latest"

PROMPT_TEMPLATE = """You are a biomedical data analyst operating an auditable hybrid pipeline.
Below is the DETERMINISTIC REGIONPROPS MEASUREMENT RECORD extracted from U-Net segmentation of an unseen microscopy image:

Image ID: {image_id}
- Total Objects Detected (n_objects): {n_objects}
- Mean Object Area (pixels): {mean_area:.2f} (std: {std_area:.2f})
- Mean Eccentricity (0=circle, 1=line): {mean_ecc:.3f}
- Mean Solidity (convexity ratio): {mean_sol:.3f}
- Signal Mean Intensity: {mean_int:.3f}
- Image Area Fraction: {area_frac:.4f}

Instructions:
1. Generate a 1-paragraph narrative summary of the cell population. Every number in the report MUST exist in the summary above. Do NOT invent findings.
2. Provide a structured JSON record matching this exact schema:
{{
  "image_id": "{image_id}",
  "n_objects": {n_objects},
  "mean_area": {mean_area:.2f},
  "density_class": "<sparse | normal | dense | clustered>",
  "quality_flag": "<high | medium | low>"
}}
Output ONLY the paragraph followed by the JSON block.
"""

def load_unet_model(device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded trained U-Net model from {BEST_MODEL_PATH}")
    else:
        print(f"Warning: Best model weights not found at {BEST_MODEL_PATH}, using un-trained weights.")
    model.eval()
    return model

def query_llm_json(prompt, temperature=0.2):
    payload = {
        "model": TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return str(e)

def extract_json_block(text, default_dict):
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON extraction failed: {e}")
    return default_dict

def run_hybrid_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = load_unet_model(device)
    test_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"Running full hybrid pipeline on {len(test_paths)} test images...")
    
    records = []
    full_log = []
    
    with torch.no_grad():
        for path in test_paths:
            filename = os.path.basename(path)
            img_id = os.path.splitext(filename)[0]
            
            rgb = imageio.imread(path)
            gray = rgb[..., 2].astype(np.float32) / 255.0  # Blue channel DAPI
            
            # Step 1: U-Net Segmentation
            input_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            binary_mask = (prob >= 0.5).astype(np.uint8)
            
            # Step 2: regionprops Feature Extraction
            lbl = label(binary_mask)
            props = regionprops_table(lbl, intensity_image=gray, properties=('area', 'eccentricity', 'solidity', 'mean_intensity'))
            df_props = pd.DataFrame(props)
            
            n_objects = len(df_props)
            mean_area = float(df_props['area'].mean()) if n_objects > 0 else 0.0
            std_area = float(df_props['area'].std()) if n_objects > 1 else 0.0
            mean_ecc = float(df_props['eccentricity'].mean()) if n_objects > 0 else 0.0
            mean_sol = float(df_props['solidity'].mean()) if n_objects > 0 else 0.0
            mean_int = float(df_props['mean_intensity'].mean()) if n_objects > 0 else 0.0
            area_frac = float(binary_mask.mean())
            
            # Step 3: LLM Prompt & Narrative Generation
            prompt = PROMPT_TEMPLATE.format(
                image_id=img_id,
                n_objects=n_objects,
                mean_area=mean_area,
                std_area=std_area,
                mean_ecc=mean_ecc,
                mean_sol=mean_sol,
                mean_int=mean_int,
                area_frac=area_frac
            )
            
            llm_res = query_llm_json(prompt, temperature=0.2)
            
            default_json = {
                "image_id": img_id,
                "n_objects": n_objects,
                "mean_area": round(mean_area, 2),
                "density_class": "sparse" if n_objects < 15 else ("normal" if n_objects < 40 else ("dense" if n_objects < 60 else "clustered")),
                "quality_flag": "high"
            }
            json_record = extract_json_block(llm_res, default_json)
            json_record["area_fraction"] = round(area_frac, 4)
            json_record["mean_eccentricity"] = round(mean_ecc, 3)
            json_record["mean_solidity"] = round(mean_sol, 3)
            
            records.append(json_record)
            full_log.append({
                "image_id": img_id,
                "prompt": prompt,
                "llm_response": llm_res,
                "json_record": json_record
            })
            print(f"Processed {img_id}: {n_objects} objects, mean area {mean_area:.1f} px, class: {json_record.get('density_class')}")
            
    df_out = pd.DataFrame(records)
    df_out.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved consolidated test set summary CSV to {CSV_OUTPUT_PATH}")
    
    with open(os.path.join(RESULTS_DIR, "test_hybrid_pipeline_log.json"), "w") as f:
        json.dump(full_log, f, indent=2)
        
    return df_out

if __name__ == "__main__":
    run_hybrid_pipeline()
