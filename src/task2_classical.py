"""
src/task2_classical.py
----------------------
Task 2: Classical Features & LLM Interpretation (Numbers-First).
- Uses scikit-image for Otsu thresholding, morphological cleanup, and regionprops_table.
- Formats region statistics into a numbers-only summary.
- Passes numerical summary to local LLM (`llama3.2:latest`) via Ollama.
- Requests 1-paragraph narrative + JSON record.
"""
import os
import json
import requests
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops_table

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEXT_MODEL = "llama3.2:latest"
SAMPLE_IMG = "nuclei_dataset/train/images/train_001.png"
RESULTS_DIR = "results"

def otsu_segmentation(gray_img):
    thresh = threshold_otsu(gray_img)
    binary = gray_img > thresh
    # Morphological cleanup
    cleaned = binary_opening(binary, disk(1))
    cleaned = binary_closing(cleaned, disk(1))
    cleaned = binary_fill_holes(cleaned)
    cleaned = remove_small_objects(cleaned, min_size=10)
    lbl = label(cleaned)
    return cleaned, lbl

def extract_regionprops(lbl, gray_img):
    properties = ('label', 'area', 'eccentricity', 'solidity', 'mean_intensity', 'perimeter', 'equivalent_diameter_area')
    table = regionprops_table(lbl, intensity_image=gray_img, properties=properties)
    df = pd.DataFrame(table)
    return df

def generate_numerical_summary(df):
    if len(df) == 0:
        return "Numerical Summary: 0 objects detected."
    n_objects = len(df)
    mean_area = df['area'].mean()
    std_area = df['area'].std() if n_objects > 1 else 0.0
    mean_ecc = df['eccentricity'].mean()
    mean_sol = df['solidity'].mean()
    mean_int = df['mean_intensity'].mean()
    
    summary = (
        f"QUANTITATIVE REGIONPROPS FEATURE SUMMARY:\n"
        f"- Total Objects Detected (n_objects): {n_objects}\n"
        f"- Mean Object Area (pixels): {mean_area:.2f} (std: {std_area:.2f}, min: {df['area'].min()}, max: {df['area'].max()})\n"
        f"- Mean Eccentricity (0=circle, 1=line): {mean_ecc:.3f}\n"
        f"- Mean Solidity (convexity ratio): {mean_sol:.3f}\n"
        f"- Mean Signal Intensity: {mean_int:.3f}\n"
        f"- Total Area Coverage Fraction: {(df['area'].sum() / (256*256)):.4f}\n"
    )
    return summary

PROMPT_TEMPLATE = """You are a biomedical data analyst interpreting deterministic image metrics.
Below is a NUMERICAL FEATURE SUMMARY extracted from a fluorescence microscopy image via classical segmentation (Otsu thresholding + regionprops). You must NOT guess features outside this table.

{summary}

Tasks:
1. Write a clear, objective 1-paragraph narrative describing the cell population based strictly on the numbers above.
2. Provide a structured JSON record with the exact keys:
{{
  "n_objects": <integer>,
  "density_class": "<sparse | normal | dense | clustered>",
  "shape_regularity": "<regular | irregular | mixed>",
  "quality_flag": "<high | medium | low>"
}}
Output ONLY the paragraph followed by the JSON record. Do NOT add ungrounded claims.
"""

def query_ollama_text(prompt, temperature=0.2):
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
        print(f"Error querying text LLM ({TEXT_MODEL}): {e}")
        return str(e)

def run_task2():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rgb = imageio.imread(SAMPLE_IMG)
    gray = rgb[..., 2] / 255.0  # Blue channel DAPI signal
    
    cleaned_mask, lbl = otsu_segmentation(gray)
    df_props = extract_regionprops(lbl, gray)
    print(f"Classical Otsu detected {len(df_props)} objects.")
    
    num_summary = generate_numerical_summary(df_props)
    print("\nGenerated Numerical Summary:\n", num_summary)
    
    prompt = PROMPT_TEMPLATE.format(summary=num_summary)
    print("\nQuerying llama3.2 with Numbers-First prompt...")
    llm_output = query_ollama_text(prompt, temperature=0.2)
    print("\nLLM Output:\n", llm_output)
    
    task2_data = {
        "sample_image": SAMPLE_IMG,
        "otsu_n_objects": int(len(df_props)),
        "numerical_summary": num_summary,
        "llm_prompt": prompt,
        "llm_output": llm_output,
        "regionprops": df_props.to_dict(orient='records')
    }
    out_json = os.path.join(RESULTS_DIR, "task2_classical_results.json")
    with open(out_json, "w") as f:
        json.dump(task2_data, f, indent=2)
    print(f"\nSaved Task 2 results to {out_json}")
    return task2_data

if __name__ == "__main__":
    run_task2()
