"""
src/task1_vlm.py
----------------
Task 1: Direct Multimodal Large Language Model (VLM) Image Description.
- Interfaces with local Ollama VLM (`llama3.2-vision:latest`).
- Compares Naive Prompt vs Engineered Structured Prompt.
- Demonstrates output non-determinism across repeated runs.
"""
import os
import json
import base64
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llava:7b"
SAMPLE_IMG = "nuclei_dataset/train/images/train_001.png"
RESULTS_DIR = "results"

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def query_ollama_vlm(prompt, image_base64, temperature=0.7, model=MODEL_NAME):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying Ollama VLM ({model}): {e}")
        return str(e)

NAIVE_PROMPT = "Describe this medical image in detail and identify any diagnostic findings or abnormalities."

STRUCTURED_PROMPT = """You are an expert biomedical image analysis assistant. Your role is strictly descriptive and auditable. Do NOT make clinical diagnoses.
Analyze the provided fluorescence microscopy image and output ONLY a valid JSON record matching this schema:
{
  "modality": "<fluorescence_microscopy | brightfield | unknown>",
  "tissue_type": "<stained_nuclei | histology | cell_culture | uncertain>",
  "notable_features": "<short string summarizing visual characteristics, object density, contrast, and spatial distribution>",
  "image_quality": "<high | medium | low | uncertain>"
}
Rules:
1. If any field is ambiguous or unclear, explicitly set the value to "uncertain".
2. Do NOT output any markdown formatting, preambles, or postscript prose. Output ONLY the JSON object.
"""

def run_task1():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(SAMPLE_IMG):
        raise FileNotFoundError(f"Sample image not found at {SAMPLE_IMG}")
        
    print(f"Encoding sample image: {SAMPLE_IMG}")
    img_b64 = encode_image_base64(SAMPLE_IMG)
    
    print("\n--- 1. Running Naive Prompt ---")
    naive_out = query_ollama_vlm(NAIVE_PROMPT, img_b64, temperature=0.7)
    print("Naive Output:\n", naive_out)
    
    print("\n--- 2. Running Engineered Structured Prompt ---")
    struct_out = query_ollama_vlm(STRUCTURED_PROMPT, img_b64, temperature=0.2)
    print("Structured Output:\n", struct_out)
    
    print("\n--- 3. Running Repeated Trials (Non-determinism Demo) ---")
    trials = []
    for i in range(1, 4):
        print(f"Trial {i}...")
        out_i = query_ollama_vlm(STRUCTURED_PROMPT, img_b64, temperature=0.7)
        trials.append({"trial_id": i, "response": out_i})
        print(f"Trial {i} output:\n", out_i)
        
    task1_data = {
        "sample_image": SAMPLE_IMG,
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_out,
        "structured_prompt": STRUCTURED_PROMPT,
        "structured_response": struct_out,
        "repeated_trials": trials
    }
    
    out_json = os.path.join(RESULTS_DIR, "task1_vlm_results.json")
    with open(out_json, "w") as f:
        json.dump(task1_data, f, indent=2)
    print(f"\nSaved Task 1 results to {out_json}")
    return task1_data

if __name__ == "__main__":
    run_task1()
