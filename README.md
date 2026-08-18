# Compact Hybrid Biomedical Image-Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-orange.svg)](https://pytorch.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-Otsu%2FRegionprops-green.svg)](https://scikit-image.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama--3.2-purple.svg)](https://ollama.ai/)

An auditable, reproducible, hybrid biomedical image-analysis pipeline that combines local Multimodal Large Language Models (VLMs), classical morphological feature extraction (`scikit-image`), and a deep PyTorch **U-Net** segmentation network.

---

## 📌 Pipeline Architecture & Auditable Boundary

```
[Raw DAPI Image (256x256)] ───► [PyTorch U-Net / Otsu Mask] ───► [regionprops Feature Table]
                                                                        │
                                                                        ▼
[Human Review / Sign-off] ◄─── [Grounded Narrative] ◄─── [Structured JSON Record (SOURCE OF TRUTH)]
```

### Auditable Boundary Paradigm
- **Auditable Region**: Preprocessing $\rightarrow$ U-Net Segmentation $\rightarrow$ `regionprops_table` extraction $\rightarrow$ Structured JSON Record. Fully deterministic, unit-testable, machine-checkable, and reproducible.
- **Narrative Region**: Structured JSON Record $\rightarrow$ Natural language report. Constrained by strict grounding prompts ("Never let the narrative invent facts. Every number/finding must exist in the JSON record").

---

## 🚀 Key Results & Performance Summary

| Metric / Experiment | Classical Otsu | PyTorch U-Net (Dice Loss) | PyTorch U-Net (Combined Loss) |
| :--- | :---: | :---: | :---: |
| **Validation Mean Dice** | 0.8124 | **0.9958** | 0.9948 |
| **Validation Mean IoU** | 0.7105 | **0.9916** | 0.9897 |
| **Touching Nuclei Separation** | Merges adjacent cells | Successfully resolves boundaries | Successfully resolves boundaries |
| **Auditability** | 100% Deterministic | Learned Features + Deterministic regionprops | Learned Features + Deterministic regionprops |

---

## 📂 Repository Structure

```
.
├── src/                        # Modular Python Source Code
│   ├── eda.py                  # Task 1: Preprocessing & Exploratory Data Analysis
│   ├── task1_vlm.py            # Task 1: Direct VLM Prompting & Non-determinism analysis
│   ├── task2_classical.py      # Task 2: Classical Otsu Thresholding & regionprops LLM interpretation
│   ├── unet.py                 # Task 3: PyTorch U-Net Architecture & Loss Functions
│   ├── train_unet.py           # Task 3: U-Net Training, Loss Ablation & Validation Panel Generation
│   ├── hybrid_pipeline.py      # Task 4: End-to-End Hybrid Pipeline Batch Execution on Test Set
│   └── extensions.py           # Extra Credit: Robustness Corruption Trace Analysis
├── report/                     # Academic Report & Figures
│   ├── biomedical_image_analysis_report.md  # Complete 4-page Academic Report
│   └── figures/                # Output plots (EDA, U-Net Curves, Predictions, Robustness)
├── results/                    # Generated Outputs, JSON Records & CSV Export
│   ├── test_pipeline_summary.csv
│   └── task1_vlm_results.json
├── dataset/                    # Stained-Nuclei Fluorescence Dataset (256x256 DAPI)
└── README.md
```

---

## ⚙️ Quick Start & Usage

### 1. Prerequisites & Environment Setup
```bash
git clone https://github.com/minhajul3557/bio-medical-image-analyzer.git
cd bio-medical-image-analyzer
pip install torch torchvision scikit-image matplotlib pandas requests imageio pillow
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
python3 src/eda.py
```

### 3. Train U-Net & Run Loss Ablation Study
```bash
python3 src/train_unet.py
```

### 4. Execute End-to-End Hybrid Pipeline on Test Set
```bash
python3 src/hybrid_pipeline.py
```

### 5. Run Extra Credit Robustness Analysis
```bash
python3 src/extensions.py
```

---

## 📄 License & Academic Reference
Educational use only. Models and outputs are not cleared for clinical diagnostic use.

- **Ronneberger et al. (2015)**: *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- **Isensee et al. (2021)**: *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods.
