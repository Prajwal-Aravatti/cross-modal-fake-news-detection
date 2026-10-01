# Cross-Modal Fake News Detection Using Image-Text Consistency

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP%20ViT--L%2F14-412991?style=for-the-badge&logo=openai)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A multimodal deep learning system that detects fake news by measuring image-text consistency using CLIP ViT-L/14.**

*KLE Technological University — Department of Computer Science & Engineering*

</div>

---

## 📌 Overview

Most fake news detectors analyze images and text **separately** and merge them at the end — never asking the central question: *do the image and the text actually belong together?*

We present the **Cross-Modal Consistency Network (CMCN)**, a model built specifically to measure how consistent a given image-text pair is. It uses **CLIP ViT-L/14** as a shared encoder and adds three parallel consistency measurement branches on top:

| Branch | What it measures |
|--------|-----------------|
| **Global Cosine Similarity** | Overall semantic alignment between image & text embeddings |
| **Cross-Modal Attention** | Fine-grained word-to-image-patch correspondences (8-head MHA) |
| **KL Ambiguity Estimator** | Disagreement between image-only and text-only predictions |

An **adaptive fusion gate** learns to weight text vs. image embeddings dynamically based on the cosine and ambiguity scores.

---

## 🏆 Results on Fakeddit (10K Test Set)

| Configuration | Accuracy | Macro F1 | Weighted F1 |
|--------------|----------|----------|-------------|
| CMCN (no bias) | 0.8363 | 0.7657 | 0.8397 |
| CMCN (bias −0.25) | 0.8424 | 0.7693 | 0.8459 |
| **CMCN (bias −0.30)** | **0.8429** | **0.7696** | **0.8464** |

### Per-Class Performance (Best Configuration)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| True | 0.92 | 0.74 | 0.82 | 3,980 |
| Satire/Parody | 0.66 | 0.86 | 0.75 | 600 |
| Misleading Content | 0.71 | 0.88 | 0.78 | 1,907 |
| Imposter Content | 0.43 | 0.61 | 0.50 | 209 |
| **False Connection** | **0.99** | **0.97** | **0.98** | 2,903 |
| Manipulated Content | 0.70 | 0.91 | 0.79 | 401 |

> **False Connection (cheapfakes)** — the hardest detection scenario — achieved a near-perfect **F1 of 0.98**, validating that CLIP cosine similarity catches out-of-context image-text pairs almost perfectly.

---

## 🏗️ Architecture

```
Input: (Image, Text Post Title)
         │
         ▼
┌──────────────────────────────────┐
│   CLIP ViT-L/14 Dual Encoder     │  openai/clip-vit-large-patch14
│   Text Encoder  (12-layer Transformer) → f_text ∈ R^768          │
│   Vision Encoder(24-layer ViT)         → f_image ∈ R^768         │
│   + sequence hidden states for cross-attention                    │
└─────────────────┬────────────────┘
                  │
    ┌─────────────┴─────────────────┐
    │    3-Branch Consistency Module │
    │                               │
    │  A: Cosine Similarity         │  s = f_text · f_image  ∈ [-1,+1]
    │  B: Cross-Modal Attention     │  f_cross ∈ R^768  (8-head MHA)
    │  C: KL Ambiguity              │  a = sym-KL(p_text ‖ p_image)
    └─────────────┬─────────────────┘
                  │
       ┌──────────▼──────────┐
       │  Adaptive Gate       │  α = σ(W₂·GELU(W₁·[s; a]))
       │  f_gated = α·f_text + (1-α)·f_image
       └──────────┬──────────┘
                  │
       ┌──────────▼──────────────────────────────────────┐
       │  3-Layer MLP Classifier                          │
       │  Input: [f_gated; f_cross; f_text; s; a] ∈ R^2306 │
       │  2306 → 512 → 256 → 6 classes                    │
       │  Dropout(0.30) → Dropout(0.20)                   │
       └──────────────────────────────────────────────────┘
```

### Multi-Task Loss
```
L_total = L_cls  +  0.3 × L_contrastive  +  0.1 × L_consistency
```

- **L_cls**: Class-weighted cross-entropy (handles 38.7% True vs 2.1% Imposter imbalance)
- **L_contrastive**: InfoNCE for True pairs; margin penalty for fake pairs (margin=0.20, τ=0.07)
- **L_consistency**: MSE supervision on cosine score — high for True, low for all fake classes

### 3-Stage Training Strategy

| Stage | Epochs | CLIP | Loss | Trainable Params | LR (Head) |
|-------|--------|------|------|-----------------|-----------|
| 1 | 1–3 | Frozen | L_cls only | ~5.46M (1.26%) | 5×10⁻⁵ |
| 2 | 4–5 | Frozen | L_total | ~5.46M | 2.5×10⁻⁵ |
| 3 | 6–8 | Last layer unfrozen | L_total | ~26.5M (6.1%) | 5×10⁻⁷ (CLIP) |

All stages use: **AdamW** + cosine LR schedule + 6% warmup + batch size 32 + **bfloat16 autocast**

---

## 📂 Project Structure

```
cross-modal-fake-news-detection/
│
├── README.md                          ← You are here
├── requirements.txt                   ← All Python dependencies
├── .gitignore
│
├── 01_data_loading_and_caching.ipynb  ← Download & cache Fakeddit images
├── 02_training.ipynb                  ← Train CMCN (3-stage pipeline)
└── 03_evaluation.ipynb                ← Test set evaluation + inference demo
```

---

## 📦 Dataset — Fakeddit

We use the **[Fakeddit](https://github.com/entitize/Fakeddit)** dataset: a large-scale multimodal benchmark from Reddit with 6 fine-grained misinformation categories.

| Label | Category | Train | Val | Test |
|-------|----------|-------|-----|------|
| 0 | True | 38,735 | 3,848 | 3,980 |
| 1 | Satire/Parody | 5,880 | 583 | 600 |
| 2 | Misleading Content | 18,484 | 1,855 | 1,907 |
| 3 | Imposter Content | 2,093 | 214 | 209 |
| 4 | False Connection | 30,796 | 3,122 | 2,903 |
| 5 | Manipulated Content | 4,012 | 378 | 401 |
| | **Total** | **100,000** | **10,000** | **10,000** |

### How to download
1. Go to the official repo: **https://github.com/entitize/Fakeddit**
2. Download the metadata TSV files (train/validate/test)
3. Run `01_data_loading_and_caching.ipynb` — it downloads and caches all images locally

> **Note**: The raw dataset and cached images are NOT included in this repo (too large). Only the code is provided.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.9+**
- GPU with **16GB+ VRAM** recommended (CLIP ViT-L/14 is 427M params)
- CUDA 11.8+ with **bfloat16** support (e.g., A100, RTX 3090/4090, T4)

> The notebooks were developed and run on **Google Colab / Lightning.ai** with a T4/A100 GPU.

### Install

```bash
pip install -r requirements.txt
```

Or directly (as used in the notebooks):

```bash
pip install transformers scikit-learn seaborn matplotlib pillow tqdm torch torchvision
```

The CLIP model (`openai/clip-vit-large-patch14`) downloads automatically from HuggingFace on first run.

---

## 🚀 How to Run

**Run the notebooks in order:**

### 1️⃣ `01_data_loading_and_caching.ipynb`
- Loads Fakeddit metadata TSV files
- Downloads and caches images locally (avoids repeated network calls during training)
- Handles corrupt/missing images with a `224×224` zero-tensor black placeholder
- Saves cached CSVs: `train_100k_cached.csv`, `validate_10k_cached.csv`, `test_10k_cached.csv`

### 2️⃣ `02_training.ipynb`
- Loads cached CSVs and locally stored images
- Builds `FakedditDataset` with `WeightedRandomSampler` for class imbalance
- Defines the full `CMCN` model (CLIP + cross-attention + gate + classifier)
- Runs Stage 1 → Stage 2 → Stage 3 training, saving the best checkpoint each stage
- Each epoch over 100K samples takes ~**2.2 minutes** on a T4 GPU
- **Known issue handled**: NaN loss from bfloat16 overflow in cross-entropy — fixed by casting logits to float32 before loss computation + finite-loss checking

### 3️⃣ `03_evaluation.ipynb`
- Loads the best Stage 3 checkpoint (`best_cmcn.pt`)
- Evaluates on 10K test set with three calibration settings
- Final best: **bias −0.30** on Misleading Content logit
- Outputs: accuracy, macro-F1, weighted-F1, per-class report, confusion matrix
- Includes a `predict_one_calibrated(text, image_path)` function for single-sample inference

---

## 🔬 Key Implementation Details

| Detail | Value |
|--------|-------|
| Backbone | `openai/clip-vit-large-patch14` (HuggingFace Transformers) |
| Text token length | 77 (CLIP standard, BPE tokenizer) |
| Image size | 224×224, CLIP standard normalization |
| Cross-attention heads | 8, dropout=0.10 |
| Unimodal MLP heads | LayerNorm → Linear(768→256) → GELU → Dropout(0.15) → Linear(256→6) |
| Classifier input dim | 2306 = 768×3 + 2 (cosine + ambiguity scalars) |
| Optimizer | AdamW, weight_decay=1e-4 |
| LR schedule | Cosine decay + 6% linear warmup |
| Mixed precision | bfloat16 via `torch.autocast` |
| Batch size | 32 (training), 64 (evaluation) |
| Gradient clipping | max norm = 0.5 (Stage 1) |
| Seed | 42 (torch, numpy, random, CUDA) |
| Platform | Google Colab / Lightning.ai Studio |

---

## 👥 Team

| Name | Roll No. | Email |
|------|----------|-------|
| Sumanth Shet | 01FE23BCS280 | 01fe23bcs280@kletech.ac.in |
| Prajwal Aravatti | 01FE23BCS299 | 01fe23bcs299@kletech.ac.in |
| Mallikarjun G Patil | 01FE23BCS283 | 01fe23bcs283@kletech.ac.in |
| Mohammad Imran | 01FE23BCS378 | 01fe23bcs378@kletech.ac.in |

**Guide**: Prof. Suvarna Kanakaraddi — suvarna_gk@kletech.ac.in

*KLE Technological University, Hubballi, India — 5th Semester, Generative AI Course*

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{cmcn2024,
  title={Cross-Modal Fake News Detection Using Image-Text Consistency},
  author={Shet, Sumanth and Aravatti, Prajwal and Patil, Mallikarjun G and Imran, Mohammad},
  institution={KLE Technological University},
  year={2024}
}
```

### Key References

| Paper | Used for |
|-------|---------|
| Radford et al. (2021) — CLIP, ICML | Backbone encoder |
| Nakamura et al. (2020) — Fakeddit, LREC | Dataset |
| Vaswani et al. (2017) — Attention, NeurIPS | Cross-modal attention |
| Chen et al. (2020) — SimCLR, ICML | InfoNCE contrastive loss |
| Loshchilov & Hutter (2019) — AdamW, ICLR | Optimizer |
| Guo et al. (2017) — Calibration, ICML | Logit bias calibration |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
