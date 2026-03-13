---
title: spotNUMT
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.13
app_file: app.py
pinned: false
---

# spotNUMT: Mitochondrial Pseudogene Detector

**spotNUMT** is a deep learning pipeline designed to distinguish true mitochondrial DNA (mtDNA) sequences from **NuMTs** (Nuclear Mitochondrial DNA sequences). 

NuMTs are ancient fragments of mitochondrial DNA that have transposed into the nuclear genome over evolutionary time. Because they share extreme sequence similarity with true mtDNA, they frequently act as confounders in genomic analysis, falsely presenting as mitochondrial mutations. This project utilizes deep neural networks to learn the subtle biological motifs and contexts required to confidently separate the two.

# Usage (Gradio Web Interface)
You can directly interact with the pre-trained Hybrid Model locally through a Gradio interface. Provide a raw string of DNA nucleotides (`A, C, G, T`) and get a prediction probability.

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Launch Interface
python app.py
```

## Architecture Experiments & Validation
The dataset exhibits extreme class imbalance (roughly 1:33, true mtDNA to NuMTs). Standard metrics like Accuracy and AUROC can be misleadingly high due to the high True Negative rate. **AUPRC (Area Under the Precision-Recall Curve)** is utilized as the primary evaluation metric because it directly measures the trade-off between Precision and Recall.

We systematically explored different architectures and class-balancing strategies to optimize detection:

| Experiment | Architecture / Strategy | Precision | Recall | AUPRC | Outcome & Trade-offs |
|------------|----------|-----------|--------|-------|---------|
| **Baseline** | Transformer Encoder | 18.4% | 87.5% | — | High recall but struggled with false positives. |
| **Exp 1** | CNN + 2-Layer BiLSTM | ~12% | **100%** | **0.40** | **Best sequential architecture.** Incredible recall (found all true mtDNA) at the cost of precision. |
| **Exp 2** | Deep Dilated CNN (TCN-style) | 13.1% | 100% | 0.28 | Expanded receptive field failed to capture long-range dependencies better than BiLSTM. |
| **Exp 3** | CNN + BiLSTM + Attention | 10.8% | 100% | 0.39 | Attention weighting yielded nearly identical performance to pure BiLSTM. |
| **Exp 4** | Reduced `pos_weight` (8.0 limit) | **17.2%** | 62.5% | 0.20 | Most *balanced* model. Sacrificed recall to increase precision (confidence in True Positives). |
| **Exp 5** | Focal Loss ($\alpha=0.75, \gamma=2.0$) | 25.0% | 12.5% | 0.22 | Dynamically down-weighting "easy" negatives made the model overly conservative. |
| **Exp 6** | Reverse Complement Augmentation | 0% | 0% | 0.19 | Synthetically doubling the minority class coupled with Focal Loss led to model collapse. |

*Conclusion*: Attempting to force the network to look *too* hard at minority samples destroys the discriminative boundary for NuMTs. The raw **CNN + BiLSTM (Experiment 1)** architecture achieves the highest theoretical performance ceiling for capturing the complex sequence motifs.

# Pipeline Components
- `data_pipeline.py`: Slices FASTA genomes into uniform 200bp sequence windows, dropping fragments and unknown `N` bases.
- `dataset.py`: Converts sequence subsets into one-hot encoded PyTorch tensors `[A,C,G,T]` and establishes the `80/10/10` DataLoaders.
- `model.py`: Natively defines the PyTorch `nn.Module` hybrid sequence classification architecture.
- `train.py`: Handles model orchestration, dynamic optimization (AdamW), and rigorous metric validation.
- `inference.py`: Standalone script to load weights (`best_model.pt`) and process single strings.

## Credits & References
This pipeline was trained using reference data generously provided by:
- **NCBI**: The True Human Mitochondrial Reference Genome (`NC_012920.1`)
- **UCSC Genome Browser**: The `hg38` NuMTs track dataset.
