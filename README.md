# Categorical Naive Bayes Model of Morphological Slot Dynamics in the Voynich Manuscript (MS 408)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23179440.svg)](https://doi.org/10.5281/zenodo.23179440)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the computational pipeline, statistical validation models, and dataset parsers for the academic paper:

> **"A Categorical Naive Bayes Model of Morphological Slot Dynamics in the Voynich Manuscript: A Rigorous Attribute-Tagging Architecture and Corpus Audit"**  
> *Denis Ricardo Tavares Ferreira (In Computational Collaboration with Gemini, Google AI)*

---

## 📌 Abstract & Research Focus

The Voynich Manuscript (MS 408) has long resisted decipherment under standard natural language translation paradigms and classical substitution ciphers. This project reframes MS 408 not as an unmapped spoken natural language, but as a **synthetic, low-entropy attribute-tagging system** operating under strict phonotactic constraints.

We establish a **16-slot morphological decomposition architecture ($S_0 - S_{15}$)** structured into three functional macro-regions ($4 + 8 + 4$), coupled with a Laplace-smoothed ($\alpha = 1.0$) Categorical Naive Bayes classifier.

### Key Findings
1. **Out-of-Sample Accuracy:** Leave-One-Out Cross-Validation (LOOCV) achieves **76.44%** predictive accuracy across the full corpus.
2. **Refutation of the Hoax Hypothesis:** A Null Baseline test ($1,000$ stochastic permutations) proves structural divergence from pseudo-random text ($H_{\text{null}} \approx 0.0488$ vs. $H_{\text{real}} \approx 0.0028$, $p < 0.001$).
3. **Codicological Auditing:** Sliding-window probability dynamics identify structural hybrid zones and codicological misplacements (e.g., folios `f26v` and `f96v`).

---

## 📐 16-Slot Morphological Architecture

Tokens in European Voynich Alphabet (EVA) are factorized into three distinct functional macro-regions:

```
[ S0  S1  S2  S3 ] [ S4  S5  S6  S7  S8  S9  S10 S11 ] [ S12 S13 S14 S15 ]
|------------------| |-----------------------------------| |------------------|
   PREFIX REGION                CORE REGION                   SUFFIX REGION
 (Modifiers/Markers)      (Root & Primary Identity)       (State/Fluidity Tags)
```

- **Prefix Region ($S_0 - S_3$, 4 slots):** Encodes structural modifiers, spatial markers, and domain entry tags (e.g., `qok-`, `p-`, `f-`).
- **Core Region ($S_4 - S_{11}$, 8 slots):** Allocates double resolution to the central operational modes and semantic variation (e.g., `che`, `sh`, `ot`).
- **Suffix Region ($S_{12} - S_{15}$, 4 slots):** Captures state variables, fluidity measures, and terminal indicators (e.g., `-ol`, `-al`, `-y`).

---

## 📁 Repository Structure

```
voynich-naive-bayes-model/
├── gerar_corpus_completo.py    # Automated corpus parsing & JSON generation
├── voynich_corpus.json         # Preprocessed 16-slot structured dataset
├── voynich_naive_bayes_real.py # Interactive Streamlit testing environment & Naive Bayes engine
├── README.md                   # Repository documentation
└── LICENSE                     # Open-source license (MIT)
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install streamlit pandas numpy matplotlib
```

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dricardoferreira/voynich-naive-bayes-model.git
   cd voynich-naive-bayes-model
   ```

2. **(Optional) Re-generate the full corpus:**
   ```bash
   python gerar_corpus_completo.py
   ```

3. **Run the Interactive Testing Environment:**
   ```bash
   streamlit run voynich_naive_bayes_real.py
   ```

---

## 🔬 Experimental Suite (Interactive Features)

The primary testing engine (`voynich_naive_bayes_real.py`) provides an interactive GUI to perform and verify all experiments reported in the paper:

- **Individual Folio & Global Scan:** Analyze normalized Shannon entropy and class posteriors ($C_{\text{bot}}$, $C_{\text{pha}}$, $C_{\text{bal}}$, $C_{\text{ast}}$) for any page.
- **Sliding-Window Audit:** Track micro-transitions across sequential blocks ($W=50, S=10$) to locate hybrid domain boundaries and misbound pages.
- **Leave-One-Out Cross-Validation (LOOCV):** Evaluate full out-of-sample prediction accuracy across the entire manuscript.
- **Null Baseline (Control Test):** Run up to 1,000 stochastic permutations to benchmark real text entropy against random noise.

---

## 📜 Citation & Open Science

If you use this model or dataset in your research, please cite:

```bibtex
@article{ferreira2026voynich,
  title={A Categorical Naive Bayes Model of Morphological Slot Dynamics in the Voynich Manuscript: A Rigorous Attribute-Tagging Architecture and Corpus Audit},
  author={Ferreira, Denis Ricardo Tavares},
  note={In Computational Collaboration with Gemini (Google AI)},
  year={2026},
  doi={10.5281/zenodo.23179440}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
