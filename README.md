# voynich-naive-bayes-model
Categorical Naive Bayes Model &amp; Slot Parsing Engine for MS 408 Research.

# Voynich Naive Bayes Model & Slot Parsing Engine

This repository contains the official computational implementation for the paper:
> **"Categorical Naive Bayes Model & Slot Parsing Engine for MS 408 (Voynich Manuscript) Functional Structural Analysis"**

## Overview

This project provides a robust, reproducible probabilistic classifier designed to analyze structural token distributions within the Voynich Manuscript (MS 408). By decomposing EVA (Extensible Manuscript Alphabet) tokens into a 16-slot morphological schema, the model evaluates domain likelihoods across defined functional categories (Botanical, Pharmaceutical, Balneological, Astronomical) using Laplace-smoothed Naive Bayes and calculates normalized Shannon Entropy ($H$).

## Features

- **Regex Morphological Slot Parser**: Maps EVA tokens into a 16-position array ($S_0$ to $S_{15}$) covering prefixes, core gallows/minim structures, and terminal suffixes.
- **Categorical Naive Bayes Classifier**: Computes log-likelihood scores with customizable Laplace smoothing ($\alpha = 1.0$).
- **Log-Sum-Exp Numerical Stability**: Prevents floating-point underflow when processing long token sequences.
- **Shannon Entropy Validation**: Calculates normalized entropy ($H \in [0, 1]$) to measure target folio domain specificity vs. distributional noise.

## Repository Structure

```text
├── voynich_bayes.py       # Core classifier pipeline & execution script
├── data/
│   └── test_transcriptions.json  # Structural EVA sample datasets by domain/folio
├── README.md              # Project documentation
└── LICENSE                # MIT License
