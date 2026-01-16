# Enhancing LSTMs with Long Memory Properties

Repository for the Advanced Machine Learning class (3rd year) at ENSAE.

## Overview

This project investigates the limitations of Long Short-Term Memory (LSTM) networks in capturing long-range dependencies in time series data. We evaluate enhanced architectures (mLSTM/MRNN) to improve their long memory properties, demonstrated through the analysis of ocean surface measurements—a time series characterized by strong cyclicity and long-term dependencies.

---

## Project Structure

```text
.
├── data/
│   └── ocean_level_data.csv      # Ocean level time series dataset
├── mRNNmLSTM/
│   ├── models.py                # Implementation of mLSTM and MRNN architectures
│   └── train.py                 # Training pipeline for the models
├── notebooks/
│   ├── climate_data_analysis.ipynb # Dataset validation and long memory testing
│   ├── lstm_memory_analysis.ipynb  # Empirical evidence of traditional LSTM limitations
│   ├── lstm_mlstm_analysis.ipynb   # mLSTM/MRNN implementation and performance evaluation
│   └── music_data_analysis.ipynb   # Statistical testing of long memory in music series
├── scripts/
│   ├── _arfima.py               # Utilities for ARFIMA processes
│   ├── _varfima.py              # Utilities for VARFIMA processes
│   ├── d_test.py                # Statistical testing scripts
│   ├── layers.py                # Custom layer implementations
│   └── utils.py                 # General utility functions
├── ts2vec/
│   ├── dilated_conv.py          # Dilated convolutions for feature extraction
│   └── encoder.py               # Temporal encoder architecture
├── .gitignore                   # Files and folders ignored by Git
├── .gitmodules                  # Git submodule configuration
├── LICENSE                      # Project license
└── README.md                    # Project documentation

## Main notebooks

### 1. **lstm_memory_analysis.ipynb**
Based on Greaves-Tunnell & Harchaoui's methodology, this notebook provides empirical evidence that traditional LSTM architectures struggle to capture long-range temporal dependencies.

### 2. **climate_data_analysis.ipynb**
**Purpose:** Comprehensive analysis and validation of the ocean level dataset.

### 3. **lstm_mlstm_analysis.ipynb**
Presentation of **mLSTM/MRNN** (Memory-Enhanced LSTM/RNN) techniques and their performance to capture long-memory

### 4. **music_data_analysis.ipynb**
Advanced statistical testing of long memory properties in music series.


---

## Data

### ocean_level_data.csv
A real-world time series representing ocean level measurements. Downloaded from https://nsidc.org/data. 

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, SciPy
- Matplotlib
- Statsmodel

Install dependencies using:
```bash
pip install torch numpy pandas scipy matplotlib seaborn statsmodels
```

---

## References

- Greaves-Tunnell, A., & Harchaoui, Z. (2019). A statistical investigation of long memory in language and music.

- Zhao & al. (2020). Do RNN and LSTM have long memory?. In International
Conference on Machine Learning 
---

