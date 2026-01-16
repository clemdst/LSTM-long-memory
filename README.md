# LSTM-long-memory: Enhancing RNNs with Long Memory Properties

Repository for the advanced ML class for 3rd year at ENSAE

## Overview

This project investigates the limitations of Long Short-Term Memory (LSTM) networks in capturing long-range dependencies in time series data and proposes enhanced architectures to improve their long memory properties. We demonstrate this using ocean level data, a time series with known cyclicity and long-term dependencies.

---

## Principal Notebooks

### 1. **LSTM_short_memory.ipynb**
Based on Greaves-Tunnell & Harchaoui's methodology, this notebook provides empirical evidence that traditional LSTM architectures struggle to capture long-range temporal dependencies.


### 2. **climate_data_analysis.ipynb**
**Purpose:** Comprehensive analysis and validation of the ocean level dataset.

**Contents:**
- **Data Quality Assessment:** Statistical properties, missing values, outliers
- **Long Memory Testing:** Analysis of long-range dependencies using:
  - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
  - ARFIMA (AutoRegressive Fractionally Integrated Moving Average) tests
  - Hurst exponent calculation
- **Cyclicity Analysis:** Seasonal patterns and periodic behavior identification
- **Dataset Selection Justification:** Why ocean level data is suitable for testing long memory properties

### 3. **LSTM_updated.ipynb**
Presentation of **mLSTM/MRNN** (Memory-Enhanced LSTM/RNN) techniques and their performance to capture long-memory

### 4. **music2.ipynb**
Advanced statistical testing of long memory properties in music series.


---

## Data

### ocean_level_data.csv
A real-world time series representing ocean level measurements 
---

## Project Structure

```
.
├── LSTM_short_memory.ipynb          # Baseline LSTM memory analysis
├── climate_data_analysis.ipynb       # Dataset validation and long memory testing
├── LSTM_updated.ipynb                # mLSTM/MRNN implementation and evaluation
├── music2.ipynb                      # Time and frequency-domain long memory testing
├── ocean_level_data.csv              # Ocean level time series dataset
├── layers.py                         # Custom layer implementations
├── utils.py                          # Utility functions
├── models.py                         # Model architectures
├── train.py                          # Training pipeline
└── time_series_prediction/           # Supporting module for time series tasks
```

---

## Key Concepts

### Long Memory in Time Series
Long memory refers to the ability of a model to capture and utilize information from distant past events. Ocean level data exhibits strong long memory properties due to climate and oceanic patterns.

### LSTM Limitations
Despite their theoretical advantages, standard LSTMs have limitations in capturing very long-range dependencies, leading to performance degradation on series with strong long memory properties.

### mLSTM/MRNN Solution
Enhanced RNN architectures that incorporate mechanisms to improve the flow and retention of long-term dependencies, providing better performance on long memory time series.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn (for visualization)
- Statsmodels (for time series analysis)

Install dependencies using:
```bash
pip install torch numpy pandas scipy matplotlib seaborn statsmodels
```

---

## Usage

1. **Start with Data Analysis:**
   - Run `climate_data_analysis.ipynb` to understand the dataset and confirm long memory properties

2. **Understand the Baseline:**
   - Execute `LSTM_short_memory.ipynb` to see LSTM limitations

3. **Explore Improvements:**
   - Work through `LSTM_updated.ipynb` to see the mLSTM/MRNN methodology and results

4. **Advanced Statistical Validation:**
   - Use `music2.ipynb` for formal testing of long memory properties

---

## References

- Greaves-Tunnell, A., & Harchaoui, Z. (2019). [Long Memory Estimation in Nonstationary Stochastic Processes](https://arxiv.org/abs/1708.00695)
- Ocean level data sources: [Cite your data source]

---

## Authors

Advanced ML Project - ENSAE 3rd Year
