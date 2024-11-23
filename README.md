# Long-Term Series Forecasting (LTSF) on Weather Dataset

This repository contains the implementation of various models for **long-term time series forecasting (LTSF)** using the Weather dataset. The project is focused on evaluating the effectiveness of linear and transformer-based models for forecasting meteorological indicators over extended periods.

---

##  Weather Dataset

 Weather statistics recorded every 10 minutes for the entire year 2020.
- **Features**: 
    21 meteorological indicators,such as air temperature, wind speed, radiation, etc.
- **Splits**: 
    The dataset is split in a chronological order to maintain the time dependency of the series: **70% / 10% / 20%** (first 70% for training, next 10% for validation, and the final 20% for testing).

---

## Task

### Multivariate Forecasting
- **Objective**: Use multiple weather indicators (features) as input to predict future values of all features simultaneously.

---

## 🧪 Models Implemented

### From ["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504) (AAAI 2023)
1. **Linear**: A simple linear layer for forecasting.
2. **DLinear**: Decomposition Linear for handling trend and seasonality patterns.
3. **NLinear**: Normalized Linear to mitigate train-test set distribution shifts.

### From ["Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting "](https://arxiv.org/abs/2106.13008)(NeurIPS 2017) 
4. **Autoformer**: Transformer-based model designed for time series decomposition with auto-correlation to capture periodic patterns efficiently.

### Custom Implementation
5. **Custom Linear Model**: A linear model specifically designed for this experiment.

---

## Evaluation Metric

- Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used to evaluate model performance.