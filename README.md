# Stock-Market-Prediction-System

## Overview

This project demonstrates the development of a comprehensive stock market analysis and prediction system spanning the NASDAQ and Vietnamese markets. It leverages advanced deep learning techniques to predict stock prices, generate trading signals, and create balanced investment portfolios. The system has been designed to handle real-world complexities in financial data processing and market analysis.

---

## Features

### 1. Data Processing
- Robust handling of multi-format datasets from both the NASDAQ and Vietnamese markets.
- Features include:
  - Extraction and preprocessing of thousands of compressed files.
  - Rigorous filtering to ensure data quality and reliability.

### 2. Prediction Models
- Multi-feature analysis incorporating market indicators (e.g., Open Price, Close Price, Volume).
- LSTM-based architecture for:
  - k-th day forecasts.
  - Consecutive days forecasting.
- Sector-specific model applications for improved accuracy.

### 3. Trading Signal System
- **Buy Signals**:
  - Identifies optimal buying opportunities based on predicted price trends.
- **Sell Signals**:
  - Detects high-price windows for profitable sales.
- High accuracy: ~70% for buy signals, ~65% for sell signals.

### 4. Portfolio Management
- Filters high-potential stocks using:
  - Volatility analysis.
  - Performance evaluation.
  - Risk assessment.
- Successfully identifies balanced investment opportunities.

---

## Architecture

- Data normalization pipeline with MinMax scaling.
- Deep learning architecture:
  - Input layer for multi-feature processing.
  - LSTM layers for temporal pattern recognition.
  - Dense layers for non-linear modeling.
  - Configurable output layers for single or multi-day predictions.

---

## Performance

- Test MSE:
  - Single-day predictions: **0.0119**.
  - 25-day predictions: **0.776**.
- Trading signal results:
  - Buy signals: Average ROI ~8.5%.
  - Sell signals: Loss prevention ~6.2%.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - `TensorFlow`
  - `Pandas`
  - `NumPy`
  - `zipfile`

### Installation
Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
