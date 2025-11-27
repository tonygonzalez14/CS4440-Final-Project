# S&P 500 High Volatility Prediction

A machine learning project that uses Long Short-Term Memory (LSTM) neural networks to predict high volatility days in the S&P 500 index. This project focuses on predicting the **magnitude** of price movements (volatility) rather than direction, which demonstrates significantly stronger predictive signals.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Key Insights](#key-insights)
- [Data](#data)

## 🎯 Overview

This project implements a binary classification model to predict whether the next trading day will experience high volatility (absolute return above median) or low volatility. The model uses a two-layer LSTM architecture trained on 30-day sequences of technical indicators and market features.

**Why Volatility Prediction?**
- Volatility prediction shows **7.2x stronger correlation signals** compared to direction prediction
- Maximum feature correlation: **0.29** (vs. ~0.04 for direction prediction)
- Demonstrates strong volatility clustering behavior (high volatility tends to follow high volatility)

## 📁 Project Structure

```
CS4440-Final-Project/
├── data/
│   └── sp500_2015_2025.csv          # S&P 500 historical data
├── models/
│   └── sp500_volatility_best.h5     # Trained model weights
├── download_sp500_data.py            # Data download script
├── model_training.ipynb              # Main model training notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
```

## 🔧 Features

### Technical Indicators
- **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)
- **Momentum Indicators**: RSI, MACD, Momentum (5, 10, 20), ROC
- **Volatility Measures**: ATR, Bollinger Bands, Rolling volatility (5, 10, 20-day)
- **Trend Indicators**: ADX, Plus DI, Minus DI
- **Volume Indicators**: Volume ratios, OBV, VWAP
- **Price Features**: High-Low range, Open-Close range, Price position in range
- **Statistical Features**: Skewness, Kurtosis

### Feature Selection
The model uses **18 top features** selected by correlation with the volatility target:
1. High_Low_Range (+0.44)
2. Volatility_5 (+0.35)
3. Volatility_10 (+0.32)
4. ATR (+0.31)
5. Volatility_20 (+0.29)
6. Volume (+0.28)
7. RSI_14 (-0.27)
8. Minus_DI (+0.26)
9. Price_Position_20 (-0.26)
10. BB_Position (-0.23)
... and 8 more features

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CS4440-Final-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download data (if not already present):
```bash
python download_sp500_data.py
```

## 💻 Usage

### Data Download

Download S&P 500 data from Yahoo Finance:
```bash
python download_sp500_data.py
# Or specify custom output path:
python download_sp500_data.py --out_path data/custom_path.csv
```

### Exploratory Data Analysis

Open `volatility_exploration.ipynb` to explore:
- Volatility distributions and patterns
- Volatility clustering analysis
- Feature correlations
- Volatility regimes (high/normal/low)
- Volume-volatility relationships
- Train/validation/test split analysis

### Model Training

Open `model_training.ipynb` and run all cells. The notebook includes:
1. **Data Loading & Preparation**: Loads S&P 500 data and handles missing values
2. **Feature Engineering**: Creates 40+ technical indicators
3. **Feature Selection**: Selects top 18 features by correlation
4. **Data Splitting**: Temporal splits (Train: ≤2022, Val: 2023, Test: 2024)
5. **Sequence Creation**: Creates 30-day sequences for LSTM input
6. **Model Training**: Trains LSTM with early stopping and checkpointing
7. **Evaluation**: Comprehensive metrics and visualizations
8. **Cross-Validation**: 10-fold time series cross-validation

### Model Configuration

Key hyperparameters (configurable in the notebook):
```python
WINDOW_SIZE = 30          # Sequence length (days)
BATCH_SIZE = 64           # Training batch size
EPOCHS = 50               # Maximum epochs
LEARNING_RATE = 0.001     # Adam optimizer learning rate
LSTM_UNITS_1 = 50         # First LSTM layer units
LSTM_UNITS_2 = 100        # Second LSTM layer units
DROPOUT_RATE = 0.2        # Dropout regularization
DENSE_UNITS = 32          # Dense layer units
N_TOP_FEATURES = 18       # Number of features to select
```

## 🏗️ Model Architecture

```
Input: (batch_size, 30, 18)
    ↓
LSTM(50 units, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(100 units, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dense(1, activation='sigmoid')
    ↓
Output: Binary probability (high/low volatility)
```

**Key Design Decisions:**
- **Two LSTM layers**: Capture both short-term and longer-term patterns
- **Dropout regularization**: Prevents overfitting (20% dropout rate)
- **Class weights**: Handles slight class imbalance
- **Early stopping**: Monitors validation accuracy with patience=10
- **MinMaxScaler**: Normalizes features to [0, 1] range

## 📊 Results

### Test Set Performance
- **Accuracy**: 59.46%
- **Precision**: 62.12%
- **Recall**: 38.68%
- **F1-Score**: 47.67%
- **ROC-AUC**: 62.86%

### Cross-Validation Results (10-fold)
- **Mean Accuracy**: 69.28% ± 5.96%
- **Mean F1-Score**: 57.02% ± 24.77%
- **Mean ROC-AUC**: 64.74% ± 8.05%

### Performance Analysis
- The model shows moderate predictive capability for volatility
- Better at identifying low volatility days (78% recall) than high volatility days (39% recall)
- ROC-AUC of 0.63 indicates the model has predictive power above random chance
- Cross-validation shows consistent performance across different time periods

## 🔍 Key Insights

### Volatility Clustering
- **Strong clustering effect**: High volatility days are 55.7% likely to be followed by high volatility (vs. 43.8% after low volatility days)
- Correlation between today's and tomorrow's volatility: **0.38**

### Feature Correlations
- **Top predictors**: High-Low Range (0.44), Volatility measures (0.29-0.35), ATR (0.31)
- **15 features** show correlation > 0.2 with volatility target
- Volume shows positive correlation (0.28) with volatility

### Volatility Regimes
- **High Volatility Regime**: 25% of days, 70% chance of next day being high volatility
- **Normal Volatility Regime**: 50% of days, 48% chance of next day being high volatility
- **Low Volatility Regime**: 25% of days, 33% chance of next day being high volatility

### Yearly Patterns
- **2020**: Highest volatility (1.35% mean daily absolute return) - COVID-19 pandemic
- **2022**: Second highest (1.20%) - Market corrections
- **2017**: Lowest volatility (0.30%) - Bull market period

## 📈 Data

### Dataset
- **Source**: Yahoo Finance (^GSPC)
- **Period**: January 2015 - December 2024
- **Frequency**: Daily
- **Features**: Open, High, Low, Close, Volume

### Data Splits
- **Training**: 2015-02-02 to 2022-12-30 (1,935 sequences)
- **Validation**: 2023-01-03 to 2023-12-29 (219 sequences)
- **Test**: 2024-01-02 to 2024-12-30 (222 sequences)

### Target Distribution
- **Balanced dataset**: ~50% high volatility, ~50% low volatility
- **Threshold**: Median absolute return (0.48% daily)

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Preprocessing and metrics
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **yfinance**: Financial data download

## 📝 Notes

- The model uses **temporal cross-validation** (TimeSeriesSplit) to respect time series structure
- All features are normalized using MinMaxScaler fitted on training data only
- The model saves the best weights based on validation accuracy
- GPU acceleration is supported (automatically detected)

## 🔮 Future Improvements

- Experiment with different sequence lengths (15, 45, 60 days)
- Try attention mechanisms or Transformer architectures
- Incorporate external features (VIX, economic indicators, news sentiment)
- Ensemble multiple models for better generalization
- Hyperparameter tuning with Optuna or similar tools
- Feature engineering with more advanced technical indicators

## 📄 License

This project is for educational purposes as part of CS4440 Data Mining course.

## 👤 Author

Created as part of the Fall 2025 Data Mining Final Project.

---

**Disclaimer**: This model is for educational and research purposes only. Past performance does not guarantee future results. Do not use for actual trading decisions without proper risk management and validation.

