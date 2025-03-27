# 📈 Stock Price Prediction Using SVR and DNN

This project explores two machine learning techniques—**Support Vector Regression (SVR)** and **Deep Neural Networks (DNNs)**—to predict short-term stock prices for AAPL, COKE, and GOOG. It compares kernelized SVR models (linear, RBF, polynomial) and a feedforward DNN architecture in terms of prediction accuracy using historical price data.

## 🧠 Techniques Used
- Support Vector Regression (SVR) with:
  - Linear Kernel
  - Radial Basis Function (RBF) Kernel
  - Polynomial Kernel (Degree 2–4)
- Deep Neural Networks (DNN) with ReLU activations and the Adam optimizer

## 📁 Files
- `StockSVM.pdf`: Full project report detailing methodology, math background, and results
- `StockSVMCode.py`: Python implementation of kernelized SVR models
- `stocksDNN.py`: Python implementation of a deep neural network for time series forecasting

## 📊 Results Summary
- **Linear SVR** achieved the lowest mean ℓ²-loss across most stocks.
- **DNN** models showed competitive performance, particularly for GOOG.
- Polynomial kernels generally underperformed for this task.

## 📦 Dependencies
- `numpy`, `pandas`, `matplotlib`, `yfinance`
- `scikit-learn`, `tensorflow`

## 🔍 Purpose
This repository serves as a demonstration of applying classic and deep learning models to a real-world regression problem. It was developed as part of a portfolio project.
