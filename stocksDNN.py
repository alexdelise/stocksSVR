import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to extract features and target variable
def extract_features_target(data, target_col='Close', window_size=10):
    features = []
    target = []

    for i in range(len(data) - window_size):
        features.append(data[target_col][i:i+window_size])
        target.append(data[target_col][i+window_size])
    
    return np.array(features), np.array(target)

# Function to train DNN model
def train_dnn(X_train_scaled, y_train, num_epochs=1000, batch_size=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer, single value for regression
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train_scaled, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)
    
    return model

# Function to perform prediction using DNN model
def predict_with_dnn(model, X_extrapolation_scaled):
    y_pred = model.predict(X_extrapolation_scaled)
    return y_pred.flatten()

# Define date ranges
train_start_date = '2010-01-01'
train_end_date = '2016-12-31'
extrapolation_start_date = '2017-01-01'
extrapolation_end_date = '2017-01-31'

# Define tickers
tickers = ['AAPL', 'COKE', 'GOOG']

for ticker in tickers:
    # Load data for training
    training_data = fetch_stock_data(ticker, train_start_date, train_end_date)

    # Extract features and target variable for training data
    X_train, y_train = extract_features_target(training_data)

    # Feature scaling for training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train DNN model
    dnn_model = train_dnn(X_train_scaled, y_train)

    # Load data for extrapolation
    extrapolation_data = fetch_stock_data(ticker, extrapolation_start_date, extrapolation_end_date)

    # Extract features and target variable for extrapolation data
    X_extrapolation, y_extrapolation = extract_features_target(extrapolation_data)

    # Feature scaling for extrapolation data
    X_extrapolation_scaled = scaler.transform(X_extrapolation)

    # Predictions using DNN model
    y_pred = predict_with_dnn(dnn_model, X_extrapolation_scaled)

    # Calculate L2 loss (mean squared error) for evaluation
    l2_loss = np.linalg.norm((y_extrapolation - y_pred) ** 2) / len(y_pred)

    # Print L2 loss result
    print(f"\n{ticker} DNN L2 Loss:", l2_loss)

    # Plotting training data
    plt.figure(figsize=(12, 6))
    plt.plot(training_data.index, training_data['Close'], label='Actual (Training)', color='blue')
    plt.title(f'Stock Price Trend for {ticker} (Training Period)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plotting extrapolated results
    plt.figure(figsize=(12, 6))
    plt.plot(extrapolation_data.index[:len(y_extrapolation)], y_extrapolation, label='Actual')
    plt.plot(extrapolation_data.index[:len(y_extrapolation)], y_pred, label='Predicted (DNN)', linestyle='-')
    plt.title(f'Extrapolated Stock Price Prediction for {ticker} using DNN')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
