import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download stock price data for Apple Inc.
df = yf.download('AAPL', start='2010-01-01', end='2016-12-31')
df.to_csv('AAPL.csv')

forecast_days = 90
df = df[['Close']]
df.loc[:, 'Prediction'] = df[['Close']].shift(-forecast_days)

X = np.array(df.drop(['Prediction'], axis=1))
X = X[:-forecast_days]
y = np.array(df['Prediction'])
y = y[:-forecast_days]

split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Train linear SVM
model_linear = SVR(kernel='linear')
model_linear.fit(X_train, y_train)

# Train RBF SVM
model_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
model_rbf.fit(X_train, y_train)

# Train polynomial SVM
model_poly4 = SVR(kernel='poly', degree=4)
model_poly4.fit(X_train, y_train)

# Calculate MSEs for each model
mse_linear = mean_squared_error(y_test, model_linear.predict(X_test))
mse_rbf = mean_squared_error(y_test, model_rbf.predict(X_test))
mse_poly4 = mean_squared_error(y_test, model_poly4.predict(X_test))

print('Linear SVM MSE:', mse_linear)
print('RBF SVM MSE:', mse_rbf)
print('Polynomial SVM (degree 4) MSE:', mse_poly4)

# Predict next forecast_days days
next_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
X_forecast = np.array([df['Close'][-1]] * forecast_days).reshape(-1, 1)

forecast_set_linear = model_linear.predict(X_forecast)
forecast_set_rbf = model_rbf.predict(X_forecast)
forecast_set_poly4 = model_poly4.predict(X_forecast)

df_forecast = pd.DataFrame({
    'Date': next_dates,
    'Forecast_Linear': forecast_set_linear,
    'Forecast_RBF': forecast_set_rbf,
    'Forecast_Poly4': forecast_set_poly4
})
df_forecast.set_index('Date', inplace=True)

plt.figure(figsize=(10, 6))

# Download stock price data for Apple Inc. for the extrapolation period
df2 = yf.download('AAPL', start='2017-01-01', end='2017-03-31')
df2 = df2[['Close']]

plt.plot(df.index, df['Close'], color='blue', label='Close')
plt.plot(df_forecast.index, df_forecast['Forecast_Linear'], color='green', label='Linear SVM Forecast')
plt.plot(df_forecast.index, df_forecast['Forecast_RBF'], color='red', label='RBF SVM Forecast')
plt.plot(df_forecast.index, df_forecast['Forecast_Poly4'], color='purple', label='Polynomial (Degree 4) SVM Forecast')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Format x-axis to display only month and day
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price Prediction')
plt.grid(False)

plt.show()
