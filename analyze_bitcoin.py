import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bitcoin = pd.read_csv("Bitcoin.csv")
print(bitcoin.head())

print(bitcoin.isnull().sum())
print(bitcoin.describe())

bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
bitcoin.sort_values(by='Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(bitcoin['Date'], bitcoin['Close'], label='Close Price')
plt.title("Bitcoin Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

bitcoin['Month'] = bitcoin['Date'].dt.month
monthly_trend = bitcoin.groupby('Month')['Close'].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_trend.index, y=monthly_trend.values)
plt.title("Average Monthly Bitcoin Prices")
plt.xlabel("Month")
plt.ylabel("Average Close Price (USD)")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("bitcoin.csv")  # Adjust the filename if necessary

# Ensure Date column is datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Sort by date
data.sort_values(by="Date", inplace=True)

data['MA_7'] = data['Close'].rolling(window=7).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=7).std()
X = data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Volatility']].dropna()
y = data['Close'][len(data) - len(X):]

# Visualize the data
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Close'], label="Closing Price")
plt.title("Bitcoin Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Extract relevant features
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define the model
gbr = GradientBoostingRegressor(random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on test data
y_pred_gbr = gbr.predict(X_test)

# Evaluate the model
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

print(f"Gradient Boosting Regressor - MSE: {mse_gbr}, MAE: {mae_gbr}")

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Prices", marker='o')
plt.plot(y_pred_gbr, label="Predicted Prices", marker='x', color='red')
plt.title("Gradient Boosting Regressor: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()