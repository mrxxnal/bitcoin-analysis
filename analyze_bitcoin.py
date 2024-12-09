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

git add analyze_bitcoin.py
git commit -m "Add Bitcoin data analysis script"
git push -u origin main

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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