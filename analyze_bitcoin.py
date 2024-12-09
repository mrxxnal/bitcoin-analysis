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

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("bitcoin.csv")  # Adjust the filename if necessary

# Ensure Date column is datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Sort by date
data.sort_values(by="Date", inplace=True)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Prepare training data
sequence_length = 60
X_lstm, y_lstm = [], []
for i in range(sequence_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-sequence_length:i, 0])
    y_lstm.append(scaled_data[i, 0])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Split into training and test sets
train_size = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

data['MA_7'] = data['Close'].rolling(window=7).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=7).std()
X = data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'Volatility']].dropna()
y = data['Close'][len(data) - len(X):]

from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X, y, scoring='neg_mean_squared_error', cv=5)
print("Cross-Validated MSE:", -np.mean(scores))

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Build Enhanced LSTM model
model_lstm_advanced = Sequential([
    Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))),
    Dropout(0.2),  # Add dropout for regularization
    Bidirectional(LSTM(100, return_sequences=False)),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

model_lstm_advanced.compile(optimizer='adam', loss='mean_squared_error')

# Set early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the advanced model
history = model_lstm_advanced.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
predictions_advanced = model_lstm_advanced.predict(X_test)
predictions_advanced = scaler.inverse_transform(predictions_advanced.reshape(-1, 1))

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(data['Close'].values[-len(y_test):], label="Actual Prices")
plt.plot(predictions_advanced, label="Enhanced LSTM Predictions", color='red')
plt.title("Enhanced LSTM: Actual vs Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()