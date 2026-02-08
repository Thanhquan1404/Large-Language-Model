# implement libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Read data
data_path = './monthly_milk_production.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
production = data['Production'].astype(float).values.reshape(-1, 1) # Take production column and resize it into an array (-1,1)

# Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(production)

# Create squence and train-test split
window_size = 12
X = []
y = []
target_dates = data.index[window_size:] # Take 12 months as a year for next prediction

for i in range(window_size, len(scaled_data)):
  X.append(scaled_data[i-window_size:i, 0])
  y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, target_dates, test_size=0.2, shuffle=False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
# 1. Initialize a linear stack of layers. 
# Think of this as an empty train track where we will add cars (layers) one by one.
model = Sequential()

# 2. First LSTM Layer
# units=128: The "width" of the layer (128 memory cells).
# return_sequences=True: THIS IS KEY. Since there is another LSTM layer after this, 
# it must pass the full sequence of its findings, not just the final result.
# input_shape: Tells the model (number of time steps, number of features).
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# 3. First Dropout Layer
# 0.2 means 20% of the neurons are randomly "turned off" during each training step.
# This prevents the model from "memorizing" (overfitting) the training data.
model.add(Dropout(0.2))

# 4. Second LSTM Layer
# return_sequences=False (Default): Since this is the last LSTM layer, 
# it summarizes everything it learned into a single vector to pass to the final output.
model.add(LSTM(units=128))

# 5. Second Dropout Layer
model.add(Dropout(0.2))

# 6. Dense Layer
# The output. units=1 means we are predicting a single number (like tomorrow's price).
model.add(Dense(1))

# 7. Compilation
# optimizer='adam': The "brain" that adjusts weights. Adam is the gold standard.
# loss='mean_squared_error': How we measure "wrongness" for numerical predictions.
model.compile(optimizer='adam', loss='mean_squared_error')

# Training and Evaluating the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(np.mean(y_test - predictions) **  2)
print(f'RMSE: {rmse:.2f}')

# Visualize model prediction
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Production')
plt.plot(dates_test, predictions, label='Predicted Production')
plt.title('Actual vs Predicted Milk Production')
plt.xlabel('Date')
plt.ylabel('Production (pounds per cow)')
plt.legend()
plt.show()