import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from train_test_split import train_test_split

#define price inputs
symbol = 'BTC-USD'
indicators = ['close', 'fisher', 'macd', 'sma', 'wma']
data_frequency = 'hour'
# Define hyperparameters
n_steps = 30
n_epochs = 50
n_batch_size = 32
n_features = len(indicators)
train_percentage = 0.8

# Load the historical price data
data = pd.read_csv(f'data/{symbol}_{data_frequency}_financial_indicators.csv')
#select a subset of the columns of data based on indicators
data = data[indicators]
#create model name
model_name = f'{symbol}_{n_steps}_{n_features}'

#split into training and testing sets
train_data, test_data, X_train, y_train, X_test, y_test, scaler = \
    train_test_split(data,train_percentage=train_percentage, n_steps=n_steps)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) 

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data=(X_test, y_test))

# Make predictions on the testing data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((len(predictions), n_features-1))), axis=1))[:, 0]

# Calculate the root mean squared error
rmse = np.sqrt(np.mean((predictions - test_data['close'].values[n_steps:]) ** 2))
print('RMSE:', rmse)

#save model
model.save(f'models/{model_name}.h5')