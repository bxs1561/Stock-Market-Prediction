import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf


# read the dataset
dataset_train = pd.read_csv('google_stock_price.csv')
dataset_train.head()

# Normalizing the dataset for close
# training_set = dataset_train.iloc[:,1:2].values
training_set = dataset_train.iloc[:, 4:5].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(training_set)

# creating x_train and y_train
train_data=training_set[0:3102,:]
valid_data=training_set[3102:,:]

X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])  # The next day's opening price

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape X_train to be 3-dimensional (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))  # Add dropout with 20% probability

lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))  # Add dropout with 20% probability

lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))  # Add dropout with 20% probability

lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))  # Add dropout with 20% probability

lstm_model.add(Dense(units=1))

# compile the model
lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# input for the model
dataset_test = dataset_train.iloc[:, 4:5].values
inputs = dataset_test[len(dataset_test) - len(valid_data) - 60:]

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

#predicit values
predicted_stock_price=lstm_model.predict(X_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)

#save the model
lstm_model.save("saved_model.h5")


# Prepare the data for plotting
train_data = dataset_train.iloc[:3102, :]
valid_data = dataset_train.iloc[3102: len(dataset_train), :]

valid_data['Predictions'] = predicted_stock_price

plt.figure(figsize=(14, 5))
plt.plot(train_data["Date"], train_data["Close"], label='Training Data')
plt.plot(valid_data["Date"], valid_data["Close"], label='Actual Stock Price')
plt.plot(valid_data["Date"], valid_data["Predictions"], label='Predicted Stock Price', linestyle='dashed')
plt.title('Google Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Google Stock Price')
plt.xticks(np.arange(0, len(dataset_train), step=300), rotation=45)
plt.legend()
plt.show()



# plt.plot(dataset_test[-len(valid_data):], color='red', label='Actual Google Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()
