#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 22:32:26 2023

@author: candilsiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

def window_data(data, n=3):
    windowed_data = pd.DataFrame()
    for i in range(n, 0, -1):
        windowed_data[f'Target-{i}'] = data['Close'].shift(i)
    windowed_data['Target'] = data['Close']
    return windowed_data.dropna()

def windowed_df_to_date_X_y(windowed_dataframe):
    
  df_as_np = windowed_dataframe.to_numpy()
  dates = windowed_dataframe.index.to_numpy()
  middle_matrix = df_as_np[:, 0:-1]
  
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
  Y = df_as_np[:, -1]
  
  return dates, X.astype(np.float32), Y.astype(np.float32)

                
df = pd.read_csv("MSFT.csv")

df = df[["Date","Close"]]
df['Date']= pd.to_datetime(df['Date'])
df.set_index("Date", inplace = True)
df = df.loc['2022-03-22':'2023-03-27']

plt.plot(df.index, df["Close"])
plt.show()

windowed_df = window_data(df)
dates, X, y = windowed_df_to_date_X_y(windowed_df)

print(dates.shape, X.shape, y.shape)

per80 = int(len(dates) * 0.8)
per90 = int(len(dates) * 0.9)

dates_train, feature_train, label_train = dates[:per80], X[:per80], y[:per80]
dates_val, feature_val, label_val = dates[per80:per90], X[per80:per90], y[per80:per90]
dates_test, feature_test, label_test = dates[per90:], X[per90:], y[per90:]

plt.plot(dates_train, label_train)
plt.plot(dates_val, label_val)
plt.plot(dates_test, label_test)
plt.legend(['Train', 'Validation', 'Test'])
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(feature_train, label_train, validation_data=(feature_val, label_val), epochs=100)

train_predictions = model.predict(feature_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, label_train)
plt.legend(['Training Predictions', 'Training Observations'])
plt.show()

val_predictions = model.predict(feature_val).flatten()

plt.plot(dates_val, val_predictions)
plt.plot(dates_val, label_val)
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.show()

test_predictions = model.predict(feature_test).flatten()

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, label_test)
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.show()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, label_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, label_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, label_test)
plt.legend(['Training Predictions', 
            'Training Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations'])
plt.show()
