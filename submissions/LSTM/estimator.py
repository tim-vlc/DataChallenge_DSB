import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from pathlib import Path
import os
import numpy as np

def _read_data(path, f_name):
    _target_column_name = "log_bike_count"

    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X_final = pd.read_parquet(Path('../../data/final_test.parquet'))
X_train, y_train = _read_data(path='../../', f_name='train.parquet')
X_test, y_test = _read_data(path='../../', f_name='test.parquet')

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _encode_cat(X):
    X = X.copy()  # modify a copy of X

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["counter_id", "site_id", "counter_technical_id", "latitude", "longitude"])

def _encode_all(data):
    date_encoder = FunctionTransformer(_encode_dates, validate=False)
    categorical_encoder = FunctionTransformer(_encode_cat)

    data = date_encoder.fit_transform(data)
    data = categorical_encoder.fit_transform(data)

    date_cols = ["year", "month", "day", "weekday", "hour"]
    categorical_cols = ["counter_name", "site_name", "counter_installation_date"]

    encoder = OrdinalEncoder()
    scaler = StandardScaler()

    data = encoder.fit_transform(data)
    data = scaler.fit_transform(data)
    return data

X_train = _encode_all(X_train)
X_test = _encode_all(X_test)
X_final = _encode_all(X_final)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_final = X_final.reshape((X_final.shape[0], 1, X_final.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(X_train[:2, :])

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('History.png')