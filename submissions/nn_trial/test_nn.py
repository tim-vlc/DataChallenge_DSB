import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from pathlib import Path
import os
import numpy as np
from xgboost import XGBRegressor

from nns import *
import torch.optim as optim

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
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _encode_cat(X):
    X = X.copy()  # modify a copy of X

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["counter_id", "site_id", "counter_technical_id", "latitude", "longitude"])


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = FunctionTransformer(_encode_cat)
    categorical_cols = ["counter_name", "site_name", "counter_installation_date"]

    preprocessor = ColumnTransformer(
        [
            ("date", OrdinalEncoder(), date_cols),
            ("cat", OrdinalEncoder(), categorical_cols),
        ]
    )

    scaler = ColumnTransformer(
        [
            ("scaler", StandardScaler(), date_cols + categorical_cols),
        ]
    )

    pipe = make_pipeline(date_encoder, categorical_encoder, preprocessor, scaler)

    return pipe

pipe = get_estimator()
X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)
X_final = pipe.transform(X_final)

print(X_train[:2, :])

input_size = 8
output_size = 1
dense1_output = 128
dense2_output = 256
dense3_output = 512
dense4_output = 128

dropratio = 0.
alpha = 0.0001 # learning rate
batch = 300
ep = 20 # epoch

X_train, y_train, X_test, y_test = (torch.tensor(X_train), torch.tensor(y_train),
                                    torch.tensor(X_test), torch.tensor(y_test))

model = NN(input_size, output_size, dense1_output, dense2_output, dense3_output, dense4_output, dropratio)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)

model.train_nn(X_train, y_train, ep, batch, optimizer, criterion)
model.test_nn(X_test, y_test, batch, criterion)
