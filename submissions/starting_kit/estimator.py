import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from pathlib import Path
import os
import numpy as np
from xgboost import XGBRegressor

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


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    #regressor = Ridge()
    regressor = XGBRegressor(n_estimators=1000, max_depth=11, eta=0.1, subsample=0.9, colsample_bytree=0.6)

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe

pipe = get_estimator()
pipe.fit(X_train, y_train)

print(
    f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}"
)
print(
    f"Test set, RMSE={mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}"
)

y_pred = pipe.predict(X_final)

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)