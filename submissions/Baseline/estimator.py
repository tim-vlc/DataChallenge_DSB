from pathlib import Path
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {
    'baseline':None,
    'ridgeregression':Ridge(),
    'randomforestregressor':RandomForestRegressor(max_depth=10),
    'lgbmregressor': LGBMRegressor(),
    'catboostregressor':CatBoostRegressor(),
    'xgbregressor':XGBRegressor()
}

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

train_data = X_train.copy()
train_data['log_bike_count'] = y_train

counters_dict = {"20 Avenue de Clichy NO-SE":(pd.to_datetime("2021/04/09"), pd.to_datetime("2021/07/21")),
            "20 Avenue de Clichy SE-NO":(pd.to_datetime("2021/04/09"), pd.to_datetime("2021/07/21")),
            "152 boulevard du Montparnasse E-O":(pd.to_datetime("2021/01/26"), pd.to_datetime("2021/02/24")),
            "152 boulevard du Montparnasse O-E":(pd.to_datetime("2021/01/26"), pd.to_datetime("2021/02/24"))}

for counter, (start, end) in counters_dict.items():
    mask = (
        (train_data["counter_name"] == counter)
        & (train_data["date"] > start)
        & (train_data["date"] < end)
    )

    train_data = train_data[~mask]

X_train = train_data[X_train.columns]
y_train = train_data['log_bike_count'].to_numpy()

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

def get_estimator(regressor):
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe

for regressor_name, regressor in models.items():
    start_time = time.time()
    if regressor is not None:
        pipe = get_estimator(regressor)
        pipe.fit(X_train, y_train)

        print(
            f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}"
        )

        cv = TimeSeriesSplit(n_splits=5)

        # When using a scorer in scikit-learn it always needs to be better when smaller, hence the minus sign.
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        print("RMSE: ", scores)
        print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")
        end_time = time.time()
        elapsed_time = end_time - start_time

        with open("results.txt", "a") as file:
            file.write(f"The 5-CV score for {regressor_name} is: {(-scores.mean()):.3} ± {(-scores).std():.3}\n")
            file.write(f"The test score for {regressor_name} is: {mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}\n")
            file.write(f"{regressor_name} executed in: {elapsed_time} seconds\n")
    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open("results.txt", "a") as file:
                file.write(f"The test score for {regressor_name} is: {mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)), squared=False):.2f}\n")
                file.write(f"{regressor_name} executed in: {elapsed_time} seconds\n")