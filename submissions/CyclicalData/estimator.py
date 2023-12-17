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

def _encode_dates1(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "week_of_year"] = X["date"].dt.isocalendar().week
    X.loc[:, "season"] = X["week_of_year"].apply(lambda w: (w // 13) % 4 + 1)
    X.loc[: ,'weekend'] = X['date'].apply(lambda x: x.weekday() >= 5).astype(int)
    
    # Month
    months_in_year = 12
    X['sin_month'] = np.sin(2*np.pi*X.month/months_in_year)
    X['cos_month'] = np.cos(2*np.pi*X.month/months_in_year)
    X.drop('month', axis=1, inplace=True)
    
    # Day
    day_in_month = 31
    X['sin_day'] = np.sin(2*np.pi*X.day/day_in_month)
    X['cos_day'] = np.cos(2*np.pi*X.day/day_in_month)
    X.drop('day', axis=1, inplace=True)
    
    # Weekday
    day_in_week = 7
    X['sin_weekday'] = np.sin(2*np.pi*X.weekday/day_in_week)
    X['cos_weekday'] = np.cos(2*np.pi*X.weekday/day_in_week)
    X.drop('weekday', axis=1, inplace=True)
    
    # Hour
    hours_in_day = 24
    X['sin_hour'] = np.sin(2*np.pi*X.hour/hours_in_day)
    X['cos_hour'] = np.cos(2*np.pi*X.hour/hours_in_day)
    X.drop('hour', axis=1, inplace=True)
    
    # Week of year
    weeks_in_year = 52
    X['sin_weekyear'] = np.sin(2*np.pi*X.week_of_year/weeks_in_year)
    X['cos_weekyear'] = np.cos(2*np.pi*X.week_of_year/weeks_in_year)
    
    # Season
    seasons_in_year = 4
    X['sin_season'] = np.sin(2*np.pi*X.season/seasons_in_year)
    X['cos_season'] = np.cos(2*np.pi*X.season/seasons_in_year)
    X.drop('season', axis=1, inplace=True)

    return X.drop(columns=["date"])

def _encode_dates2(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "week_of_year"] = X["date"].dt.isocalendar().week
    X.loc[:, "season"] = X["week_of_year"].apply(lambda w: (w // 13) % 4 + 1)
    X.loc[: ,'weekend'] = X['date'].apply(lambda x: x.weekday() >= 5).astype(int)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def get_estimator(_encode_dates, date_cols):
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_cols = ["counter_name", "site_name", "year", "weekend"]

    preprocessor = ColumnTransformer(
        [
            ("date", 'passthrough', date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    regressor = CatBoostRegressor()

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe

date_cols1 = ["sin_month", "sin_day", "sin_hour", "sin_weekday", "sin_season", "sin_weekyear",
                 "cos_month", "cos_day", "cos_hour", "cos_weekday", "cos_season", "cos_weekyear"]
date_cols2 = ["month", "day", "weekday", "hour", "week_of_year", "season"]

cyclical = {
    'fourier':(_encode_dates1, date_cols1),
    'none':(_encode_dates2, date_cols2)
}

for cyclical_name, (function, date_cols) in cyclical.items():    
    pipe = get_estimator(function, date_cols)
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

    with open("results.txt", "a") as file:
        file.write(f"The 5-CV score for {cyclical_name} is: {(-scores.mean()):.3} ± {(-scores).std():.3}\n")
        file.write(f"The test score for {cyclical_name} is: {mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}\n")