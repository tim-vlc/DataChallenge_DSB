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
from sklearn.decomposition import PCA

from meteostat import Point, Hourly
from datetime import datetime

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

def _encode_dates(X):
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
    X.loc[:, "date_ws"] = X["date"].dt.date.astype('datetime64[ns]') # To be used for merging

    # Finally we can drop the original columns from the dataframe
    return X

def _add_weather_data(X):
    X = X.copy()
    dfs = []

    for counter_id in X['counter_id'].unique():
        # Get the coordinates of the counter
        coordinates_counter = (
            X.loc[X['counter_id'] == counter_id, 'latitude'].values[0],
            X.loc[X['counter_id'] == counter_id, 'longitude'].values[0]
        )

        # Create a Point object with the counter's coordinates
        counter_point = Point(*coordinates_counter)

        # Define the time range (start and end dates)
        start = datetime(2020, 8, 1)
        end = datetime(2022, 1, 1)

        # Create a Hourly object and fetch the weather data
        weather_data = Hourly(counter_point, start, end).interpolate().fetch()
        selected_columns = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
        weather_data = weather_data[selected_columns].shift(-1)

        # Append counter_id to the weather_data DataFrame
        weather_data['counter_id'] = counter_id

        # Append the DataFrame to the list
        dfs.append(weather_data)

    # Concatenate all DataFrames into a single result_df
    result_df = pd.concat([df for df in dfs if not df.empty])
    
    # Convert 'time' index to 'date' column for merging
    result_df['date'] = result_df.index
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Merge the result DataFrame with the original DataFrame on 'counter_id' and 'date'
    X = X.reset_index().merge(result_df, how='left', on=['counter_id', 'date']).set_index('index')

    return X.drop(columns=["date_ws", "date"])

def get_estimator(pca_dim):
    date_encoder = FunctionTransformer(_encode_dates)
    weather_encoder = FunctionTransformer(_add_weather_data)

    categorical_cols = ["counter_name", "site_name", "year", "weekend"]
    date_cols = ["month", "day", "weekday", "hour", "week_of_year", "season"]
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    preprocessor = ColumnTransformer(
        [
            ("date", 'passthrough', date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("pca_weather", make_pipeline(StandardScaler(), PCA(n_components=pca_dim)), weather_cols),
            ("drop_weather", 'drop', weather_cols)
        ]
    )

    regressor = CatBoostRegressor()

    pipe = make_pipeline(date_encoder, weather_encoder, preprocessor, regressor)

    return pipe

PCA_dims = np.arange(2, 9, 1)
print(PCA_dims)

for pca_dim in PCA_dims:    
    pipe = get_estimator(pca_dim)
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
        file.write(f"The 5-CV score for PCA={pca_dim} is: {(-scores.mean()):.3} ± {(-scores).std():.3}\n")
        file.write(f"The test score for PCA={pca_dim} is: {mean_squared_error(y_test, pipe.predict(X_test), squared=False):.3f}\n")