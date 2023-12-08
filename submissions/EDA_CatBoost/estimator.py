import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

import os

import holidays
from lockdowndates.core import LockdownDates
import haversine as hs
from datetime import datetime
from meteostat import Point, Hourly
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np


def _read_data(path):
    _target_column_name = 'log_bike_count'
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X_train, y_train = _read_data('../../data/train.parquet')
X_test, y_test = _read_data('../../data/test.parquet')
X_final = pd.read_parquet('../../data/final_test.parquet')

X_train = pd.concat([X_train, X_test], ignore_index=True, sort=False)
y_train = np.concatenate((y_train, y_test), axis=0)

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "date_ws"] = X["date"].dt.date.astype('datetime64[ns]') # To be used for merging
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "week_of_year"] = X["date"].dt.isocalendar().week
    X.loc[:, "season"] = X["week_of_year"].apply(lambda w: (w // 13) % 4 + 1)

    # Add the column corresponding to holidays
    fr_holidays = holidays.FR(years=X["year"].unique().tolist())
    X.loc[: ,'holidays'] = X['date'].apply(lambda x: x in fr_holidays or x.weekday() >= 5).astype(int)

    # Add covid restrictions
    ld = LockdownDates("France", "2020-09-01", "2022-01-01", ("stay_at_home", "masks"))
    lockdown_dates = ld.dates()
    X = X.reset_index().merge(lockdown_dates['france_stay_at_home'], how='left', left_on='date_ws', right_index=True).set_index('index')
    
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
    X.drop('week_of_year', axis=1, inplace=True)

    # Seasons
    seasons_in_year = 4
    X['sin_season'] = np.sin(2*np.pi*X.season/seasons_in_year)
    X['cos_season'] = np.cos(2*np.pi*X.season/seasons_in_year)
    X.drop('season', axis=1, inplace=True)

    return X

def _closest_transport(X): 
    column_names = ['longitude', 'latitude', 'station_name']  # Replace with your actual column names
    idf_stations = pd.read_csv('../../data/Stations_IDF.csv', delimiter=';', header=None, names=column_names)

    X = X.copy()
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['counter_id', 'closest_metro_distance'])

    # Iterate over unique counter_ids in X
    for counter_id in X['counter_id'].unique():
        coordinates_counter = (X.loc[X['counter_id'] == counter_id, 'latitude'].values[0],
                               X.loc[X['counter_id'] == counter_id, 'longitude'].values[0])

        # Calculate distances to all metro stations
        distances = []
        for _, station_row in idf_stations.iterrows():
            coordinates_station = (station_row['latitude'], station_row['longitude'])
            distance = hs.haversine(coordinates_counter, coordinates_station)
            distances.append(distance)

        # Get the k closest distances
        closest_distance = sorted(distances)[0]

        # Append to the result DataFrame
        result_df = pd.concat([result_df, pd.DataFrame({'counter_id': [counter_id], 'closest_metro_distance': [closest_distance]})])
    
    result_df = result_df.set_index('counter_id')
    X = X.reset_index().merge(result_df['closest_metro_distance'], how='left', left_on='counter_id', right_index=True).set_index('index')
        
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
        weather_data = Hourly(counter_point, start, end).fetch()
        selected_columns = ['temp', 'rhum', 'wspd', 'prcp']
        weather_data['prcp'] = weather_data['prcp'].fillna(method='ffill')
        weather_data = weather_data[selected_columns].shift(-1)

        # Append counter_id to the weather_data DataFrame
        weather_data['counter_id'] = counter_id

        # Append the DataFrame to the list
        dfs.append(weather_data)

    # Concatenate all DataFrames into a single result_df
    result_df = pd.concat(dfs)
    
    # Convert 'time' index to 'date' column for merging
    result_df['date'] = result_df.index
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Merge the result DataFrame with the original DataFrame on 'counter_id' and 'date'
    X = X.reset_index().merge(result_df, how='left', on=['counter_id', 'date']).set_index('index')

    return X.drop(columns=["date", "counter_name", "site_id", "site_name", 
                           "counter_installation_date", "counter_technical_id",
                           "longitude", "latitude", "date_ws"])

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    transport_encoder = FunctionTransformer(_closest_transport)
    weather_encoder = FunctionTransformer(_add_weather_data)
    
    date_cols = ["sin_month", "sin_day", "sin_hour", "sin_weekyear", "sin_weekday", "sin_season",
                 "cos_month", "cos_day", "cos_hour", "cos_weekyear", "cos_weekday", "cos_season"]

    categorical_cols = ["counter_id", "closest_metro_distance", "holidays", "france_stay_at_home", "year"]
    numerical_cols = ['rhum', 'wspd', 'prcp', 'temp']

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", 'passthrough', numerical_cols + date_cols)
        ]
    )
    regressor = CatBoostRegressor()

    pipe = make_pipeline(date_encoder, transport_encoder, weather_encoder,
                         preprocessor, regressor)

    return pipe

def get_estimator_counter(X_train, y_train):
    counters = X_train['counter_id'].unique()
    estimator_counter = {}

    for counter_id in counters:
        X_subset = X_train[X_train['counter_id'] == counter_id]
        y_subset = y_train[X_train['counter_id'] == counter_id]

        pipe_counter = get_estimator()

        pipe_counter.fit(X_subset, y_subset)

        estimator_counter[counter_id] = pipe_counter

    return estimator_counter

def predict_counters(X_final, estimator_counter):
    predictions = []

    for counter_id, estimator in estimator_counter.items():
        X_subset = X_final[X_final['counter_id'] == counter_id]
        indices_subset = X_subset.index

        y_pred_subset = estimator.predict(X_subset)
        y_pred_subset[y_pred_subset < 0] = 0

        predictions.extend(zip(indices_subset, y_pred_subset))

    predictions = sorted(predictions, key=lambda x: x[0])

    indices, y_pred = zip(*predictions)

    return np.array(y_pred), np.array(indices)

estimators = get_estimator_counter(X_train, y_train)

y_pred, indices = predict_counters(X_final, estimators)

results = pd.DataFrame(

    dict(
        Id=indices,
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)