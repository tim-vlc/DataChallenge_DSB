import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA

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

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "date_ws"] = X["date"].dt.date.astype('datetime64[ns]') # To be used for merging
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = (X["date"].dt.day / X["date"].dt.to_period("M").dt.daysinmonth) * 31
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "week_of_year"] = X["date"].dt.isocalendar().week
    X.loc[:, "season"] = X["week_of_year"].apply(lambda w: (w // 13) % 4 + 1)

    # Add the column corresponding to jours fériés + week-ends
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
    weeks_in_year = 53
    X['sin_weekyear'] = np.sin(2*np.pi*X.week_of_year/weeks_in_year)
    X['cos_weekyear'] = np.cos(2*np.pi*X.week_of_year/weeks_in_year)
    X.drop('week_of_year', axis=1, inplace=True)
    
    # Season
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
        new_df = pd.DataFrame({'counter_id': [counter_id], 'closest_metro_distance': [closest_distance]})
        result_df = ( new_df if result_df.empty else pd.concat([result_df, new_df]) )
    
    result_df = result_df.set_index('counter_id')
    X = X.reset_index().merge(result_df['closest_metro_distance'], how='left', left_on='counter_id', right_index=True).set_index('index')
        
    return X

def _add_greves_data(X):
    X = X.copy()

    greves = pd.read_csv('../../data/greves-sncf2002.csv',sep = ";", encoding='latin').iloc[:35, :]
    greves['Date'] = pd.to_datetime(greves['Date'], format="%d/%m/%Y")

    X = pd.merge(X, greves, left_on='date_ws', right_on='Date', how='left')
    X['greves'] = X['Taux impact grevistes'].fillna(0)

    # Drop unnecessary columns
    X = X.drop(['Date', 'Taux impact grevistes'], axis=1)
    
    # X.loc[:, 'greves'] = X['date_ws'].isin(greves['Date'].dt.date).astype(int)

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
        selected_columns = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']
        weather_data['prcp'] = weather_data['prcp'].ffill()
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

    return X.drop(columns=["date", "counter_name", "site_id", "site_name", 
                           "counter_installation_date", "counter_technical_id",
                           "longitude", "latitude", "date_ws"])

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    transport_encoder = FunctionTransformer(_closest_transport)
    weather_encoder = FunctionTransformer(_add_weather_data)
    greves_encoder = FunctionTransformer(_add_greves_data)
    
    date_cols = ["sin_month", "sin_day", "sin_hour", "sin_weekday", "sin_season", "sin_weekyear",
                 "cos_month", "cos_day", "cos_hour", "cos_weekday", "cos_season", "cos_weekyear"]

    categorical_cols = ["counter_id", "holidays", "france_stay_at_home", "year"]
    numerical_cols = ["closest_metro_distance", "greves"]
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres', 'coco']

    preprocessor = ColumnTransformer(
        [
            ("date", 'passthrough', date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", 'passthrough', numerical_cols),
            ("pca_weather", make_pipeline(StandardScaler(), PCA(n_components=5)), weather_cols),
            ("drop_weather", 'drop', weather_cols),
        ]
    )

    regressor = CatBoostRegressor()

    pipe = make_pipeline(date_encoder, transport_encoder, greves_encoder, weather_encoder,
                         preprocessor, regressor)

    return pipe

pipe = get_estimator()
pipe.fit(X_train, y_train)

# Generate Predictions

cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
)
print("RMSE: ", scores)
print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")

print(
    f"Test set, RMSE={mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}"
)

# Feature Importance
catboost_regressor = pipe.named_steps['catboostregressor']  # Adjust the name if necessary

feature_importance = catboost_regressor.get_feature_importance()
feature_names = pipe.named_steps['columntransformer'].get_feature_names_out()
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Print or visualize feature importance
print("Feature Importance:")
for feature, importance in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")

y_pred = pipe.predict(X_final)
results = pd.DataFrame(

    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)