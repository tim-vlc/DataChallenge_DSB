import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
import holidays
from lockdowndates.core import LockdownDates
import haversine as hs
from datetime import datetime
from meteostat import Point, Hourly
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
from pathlib import Path
from datetime import datetime
from meteostat import Point, Hourly


def _read_data():
    _target_column_name = 'log_bike_count'
    data = pd.read_parquet(Path("data") / "train.parquet")
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X_train, y_train = _read_data()
X_final = pd.read_parquet(Path("data") / "final_test.parquet")


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
    
    # Season
    seasons_in_year = 4
    X['sin_season'] = np.sin(2*np.pi*X.season/seasons_in_year)
    X['cos_season'] = np.cos(2*np.pi*X.season/seasons_in_year)
    X.drop('season', axis=1, inplace=True)

    return X

def _closest_transport(X): 
    column_names = ['longitude', 'latitude', 'station_name']  # Replace with your actual column names
    idf_stations = pd.read_csv(Path("data") / "Stations_IDF.csv", delimiter=';', header=None, names=column_names)

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

from datetime import datetime
from meteostat import Point, Hourly

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
        selected_columns = ['temp', 'rhum', 'wspd', 'prcp', 'coco']
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
                           "counter_installation_date", "coordinates", "counter_technical_id",
                           "longitude", "latitude", "date_ws"])

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    transport_encoder = FunctionTransformer(_closest_transport)
    weather_encoder = FunctionTransformer(_add_weather_data)
    
    date_cols = ["sin_month", "sin_day", "sin_hour", "sin_weekyear", "sin_weekday",
                 "cos_month", "cos_day", "cos_hour", "cos_weekyear", "cos_weekday"]

    categorical_cols = ["counter_id", "closest_metro_distance", "holidays", "france_stay_at_home", "year", "coco"]
    numerical_cols = ['temp', 'rhum', 'wspd', 'prcp']

    preprocessor = ColumnTransformer(
        [
            ("date", 'passthrough', date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )
    regressor = CatBoostRegressor(learning_rate=0.03335437372136638, depth=8, subsample=0.3001434590043754, colsample_bylevel=0.3053639610152637, min_data_in_leaf=17)

    pipe = make_pipeline(date_encoder, transport_encoder, weather_encoder, preprocessor, regressor)

    return pipe

pipe = get_estimator()

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_final)
results = pd.DataFrame(

    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
