from typing import Tuple, Optional
import os

import pandas as pd

COL_MINTEMP = "MinTemp"
COL_MAXTEMP = "MaxTemp"
COL_RAINFALL = "Rainfall"
COL_EVAPORATION = "Evaporation"
COL_SUNSHINE = "Sunshine"
COL_WINDGUSTDIR = "WindGustDir"
COL_WINDGUSTSPEED = "WindGustSpeed"
COL_WINDDIR9AM = "WindDir9am"
COL_WINDDIR3PM = "WindDir3pm"
COL_WINDSPEED9AM = "WindSpeed9am"
COL_WINDSPEED3PM = "WindSpeed3pm"
COL_HUMIDITY9AM = "Humidity9am"
COL_HUMIDITY3PM = "Humidity3pm"
COL_PRESSURE9AM = "Pressure9am"
COL_PRESSURE3PM = "Pressure3pm"
COL_CLOUD9AM = "Cloud9am"
COL_CLOUD3PM = "Cloud3pm"
COL_TEMP9AM = "Temp9am"
COL_TEMP3PM = "Temp3pm"
COL_RAINTODAY = "RainToday"
COL_RAINTOMORROW = "RainTomorrow"
COL_DAYOFYEAR = "DayOfYear"
COL_MONTH = "Month"
COL_DATE = "Date"

class Dataset:
    COLUMS = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_WINDSPEED9AM,
     COL_WINDSPEED3PM, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP9AM, COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_MONTH, COL_WINDGUSTDIR, COL_WINDDIR9AM, COL_WINDDIR3PM]


    def __init__(self, num_samples: Optional[int] = None, random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.df_origin = pd.read_csv(os.path.join(os.getcwd(), "data", "weatherAUS.csv")).dropna()


    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        df = self.df_origin
        if self.num_samples is not None:
            df = self.df_origin.sample(self.num_samples, random_state=self.random_seed)
        #df[COL_GEN_POPULARITY_CLASS] = df[COL_POPULARITY].apply(lambda x: CLASS_POPULAR if x >= self.threshold_popular else CLASS_UNPOPULAR)
        return df

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ##%%
        df_transformed = df.copy(deep=True)

        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW",
                      "NNW"]

        # Convert directions into degrees
        # Each step is 360 / 16 = 22.5 degrees
        direction_to_number = {direction: i * 22.5 for i, direction in enumerate(directions)}

        numbers = []

        # Print the mapping
        for direction, number in direction_to_number.items():
            numbers.append(number)

        df_transformed[COL_WINDGUSTDIR] = df_transformed[COL_WINDGUSTDIR].replace(to_replace=directions, value=numbers)
        df_transformed[COL_WINDDIR3PM] = df_transformed[COL_WINDDIR3PM].replace(to_replace=directions, value=numbers)
        df_transformed[COL_WINDDIR9AM] = df_transformed[COL_WINDDIR9AM].replace(to_replace=directions, value=numbers)

        df_transformed[COL_RAINTOMORROW] = df_transformed[COL_RAINTOMORROW].replace(to_replace=['No', 'Yes'], value=[0, 1])
        df_transformed[COL_RAINTODAY] = df_transformed[COL_RAINTODAY].replace(to_replace=['No', 'Yes'], value=[0, 1])

        df_transformed[COL_DATE] = pd.to_datetime(df_transformed[COL_DATE])
        df_transformed[COL_DAYOFYEAR] = df_transformed.Date.dt.dayofyear
        df_transformed[COL_MONTH] = df_transformed.Date.dt.month

        return df_transformed

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        df_transformed = self.transform_data(df)
        return df_transformed.drop(columns=COL_RAINTOMORROW), df_transformed[COL_RAINTOMORROW]
