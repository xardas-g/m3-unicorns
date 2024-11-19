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
COL_RAINTODAYBOOL = "RainTodayBool"
COL_RAINTOMORROWBOOL = "RainTomorrowBool"
COL_DAYOFYEAR = "DayOfYear"
COL_MONTH = "Month"

class Dataset:
    def __init__(self, num_samples: Optional[int] = None, drop_zero_popularity: bool = False, threshold_popular: int = 50,
            random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param drop_zero_popularity: whether to drop data points where the popularity is zero
        :param threshold_popular: the threshold below which a song is considered as unpopular
        :param random_seed: the random seed to use when sampling data points
        """
        self.num_samples = num_samples
        self.threshold_popular = threshold_popular
        self.drop_zero_popularity = drop_zero_popularity
        self.random_seed = random_seed

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        data_path = os.path.join(os.getcwd(), "data", "weatherAUS.csv")
        df = pd.read_csv(data_path).dropna()
        #if self.drop_zero_popularity:
        #    df = df[df[COL_POPULARITY] > 0]
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        #df[COL_GEN_POPULARITY_CLASS] = df[COL_POPULARITY].apply(lambda x: CLASS_POPULAR if x >= self.threshold_popular else CLASS_UNPOPULAR)
        return df

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=COL_RAINTOMORROWBOOL), df[COL_RAINTOMORROWBOOL]
