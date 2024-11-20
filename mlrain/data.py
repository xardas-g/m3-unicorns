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
COL_LOCATION = "Location"
COL_LONG = "Longitude"
COL_LAT = "Latitude"

class Dataset:


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

        df_transformed = df_transformed.drop(COL_DATE, axis=1)

        locations_coordinates = [
            {"Ort": "Albury", "Breitengrad": -36.080780, "Längengrad": 146.916473},
            {"Ort": "BadgerysCreek", "Breitengrad": -33.87972, "Längengrad": 150.75222},
            {"Ort": "Cobar", "Breitengrad": -31.49972, "Längengrad": 145.83194},
            {"Ort": "CoffsHarbour", "Breitengrad": -30.30222, "Längengrad": 153.11889},
            {"Ort": "Moree", "Breitengrad": -29.46583, "Längengrad": 149.83389},
            {"Ort": "Newcastle", "Breitengrad": -32.92953, "Längengrad": 151.7801},
            {"Ort": "NorahHead", "Breitengrad": -33.28250, "Längengrad": 151.57417},
            {"Ort": "NorfolkIsland", "Breitengrad": -29.0328, "Längengrad": 167.9544},
            {"Ort": "Penrith", "Breitengrad": -33.758011, "Längengrad": 150.705444},
            {"Ort": "Richmond", "Breitengrad": -37.541290, "Längengrad": -77.434769},
            {"Ort": "Sydney", "Breitengrad": -33.865143, "Längengrad": 151.209900},
            {"Ort": "SydneyAirport", "Breitengrad": -33.947346, "Längengrad": 151.179428},
            {"Ort": "WaggaWagga", "Breitengrad": -35.12577, "Längengrad": 147.35375},
            {"Ort": "Williamtown", "Breitengrad": -32.81500, "Längengrad": 151.84278},
            {"Ort": "Wollongong", "Breitengrad": -34.424, "Längengrad": 150.89345},
            {"Ort": "Canberra", "Breitengrad": -35.282001, "Längengrad": 149.128998},
            {"Ort": "Tuggeranong", "Breitengrad": -35.4244, "Längengrad": 149.0888},
            {"Ort": "MountGinini", "Breitengrad": -35.5333, "Längengrad": 148.7833},
            {"Ort": "Ballarat", "Breitengrad": -37.56622, "Längengrad": 143.84957},
            {"Ort": "Bendigo", "Breitengrad": -36.75818, "Längengrad": 144.28024},
            {"Ort": "Sale", "Breitengrad": -38.1069, "Längengrad": 147.0637},
            {"Ort": "MelbourneAirport", "Breitengrad": -37.6690, "Längengrad": 144.8410},
            {"Ort": "Melbourne", "Breitengrad": -37.8136, "Längengrad": 144.9631},
            {"Ort": "Mildura", "Breitengrad": -34.1850, "Längengrad": 142.1625},
            {"Ort": "Nhil", "Breitengrad": -36.3340, "Längengrad": 141.6500},
            {"Ort": "Portland", "Breitengrad": -38.3496, "Längengrad": 141.6043},
            {"Ort": "Watsonia", "Breitengrad": -37.7167, "Längengrad": 145.0833},
            {"Ort": "Dartmoor", "Breitengrad": -37.9167, "Längengrad": 141.2833},
            {"Ort": "Brisbane", "Breitengrad": -27.4698, "Längengrad": 153.0251},
            {"Ort": "Cairns", "Breitengrad": -16.9203, "Längengrad": 145.7710},
            {"Ort": "GoldCoast", "Breitengrad": -28.0167, "Längengrad": 153.4000},
            {"Ort": "Townsville", "Breitengrad": -19.2589, "Längengrad": 146.8169},
            {"Ort": "Adelaide", "Breitengrad": -34.9285, "Längengrad": 138.6007},
            {"Ort": "MountGambier", "Breitengrad": -37.8318, "Längengrad": 140.7792},
            {"Ort": "Nuriootpa", "Breitengrad": -34.4700, "Längengrad": 138.9960},
            {"Ort": "Woomera", "Breitengrad": -31.1983, "Längengrad": 136.8250},
            {"Ort": "Albany", "Breitengrad": -35.0228, "Längengrad": 117.8814},
            {"Ort": "Witchcliffe", "Breitengrad": -34.0167, "Längengrad": 115.1000},
            {"Ort": "PearceRAAF", "Breitengrad": -31.6670, "Längengrad": 116.0170},
            {"Ort": "PerthAirport", "Breitengrad": -31.9403, "Längengrad": 115.9668},
            {"Ort": "Perth", "Breitengrad": -31.9505, "Längengrad": 115.8605},
            {"Ort": "SalmonGums", "Breitengrad": -32.9833, "Längengrad": 121.6333},
            {"Ort": "Walpole", "Breitengrad": -34.9500, "Längengrad": 116.7333},
            {"Ort": "Hobart", "Breitengrad": -42.8821, "Längengrad": 147.3272},
            {"Ort": "Launceston", "Breitengrad": -41.4332, "Längengrad": 147.1441},
            {"Ort": "AliceSprings", "Breitengrad": -23.6980, "Längengrad": 133.8807},
            {"Ort": "Darwin", "Breitengrad": -12.4634, "Längengrad": 130.8456},
            {"Ort": "Katherine", "Breitengrad": -14.4650, "Längengrad": 132.2635},
            {"Ort": "Uluru", "Breitengrad": -25.3444, "Längengrad": 131.0369}
        ]

        df_locations = pd.DataFrame(locations_coordinates)

        df_transformed = df_transformed.merge(df_locations, how='left', left_on=COL_LOCATION, right_on='Ort')

        # Aktualisiere die lat und lng Spalten mit den entsprechenden Werten
        df_transformed[COL_LAT] = df_transformed['Breitengrad']
        df_transformed[COL_LONG] = df_transformed['Längengrad']

        # Entferne unnötige Spalten
        df_transformed = df_transformed.drop(columns=['Ort', 'Breitengrad', 'Längengrad'])

        # df_transformed = df_transformed.dropna()

        return df_transformed

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        df_transformed = self.transform_data(df)
        return df_transformed.drop(columns=COL_RAINTOMORROW), df_transformed[COL_RAINTOMORROW]
