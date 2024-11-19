from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import *

class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_MINTEMP,COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTDIR, COL_WINDGUSTSPEED,
     COL_WINDDIR9AM, COL_WINDDIR3PM, COL_WINDSPEED9AM, COL_WINDSPEED3PM, COL_HUMIDITY9AM, COL_HUMIDITY3PM,
     COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM, COL_TEMP9AM, COL_TEMP3PM, COL_RAINTODAYBOOL,
     COL_RAINTOMORROWBOOL, COL_DAYOFYEAR, COL_MONTH, ]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])

    @classmethod
    def create_knn_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", KNeighborsClassifier(n_neighbors=1))])

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", RandomForestClassifier(n_estimators=100))])

    @classmethod
    def create_decision_tree_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", DecisionTreeClassifier(random_state=42, max_depth=2))])