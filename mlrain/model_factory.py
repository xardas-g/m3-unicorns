from sensai.xgboost import XGBGradientBoostedVectorClassificationModel
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sensai.data_transformation import DFTSkLearnTransformer, DFTOneHotEncoder
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, SkLearnDecisionTreeVectorClassificationModel

from typing import Sequence

from .data import *
from .features import FeatureName, registry

class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_WINDSPEED9AM,
     COL_WINDSPEED3PM, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP9AM, COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_MONTH, COL_YEAR, COL_WINDGUSTDIR, COL_WINDDIR9AM, COL_WINDDIR3PM, COL_LONG, COL_LAT, COL_ALTITUDE]

    COLS_REDUCED = [COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_LONG, COL_LAT, COL_ALTITUDE]

    DEFAULT_FEATURES = (FeatureName.LOCATION_PARAMS, FeatureName.SUNSHINE_HOURS, FeatureName.MONTH, FeatureName.WINDDIR, FeatureName.HUMIDITY3PM, FeatureName.MEAN_RAIN_DAYS)

#    @classmethod
#    def create_logistic_regression_orig(cls, columnsUsed):
#        return Pipeline([
#            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), columnsUsed)])),
#            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])
#
#    @classmethod
#    def create_knn_orig(cls, columnsUsed):
#        return Pipeline([
#            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), columnsUsed)])),
#            ("model", KNeighborsClassifier(n_neighbors=20))])
#
#    @classmethod
#    def create_random_forest_orig(cls, columnsUsed):
#        return Pipeline([
#            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), columnsUsed)])),
#            ("model", RandomForestClassifier(n_estimators=100))])
#
#    @classmethod
#    def create_decision_tree_orig(cls, columnsUsed):
#        return Pipeline([
#            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), columnsUsed)])),
#            ("model", DecisionTreeClassifier(random_state=42, max_depth=50))])
    @classmethod
    def create_decision_tree_orig(cls, columnsUsed):
        return SkLearnDecisionTreeVectorClassificationModel(max_depth=40) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("DecisionTree-Deep-40")
    
    @classmethod
    def create_logistic_regression_orig(cls, columnsUsed):
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LogisticRegression-lbfgs")

    @classmethod
    def create_knn_orig(cls, columnsUsed):
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=20) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("KNeighbors-orig-n=20")

    @classmethod
    def create_random_forest_orig(cls, columnsUsed):
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("RandomForest-orig-est=100")


    @classmethod
    def create_xgb_gradient_with_hotencode_location(cls, columnsUsed:list):
        return (XGBGradientBoostedVectorClassificationModel() \
            .with_feature_generator(FeatureGeneratorTakeColumns([*columnsUsed, COL_LOCATION])) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler(), columnsUsed))
            .with_feature_transformers(DFTOneHotEncoder(COL_LOCATION), add=True)\
            .with_name("XGBGradientBoostedVector-hotencode-location"))

    @classmethod
    def create_xgb(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, add_features: Sequence[FeatureName] = (),
            min_child_weight: Optional[float] = None, **kwargs):
        fc = FeatureCollector(*features, *add_features, registry=registry)
        return XGBGradientBoostedVectorClassificationModel(min_child_weight=min_child_weight, **kwargs) \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name(f"XGBoost{name_suffix}")
