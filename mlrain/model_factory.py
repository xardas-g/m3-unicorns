from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns
from sensai.sklearn.sklearn_classification import SkLearnLogisticRegressionVectorClassificationModel, \
    SkLearnKNeighborsVectorClassificationModel, SkLearnRandomForestVectorClassificationModel, SkLearnDecisionTreeVectorClassificationModel

from .data import *

class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_WINDSPEED9AM,
     COL_WINDSPEED3PM, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP9AM, COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_MONTH, COL_YEAR, COL_WINDGUSTDIR, COL_WINDDIR9AM, COL_WINDDIR3PM, COL_LONG, COL_LAT]

    COLS_REDUCED = [COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_LONG, COL_LAT]

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
            .with_name("DecisionTree-orig")
    
    @classmethod
    def create_logistic_regression_orig(cls, columnsUsed):
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LogisticRegression-orig")

    @classmethod
    def create_knn_orig(cls, columnsUsed):
        return SkLearnKNeighborsVectorClassificationModel(n_neighbors=20) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("KNeighbors-orig")

    @classmethod
    def create_random_forest_orig(cls, columnsUsed):
        return SkLearnRandomForestVectorClassificationModel(n_estimators=100) \
            .with_feature_generator(FeatureGeneratorTakeColumns(columnsUsed)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("RandomForest-orig")