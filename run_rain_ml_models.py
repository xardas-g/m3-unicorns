from pprint import pformat

from sensai.util.string import pretty_string_repr
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from mlrain.eval import evaluate_model
from mlrain.logger import init_logger
from mlrain.data import *
from mlrain.model_factory import ModelFactory

logger = init_logger(__name__)

dset = Dataset()

X,y = dset.load_xy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)
#model = ModelFactory.create_decision_tree_orig()

#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
models = [
    # ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_REDUCED),
    # ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    # ModelFactory.create_knn_orig(ModelFactory.COLS_REDUCED),
    # ModelFactory.create_knn_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    # ModelFactory.create_random_forest_orig(ModelFactory.COLS_REDUCED),
    # ModelFactory.create_random_forest_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    ModelFactory.create_decision_tree_orig(ModelFactory.COLS_REDUCED),
    ModelFactory.create_decision_tree_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    ModelFactory.create_xgb_gradient_with_hotencode_location([COL_MINTEMP, COL_MAXTEMP, COL_RAINFALL, COL_EVAPORATION, COL_SUNSHINE, COL_WINDGUSTSPEED, COL_WINDSPEED9AM,
     COL_WINDSPEED3PM, COL_HUMIDITY9AM, COL_HUMIDITY3PM, COL_PRESSURE9AM, COL_PRESSURE3PM, COL_CLOUD9AM, COL_CLOUD3PM,
     COL_TEMP9AM, COL_TEMP3PM, COL_RAINTODAY, COL_DAYOFYEAR, COL_MONTH, COL_YEAR, COL_WINDGUSTDIR, COL_WINDDIR9AM, COL_WINDDIR3PM]),
]

model_result = []

    # evaluate models
for model in models:
    logger.info(f"Evaluating model: {model.get_name()} \nConfig: {model.get_feature_generator()}\n{model.get_feature_transformer_chain()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_result.append([evaluate_model(y_pred,y_test),model.get_name()])





logger.info(f"Result: \n {pd.DataFrame(model_result, columns=["F1-Score", "Model"])}")
