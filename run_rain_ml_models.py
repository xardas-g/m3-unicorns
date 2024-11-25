from unittest import main

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from mlrain.eval import evaluate_model
from mlrain.logger import init_logger
from mlrain.data import *
from mlrain.model_factory import ModelFactory

logger = init_logger(__name__)


def run_ml_experiment():
    dset = Dataset()

    X,y = dset.load_xy()
    # balancer = SMOTE()
    balancer = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X,y = balancer.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

    #model = ModelFactory.create_decision_tree_orig()

    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))

    models = [
        ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_REDUCED),
        ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_knn_orig(ModelFactory.COLS_REDUCED),
        ModelFactory.create_knn_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_random_forest_orig(ModelFactory.COLS_REDUCED),
        ModelFactory.create_random_forest_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_decision_tree_orig(ModelFactory.COLS_REDUCED),
        ModelFactory.create_decision_tree_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_xgb_gradient_with_hotencode_location(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_xgb_gradient_no_hotencode_location(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
        ModelFactory.create_xgb()
    ]

    model_result = []
    run_id = 0


        # evaluate models
    for model in models:
        logger.info(f"RUN_ID: {run_id}\nEvaluating model: {model.get_name()} \nConfig: {model.get_feature_generator()}\n{model.get_feature_transformer_chain()}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_result.append([evaluate_model(y_pred,y_test, run_id),model.get_name()])
        run_id += 1




    logger.info(f"Result: \n {pd.DataFrame(model_result, columns=["F1-Score", "Model"])}")


if __name__ == "__main__":
    run_ml_experiment()