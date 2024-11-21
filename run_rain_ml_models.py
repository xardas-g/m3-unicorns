from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from mlrain.eval import evaluate_model
from mlrain.logger import init_logger
from mlrain.data import Dataset
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
    ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_REDUCED),
    ModelFactory.create_logistic_regression_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    ModelFactory.create_knn_orig(ModelFactory.COLS_REDUCED),
    ModelFactory.create_knn_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    ModelFactory.create_random_forest_orig(ModelFactory.COLS_REDUCED),
    ModelFactory.create_random_forest_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
    ModelFactory.create_decision_tree_orig(ModelFactory.COLS_REDUCED),
    ModelFactory.create_decision_tree_orig(ModelFactory.COLS_USED_BY_ORIGINAL_MODELS),
]

    # evaluate models
for model in models:
    logger.info(f"Evaluating model: \n {model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_pred,y_test)