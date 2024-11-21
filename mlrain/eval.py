from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report

from .logger import init_logger

logger = init_logger(__name__)

def evaluate_model(y_prediction: DataFrame, y_true_label: DataFrame):


    logger.info(f"\n  {confusion_matrix(y_true_label, y_prediction)}")
    logger.info(f"\n {classification_report(y_true_label, y_prediction)}")

