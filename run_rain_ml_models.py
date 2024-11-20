from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from Module3.gruppenarbeit.mlrain.data import Dataset
from Module3.gruppenarbeit.mlrain.model_factory import ModelFactory

dset = Dataset()

X,y = dset.load_xy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)
model = ModelFactory.create_decision_tree_orig()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

