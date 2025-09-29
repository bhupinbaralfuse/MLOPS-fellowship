from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
import numpy as np

inference_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inference', 'model')

os.makedirs(inference_folder, exist_ok=True)

os.environ["MLFLOW_ENABLE_MODEL_REGISTRY"] = "false"

iris = load_iris()
X = iris.data
y = (iris.target).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=150)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://mlflow-server:5000")

example = np.array([[5.1, 3.5, 1.4, 0.2]])

mlflow.set_experiment("iris-model")

with mlflow.start_run():
    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.log_param('max_iter', 150)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, name="iris_model", input_example=example)

joblib.dump(model, os.path.join(inference_folder, 'logistic_model.pkl'))