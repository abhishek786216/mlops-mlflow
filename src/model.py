import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='abhishek786216', repo_name='mlops-mlflow', mlflow=True)
# Set tracking URI
# mlflow.set_tracking_uri("https://dagshub.com/abhishek786216/mlops-mlflow/experiments")

# Prepare data
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Parameters
max_depth = 5
n_estimators = 30
mlflow.autolog()

mlflow.set_experiment('exp1')

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_artifact(__file__)


    print("Run logged with accuracy:", accuracy)
