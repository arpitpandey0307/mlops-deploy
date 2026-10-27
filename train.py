from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import sys
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score
#Client API lets us manage registered models and their aliases
from mlflow import MlflowClient

# Set the tracking URI for MLflow
mlflow.set_tracking_uri("http://localhost:5000")

#group related runs together
mlflow.set_experiment("Iris_Classification")


#load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 10
random_state = 42

model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}

#start the tracked training run
with mlflow.start_run() as run:
    #Record the training parameters before model fitting
    mlflow.log_params(model_parameters)

    #Build classifier using selected parameters
    #Initialise the Random Forest Classifier with the specified parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    #make predictions
    y_pred = model.predict(X_test)

    #calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #calculate the precision and recall metrics and print them
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    #model quality gate
    #define the minimum acceptable accuracy for now
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(f"Model rejected :Model accuracy {accuracy} is below the minimum threshold of {MIN_TEST_ACCURACY}")
    else:
        print("Model accepted: Model accuracy meets the minimum threshold.")

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    #log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    #save the model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    #log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_random_forest_model",
        registered_model_name="IrisRandomForestModel",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    #Assign the champion alias to the newly registered model version
    client = MlflowClient()
    #identify current training run
    #we use this rum id to find the model version that was just registered for particular run
    current_run_id = mlflow.active_run().info.run_id
    #Retirive all the version registered under this model name
    registered_versions = client.search_model_versions(
        "name = 'IrisRandomForestModel'"
    )

    #find the version that was registered in this run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    client.set_registered_model_alias(
        name="IrisRandomForestModel",
        alias="champion",
        version=current_model_version.version
    )
    print("Registered model in MLflow Model Registry as 'IrisRandomForestModel'")