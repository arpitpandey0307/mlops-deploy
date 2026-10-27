# Iris ML API

An end-to-end machine learning project that trains a Random Forest classifier on
the scikit-learn Iris dataset and serves predictions through a FastAPI
application.

The project includes:

- Reproducible model training with a fixed train/test split.
- Accuracy, precision, and recall evaluation.
- A model-quality gate that rejects models below 90% test accuracy.
- MLflow experiment tracking and model registration.
- A `champion` alias for the newly registered MLflow model version.
- A REST API for Iris predictions.
- Automated tests and Docker image builds through GitHub Actions.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml              # Test and Docker build workflow
├── models/
│   └── model.pkl               # Serialized model used by the API
├── tests/
│   └── test_api.py             # FastAPI endpoint tests
├── Dockerfile                  # Container image definition
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
└── train.py                    # Training, evaluation, and MLflow logging
```

## Requirements

- Python 3.14
- pip
- Docker (optional, for containerized serving)
- MLflow 3.x with a running tracking server (required when training)

## Local setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start MLflow

`train.py` sends tracking requests to `http://localhost:5000`. Start an MLflow
server in a separate terminal before running training:

```powershell
mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlartifacts
```

The MLflow UI is available at <http://localhost:5000>.

## Train and register the model

With the MLflow server running:

```powershell
python train.py
```

The training script:

1. Loads the Iris dataset.
2. Splits it into training and test sets using `random_state=42`.
3. Trains a `RandomForestClassifier` with 10 estimators.
4. Rejects the model if test accuracy is below `0.90`.
5. Saves the model to `models/model.pkl`.
6. Logs parameters and metrics to the `Iris_Classification` MLflow experiment.
7. Registers the model as `IrisRandomForestModel` and assigns the `champion`
   alias to the new version.

## Run the API

The API loads `models/model.pkl` when it starts:

```powershell
uvicorn main:app --reload
```

The service is available at <http://127.0.0.1:8000>.

Interactive API documentation is available at:

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Returns a basic readiness response |
| `GET` | `/health` | Returns the service health status |
| `POST` | `/predict` | Predicts an Iris class from four measurements |

Example request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}'
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

All measurements must be positive numbers. Invalid input returns HTTP 422.

## Run tests

```powershell
python -m pytest -v
```

## Run with Docker

Build the image:

```powershell
docker build -t iris-ml-api .
```

Run the API:

```powershell
docker run --rm -p 8000:8000 iris-ml-api
```

Then open <http://127.0.0.1:8000/docs>.

## Continuous integration

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests
targeting `main`. It:

1. Sets up Python 3.14.
2. Installs `requirements.txt`.
3. Runs the pytest suite.
4. Builds the Docker image.

